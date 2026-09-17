"""Operational simulation: replay leads in time order, offer each to its
top-K ranked candidates subject to weekly provider capacity, and count
expected purchases. Used to compare ranking strategies and to stress-test
capacity / demand changes."""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd

from .evaluation import order_by_score


def simulate(df: pd.DataFrame, ins: pd.DataFrame, score_col: str, k: int = 3,
             capacity_scale: float = 1.0, demand_scale: float = 1.0, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    cap = (ins.set_index("provider_id").weekly_lead_capacity * capacity_scale).round().clip(lower=0).to_dict()
    leads = df[["lead_id", "created_at"]].drop_duplicates().sort_values("created_at")
    if demand_scale != 1.0:
        n = int(len(leads) * demand_scale)
        leads = leads.sample(n, replace=demand_scale > 1, random_state=seed).sort_values("created_at")
    used = defaultdict(lambda: defaultdict(int))  # week -> provider -> offers made
    exp_purchases, offered, unfilled = 0.0, 0, 0
    groups = {lid: order_by_score(g, score_col) for lid, g in df.groupby("lead_id")}
    for _, ld in leads.iterrows():
        wk = ld.created_at.to_period("W")
        g = groups[ld.lead_id]
        made = 0
        for _, row in g.iterrows():
            if made == k:
                break
            if used[wk][row.provider_id] < cap.get(row.provider_id, 0):
                used[wk][row.provider_id] += 1
                exp_purchases += row.purchased  # observed outcome of a pair that was actually shown
                made += 1
        offered += made
        unfilled += int(made == 0)
    return {"strategy": score_col, "k": k, "capacity_scale": capacity_scale, "demand_scale": demand_scale,
            "leads": len(leads), "offers_made": offered, "leads_unfilled": unfilled,
            "expected_purchases": float(exp_purchases), "purchases_per_lead": float(exp_purchases / len(leads))}
