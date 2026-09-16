"""Greedy baseline = what the platform does today, formalised: leads in
arrival order, each notified to its best M feasible providers with spare
capacity. Myopic: an early lead can consume a scarce provider that a later,
better-matched (or priority) lead needed."""
import pandas as pd
from .formulation import Instance


def solve(inst: Instance) -> pd.DataFrame:
    cap = inst.capacity.to_dict()
    pairs = inst.pairs.sort_values("q", ascending=False)
    by_lead = {l: g for l, g in pairs.groupby("lead_id")}
    rows = []
    for lid in inst.leads.sort_values("created_at").lead_id:
        n = 0
        for _, r in by_lead.get(lid, pd.DataFrame()).iterrows():
            if n == inst.M:
                break
            if cap.get(r.provider_id, 0) > 0:
                cap[r.provider_id] -= 1; n += 1
                rows.append((lid, r.provider_id))
    return pd.DataFrame(rows, columns=["lead_id", "provider_id"])
