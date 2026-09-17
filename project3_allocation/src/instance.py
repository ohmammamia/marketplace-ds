"""Build a weekly allocation instance from the cleaned data, reusing Project 1's
candidate generation and ranker as the match-quality signal (a real
cross-project data product dependency)."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from common.monitoring import get_logger
from project1_matching.src import data as p1data, candidates, features, baseline, ranker
from .formulation import Instance

log = get_logger("p3.instance")


def load_quality_model(path: str):
    if os.path.exists(path):
        import joblib
        return joblib.load(path)
    log.warning("no Project 1 model at %s — falling back to rule score", path)
    return None


def build(data_dir: str, week_start: str, model_path: str, M: int = 3, weights: dict | None = None,
          demand_scale: float = 1.0, capacity_scale: float = 1.0, seed: int = 0) -> Instance:
    """Input validation runs inside `p1data.clean`, which raises on any error
    in the cleaned frames. The reports are kept on the instance so the pipeline
    can publish them as an artefact rather than discarding the evidence."""
    ins, leads, offers, reports = p1data.clean(*p1data.load(data_dir))
    ws = pd.Timestamp(week_start)
    wk = leads[(leads.created_at >= ws) & (leads.created_at < ws + pd.Timedelta(days=7))].copy()
    if demand_scale != 1.0:
        wk = wk.sample(int(len(wk) * demand_scale), replace=demand_scale > 1, random_state=seed)
        wk["lead_id"] = [f"{l}_{k}" for k, l in enumerate(wk.lead_id)]     # resampled leads need unique ids
    ins = ins.copy()
    ins["weekly_lead_capacity"] = (ins.weekly_lead_capacity * capacity_scale).round().astype(int)
    model = load_quality_model(model_path)
    # scoring week `ws` must not see offers from `ws` onwards: taking the prior
    # from the whole table let future weeks inform the current allocation
    prior = features.prevailing_prior(offers, ws)
    joined = ins[ins.joined_at < ws]
    med_price = (joined if len(joined) else ins).price_per_hour.median()
    hist = features.provider_history(offers, ws, prior)
    rows = []
    for _, ld in wk.iterrows():
        c = candidates.generate(ld, ins)
        if c.empty:
            continue
        f = features.build_pair_features(ld, c, hist, prior, med_price)
        q = ranker.score(model, f) if model is not None else 1 / (1 + np.exp(-baseline.rule_score(f)))
        rows.append(pd.DataFrame({"lead_id": ld.lead_id, "provider_id": c.provider_id.values, "q": q, "distance_km": c.distance_km.values}))
    pairs = pd.concat(rows, ignore_index=True)
    log.info("week %s: %d leads, %d providers, %d feasible pairs", ws.date(), len(wk), len(ins), len(pairs))
    return Instance(leads=wk.reset_index(drop=True), providers=ins.reset_index(drop=True), pairs=pairs, M=M,
                    weights=weights or Instance.__dataclass_fields__["weights"].default_factory(),
                    validation=reports)
