"""Exposure fairness: does the ranker systematically push a group of
providers out of the top-K relative to their share of the eligible pool?
Groups here are supply-side attributes the business cares about (apprentice vs accredited
— new entrants — and gender). Customer gender preference is a legitimate
constraint and is honoured at candidate generation, not here."""
from __future__ import annotations
import pandas as pd

from .evaluation import order_by_score


def exposure_by_group(df: pd.DataFrame, ins: pd.DataFrame, score_col: str, group: str, k: int = 3) -> pd.DataFrame:
    d = df.merge(ins[["provider_id", group]], on="provider_id", how="left")
    pool = d.groupby(group).size() / len(d)
    top = order_by_score(d, score_col).groupby("lead_id").head(k)
    exp = top.groupby(group).size() / len(top)
    out = pd.DataFrame({"pool_share": pool, "topk_share": exp}).fillna(0)
    out["exposure_ratio"] = out.topk_share / out.pool_share
    out["observed_purchase_rate"] = d.groupby(group).purchased.mean()
    return out
