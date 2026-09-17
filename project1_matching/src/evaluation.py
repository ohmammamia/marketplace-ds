"""Ranking evaluation grouped by lead. Accuracy is deliberately absent."""
from __future__ import annotations
import numpy as np
import pandas as pd


def order_by_score(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Rank rows by score, breaking ties deterministically on `provider_id`.

    `sort_values` leaves tied rows in input order, which here is the order
    offers were logged — i.e. the legacy `rank_shown` order. Any heavily-tied
    score therefore silently collapses onto the legacy baseline (`s_nearest`
    takes ~2 distinct distances per lead, so it reproduced legacy order on
    100% of leads), and its metrics move with row order rather than with the
    score. An explicit tie-break makes every strategy independent and stable.
    """
    return df.sort_values([score_col, "provider_id"], ascending=[False, True], kind="mergesort")


def _dcg(rel: np.ndarray) -> float:
    return float(np.sum(rel / np.log2(np.arange(2, len(rel) + 2))))


def ranking_metrics(df: pd.DataFrame, score_col: str, label: str = "purchased", k: int = 3) -> dict:
    p, r, n, hits = [], [], [], 0
    for _, g in df.groupby("lead_id"):
        if g[label].sum() == 0:
            continue                       # no positive: metric undefined for this query
        g = order_by_score(g, score_col)
        top = g[label].values[:k]
        # precision@k divides by k, not by len(top): a lead with fewer than k
        # candidates must not score a full 1.0 for a single hit.
        p.append(top.sum() / k); r.append(top.sum() / g[label].sum())
        ideal = np.sort(g[label].values)[::-1][:k]
        n.append(_dcg(top) / (_dcg(ideal) or 1))
    return {f"precision@{k}": float(np.mean(p)), f"recall@{k}": float(np.mean(r)),
            f"ndcg@{k}": float(np.mean(n)), "n_queries": len(p)}


def coverage(df: pd.DataFrame, score_col: str, k: int = 3) -> float:
    """Share of providers that appear at least once in any top-k list."""
    top = order_by_score(df, score_col).groupby("lead_id").head(k)
    return float(top.provider_id.nunique() / df.provider_id.nunique())


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return pd.DataFrame({"bin": idx, "y": y, "p": p}).groupby("bin").agg(n=("y", "size"), observed=("y", "mean"), predicted=("p", "mean"))
