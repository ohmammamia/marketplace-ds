"""Emerging-issue detection: is a category growing faster than the whole?

For each category compare the count in the recent window with the count
expected from the baseline window *scaled by total volume* (so a marketing
push that lifts everything is not an 'emerging issue'). Significance via a
one-sided Poisson test on the recent count against the expected rate;
we report the rate ratio and its p-value, and flag ratio ≥ threshold with
p < alpha. Weekly resolution; recent = 4 weeks, baseline = the 12 before."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import poisson


def weekly_counts(df: pd.DataFrame, cat_col: str = "category") -> pd.DataFrame:
    w = df.created_at.dt.to_period("W").dt.start_time
    return pd.crosstab(w, df[cat_col]).sort_index()


def detect(counts: pd.DataFrame, recent_weeks: int = 4, baseline_weeks: int = 12,
           ratio_threshold: float = 1.5, alpha: float = 0.05) -> pd.DataFrame:
    recent = counts.iloc[-recent_weeks:]
    base = counts.iloc[-(recent_weeks + baseline_weeks):-recent_weeks]
    tot_r, tot_b = recent.values.sum(), base.values.sum()
    rows = []
    for c in counts.columns:
        r, b = recent[c].sum(), base[c].sum()
        share_b = (b + 0.5) / (tot_b + 0.5 * counts.shape[1])
        expected = share_b * tot_r
        ratio = (r + 0.5) / (expected + 0.5)
        p = 1 - poisson.cdf(r - 1, expected)
        rows.append({"category": c, "recent_count": int(r), "baseline_count": int(b), "expected_recent": round(expected, 1),
                     "rate_ratio": round(ratio, 2), "p_value": round(float(p), 4),
                     "emerging": bool(ratio >= ratio_threshold and p < alpha)})
    return pd.DataFrame(rows).sort_values("rate_ratio", ascending=False)
