"""Emerging-issue detection: is a category growing faster than the whole?

For each category compare the count in the recent window with the count
expected from the baseline window *scaled by total volume* (so a marketing
push that lifts everything is not an 'emerging issue'). Significance via a
one-sided Poisson test on the recent count against the expected rate; we
report the rate ratio, its p-value and a Benjamini-Hochberg q-value, and flag
ratio ≥ threshold with q < alpha. Weekly resolution; recent = 4 weeks,
baseline = the 12 before.

One test is run per category, so the raw p-values cannot be read at face
value: at eight categories and alpha = 0.05 the chance of at least one false
flag is ~34%. Flagging is therefore driven by the BH q-value, which controls
the false-discovery rate across the family. The Poisson model also treats the
baseline rate as known and assumes no overdispersion, both of which make the
p-values slightly optimistic — see docs/limitations.md."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, poisson


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
                     "rate_ratio": round(ratio, 2), "p_value": round(float(p), 4)})
    df = pd.DataFrame(rows)
    # BH across the one-test-per-category family
    df["q_value"] = false_discovery_control(df.p_value.values, method="bh").round(4)
    df["emerging"] = (df.rate_ratio >= ratio_threshold) & (df.q_value < alpha)
    return df.sort_values("rate_ratio", ascending=False)
