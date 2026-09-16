"""Operational prioritisation.

priority = share × severity × growth × confidence

  share      — recent share of categorised feedback (frequency, volume-normalised)
  severity   — 1..5 from a written rubric (docs/methodology.md): 5 = safety,
               4 = money / trust, 3 = service failure, 2 = experience, 1 = cosmetic.
               The rubric is the *business's* judgement, made explicit and versioned,
               not a learnt weight.
  growth     — max(1, rate_ratio) from emerging-issue detection; growth never
               *reduces* priority, it only amplifies it.
  confidence — mean classifier confidence for the category: we do not want to
               escalate a theme the model itself is unsure about.

Multiplicative so that a zero on any axis cannot be compensated by the others.
Weights are not tuned — there is no ground truth for 'correct priority'; the
formula is a transparent policy, reviewable by the ops lead."""
from __future__ import annotations
import pandas as pd
from .taxonomy import SEVERITY


def compute(pred: pd.DataFrame, emerging: pd.DataFrame, recent_weeks: int = 4) -> pd.DataFrame:
    d = pred[pred.category != "needs_review"]
    cutoff = d.created_at.max() - pd.Timedelta(weeks=recent_weeks)
    rec = d[d.created_at > cutoff]
    share = rec.category.value_counts(normalize=True).rename("share")
    conf = rec.groupby("category").confidence.mean().rename("confidence")
    out = pd.concat([share, conf], axis=1).join(emerging.set_index("category")[["rate_ratio", "emerging"]]).fillna({"rate_ratio": 1})
    out["severity"] = out.index.map(lambda c: SEVERITY.get(c, 2))
    out["growth"] = out.rate_ratio.clip(lower=1)
    out["priority"] = out.share * out.severity * out.growth * out.confidence
    out["rank"] = out.priority.rank(ascending=False).astype(int)
    return out.sort_values("priority", ascending=False).round(3)
