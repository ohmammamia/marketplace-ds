"""Text quality layer: every exclusion is counted and reported, nothing is
silently dropped."""
from __future__ import annotations
import re
import pandas as pd
from common.synthetic import BOILERPLATE

_WS = re.compile(r"\s+")
STOP_EN = {"the", "a", "and", "to", "i", "my", "for", "of", "it", "was", "me", "on", "in", "he", "she", "not", "no"}


def normalise(t: str) -> str:
    t = str(t) if pd.notna(t) else ""
    for b in BOILERPLATE:
        t = t.replace(b, " ")
    return _WS.sub(" ", t).strip()


def looks_english(t: str) -> bool:
    toks = re.findall(r"[a-z']+", t.lower())
    return len(toks) == 0 or sum(w in STOP_EN for w in toks) / len(toks) > 0.04 or len(toks) < 6


def apply(df: pd.DataFrame, min_tokens: int = 3) -> tuple[pd.DataFrame, dict]:
    d = df.copy()
    d["text_clean"] = d.text.map(normalise)
    d["n_tokens"] = d.text_clean.str.split().str.len().fillna(0).astype(int)
    d["_key"] = d.text_clean.str.lower()
    reasons = pd.Series("", index=d.index)
    reasons[d.text_clean == ""] = "empty"
    reasons[(reasons == "") & (d.n_tokens < min_tokens)] = "too_short"
    reasons[(reasons == "") & d._key.duplicated(keep="first")] = "exact_duplicate"
    reasons[(reasons == "") & ~d.text_clean.map(looks_english)] = "non_english"
    d["exclusion_reason"] = reasons
    report = {"n_input": len(d), "n_kept": int((reasons == "").sum()),
              "exclusions": reasons[reasons != ""].value_counts().to_dict(),
              "median_tokens_kept": float(d.loc[reasons == "", "n_tokens"].median())}
    return d.drop(columns="_key"), report
