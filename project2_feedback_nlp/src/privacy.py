"""Privacy-aware preprocessing. Free text on a marketplace routinely contains
phone numbers, emails, postcodes and names. Detection is rule-based (high
recall on structured identifiers) plus a curated name gazetteer; anything
flagged is replaced by a typed placeholder BEFORE the text reaches any model,
notebook output or example list. Counts are reported; raw matches never are."""
from __future__ import annotations
import re
import pandas as pd

PATTERNS = {
    "PHONE": re.compile(r"(?:\+44\s?7\d{3}|\(?07\d{3}\)?)\s?\d{3}\s?\d{3}|\b0\d{10}\b"),
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "POSTCODE": re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b"),
    "URL": re.compile(r"https?://\S+|www\.\S+"),
}
# In production this gazetteer would be replaced by a NER model + the
# platform's own user-name table (names of *registered* users are exactly
# the ones most likely to appear). Kept simple and inspectable here.
NAME_GAZETTEER = {"ahmed", "priya", "tom", "olga", "kwame", "sophie", "marek", "fatima"}
_NAME_CTX = re.compile(r"\b(provider|customer|called|named|mr|mrs|ms)\s+([A-Z][a-z]+)\b")


def redact(t: str) -> tuple[str, dict]:
    counts = {}
    for label, pat in PATTERNS.items():
        t, n = pat.subn(f"[{label}]", t)
        if n:
            counts[label] = n
    def _name(m):
        counts["NAME"] = counts.get("NAME", 0) + 1
        return f"{m.group(1)} [NAME]"
    t = _NAME_CTX.sub(_name, t)
    toks = t.split()
    out = []
    for w in toks:
        if w.strip(",.!?").lower() in NAME_GAZETTEER:
            counts["NAME"] = counts.get("NAME", 0) + 1
            out.append("[NAME]")
        else:
            out.append(w)
    return " ".join(out), counts


def apply(df: pd.DataFrame, col: str = "text_clean") -> tuple[pd.DataFrame, dict]:
    d = df.copy()
    res = d[col].map(redact)
    d["text_redacted"] = res.map(lambda r: r[0])
    d["pii_types"] = res.map(lambda r: ",".join(sorted(r[1])) if r[1] else "")
    flagged = d.pii_types != ""
    per_type = {}
    for r in res:
        for k, v in r[1].items():
            per_type[k] = per_type.get(k, 0) + v
    report = {"n_docs": len(d), "share_docs_with_pii": float(flagged.mean()), "entities_by_type": per_type}
    return d, report
