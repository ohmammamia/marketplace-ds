"""Privacy-aware preprocessing. Free text on a marketplace routinely contains
phone numbers, emails, postcodes and names. Detection is rule-based (high
recall on structured identifiers) plus a curated name gazetteer; anything
flagged is replaced by a typed placeholder BEFORE the text reaches any model,
notebook output or example list. Counts are reported; raw matches never are."""
from __future__ import annotations
import re
import pandas as pd

# Separators: UK numbers are written with spaces, hyphens, dots or brackets.
_SEP = r"[\s.\-]?"
PATTERNS = {
    # mobile (07… / +447…) and landline (+44 1–2… / 0…), each allowing separators
    "PHONE": re.compile(
        r"(?:\+44\s?\(?0?\)?\s?|\b0)"                      # +44, +44(0), or leading 0
        rf"(?:\d{{2,5}}{_SEP}\d{{3,4}}{_SEP}\d{{3,4}}|\d{{9,10}})\b"
    ),
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    # case-insensitive: users type postcodes in lower case at least as often
    "POSTCODE": re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b", re.IGNORECASE),
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
    # Gazetteer pass. Strip surrounding punctuation *and* a possessive suffix
    # ("Tom's" previously survived), and rebuild the token with its trimmings
    # so the redaction does not silently reflow the text.
    out = []
    for w in t.split():
        core = w.strip(",.!?;:()\"'")
        stem = core[:-2] if core.lower().endswith("'s") else core
        if stem.lower() in NAME_GAZETTEER:
            counts["NAME"] = counts.get("NAME", 0) + 1
            out.append(w.replace(stem, "[NAME]", 1))
        else:
            out.append(w)
    return " ".join(out), counts


def residual_pii(texts: pd.Series) -> pd.Series:
    """Rows where a structured identifier survived redaction.

    The gate reuses PATTERNS itself: the previous check looked only for
    `\d{10}|@`, which a separated phone number ("07700-900-123") or a
    lower-case postcode passes untouched, so redaction gaps could not fail
    the run they were meant to catch.
    """
    hit = pd.Series(False, index=texts.index)
    for pat in PATTERNS.values():
        hit |= texts.str.contains(pat, regex=True, na=False)
    return hit


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
