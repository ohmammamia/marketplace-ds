"""Operational taxonomy of service issues.

The taxonomy was designed from (a) the NMF topics, (b) a manual read of a
stratified sample (in the demo: 150 redacted texts) and (c) the question
"who acts on this?" — each category maps to an owner and an action. Seed
keywords give a transparent rule baseline and the weak labels used to
bootstrap the classifier. Categories are deliberately few: a taxonomy the
operations lead cannot hold in their head will not be used."""
from __future__ import annotations
import re
import pandas as pd

TAXONOMY = {
    # category: (owner, action, seed keywords)
    "provider_no_show":        ("supply ops",  "warn / suspend provider",   ["didn't turn up", "no show", "nobody came", "cancelled", "late", "waited"]),
    "provider_communication":  ("supply ops",  "nudge provider, auto-refund lead", ["never contacted", "no reply", "never heard", "got in touch", "messages"]),
    "price_dispute":             ("trust & safety", "profile price audit",     ["charged", "price", "fees", "rates", "quoted", "£"]),
    "workmanship_quality":          ("product",     "surface in ratings",         ["thorough", "explained", "recommend", "rushed", "methodical", "clear plan", "worth every penny", "progress"]),
    "safety_issue":         ("trust & safety", "site safety check",    ["exposed wiring", "unsafe", "warning light", "dust sheets", "ladder", "certified", "debris"]),
    "lead_quality":              ("demand ops",  "lead validation rules",      ["lead", "customer never", "disconnected", "wasted credits", "don't even cover", "found someone", "sent to me"]),
    "credits_refund":            ("finance ops", "refund SLA / bug ticket",    ["refund", "credits", "charged for a lead", "disappeared"]),
    "app_bug":                   ("engineering", "bug ticket",                 ["app", "crashes", "button", "notifications", "greyed", "save"]),
}
CATEGORIES = list(TAXONOMY)

# Severity rubric (business policy, versioned with the taxonomy):
#   5 safety · 4 money/trust · 3 service failure · 2 experience · 1 cosmetic
SEVERITY = {
    "safety_issue": 5, "price_dispute": 4, "credits_refund": 4, "provider_no_show": 4,
    "provider_communication": 3, "lead_quality": 3, "app_bug": 3, "workmanship_quality": 2,
}


def keyword_scores(text: str) -> dict[str, int]:
    t = text.lower()
    return {c: sum(1 for kw in spec[2] if kw in t) for c, spec in TAXONOMY.items()}


def rule_label(text: str) -> str:
    s = keyword_scores(text)
    best = max(s, key=s.get)
    return best if s[best] > 0 else "unassigned"


def map_topics_to_taxonomy(topic_terms: list[list[str]]) -> pd.DataFrame:
    rows = []
    for k, terms in enumerate(topic_terms):
        joined = " ".join(terms)
        hits = {c: sum(1 for kw in spec[2] if any(w in kw or kw in w for w in terms)) for c, spec in TAXONOMY.items()}
        best = max(hits, key=hits.get)
        rows.append({"topic": k, "top_terms": ", ".join(terms), "mapped_category": best if hits[best] else "NEW/uncovered", "overlap": hits[best]})
    return pd.DataFrame(rows)
