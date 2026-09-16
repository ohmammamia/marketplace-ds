"""Candidate generation: hard eligibility filters applied BEFORE ranking.
Ranking every provider for every lead is wasteful and produces infeasible
matches (wrong job_type, outside coverage). The filters are transparent
business rules; the ranker only reorders what survives them."""
from __future__ import annotations
import pandas as pd
from common.synthetic import _dist


def job_type_ok(lead_t: str, ins_t: str) -> bool:
    return ins_t == "both" or ins_t == lead_t


def availability_overlap(lead: pd.Series, ins: pd.Series) -> int:
    return int(sum(int(lead[c]) & int(ins[c]) for c in ["weekday_day", "weekday_eve", "weekend"]))


def generate(lead: pd.Series, ins: pd.DataFrame, max_distance_km: float = 8.0,
             relax_if_empty: bool = True) -> pd.DataFrame:
    """Eligible providers for one lead, with a `candidate_reason` column.
    Filters: coverage OR within max_distance; job_type compatible; gender
    preference honoured; at least one availability slot in common (relaxed
    if it empties the set — a lead with no candidates is a lost lead)."""
    c = ins[ins.job_type.apply(lambda t: job_type_ok(lead.job_type, t))].copy()
    c["distance_km"] = c.home_zone.apply(lambda h: _dist(h, lead.zone))
    c = c[c.coverage.str.contains(lead.zone) | (c.distance_km <= max_distance_km)]
    if lead.pref_gender != "any":
        c = c[c.gender == lead.pref_gender]
    c["availability_overlap"] = c.apply(lambda r: availability_overlap(lead, r), axis=1) if len(c) else 0
    strict = c[c.availability_overlap > 0]
    if len(strict) or not relax_if_empty:
        return strict.assign(candidate_reason="strict")
    return c.assign(candidate_reason="availability_relaxed")


def coverage_report(leads: pd.DataFrame, ins: pd.DataFrame, **kw) -> dict:
    sizes, relaxed = [], 0
    for _, ld in leads.iterrows():
        c = generate(ld, ins, **kw)
        sizes.append(len(c))
        relaxed += int(len(c) > 0 and (c.candidate_reason == "availability_relaxed").any())
    s = pd.Series(sizes)
    return {"n_leads": len(leads), "share_with_candidates": float((s > 0).mean()),
            "share_relaxed": relaxed / len(leads), "candidate_size_median": float(s.median()),
            "candidate_size_p10": float(s.quantile(.1)), "candidate_size_p90": float(s.quantile(.9))}
