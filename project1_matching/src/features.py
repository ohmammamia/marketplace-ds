"""Pairwise (lead, provider) features.

Leakage guard: provider history features (purchase / conversion rates) are
computed ONLY from offers strictly before the lead's `created_at`, and are
shrunk towards the prevailing prior so that new providers are not penalised
or inflated by 1–2 observations (empirical-Bayes cold start).

The shrinkage prior is itself computed strictly before the lead's week and
shrunk towards 0.5. Taking it from the whole offer table — as this module
previously did — leaked the test-period purchase rate into training features,
which contradicted the guarantee above even though the per-provider history
respected it.

`hist_n_offers_log` is computed for diagnostics but excluded from FEATURES:
it grows monotonically with platform age, so it drifts by construction
(PSI 2.7 train→test in the first run) — see decision log D7."""
from __future__ import annotations
import numpy as np
import pandas as pd
from common.synthetic import _dist
from .candidates import availability_overlap

FEATURES = ["distance_km", "availability_overlap", "price_gap_pos", "price_missing", "has_credit",
            "credit_balance_log", "urgent", "emergency", "lang_match", "gender_pref_met",
            "rating_f", "years_exp_cap", "is_apprentice", "hist_purchase_rate",
            "tenure_days_log", "hours_wanted"]


def prevailing_prior(offers: pd.DataFrame, as_of: pd.Timestamp, prior_n: float = 10.0) -> float:
    """Purchase rate over offers strictly before `as_of`, shrunk towards 0.5.

    Shrinkage keeps the earliest weeks usable: week 1 sees no offers at all and
    week 2 only ~10, so a raw mean would swing wildly before settling.
    """
    past = offers.loc[offers.offered_at < as_of, "purchased"]
    return float((past.sum() + 0.5 * prior_n) / (len(past) + prior_n))


def provider_history(offers: pd.DataFrame, as_of: pd.Timestamp, prior_rate: float, prior_n: float = 10.0) -> pd.DataFrame:
    """prior_n=10 ≈ two weeks of offers for a typical provider."""
    h = offers[offers.offered_at < as_of].groupby("provider_id").agg(n=("purchased", "size"), k=("purchased", "sum"))
    h["hist_purchase_rate"] = (h.k + prior_rate * prior_n) / (h.n + prior_n)
    h["hist_n_offers"] = h.n
    return h[["hist_purchase_rate", "hist_n_offers"]]


def build_pair_features(lead: pd.Series, cands: pd.DataFrame, history: pd.DataFrame,
                        prior_rate: float, median_price: float) -> pd.DataFrame:
    f = pd.DataFrame(index=cands.index)
    f["distance_km"] = cands.get("distance_km", cands.home_zone.apply(lambda h: _dist(h, lead.zone)))
    f["availability_overlap"] = cands.get("availability_overlap", cands.apply(lambda r: availability_overlap(lead, r), axis=1))
    budget = lead.budget_max if pd.notna(lead.budget_max) else np.nan
    price = cands.price_per_hour
    f["price_missing"] = (price.isna() | pd.isna(budget)).astype(int)
    f["price_gap_pos"] = np.clip((price.fillna(median_price) - (budget if pd.notna(budget) else median_price)) / 10, 0, None)
    f["has_credit"] = (cands.credit_balance > 0).astype(int)
    f["credit_balance_log"] = np.log1p(cands.credit_balance)
    f["urgent"] = int(lead.urgency != "none")
    f["emergency"] = int(lead.urgency == "emergency")
    f["lang_match"] = (cands.language == lead.language).astype(int)
    f["gender_pref_met"] = ((lead.pref_gender == "any") | (cands.gender == lead.pref_gender)).astype(int)
    f["rating_f"] = cands.rating.fillna(4.5)
    f["years_exp_cap"] = np.minimum(cands.years_experience, 10)
    f["is_apprentice"] = (cands.tier == "apprentice").astype(int)
    h = history.reindex(cands.provider_id)
    f["hist_purchase_rate"] = h.hist_purchase_rate.fillna(prior_rate).values
    f["hist_n_offers_log"] = np.log1p(h.hist_n_offers.fillna(0).values)
    f["tenure_days_log"] = np.log1p((lead.created_at - cands.joined_at).dt.days.clip(lower=0))
    f["hours_wanted"] = lead.hours_wanted
    f["lead_id"] = lead.lead_id
    f["provider_id"] = cands.provider_id.values
    return f


def build_training_set(leads: pd.DataFrame, ins: pd.DataFrame, offers: pd.DataFrame) -> pd.DataFrame:
    """Training rows = historical offers (the only pairs with observed outcomes).
    Selection bias caveat is discussed in docs/limitations.md."""
    ins_idx = ins.set_index("provider_id")
    rows = []
    # prior, history and price median are all recomputed per week, from data
    # strictly before that week, so no future information reaches any feature
    leads = leads.assign(week=leads.created_at.dt.to_period("W").dt.start_time)
    for week, ld_w in leads.groupby("week"):
        prior = prevailing_prior(offers, week)
        joined = ins[ins.joined_at < week]
        med_price = (joined if len(joined) else ins).price_per_hour.median()
        hist = provider_history(offers, week, prior)
        for _, ld in ld_w.iterrows():
            o = offers[offers.lead_id == ld.lead_id]
            if o.empty:
                continue
            cands = ins_idx.loc[o.provider_id].reset_index()
            f = build_pair_features(ld, cands, hist, prior, med_price)
            f["rank_shown"] = o.rank_shown.values
            f["purchased"] = o.purchased.values
            f["converted"] = o.converted.values
            f["created_at"] = ld.created_at
            rows.append(f)
    return pd.concat(rows, ignore_index=True)
