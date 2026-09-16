"""Load, validate and clean the matching inputs. Cleaning steps are logged and
counted so the decision log can cite them."""
from __future__ import annotations
import pandas as pd
from common.validation import Schema, validate
from common.monitoring import get_logger

log = get_logger("p1.data")

PROVIDER_SCHEMA = Schema(
    required={"provider_id": "O", "home_zone": "O", "coverage": "O", "job_type": "O",
              "price_per_hour": "f", "tier": "O", "years_experience": "i", "gender": "O",
              "weekly_lead_capacity": "i", "rating": "f", "credit_balance": "i"},
    key="provider_id",
    ranges={"price_per_hour": (15, 80), "rating": (1, 5), "years_experience": (0, 50), "weekly_lead_capacity": (0, 20)},
    categories={"job_type": {"repair", "install", "both"}, "tier": {"accredited", "apprentice"}},
    max_missing={"rating": 0.10, "price_per_hour": 0.10},
)
LEAD_SCHEMA = Schema(
    required={"lead_id": "O", "created_at": "M", "zone": "O", "job_type": "O",
              "budget_max": "f", "urgency": "O", "hours_wanted": "i"},
    key="lead_id",
    ranges={"budget_max": (10, 100), "hours_wanted": (1, 100)},
    categories={"job_type": {"repair", "install"}, "urgency": {"none", "deadline", "emergency"}},
    max_missing={"budget_max": 0.10},
)
OFFER_SCHEMA = Schema(
    required={"lead_id": "O", "provider_id": "O", "offered_at": "M", "rank_shown": "i",
              "purchased": "i", "contacted": "i", "converted": "i"},
    ranges={"rank_shown": (1, 20), "purchased": (0, 1), "contacted": (0, 1), "converted": (0, 1)},
)


def load(data_dir: str):
    ins = pd.read_csv(f"{data_dir}/providers.csv", parse_dates=["joined_at"])
    leads = pd.read_csv(f"{data_dir}/leads.csv", parse_dates=["created_at"])
    offers = pd.read_csv(f"{data_dir}/offers.csv", parse_dates=["offered_at"])
    return ins, leads, offers


def clean(ins: pd.DataFrame, leads: pd.DataFrame, offers: pd.DataFrame):
    reports = {}
    reports["providers_raw"] = validate(ins, PROVIDER_SCHEMA, "providers_raw")
    n0 = len(ins)
    ins = ins.copy()
    ins["job_type"] = ins["job_type"].str.lower()                   # inconsistent casing
    # duplicate registrations: same person, two ids. Resolved by an exact
    # match on stable attributes (see docs/methodology.md, "Duplicate providers").
    dup_cols = ["home_zone", "coverage", "job_type", "price_per_hour", "years_experience", "gender", "joined_at"]
    ins = ins.sort_values("provider_id").drop_duplicates(subset=dup_cols, keep="first")
    log.info("providers: %d -> %d after de-duplication", n0, len(ins))
    reports["leads_raw"] = validate(leads, LEAD_SCHEMA, "leads_raw")
    n0 = len(leads)
    leads = leads[leads.zone != "UNKNOWN"].copy()
    log.info("leads: %d -> %d after dropping unknown zone", n0, len(leads))
    reports["offers_raw"] = validate(offers, OFFER_SCHEMA, "offers_raw",
                                     references={"provider_id": ins.provider_id, "lead_id": leads.lead_id})
    n0 = len(offers)
    offers = offers[offers.provider_id.isin(ins.provider_id) & offers.lead_id.isin(leads.lead_id)].copy()
    log.info("offers: %d -> %d after referential-integrity filter", n0, len(offers))
    reports["providers_clean"] = validate(ins, PROVIDER_SCHEMA, "providers_clean")
    reports["providers_clean"].raise_if_failed()
    reports["offers_clean"] = validate(offers, OFFER_SCHEMA, "offers_clean",
                                       references={"provider_id": ins.provider_id, "lead_id": leads.lead_id})
    reports["offers_clean"].raise_if_failed()
    return ins.reset_index(drop=True), leads.reset_index(drop=True), offers.reset_index(drop=True), reports
