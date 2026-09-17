"""
SYNTHETIC / DEMO DATA GENERATOR — two-sided lead marketplace.

Everything produced here is SYNTHETIC. There is no underlying real dataset:
every row is sampled from the distributions defined in this file. The
generator builds the *schema and analytical structure* of a pay-per-lead
marketplace (customer job request -> offered to providers -> provider pays
for the lead -> contact -> conversion) so that the pipelines have a
realistic, fully reproducible substrate to run against.

Domain (home-trades marketplace, a fictional metropolitan area):
  - customers submit a job lead: service zone, job type, availability,
    budget, urgency (deadline / emergency), estimated hours, optional
    preferences (provider gender, language);
  - providers (tradespeople) have coverage zones, job types, an hourly
    rate, an accreditation tier, experience, availability, weekly capacity
    and a prepaid credit balance (they pay per lead);
  - a lead is *offered* to a shortlist of providers; a provider may
    *purchase* it, then *contact* the customer and *convert* (job booked).

The geography is a fictional grid of service zones (Z01..Z30) with (x, y)
coordinates in kilometres; it does not correspond to any real place.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ------------------------------------------------------------------ geography
# A compact, fictional grid of service zones with (x, y) km coordinates.
# Distances are Euclidean on this grid.
ZONES = {
    "Z01": (8, 8), "Z02": (7, 7), "Z03": (9, 6), "Z04": (4, 8), "Z05": (5, 10),
    "Z06": (3, 10), "Z07": (3, 6), "Z08": (1, 3), "Z09": (3, 3), "Z10": (5, 3),
    "Z11": (8, 3), "Z12": (10, 4), "Z13": (12, 2), "Z14": (13, 6), "Z15": (16, 8),
    "Z16": (6, 14), "Z17": (5, 12), "Z18": (6, 11), "Z19": (2, 12), "Z20": (1, 10),
    "Z21": (4, -6), "Z22": (4, -9), "Z23": (8, -8), "Z24": (8, -12), "Z25": (-2, -14),
    "Z26": (-4, -8), "Z27": (-2, -5), "Z28": (2, -4), "Z29": (0, -1), "Z30": (-9, 2),
}
ZONE_NAMES = list(ZONES)
LANGS = ["en", "en", "en", "en", "pl", "ur", "ro", "tr"]
THEME_START = datetime(2025, 9, 1)


def _dist(a: str, b: str) -> float:
    (x1, y1), (x2, y2) = ZONES[a], ZONES[b]
    return float(np.hypot(x1 - x2, y1 - y2))


def zone_distance_matrix() -> pd.DataFrame:
    m = np.array([[_dist(a, b) for b in ZONE_NAMES] for a in ZONE_NAMES])
    return pd.DataFrame(m, index=ZONE_NAMES, columns=ZONE_NAMES)


# ------------------------------------------------------------------ entities
def make_providers(n: int, rng: np.random.Generator) -> pd.DataFrame:
    home = rng.choice(ZONE_NAMES, n)
    rows = []
    for i in range(n):
        h = home[i]
        near = sorted(ZONE_NAMES, key=lambda d: _dist(h, d))[: rng.integers(3, 8)]
        tier = rng.choice(["accredited", "apprentice"], p=[0.8, 0.2])
        years = int(rng.integers(0, 3)) if tier == "apprentice" else int(rng.integers(1, 25))
        rows.append(dict(
            provider_id=f"P{i:04d}",
            home_zone=h,
            coverage=",".join(near),
            job_type=rng.choice(["repair", "install", "both"], p=[0.45, 0.35, 0.20]),
            price_per_hour=float(np.round(rng.normal(36 + 0.3 * years, 4), 0)),
            tier=tier,
            years_experience=years,
            gender=rng.choice(["F", "M"], p=[0.3, 0.7]),
            language=rng.choice(LANGS),
            weekday_day=int(rng.random() < 0.85),
            weekday_eve=int(rng.random() < 0.5),
            weekend=int(rng.random() < 0.6),
            weekly_lead_capacity=int(rng.integers(1, 6)),
            rating=float(np.clip(rng.normal(4.6, 0.3), 3.0, 5.0)),
            credit_balance=int(max(0, rng.integers(-8, 60))),   # ~12% of providers have no credits
            joined_at=THEME_START - timedelta(days=int(rng.integers(0, 700))),
            # latent responsiveness, unobservable in a real system: drives outcomes
            _responsiveness=float(np.clip(rng.beta(4, 3), 0.05, 0.98)),
        ))
    df = pd.DataFrame(rows)
    # data-quality realism: missing values and inconsistent categories
    df.loc[rng.random(n) < 0.06, "rating"] = np.nan
    df.loc[rng.random(n) < 0.04, "price_per_hour"] = np.nan
    mask = rng.random(n) < 0.03
    df.loc[mask, "job_type"] = df.loc[mask, "job_type"].str.upper()
    # a few duplicate provider rows (same person registered twice)
    dupes = df.sample(int(n * 0.02), random_state=int(rng.integers(0, 1e6))).copy()
    dupes["provider_id"] = dupes["provider_id"] + "_dup"
    return pd.concat([df, dupes], ignore_index=True)


def make_leads(n: int, rng: np.random.Generator, weeks: int = 26) -> pd.DataFrame:
    rows = []
    for i in range(n):
        # demand ramps up over time (marketing) with a seasonal bump
        w = int(min(weeks - 1, rng.beta(2.2, 1.6) * weeks))
        created = THEME_START + timedelta(days=w * 7 + int(rng.integers(0, 7)))
        rows.append(dict(
            lead_id=f"L{i:05d}",
            created_at=created,
            zone=rng.choice(ZONE_NAMES),
            job_type=rng.choice(["repair", "install"], p=[0.55, 0.45]),
            weekday_day=int(rng.random() < 0.4),
            weekday_eve=int(rng.random() < 0.6),
            weekend=int(rng.random() < 0.55),
            budget_max=float(np.round(rng.normal(38, 6), 0)),
            urgency=rng.choice(["none", "deadline", "emergency"], p=[0.6, 0.3, 0.1]),
            hours_wanted=int(rng.choice([10, 20, 30, 40], p=[0.3, 0.4, 0.2, 0.1])),
            pref_gender=rng.choice(["any", "F"], p=[0.85, 0.15]),
            language=rng.choice(LANGS),
            is_returning=int(rng.random() < 0.08),
            source=rng.choice(["paid_search", "organic", "referral"], p=[0.6, 0.3, 0.1]),
        ))
    df = pd.DataFrame(rows).sort_values("created_at").reset_index(drop=True)
    df.loc[rng.random(n) < 0.05, "budget_max"] = np.nan
    df.loc[rng.random(n) < 0.02, "zone"] = "UNKNOWN"
    return df


def _availability_overlap(lead: pd.Series, prov: pd.Series) -> int:
    return int(sum(int(lead[c]) & int(prov[c]) for c in ["weekday_day", "weekday_eve", "weekend"]))


def _job_type_ok(lead_t: str, prov_t: str) -> bool:
    prov_t = str(prov_t).lower()
    return prov_t == "both" or prov_t == lead_t


def make_offers(leads: pd.DataFrame, providers: pd.DataFrame, rng: np.random.Generator,
                shortlist: int = 5) -> pd.DataFrame:
    """Historical offer log produced by the marketplace's *legacy* rule (nearest
    providers with a compatible job type). Outcomes are generated from a latent
    model so that the pipelines have real signal to learn."""
    prov = providers[~providers.provider_id.str.endswith("_dup")].copy()
    prov["job_type"] = prov["job_type"].str.lower()
    prov["price_f"] = prov["price_per_hour"].fillna(prov["price_per_hour"].median())
    rows = []
    for _, ld in leads.iterrows():
        if ld.zone == "UNKNOWN":
            continue
        cands = prov[prov.job_type.apply(lambda t: _job_type_ok(ld.job_type, t))]
        cands = cands[cands.coverage.str.contains(ld.zone)]
        if cands.empty:
            continue
        d = cands.home_zone.apply(lambda h: _dist(h, ld.zone))
        # legacy rule: notify up to `shortlist` providers drawn from the
        # nearest 2*shortlist compatible ones (order of notification = rank)
        pool = cands.assign(_d=d).nsmallest(min(2 * shortlist, len(cands)), "_d")
        cands = pool.sample(min(shortlist, len(pool)), random_state=int(rng.integers(0, 1e9))).sort_values("_d")
        for rank, (_, prov_row) in enumerate(cands.iterrows(), start=1):
            av = _availability_overlap(ld, prov_row)
            budget = ld.budget_max if not np.isnan(ld.budget_max) else 38.0
            price_gap = (prov_row.price_f - budget) / 10.0
            # latent purchase propensity (provider decides to buy the lead)
            z = (0.9 - 0.18 * prov_row._d + 0.5 * av - 0.6 * max(price_gap, 0)
                 + 0.9 * (prov_row.credit_balance > 0) + 0.7 * (ld.urgency != "none")
                 - 0.10 * rank + 2.2 * prov_row._responsiveness
                 + 0.3 * (ld.language == prov_row.language)
                 - 0.8 * (ld.pref_gender == "F" and prov_row.gender != "F"))
            p_buy = 1 / (1 + np.exp(-z + 3.1))
            bought = rng.random() < p_buy
            contacted = bought and rng.random() < (0.5 + 0.45 * prov_row._responsiveness)
            rating = prov_row.rating if not np.isnan(prov_row.rating) else 4.5
            p_conv = 0.25 + 0.15 * (rating - 4.5) + 0.02 * min(prov_row.years_experience, 10) - 0.15 * max(price_gap, 0)
            converted = contacted and rng.random() < np.clip(p_conv, 0.05, 0.9)
            rows.append(dict(
                lead_id=ld.lead_id, provider_id=prov_row.provider_id,
                offered_at=ld.created_at, rank_shown=rank,
                purchased=int(bought), contacted=int(contacted), converted=int(converted),
                lead_price_gbp=float(rng.choice([8, 10, 12, 15])),
            ))
    df = pd.DataFrame(rows)
    # orphan rows to exercise referential-integrity checks
    orphan = df.sample(3, random_state=1).copy()
    orphan["provider_id"] = "P9999"
    return pd.concat([df, orphan], ignore_index=True)


# ------------------------------------------------------------------ feedback text
THEMES = {
    "provider_no_show": [
        "Provider {name} didn't turn up for the first visit and never called",
        "Booked a morning slot, waited 40 minutes, nobody came. No explanation",
        "Cancelled on me twice last minute, really unreliable",
        "Was late again by half an hour, third time this month",
        "He simply didn't show up on Saturday and went quiet afterwards",
        "Two appointments in a row moved at the last minute, I've lost a week",
        "Turned up 25 minutes into a one hour slot and still charged the full hour",
        "Left me waiting at the property in {zone} all morning, no message nothing",
    ],
    "provider_communication": [
        "Provider bought my lead but never contacted me",
        "Sent three messages, no reply for a week",
        "Paid for the lead apparently but I never heard from anyone",
        "Nobody got in touch after I filled in the form",
        "Still waiting for someone to call me back, it's been {n} days",
        "Filled in my details twice and heard nothing, is this site even active",
        "Got one text and then silence, no idea if I have a tradesperson or not",
        "Would be nice to actually get a response from the person I was matched with",
    ],
    "price_dispute": [
        "Price on the profile said £{price} but he charged £{price2} per hour",
        "Quoted one price then asked for more once the work started",
        "Hidden fees for materials and disposal, not mentioned before",
        "Rates changed after the first day of work without warning",
        "Was told £{price} an hour, invoice came in at £{price2}, no explanation",
        "The multi-day discount on the profile turned out not to exist",
        "Asked for cash up front for the full job then said the rate had gone up",
        "Expensive compared to what was advertised, felt misled",
    ],
    "workmanship_quality": [
        "Really thorough, explained the whole job clearly and finished ahead of time!",
        "Great tradesperson, tidy and methodical, highly recommend",
        "Spent most of the visit on his phone, barely explained anything",
        "Felt rushed, didn't finish the sealing properly before leaving",
        "Brilliant at talking me through the options, I was dreading this and it was painless",
        "Every visit had a clear plan and a recap at the end, finished in {n} days",
        "Snapped at me when I asked a question, made the whole thing uncomfortable",
        "Kept coming back for the same fault each week, no real progress at all",
        "Explains things in a way that actually makes sense, worth every penny",
        "Nice enough but I didn't feel much had been done after ten hours",
    ],
    "safety_issue": [
        "Left exposed wiring behind the panel and said it would be fine",
        "Isolation switch didn't seem to work properly, felt unsafe",
        "Lovely clean job, everything tested and certified before he left",
        "Warning light on the boiler the whole time, was told to ignore it",
        "No dust sheets, no mask, debris left all over a room with a baby in it",
        "The ladder he used was visibly damaged, I mentioned it and got shrugged at",
        "Site left spotless and every connection checked, felt genuinely professional",
    ],
    "lead_quality": [
        "Paid for a lead and the customer's number was disconnected",
        "Customer never answered, wasted credits again",
        "Lead was from {zone} which I don't even cover, why was it sent to me",
        "Same customer sent to me twice, charged twice",
        "Customers keep saying they already found someone by the time I call",
        "Third enquiry this week where the person says they never filled anything in",
        "Bought {n} leads this month and only one picked up the phone",
        "Someone wanting an installation sent to me, I only do repairs, that's on you",
        "The customer was actually two counties away, the address must have been typed wrong",
        "Getting enquiries from people who want work in three months, not now",
    ],
    "credits_refund": [
        "Requested a refund for a dead lead 2 weeks ago, still nothing",
        "Credits not refunded after the customer cancelled, support not replying",
        "The refund button does nothing, credits just disappeared",
        "Charged for a lead I never accepted, please refund",
        "My balance went down by {n} and I didn't buy anything, what happened",
        "Was promised my money back for the fake enquiry, {n} days later still waiting",
        "Top up went through on my card but the balance didn't change",
        "Two refunds approved by email but nothing showing in the account",
    ],
    "app_bug": [
        "The app crashes every time I open the leads tab",
        "Can't update my coverage zones, the save button is greyed out",
        "Notifications arrive hours late so I miss leads",
        "Login loop on Android, keeps sending me back to the start screen",
        "Uploaded a new profile photo and it shows sideways",
        "The availability calendar doesn't keep my changes after I close it",
        "Map view is blank on my phone since the last update",
    ],
}
PREFIX = ["", "", "", "Honestly, ", "Quick one: ", "Update - ", "Not happy. ", "Just to say ", "FYI ", "Hi, ", "Feedback: ", "So "]
SUFFIX = ["", "", "", " Thanks", " Please sort it out", " Not impressed", " Would appreciate a reply", " Otherwise fine",
          " Been using the site since {month}", " This was in {zone}", " Second time this has happened", " Cheers"]
MONTHS = ["September", "October", "November", "December", "January"]
SEVERITY = {  # prior severity used by the priority framework (justified in docs)
    "provider_no_show": 4, "provider_communication": 3, "price_dispute": 4,
    "workmanship_quality": 2, "safety_issue": 5, "lead_quality": 3,
    "credits_refund": 4, "app_bug": 3,
}
NAMES = ["Ahmed", "Priya", "Tom", "Olga", "Kwame", "Sophie", "Marek", "Fatima"]
BOILERPLATE = ["Sent from my iPhone", "-- This message was sent via the marketplace app", "Regards,"]


def make_feedback(n: int, rng: np.random.Generator, weeks: int = 26) -> pd.DataFrame:
    themes = list(THEMES)
    base_p = np.array([0.12, 0.14, 0.10, 0.22, 0.06, 0.16, 0.08, 0.12])
    rows = []
    for i in range(n):
        w = int(rng.integers(0, weeks))
        p = base_p.copy()
        if w >= weeks - 5:          # emerging issue: refund/credit complaints spike late
            p[themes.index("credits_refund")] *= 3.5
        if 8 <= w <= 11:            # transient app bug incident
            p[themes.index("app_bug")] *= 2.5
        p /= p.sum()
        theme = rng.choice(themes, p=p)
        fmt = dict(name=rng.choice(NAMES), price=int(rng.integers(30, 40)), price2=int(rng.integers(40, 50)),
                   zone=rng.choice(ZONE_NAMES), n=int(rng.integers(2, 15)), month=rng.choice(MONTHS))
        txt = rng.choice(PREFIX) + rng.choice(THEMES[theme]).format(**fmt) + rng.choice(SUFFIX).format(**fmt)
        if rng.random() < 0.3:
            txt = txt.lower()
        r = rng.random()
        if r < 0.08:
            # 07700 900000-900999 is reserved by Ofcom for fiction, so a
            # generated number can never collide with a real subscriber.
            txt += f" my number is 07700 900{rng.integers(0, 1000):03d}"
        elif r < 0.12:
            txt += f" email me at {rng.choice(NAMES).lower()}{rng.integers(1, 99)}@example.com"
        elif r < 0.16:
            txt += f" I live near {rng.choice(ZONE_NAMES)} {rng.integers(1, 9)}{rng.choice(list('ABCDEF'))}{rng.choice(list('ABCDEF'))}"
        if rng.random() < 0.15:
            txt += " " + rng.choice(BOILERPLATE)
        rows.append(dict(
            feedback_id=f"F{i:05d}",
            created_at=THEME_START + timedelta(days=w * 7 + int(rng.integers(0, 7))),
            author_type="provider" if theme in ("lead_quality", "credits_refund", "app_bug") or rng.random() < 0.1 else "customer",
            channel=rng.choice(["review", "support", "survey"], p=[0.4, 0.4, 0.2]),
            text=txt,
            true_theme=theme,     # ground truth kept ONLY for evaluation; a real system would not have it
        ))
    df = pd.DataFrame(rows)
    # quality noise: exact duplicates, very short texts, empty/malformed
    df = pd.concat([df, df.sample(int(n * 0.04), random_state=3)], ignore_index=True)
    short = df.sample(int(n * 0.03), random_state=4).index
    df.loc[short, "text"] = rng.choice(["ok", "fine", "??", "n/a", "good"], len(short))
    df.loc[short, "true_theme"] = "noise"
    df.loc[df.sample(int(n * 0.01), random_state=5).index, "text"] = ""
    return df


# ------------------------------------------------------------------ entry point
def generate_all(out_dir: str, seed: int = 42, n_providers: int = 220,
                 n_leads: int = 3000, n_feedback: int = 1500) -> dict[str, pd.DataFrame]:
    import os
    # One independent stream per table, derived from a single seed. Threading a
    # single shared `rng` through all four made every table depend on how many
    # draws the previous one happened to take, so any edit to an earlier
    # generator silently changed the tables after it — which is how the
    # committed data drifted out of step with this file.
    ss = np.random.SeedSequence(seed)
    r_prov, r_leads, r_offers, r_fb = (np.random.default_rng(s) for s in ss.spawn(4))
    os.makedirs(out_dir, exist_ok=True)
    prov = make_providers(n_providers, r_prov)
    leads = make_leads(n_leads, r_leads)
    offers = make_offers(leads, prov, r_offers)
    fb = make_feedback(n_feedback, r_fb)
    prov_public = prov.drop(columns=["_responsiveness"])  # latent variable never exported
    data = {"providers": prov_public, "leads": leads, "offers": offers, "feedback": fb}
    for k, v in data.items():
        v.to_csv(os.path.join(out_dir, f"{k}.csv"), index=False)
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write("# SYNTHETIC DEMO DATA\n\nGenerated by `common/synthetic.py` (seed=%d).\n"
                "Every row is sampled from the distributions defined in that file; "
                "there is no underlying real dataset.\n" % seed)
    return data


if __name__ == "__main__":
    import sys
    d = generate_all(sys.argv[1] if len(sys.argv) > 1 else "data/synthetic")
    for k, v in d.items():
        print(f"{k:12s} {v.shape}")
