# Methodology

```
leads + providers + offers
   ↓ validation (schema, ranges, categories, key uniqueness, referential integrity)
   ↓ cleaning (dedup, casing, unknown zones)
   ↓ candidate generation   — hard rules, coverage report
   ↓ pairwise features      — leakage-safe history, shrinkage
   ↓ baselines              — nearest-first (legacy), rule score
   ↓ ranker                 — logistic regression (prod), HGB (ceiling), position de-biasing
   ↓ evaluation             — P@K, R@K, NDCG@K, coverage, calibration; temporal split
   ↓ explainability         — additive log-odds contributions per pair
   ↓ cold start / fairness  — simulated new providers; exposure ratios by group
   ↓ scenarios              — capacity/demand replay with weekly capacity
   ↓ model artefact         — `model_logreg.joblib`, consumed by Project 3
```

## Candidate generation
Hard filters: job_type compatible; zone in coverage **or** within 8 km; customer
gender preference honoured; ≥1 availability slot in common (relaxed only if it empties the
set). Result on demo data: 100% of leads have candidates, 10.6% only after relaxation,
median candidate set 35 (p10 = 10, p90 = 70).

## Features (17)
Distance, availability overlap, positive price gap vs budget (+ missing flag), credit
availability, urgency flags, language match, gender-preference met, rating, capped
experience, apprentice flag, shrunk historical purchase rate, tenure, hours wanted.
`hist_n_offers` is computed but excluded: it drifts by construction (D7).

## Position de-biasing
Purchase rate falls with notification order (54% → 37% from position 1 to 5 in the log).
Order is chosen by the platform, not a property of the pair. `rank_shown` is included in
training and fixed to 1 at scoring, so the learnt score is "P(purchase | notified first)".
This is the standard logged-bandit correction; it does not remove all confounding
(see limitations).

## Model choice
Logistic regression (standardised features, C=0.5) is the production model. HGB was
fitted as a non-linear ceiling and did not beat it. Learning-to-rank (pairwise/listwise)
and deep models were considered and rejected: ~10k offers, 17 features, and the
downstream optimiser needs calibrated probabilities, not just order.

## Evaluation
Temporal split (last 25% of leads). Metrics per lead (query) at K = 3, purchase and
conversion labels, plus coverage (share of providers appearing in any top-3) and ECE.

| strategy | NDCG@3 (purchase) | NDCG@3 (conversion) | coverage | ECE |
|---|---|---|---|---|
| legacy order / nearest | 0.678 | 0.546 | 1.00 | – |
| rule score | 0.706 | 0.563 | 0.97 | – |
| **logistic regression** | **0.715** | **0.579** | 0.92 | 0.036 |
| HGB | 0.703 | 0.554 | 0.94 | 0.027 |

## Cold start
No genuinely new provider exists in the demo window, so cold start is *simulated*:
20% of test providers have their history reset to the prior. NDCG is unchanged
(0.715 → 0.717) and cold providers keep 92% of pool-share exposure. Recommended
production policy: shrinkage as implemented + one exploration slot in K for providers
with <10 offers.

## Fairness
Exposure ratio (top-3 share ÷ eligible-pool share) by tier: accredited 0.95, apprentice 1.18; by
gender: F 1.08, M 0.97. No group is buried; PDIs are slightly over-exposed because they
buy more eagerly — a fact, not a bias, but one to keep watching.

## Duplicate providers
Exact match on {home zone, coverage, job_type, price, experience, gender,
join date}; keep the earliest id. In production this would be replaced by the
phone-number-as-primary-key change already planned for registration.
