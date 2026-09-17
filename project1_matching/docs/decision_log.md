# Decision log — Project 1

### D1 Duplicate provider resolution
- **Context** raw table fails the unique-key check (duplicate registrations).
- **Evidence** 4 exact duplicates on stable attributes in demo data; repeat registration is a
  standard data-quality problem wherever sign-up is self-service.
- **Alternatives** fuzzy linkage; ignore.
- **Why selected** exact match on stable attributes is transparent and zero-false-positive; fuzzy linkage belongs upstream at registration, not in the modelling pipeline.
- **Risks** misses near-duplicates. **Would change if** >2% near-duplicates found by manual audit.

### D2 Train on purchase, report conversion
- **Evidence** purchase rate 46%, conversion 12%; conversion observed weeks later.
- **Alternatives** train on conversion; two-stage model.
- **Why** purchase is the revenue event and the gate; conversion NDCG is reported to keep the proxy honest.
- **Would change if** purchase- and conversion-NDCG rankings of strategies diverge.

### D3 Hard candidate filters before ranking
- **Evidence** median 35 eligible providers per lead; ranking all 220 would produce infeasible matches.
- **Alternatives** rank everything with penalties.
- **Why** feasibility is a business rule, not a learnt weight; also 6× cheaper.
- **Risks** over-filtering — mitigated by the relaxation step and the coverage report.

### D4 Empirical-Bayes shrinkage for provider history
- **Evidence** providers with <10 offers have purchase rates of 0% or 100% by chance.
- **Alternatives** drop history; raw rate.
- **Why** prior_n=10 ≈ two weeks of offers; handles cold start without a separate model.

### D5 Logistic regression over HGB / LTR
- **Evidence** HGB NDCG@3 0.729 vs LR 0.736; LR ECE 0.052, HGB 0.047.
- **Why** explainable additive contributions; calibrated probabilities for Project 3; trivial to serve.
- **Would change if** data volume ×10 and HGB gains >2 NDCG points on a temporal holdout.

### D6 Position de-biasing via `rank_shown` fixed at 1
- **Evidence** purchase rate falls 57% → 36% across positions 1→5.
- **Alternatives** inverse-propensity weighting; ignore.
- **Why** simplest correction with a single logged policy; IPW needs randomisation we do not have.
- **Risks** residual confounding — requires an online interleaving test (limitations).

### D7 Drop `hist_n_offers_log` from features
- **Evidence** excluded from FEATURES; `tenure_days_log` is now the largest surviving drift (PSI 0.36).
- **Why** a monotonic feature guarantees false drift alarms and degrades under retraining; shrinkage already encodes sample size.

### D8 Simulated cold start
- **Context** no new providers in the demo window.
- **Why** reset 20% of providers' history — tests the policy that will actually run.
- **Risks** simulated new providers still have attributes distributed like old ones.
