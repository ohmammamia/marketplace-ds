# Two-Sided Marketplace — Data Science Projects

Three commercial data-science projects built on a simulated pay-per-lead marketplace for
home trades: customers post job requests, providers (tradespeople) pay per lead to be
introduced. Together they cover **matching/ranking, unstructured text/NLP and
optimisation**, on a shared **production data-science layer** (validation,
reproducibility, testing, monitoring).

| # | project | question | core method |
|---|---|---|---|
| 1 | [Intelligent Matching and Ranking](project1_matching/) | which providers should be notified about a lead, in what order, and why? | candidate generation → leakage-safe pairwise features → calibrated logistic ranker with position de-biasing → NDCG/coverage/ECE → cold start, fairness, capacity scenarios |
| 2 | [Feedback → Operational Intelligence](project2_feedback_nlp/) | which issues are growing and which should ops act on first? | quality + PII redaction → NMF discovery → taxonomy with owners → weak supervision + human review → Poisson emergence test → priority = share × severity × growth × confidence |
| 3 | [Capacity and Resource Allocation](project3_allocation/) | how should limited provider capacity be allocated across a week's leads? | MILP (PuLP/CBC) with soft minima and fairness slack → λ sweep → binding-constraint analysis → demand/capacity scenarios → robustness under score noise |

The projects are linked: Project 3 consumes Project 1's calibrated purchase model as
match quality, and Project 2's complaint categories provide the counterweight to the
allocation levers Project 3 tunes.

## Data policy
**Everything in this repository is synthetic.** There is no underlying real dataset:
every row is sampled from distributions defined in `common/synthetic.py`, and the files
in `data/synthetic/` are the committed output of that generator. The simulation is
designed to carry realistic data-quality problems (duplicates, missing values,
inconsistent categories, orphan keys) and a latent outcome process, so the pipelines
have something genuine to learn — and to fail on. Latent generator variables are never
exported. See each project's `docs/dataset.md`.

Because the data is simulated, every number in this repository is a property of the
generator, not evidence about any real market. The methods are the point; the effect
sizes are not transferable.

## Run
```bash
pip install -r requirements.txt
make data      # regenerate synthetic data
make test      # unit tests
make p1 p2 p3  # pipelines → outputs/project{1,2,3}/summary.json + csv artefacts
make notebooks # rebuild + execute narrative notebooks
```

## Layout
```
common/            synthetic data, validation layer, monitoring (PSI, category drift), logging
configs/           one YAML per project — every threshold, weight and scenario lives here
project1_matching/ src/ (data, candidates, features, baseline, ranker, evaluation, fairness, scenarios, pipeline)
project2_feedback_nlp/ src/ (quality, privacy, discovery, taxonomy, classifier, emerging, priority, pipeline)
project3_allocation/   src/ (formulation, greedy, milp, instance, analysis, pipeline)
   each with docs/ {problem_definition, dataset, methodology, decision_log, limitations}.md
   and notebooks/ (narrative only — no business logic)
tests/             pytest
outputs/           committed demo outputs so results can be read without running
scripts/           notebook builder
```

## Production data-science layer (cross-project)
- **Validation**: `common/validation.py` — schema, types, ranges, categories, key
  uniqueness, missingness, referential integrity; explicit `ValidationReport`, blocking
  gate before modelling. All three pipelines validate their inputs, call
  `raise_if_failed()` on the cleaned frames and publish `validation_report.csv`.
- **Reproducibility**: seeded generator with an independent stream per table, **pinned**
  dependencies, YAML configs, deterministic models/solver, explicit rank tie-breaks,
  temporal splits, `summary.json` per run. `make data && make p1 p2 p3` reproduces
  every committed artefact byte-for-byte on the pinned versions.
- **Testing**: unit tests for validation, candidate rules, leakage, ranking metrics,
  redaction, quality reasons, emergence logic, MILP constraints.
- **Monitoring**: feature/score PSI (P1), category-share drift + review-queue size (P2),
  slack totals / utilisation / binding share (P3). Baselines written to `summary.json`.
- **Documentation**: problem definition, dataset, methodology, decision log (decision /
  context / evidence / alternatives / why / assumptions / risks / what would change it)
  and limitations, per project.
- **Logging**: structured, per-stage counts of what was dropped and why.

MIT licence.
