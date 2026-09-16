# Methodology

```
raw text
 ↓ validation      schema, categories, missingness
 ↓ quality         normalise, strip boilerplate, drop empty / <3 tokens / exact dup / non-English  (counted)
 ↓ privacy         regex (phone, email, postcode, URL) + name gazetteer/context → typed placeholders (counted, assert no leak)
 ↓ exploration     volume by week/channel/author, length
 ↓ discovery       TF-IDF (1–2 grams) + NMF; k chosen on coherence proxy × seed stability
 ↓ taxonomy        8 operational categories, each with owner, action, seed keywords, severity
 ↓ categorisation  rules → weak labels → TF-IDF+LR; human-reviewed labels override; abstain <0.5 → review queue
 ↓ emerging        weekly counts; recent 4 wks vs baseline 12; volume-normalised rate ratio; Poisson p-value
 ↓ priority        share × severity × growth × confidence → ranked table with owner/action
 ↓ explainability  key terms, redacted examples, mean confidence per category
 ↓ monitoring      category-share drift, review-queue size, PII rate
```

## Why NMF for discovery
Short texts (median 16 tokens) and a small corpus: LDA on sparse counts is unstable; sentence
embeddings + clustering were tried mentally and rejected as harder to explain for no
evident gain at this size. NMF on TF-IDF gives readable term lists. k = 8: coherence keeps
rising with k but stability peaks at 8 (0.90) — the compromise is documented in
`topic_k_selection.csv`.

## Taxonomy design
Topics are evidence, not categories. The taxonomy was written from (a) topic terms,
(b) a manual read of a stratified sample, (c) "who acts on this?". Each category carries
its owner, action, seed keywords and severity (rubric: 5 safety, 4 money/trust, 3 service
failure, 2 experience, 1 cosmetic). It is versioned with the code.

## Categorisation: three stages, one holdout
All evaluated on the same 150 held-out human-reviewed records (demo: hidden labels).

| stage | macro-F1 | coverage | agreement |
|---|---|---|---|
| keyword rules | 0.652 | 0.727 | 0.661 |
| classifier on weak labels only | 0.593 | 0.880 | 0.583 |
| classifier on weak + 250 reviewed labels | **0.743** | 0.860 | 0.767 |

The weak-only classifier is *worse* than the rules that produced its labels — it
generalises their errors. Human labels are the highest-leverage input, so the pipeline
budgets for them structurally (review queue → retrain).

Weakest class: `safety_issue` (F1 0.32) — small, lexically diverse, overlaps with
praise about tidy, well-finished work. First target for the next review batch.

## Emerging-issue detection
A category is emerging if its recent count exceeds what its baseline *share* would predict
at current total volume (so a marketing push lifting everything is not "emerging") by a
rate ratio ≥1.5 with a one-sided Poisson p < 0.05. On demo data `credits_refund` is
flagged (ratio 2.6, p < 0.001); the transient app-bug incident is not, because it lies in
the baseline window — by design, the method finds *current* growth.

## Prioritisation
`priority = share × severity × growth × confidence`, multiplicative so no axis can be
compensated. Weights are not learnt: there is no ground truth for "correct priority";
this is a transparent policy the ops lead can argue with. The output table carries the
owner and action for each category.

## Human-in-the-loop
Items with max class probability < 0.5 (12% of kept records) go to `review_queue.csv`;
reviewed labels are appended to the training set. Everything shown to a reviewer is
redacted text.
