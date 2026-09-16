# Dataset

## Schema expected
| column | type | notes |
|---|---|---|
| feedback_id | str, unique | |
| created_at | datetime | |
| author_type | {customer, provider} | |
| channel | {review, support, survey} | |
| text | str | raw free text — never exported from this pipeline |

## In this repository
Synthetic demo text only (`common/synthetic.py`): 8 issue themes, paraphrase variants,
prefix/suffix noise, injected identifiers (phones, emails, postcodes, names), boilerplate,
exact duplicates, empty/short/non-English records, one transient incident (app bug,
weeks 8–11) and one emerging issue (credits/refunds, last 5 weeks). A hidden theme label
exists **only** to stand in for human review in evaluation; it is removed before
processing and is never used for training beyond the simulated reviewed sample.

## Not in this repository
Any real message, even redacted. With real data the same pipeline runs on the internal
store; only aggregate outputs (counts, priorities, key terms) and redacted examples that
have passed a manual check would be shared.

## Quality profile (demo)
1,560 records → 1,329 kept: 159 exact duplicates, 39 too short, 21 empty, 12 non-English.
21% of kept records contained ≥1 identifier (118 phones, 61 emails, 63 postcodes, 50 names).
