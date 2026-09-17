# Methodology

```
cleaned leads / providers / offers  (Project 1 data layer)
 ↓ instance build      candidates + features + P1 ranker → (lead, provider, q, distance)
 ↓ baseline            greedy by arrival order, best-M with spare capacity
 ↓ MILP                PuLP + CBC; binaries x_ij; slacks u_j (unfilled), f_i (provider with none)
 ↓ objective           Σ q x − λ_u Σ u − λ_f Σ f − λ_d Σ d x / d_max
 ↓ trade-offs          sweep λ_f → expected purchases vs providers-with-zero / Gini
 ↓ constraint analysis marginal value of relaxing each constraint family by one step
 ↓ scenarios           demand +10/+25%, capacity −10/−25%, 5 km limit, quiet week
 ↓ robustness          perturb q (σ = 0.02/0.05/0.10), re-solve, assignment Jaccard + regret
 ↓ decision output     allocation + reasons, unfilled list, scenario table, solver status
```

## Solver choice
MILP via CBC: the problem is an assignment with side constraints (per-lead caps, soft
minima); ~6k binaries solve in seconds. Constraint programming was rejected (no
sequencing/scheduling logic); pure LP was rejected (integrality is lost with side
constraints). OR-Tools would be a drop-in replacement at larger scale.

## Weights
λ_u = 1.0 exceeds any single q, so an unfilled lead is always worse than a mediocre
notification. λ_f and λ_d are **swept**, not assumed; the recommended λ_f = 0.3 is the
point where no credit-holding provider is left out at negligible cost.

## Results (demo peak week)
| | greedy | MILP |
|---|---|---|
| expected purchases | 264.5 | **281.6** (+6.4%) |
| leads unfilled | 0 | 0 |
| leads below priority minimum | 2 | 0 |
| providers with zero leads | 32 | 2 |
| mean q of notifications | 0.604 | 0.633 |
| Gini of provider load | 0.41 | 0.33 |

Fairness trade-off: λ_f 0 → 0.3 takes zero-lead providers from 30 to 2 for −2.1 expected
purchases. Binding constraints: 57% of providers' capacity is saturated; **M (offers per
lead) is the most valuable relaxation** (+67.3 expected purchases for M=4), then capacity
(+9.0 for +1 each); the priority minimum costs nothing.

Scenarios: at demand +25% the MILP still fills every lead where greedy leaves 8;
at capacity −25% utilisation reaches 81% and the difference to greedy widens. The 5 km
scenario caps distance on the *instance*, so greedy and the MILP are compared on the
same feasible set (previously the cap reached only the MILP, which flattered greedy).

Robustness: with σ = 0.05 noise on q, 51% of notifications change — the Jaccard overlap
is 0.49, i.e. 49% are *retained* — and regret is 2.6%. Many near-equivalent allocations
exist, so the recommendation is stable in value even when it is not unique.
