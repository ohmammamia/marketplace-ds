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
| expected purchases | 254.4 | **283.1** (+11%) |
| leads unfilled | 5 | 0 |
| leads below priority minimum | 5 | 0 |
| providers with zero leads | 22 | 0 |
| mean q of notifications | 0.546 | 0.579 |
| Gini of provider load | 0.38 | 0.32 |

Fairness trade-off: λ_f 0 → 0.3 removes all 12 zero-lead providers for −0.7 expected
purchases. Binding constraints: 71% of providers' capacity is saturated; **M (offers per
lead) is the most valuable relaxation** (+49 expected purchases for M=4), then capacity
(+14 for +1 each); the priority minimum costs nothing.

Scenarios: at demand +25% the MILP still fills every lead where greedy leaves 18;
at capacity −25% utilisation reaches 93% and the difference to greedy widens.

Robustness: with σ = 0.05 noise on q, 55% of notifications change but regret is 2.3% —
many near-equivalent allocations exist, so the recommendation is stable in value even
when it is not unique.
