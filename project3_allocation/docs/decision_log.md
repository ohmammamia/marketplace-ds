# Decision log — Project 3

### D1 Weekly batch with binary notify decisions
- **Alternatives** continuous real-time re-ranking; assign one provider per lead.
- **Why** leads batch naturally (providers check the app daily); notifying up to M keeps the marketplace dynamic; re-solve intra-week on arrival is a straightforward extension.

### D2 MILP (CBC) over CP / LP / heuristics
- **Evidence** 5.9k binaries, optimal in <2 s.
- **Why** exact, explainable, side constraints are linear. **Would change if** scheduling/time windows enter the problem (→ CP) or instances exceed ~1M binaries (→ column generation / heuristics).

### D3 Sweep λ_f instead of fixing it
- **Evidence** frontier flat: fairness nearly free at λ_f = 0.3.
- **Why** a swept trade-off is a decision the business can make; a fixed weight is an assumption hidden in code.

### D4 Soft, not hard, minimum offers
- **Evidence** hard minima would be infeasible when a zone has fewer eligible providers than the minimum.
- **Why** slack variables keep the problem feasible and *report* the shortfall.

### D5 Reuse Project 1 ranker as q
- **Alternatives** rule score; re-fit inside P3.
- **Why** one calibrated probability model serves both products; fallback to the rule keeps P3 runnable standalone.

### D6 Discrete marginal analysis instead of shadow prices
- **Why** duals are unavailable for MILPs; relaxing each constraint family by one unit gives the ops-relevant answer ("what is +1 capacity per provider worth?").

### D7 Robustness by perturbation of q
- **Why** the ranker is noisy; regret under perturbation is the right question ("how much do I lose by committing?"), not solution uniqueness.
