# Limitations

1. **Expected value objective.** Σ q x maximises expected purchases; it does not penalise
   variance or double-selling (two providers buying the same lead). A cap on expected
   purchases per lead (Σ q x ≤ 1.5) is a one-line extension.
2. **Capacity is declared, not observed.** `weekly_lead_capacity` is what providers say;
   realised acceptance should replace it once observed.
3. **Provider preferences absent.** Only customer preferences enter; provider-side
   preferences (areas, times) are captured only via coverage/availability.
4. **Static week.** Leads inside the week are treated as known; a rolling re-solve would
   be needed for same-day routing.
5. **Fairness = "everyone gets one".** A share-based or Gini constraint would be
   stronger; the Gini is reported, not optimised.
6. **q inherits Project 1's biases** (logged-policy, position). The robustness analysis
   bounds the damage but does not remove it.
7. **Synthetic instance.** Effect sizes depend on the generator's capacity/demand balance.
