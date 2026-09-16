# Limitations

1. **Logged-policy bias.** Outcomes exist only for pairs the legacy rule chose to notify
   (nearest providers). The ranker cannot learn about far-but-excellent providers.
   Correct remedy: an exploration slot and an online interleaving/A-B test before claiming
   the offline uplift.
2. **Modest uplift.** +3.7 NDCG points over legacy order, +0.9 over a hand rule. Within
   a candidate set of near-equivalent local providers the ceiling is low; the larger win
   is in candidate generation and in Project 3's joint allocation.
3. **Proxy target.** Purchase ≠ job booked. Conversion metrics are reported but the
   model is not optimised for them.
4. **Calibration is conditional** on being notified first (position 1); probabilities for
   position 2–3 are optimistic.
5. **Synthetic data.** Effect sizes are properties of the generator, not of the market.
   The pipeline, checks and evaluation design are what transfers.
6. **Fairness scope.** Exposure by tier and gender only; no customer-side fairness analysis
   (e.g. by zone affluence) because the demo data carries no such attribute.
7. **Static provider attributes.** Credit balance and capacity are snapshots; in
   production they change hourly and must be read at scoring time.
