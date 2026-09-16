# Project 1 — Intelligent Matching and Ranking for Service Allocation

## Business problem
The setting is a two-sided lead marketplace for home trades: customers submit a job
request; the platform notifies a shortlist of providers, who **pay per lead** and then
contact the customer. Revenue is earned when a provider buys a lead; the platform's
reputation depends on the customer actually getting the job done.

Today the shortlist is "nearest compatible providers". Providers complain about
paying for leads they cannot serve or that never answer; customers complain that nobody
calls. Both are symptoms of the same thing: the shortlist is not chosen for the
probability that the pair *works*.

## Who decides, and what
The lead-routing service decides, for every new lead, **which K providers to notify and
in what order**. The ops lead needs to be able to read *why*.

## Definitions
| term | definition |
|---|---|
| customer | a person submitting a lead: zone, job_type, availability, budget, urgency, hours, preferences |
| provider | a registered tradesperson: coverage zones, job_type, price, tier (accredited/apprentice), availability, weekly lead capacity, credit balance |
| candidate | a provider that passes hard eligibility rules for a lead |
| offer | a notification of a lead to a provider (a row in the offers log) |
| purchase | the provider pays credits for the lead — **modelling target** |
| conversion | job booked — **business target**, observed later and more sparsely |
| good match | high P(purchase) *and* high P(conversion \| purchase), operationally feasible (capacity, availability), explainable |

## Business objective vs modelling objective
The business wants conversions. Purchases are ~4× more frequent, observed within hours,
and are the gate to conversion, so the ranker is trained on **purchase** and evaluated on
both purchase and conversion ranking metrics. This is a deliberate proxy; the gap is
monitored (`conv_ndcg@3` in `ranking_metrics.csv`).

## Why this is not "build a recommender"
- The candidate set is small and constrained (eligibility, capacity, geography), so
  candidate generation matters as much as ranking.
- Outcomes are decided by the *provider*, not the customer — a provider with no credits
  will not buy the best lead on the platform.
- The output feeds an allocation decision with capacity limits (Project 3), so the score
  must be a calibrated probability, not an arbitrary ranking score.

## Out of scope
Pricing of leads; customer-side ranking of providers; real-time re-ranking.
