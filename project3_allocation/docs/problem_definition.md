# Project 3 — Capacity and Resource Allocation Optimisation for Service Delivery

## Business problem
Provider capacity (the number of leads each will buy in a week) is limited and uneven
across the service area. Project 1 ranks providers *per lead*; when many leads arrive in the same
area, that myopic rule lets early leads consume providers that later — often more urgent
— leads needed, and leaves some providers with nothing while others are saturated.

## Decision
> For the leads arriving in a planning week, **which providers are notified about
> which leads** so as to maximise expected purchases (revenue) while respecting per-lead
> and per-provider limits, giving priority leads more offers, and keeping the supply
> side engaged.

## Formal statement
See `src/formulation.py` docstring: binary x_ij, per-lead max M and priority minimum,
per-provider capacity, soft penalties for unfilled leads and for credit-holding
providers that receive nothing, a mild distance penalty.

## Why optimisation and not prediction
Prediction (Project 1) tells us how good each pair is. It cannot resolve *contention*:
two leads wanting the same provider's last slot. That is an allocation problem with
constraints and trade-offs — an optimisation problem by definition.

## Objectives considered
match quality (expected purchases), utilisation, fairness of provider exposure,
geographic efficiency, priority for test-booked/emergency customers. They conflict; the
project quantifies the trade-offs rather than asserting a weighting.

## Constraints
Hard: max M offers per lead; provider capacity; feasibility (Project 1 candidate
rules). Soft: minimum offers (2 for priority leads, 1 otherwise); ≥1 lead for each
provider with credits.

## Output
Recommended allocation with a reason per notification, objective decomposition,
constraint status (binding capacity share, leads at the offer cap), scenario comparison,
unfilled-lead list.
