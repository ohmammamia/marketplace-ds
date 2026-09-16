# Project 2 — From Unstructured Service Feedback to Actionable Intelligence

## Business problem
A two-sided marketplace receives free text from both sides: customer reviews and support
messages, provider complaints about lead quality, credits and the app, survey comments.
Daily volume is modest, which is exactly why it tends to go unread — so issues surface
late and the same complaint types get re-discovered independently by different teams.

## Operational question
> How can unstructured feedback be converted, reproducibly, into **reliable themes,
> emerging issues and a prioritised list of actions with an owner**?

Not: "what is the sentiment of reviews" — sentiment does not tell anyone what to do.

## Who consumes the output, and what decision it supports
A category is only useful if somebody owns it, so the taxonomy is defined against an
operating model. The functions below are the generic ones a marketplace of this shape
would have; the point is the mapping from category → owner → action, not the org chart.

| consumer | decision |
|---|---|
| ops lead (weekly) | which issue category to act on first; whether something new is emerging |
| supply ops | which providers to warn/suspend (no-shows, silence) |
| finance ops | refund turnaround, credit bugs |
| engineering | app bugs |
| trust & safety | site safety and price disputes |

## What constitutes a meaningful issue
A category is meaningful when (a) it maps to an owner and an action, (b) the model
can assign it with measurable agreement against human review, and (c) its frequency or
growth is high enough to justify the action's cost. All three are explicit in the pipeline.

## Constraints
- **Privacy.** Free text contains phone numbers, emails, postcodes and names. Nothing
  reaches a model, notebook or example list unredacted.
- **Human-in-the-loop.** Full automation is not desirable: low-confidence items go to a
  review queue and reviewed labels feed the next retrain.
- **Reproducibility.** Same input + config → same categories, same priority table.

## Out of scope
Chatbot / RAG over feedback; replying to users automatically; provider scoring.
