"""Problem formulation for weekly lead allocation.

Sets
  L  leads arriving in the planning week
  I  providers
  E ⊆ L×I  feasible pairs (from Project 1 candidate generation)

Parameters
  q_ij  match quality = P(provider i purchases lead j | notified first), from
        the Project 1 ranker (falls back to the rule score, rescaled to [0,1])
  c_i   weekly lead capacity of provider i (leads they are willing to buy)
  M     max providers notified per lead (product rule: 3)
  m_j   min providers per lead: 2 for priority leads (test booked / emergency), else 1
  d_ij  distance km

Decision variables
  x_ij ∈ {0,1}  notify provider i about lead j
  u_j  ≥ 0      shortfall below m_j for lead j   (soft: unfilled leads are lost revenue)
  f_i  ≥ 0      shortfall below 1 lead for a provider with credits (soft: fairness /
                supply retention — providers who never receive leads churn)

Objective (maximise)
  Σ q_ij x_ij  −  λ_u Σ u_j  −  λ_f Σ f_i  −  λ_d Σ d_ij x_ij / d_max

Hard constraints
  Σ_i x_ij ≤ M            for all j
  Σ_j x_ij ≤ c_i          for all i
  x_ij = 0                for (i,j) ∉ E
Soft constraints
  Σ_i x_ij + u_j ≥ m_j    for all j
  Σ_j x_ij + f_i ≥ 1      for all i with credits > 0 and c_i ≥ 1

Weights: λ_u is set so that leaving a lead unfilled is worse than any single
notification's quality (λ_u = 1.0 > max q); λ_f and λ_d are swept in the
trade-off analysis rather than fixed by assumption (decision log D3)."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class Instance:
    leads: pd.DataFrame            # lead_id, created_at, urgency, ...
    providers: pd.DataFrame      # provider_id, weekly_lead_capacity, credit_balance, ...
    pairs: pd.DataFrame            # lead_id, provider_id, q, distance_km
    M: int = 3
    priority_min: int = 2
    weights: dict = field(default_factory=lambda: {"unfilled": 1.0, "fairness": 0.3, "distance": 0.1})

    @property
    def min_offers(self) -> pd.Series:
        return self.leads.set_index("lead_id").urgency.map(lambda u: self.priority_min if u != "none" else 1)

    @property
    def capacity(self) -> pd.Series:
        return self.providers.set_index("provider_id").weekly_lead_capacity.astype(int)

    @property
    def wants_lead(self) -> pd.Series:
        i = self.providers.set_index("provider_id")
        return (i.credit_balance > 0) & (i.weekly_lead_capacity >= 1)


def objective_value(inst: Instance, x: pd.DataFrame) -> dict:
    """Decompose the objective for any allocation x (lead_id, provider_id)."""
    pr = inst.pairs.merge(x.assign(chosen=1), on=["lead_id", "provider_id"], how="left").fillna({"chosen": 0})
    chosen = pr[pr.chosen == 1]
    per_lead = chosen.groupby("lead_id").size().reindex(inst.leads.lead_id, fill_value=0)
    per_ins = chosen.groupby("provider_id").size().reindex(inst.providers.provider_id, fill_value=0)
    u = (inst.min_offers - per_lead).clip(lower=0)
    f = ((1 - per_ins).clip(lower=0) * inst.wants_lead.astype(int))
    dmax = inst.pairs.distance_km.max() or 1
    w = inst.weights
    comp = {"quality": float(chosen.q.sum()), "unfilled_penalty": float(w["unfilled"] * u.sum()),
            "fairness_penalty": float(w["fairness"] * f.sum()), "distance_penalty": float(w["distance"] * chosen.distance_km.sum() / dmax)}
    comp["objective"] = comp["quality"] - comp["unfilled_penalty"] - comp["fairness_penalty"] - comp["distance_penalty"]
    comp.update({"leads_unfilled": int((per_lead == 0).sum()), "leads_below_min": int((u > 0).sum()),
                 "providers_with_zero": int(f.sum()), "notifications": int(len(chosen)),
                 "mean_q": float(chosen.q.mean()) if len(chosen) else 0.0,
                 "capacity_utilisation": float(per_ins.sum() / inst.capacity.sum()),
                 "expected_purchases": float(chosen.q.sum()),
                 "gini_provider_load": gini(per_ins[inst.wants_lead].values)})
    return comp


def gini(v: np.ndarray) -> float:
    v = np.sort(np.asarray(v, float))
    if len(v) == 0 or v.sum() == 0:
        return 0.0
    n = len(v); idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * v) / (n * v.sum())) - (n + 1) / n)
