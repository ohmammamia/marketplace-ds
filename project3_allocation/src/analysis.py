"""Trade-offs, constraint analysis, scenarios and robustness."""
from __future__ import annotations
import dataclasses
import numpy as np
import pandas as pd
from .formulation import Instance, objective_value
from . import milp, greedy


def compare_solvers(inst: Instance) -> pd.DataFrame:
    xg = greedy.solve(inst); xm, info = milp.solve(inst)
    rows = [dict(solver="greedy", **objective_value(inst, xg)), dict(solver="milp", **objective_value(inst, xm), **{"solver_status": info["status"]})]
    return pd.DataFrame(rows)


def fairness_tradeoff(inst: Instance, lambdas=(0.0, 0.1, 0.3, 0.6, 1.0, 2.0)) -> pd.DataFrame:
    rows = []
    for lam in lambdas:
        i2 = dataclasses.replace(inst, weights={**inst.weights, "fairness": lam})
        x, _ = milp.solve(i2)
        rows.append(dict(lambda_fairness=lam, **objective_value(i2, x)))
    return pd.DataFrame(rows)


def constraint_analysis(inst: Instance) -> pd.DataFrame:
    """Marginal value of relaxing each constraint family by a small step.
    (Shadow prices are not available from a MILP; this is the discrete analogue.)"""
    base_x, base_info = milp.solve(inst); base = objective_value(inst, base_x)["expected_purchases"]
    rows = [{"relaxation": "none", "expected_purchases": base, "gain": 0.0, "binding_share": base_info["share_capacity_binding"]}]
    # +1 capacity for every provider
    i2 = dataclasses.replace(inst, providers=inst.providers.assign(weekly_lead_capacity=inst.providers.weekly_lead_capacity + 1))
    rows.append({"relaxation": "capacity +1 per provider", "expected_purchases": objective_value(i2, milp.solve(i2)[0])["expected_purchases"]})
    # M + 1 offers per lead
    i3 = dataclasses.replace(inst, M=inst.M + 1)
    rows.append({"relaxation": "max offers per lead +1", "expected_purchases": objective_value(i3, milp.solve(i3)[0])["expected_purchases"]})
    # drop priority minimum
    i4 = dataclasses.replace(inst, priority_min=1)
    rows.append({"relaxation": "priority min offers 2→1", "expected_purchases": objective_value(i4, milp.solve(i4)[0])["expected_purchases"]})
    # widen feasibility: relax candidate distance
    i5 = dataclasses.replace(inst, pairs=inst.pairs)  # distance already embedded in E; approximated via distance weight 0
    i5 = dataclasses.replace(i5, weights={**inst.weights, "distance": 0.0})
    rows.append({"relaxation": "ignore distance penalty", "expected_purchases": objective_value(i5, milp.solve(i5)[0])["expected_purchases"]})
    df = pd.DataFrame(rows); df["gain"] = df.expected_purchases - base
    return df


def robustness(inst: Instance, sigmas=(0.02, 0.05, 0.10), n_rep: int = 5, seed: int = 0) -> pd.DataFrame:
    """Perturb q (the ranker is not exact) and re-solve: how much of the
    allocation survives, and how much objective is lost if we commit to the
    nominal allocation while the truth is the perturbed one?"""
    rng = np.random.default_rng(seed)
    x0, _ = milp.solve(inst); s0 = set(map(tuple, x0.values))
    rows = []
    for sg in sigmas:
        for r in range(n_rep):
            p2 = inst.pairs.assign(q=np.clip(inst.pairs.q + rng.normal(0, sg, len(inst.pairs)), 0, 1))
            i2 = dataclasses.replace(inst, pairs=p2)
            x1, _ = milp.solve(i2); s1 = set(map(tuple, x1.values))
            opt_true = objective_value(i2, x1)["objective"]; nominal_under_true = objective_value(i2, x0)["objective"]
            rows.append({"sigma": sg, "rep": r, "assignment_jaccard": len(s0 & s1) / len(s0 | s1),
                         "regret_pct": 100 * (opt_true - nominal_under_true) / abs(opt_true)})
    return pd.DataFrame(rows).groupby("sigma").agg(assignment_jaccard=("assignment_jaccard", "mean"), regret_pct=("regret_pct", "mean")).reset_index()
