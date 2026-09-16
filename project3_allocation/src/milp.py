"""Mixed-integer formulation solved with CBC via PuLP. Chosen over constraint
programming (no complex scheduling logic here) and over a pure LP (the
assignment polytope is not integral once side constraints are added, and we
need binary decisions). ~4k binaries per week solves in seconds."""
from __future__ import annotations
import pulp
import pandas as pd
from .formulation import Instance


def solve(inst: Instance, time_limit: int = 60, extra_max_distance: float | None = None) -> tuple[pd.DataFrame, dict]:
    pairs = inst.pairs
    if extra_max_distance is not None:
        pairs = pairs[pairs.distance_km <= extra_max_distance]
    w = inst.weights
    dmax = float(pairs.distance_km.max() or 1)
    prob = pulp.LpProblem("lead_allocation", pulp.LpMaximize)
    x = {(r.lead_id, r.provider_id): pulp.LpVariable(f"x_{k}", cat="Binary") for k, r in enumerate(pairs.itertuples())}
    leads = list(inst.leads.lead_id); ins = list(inst.providers.provider_id)
    u = {j: pulp.LpVariable(f"u_{j}", lowBound=0) for j in leads}
    wants = inst.wants_lead
    f = {i: pulp.LpVariable(f"f_{i}", lowBound=0) for i in ins if wants[i]}
    q = dict(zip(zip(pairs.lead_id, pairs.provider_id), pairs.q))
    d = dict(zip(zip(pairs.lead_id, pairs.provider_id), pairs.distance_km))
    prob += (pulp.lpSum(q[k] * v for k, v in x.items())
             - w["unfilled"] * pulp.lpSum(u.values())
             - w["fairness"] * pulp.lpSum(f.values())
             - w["distance"] * pulp.lpSum(d[k] / dmax * v for k, v in x.items()))
    by_lead, by_ins = {}, {}
    for (j, i), v in x.items():
        by_lead.setdefault(j, []).append(v); by_ins.setdefault(i, []).append(v)
    mo = inst.min_offers; cap = inst.capacity
    for j in leads:
        vs = by_lead.get(j, [])
        prob += pulp.lpSum(vs) <= inst.M, f"max_offers_{j}"
        prob += pulp.lpSum(vs) + u[j] >= int(mo[j]), f"min_offers_{j}"
    for i in ins:
        vs = by_ins.get(i, [])
        prob += pulp.lpSum(vs) <= int(cap[i]), f"capacity_{i}"
        if i in f:
            prob += pulp.lpSum(vs) + f[i] >= 1, f"fair_{i}"
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    chosen = [(j, i) for (j, i), v in x.items() if v.value() and v.value() > 0.5]
    binding_cap = [i for i in ins if by_ins.get(i) and abs(sum(v.value() for v in by_ins[i]) - cap[i]) < 1e-6 and cap[i] > 0]
    info = {"status": pulp.LpStatus[status], "objective": pulp.value(prob.objective),
            "n_binary": len(x), "binding_capacity_providers": len(binding_cap),
            "share_capacity_binding": len(binding_cap) / max(1, sum(cap > 0)),
            "leads_at_max_offers": sum(1 for j in leads if by_lead.get(j) and sum(v.value() for v in by_lead[j]) >= inst.M - 1e-6)}
    return pd.DataFrame(chosen, columns=["lead_id", "provider_id"]), info
