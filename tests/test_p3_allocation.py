import pandas as pd
from project3_allocation.src.formulation import Instance, objective_value, restrict_distance
from project3_allocation.src import milp, greedy

def _toy():
    leads = pd.DataFrame({"lead_id": ["L1", "L2"], "created_at": pd.to_datetime(["2026-01-01", "2026-01-02"]), "urgency": ["deadline", "none"]})
    ins = pd.DataFrame({"provider_id": ["A", "B"], "weekly_lead_capacity": [1, 2], "credit_balance": [5, 5]})
    pairs = pd.DataFrame({"lead_id": ["L1", "L1", "L2", "L2"], "provider_id": ["A", "B", "A", "B"],
                          "q": [0.9, 0.5, 0.8, 0.4], "distance_km": [1, 2, 1, 2]})
    return Instance(leads, ins, pairs, M=2)

def test_milp_respects_capacity_and_beats_greedy():
    inst = _toy()
    xm, info = milp.solve(inst); xg = greedy.solve(inst)
    assert info["status"] == "Optimal"
    assert (xm.groupby("provider_id").size() <= inst.capacity.reindex(xm.provider_id.unique())).all()
    assert objective_value(inst, xm)["objective"] >= objective_value(inst, xg)["objective"] - 1e-9

def test_priority_lead_gets_two_offers():
    inst = _toy()
    xm, _ = milp.solve(inst)
    assert (xm.lead_id == "L1").sum() == 2          # capacity A=1,B=2 allows it; priority min = 2

def test_infeasible_pairs_never_chosen():
    inst = restrict_distance(_toy(), 1.5)
    xm, _ = milp.solve(inst)
    assert set(xm.provider_id) == {"A"}

def test_distance_cap_applies_to_greedy_too():
    """The cap shrinks E, so both solvers see it.

    Previously it was a MILP-only argument: greedy kept allocating beyond the
    cap and the two were still compared head-to-head.
    """
    capped = restrict_distance(_toy(), 1.5)
    xg, xm = greedy.solve(capped), milp.solve(capped)[0]
    assert set(xg.provider_id) == {"A"}, "greedy must honour the distance cap"
    assert set(xm.provider_id) == {"A"}
    # and the cap must actually bind: uncapped, greedy reaches B
    assert "B" in set(greedy.solve(_toy()).provider_id)

def test_objective_matches_solver_under_cap():
    """`objective_value` and the MILP must normalise distance by the same d_max."""
    capped = restrict_distance(_toy(), 1.5)
    xm, info = milp.solve(capped)
    assert abs(info["objective"] - objective_value(capped, xm)["objective"]) < 1e-9
