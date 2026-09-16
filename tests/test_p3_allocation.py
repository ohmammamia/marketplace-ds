import pandas as pd
from project3_allocation.src.formulation import Instance, objective_value
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
    inst = _toy()
    xm, _ = milp.solve(inst, extra_max_distance=1.5)
    assert set(xm.provider_id) == {"A"}
