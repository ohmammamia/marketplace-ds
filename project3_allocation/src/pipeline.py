"""End-to-end run for Project 3. `python -m project3_allocation.src.pipeline configs/project3.yaml`"""
from __future__ import annotations
import os, sys
import pandas as pd
from common.monitoring import get_logger, save_json, load_config
from . import instance, milp, analysis
from .formulation import objective_value, restrict_distance

log = get_logger("p3.pipeline")


def run(cfg: dict) -> dict:
    out = cfg["output_dir"]; os.makedirs(out, exist_ok=True)
    S = {}
    inst = instance.build(cfg["data_dir"], cfg["week_start"], cfg["p1_model_path"], cfg["M"], cfg["weights"])
    pd.concat([r.to_frame().assign(dataset=n) for n, r in inst.validation.items()]).to_csv(
        f"{out}/validation_report.csv", index=False)
    S["validation"] = {n: {"errors": len(r.errors), "warnings": len(r.warnings)} for n, r in inst.validation.items()}
    log.info("validation: %s", S["validation"])
    S["instance"] = {"week": cfg["week_start"], "leads": len(inst.leads), "providers": len(inst.providers),
                     "feasible_pairs": len(inst.pairs), "total_capacity": int(inst.capacity.sum()),
                     "priority_leads": int((inst.leads.urgency != "none").sum()),
                     "demand_vs_capacity": float(len(inst.leads) * inst.M / inst.capacity.sum())}
    log.info("instance: %s", S["instance"])

    cmp = analysis.compare_solvers(inst); cmp.to_csv(f"{out}/solver_comparison.csv", index=False)
    S["solver_comparison"] = cmp.round(3).to_dict(orient="records")
    log.info("greedy vs milp:\n%s", cmp.round(3).T.to_string())

    tr = analysis.fairness_tradeoff(inst); tr.to_csv(f"{out}/fairness_tradeoff.csv", index=False)
    S["fairness_tradeoff"] = tr.round(3).to_dict(orient="records")
    log.info("fairness trade-off:\n%s", tr[["lambda_fairness", "expected_purchases", "providers_with_zero", "gini_provider_load", "mean_q"]].round(3).to_string())

    ca = analysis.constraint_analysis(inst); ca.to_csv(f"{out}/constraint_analysis.csv", index=False)
    S["constraint_analysis"] = ca.round(3).to_dict(orient="records")
    log.info("constraint analysis:\n%s", ca.round(3).to_string())

    sc_rows = []
    for sc in cfg["scenarios"]:
        kw = {k: sc[k] for k in ("demand_scale", "capacity_scale") if k in sc}
        i2 = instance.build(cfg["data_dir"], sc.get("week_start", cfg["week_start"]), cfg["p1_model_path"], cfg["M"], cfg["weights"], **kw)
        # the cap shrinks E for *both* solvers, so the comparison stays like-for-like
        i2 = restrict_distance(i2, sc.get("max_distance"))
        x, info = milp.solve(i2)
        xg = analysis.greedy.solve(i2)
        ov, og = objective_value(i2, x), objective_value(i2, xg)
        sc_rows.append({"scenario": sc["name"], "leads": len(i2.leads), "capacity": int(i2.capacity.sum()), "status": info["status"],
                        "milp_expected_purchases": ov["expected_purchases"], "greedy_expected_purchases": og["expected_purchases"],
                        "milp_leads_unfilled": ov["leads_unfilled"], "greedy_leads_unfilled": og["leads_unfilled"],
                        "milp_below_min": ov["leads_below_min"], "utilisation": ov["capacity_utilisation"],
                        "share_capacity_binding": info["share_capacity_binding"]})
    scd = pd.DataFrame(sc_rows); scd.to_csv(f"{out}/scenarios.csv", index=False)
    S["scenarios"] = scd.round(3).to_dict(orient="records")
    log.info("scenarios:\n%s", scd.round(3).to_string())

    rb = analysis.robustness(inst); rb.to_csv(f"{out}/robustness.csv", index=False)
    S["robustness"] = rb.round(3).to_dict(orient="records")
    log.info("robustness:\n%s", rb.round(3).to_string())

    # decision-support output: the recommended allocation with explanation per lead
    x, info = milp.solve(inst)
    rec = x.merge(inst.pairs, on=["lead_id", "provider_id"]).merge(inst.leads[["lead_id", "urgency", "zone"]], on="lead_id")
    rec["reason"] = rec.apply(lambda r: f"q={r.q:.2f}, {r.distance_km:.1f}km" + (", priority lead" if r.urgency != "none" else ""), axis=1)
    rec.sort_values(["lead_id", "q"], ascending=[True, False]).to_csv(f"{out}/recommended_allocation.csv", index=False)
    unfilled = inst.leads[~inst.leads.lead_id.isin(x.lead_id)][["lead_id", "zone", "job_type", "urgency"]]
    unfilled.to_csv(f"{out}/unfilled_leads.csv", index=False)
    S["decision_output"] = {"notifications": len(x), "unfilled_leads": len(unfilled), "solver": info}
    save_json(S, f"{out}/summary.json")
    return S


if __name__ == "__main__":
    run(load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/project3.yaml"))
