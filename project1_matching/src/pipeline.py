"""End-to-end run for Project 1. `python -m project1_matching.src.pipeline configs/project1.yaml`"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from common.monitoring import get_logger, psi, save_json, load_config
from . import data, candidates, features, baseline, ranker, evaluation, fairness, scenarios

log = get_logger("p1.pipeline")


def run(cfg: dict) -> dict:
    out = cfg["output_dir"]; os.makedirs(out, exist_ok=True)
    K, target = cfg["k"], cfg["target"]
    summary = {}

    # 1. data + validation
    ins, leads, offers, reports = data.clean(*data.load(cfg["data_dir"]))
    pd.concat([r.to_frame().assign(dataset=n) for n, r in reports.items()]).to_csv(f"{out}/validation_report.csv", index=False)
    summary["validation"] = {n: {"errors": len(r.errors), "warnings": len(r.warnings)} for n, r in reports.items()}

    # 2. candidate generation coverage
    summary["candidate_coverage"] = candidates.coverage_report(leads, ins, max_distance_km=cfg["candidate_max_distance_km"])
    log.info("candidate coverage: %s", summary["candidate_coverage"])

    # 3. training set + temporal split
    df = features.build_training_set(leads, ins, offers)
    train, test = ranker.temporal_split(df, cfg["test_frac"])
    log.info("train %d rows / test %d rows; positive rate %.3f", len(train), len(test), df[target].mean())
    summary["data"] = {"rows": len(df), "train": len(train), "test": len(test), "positive_rate": float(df[target].mean())}

    # 4. baselines + models
    test = test.copy()
    test["s_nearest"] = baseline.nearest_score(test)
    test["s_rule"] = baseline.rule_score(test)
    test["s_shown_rank"] = -test.rank_shown                     # legacy platform order
    models = ranker.fit(ranker.make_models(), train, target)
    for name, m in models.items():
        test[f"p_{name}"] = ranker.score(m, test)
    strategies = ["s_shown_rank", "s_nearest", "s_rule", "p_logreg", "p_hgb"]
    res = {}
    for s in strategies:
        res[s] = evaluation.ranking_metrics(test, s, target, K)
        res[s].update({f"conv_{k}": v for k, v in evaluation.ranking_metrics(test, s, "converted", K).items() if k != "n_queries"})
        res[s]["coverage"] = evaluation.coverage(test, s, K)
    for name in models:
        res[f"p_{name}"]["ece"] = evaluation.expected_calibration_error(test[target].values, test[f"p_{name}"].values)
    summary["ranking"] = res
    pd.DataFrame(res).T.to_csv(f"{out}/ranking_metrics.csv")
    evaluation.calibration_table(test[target].values, test["p_logreg"].values).to_csv(f"{out}/calibration_logreg.csv")
    log.info("ranking metrics:\n%s", pd.DataFrame(res).T.round(3))

    # 5. explainability (logreg contributions) — global + one worked example
    contrib = ranker.logreg_contributions(models["logreg"], test)
    summary["global_importance"] = contrib.abs().mean().sort_values(ascending=False).round(3).to_dict()
    lid = test.lead_id.iloc[0]
    ex = evaluation.order_by_score(test[test.lead_id == lid], "p_logreg").head(K)
    ex_expl = contrib.loc[ex.index].round(2)
    ex_expl.insert(0, "provider_id", ex.provider_id.values); ex_expl.insert(1, "p_purchase", ex.p_logreg.round(3).values)
    ex_expl.to_csv(f"{out}/example_explanation_{lid}.csv", index=False)

    # 6. cold start — no genuinely new providers exist in the demo window, so
    # we *simulate* it: 20% of test providers have their history reset to the
    # prior (what a new joiner looks like) and ranking quality is re-measured.
    rng = np.random.default_rng(0)
    new_ids = set(rng.choice(test.provider_id.unique(), int(0.2 * test.provider_id.nunique()), replace=False))
    tc = test.copy(); m_new = tc.provider_id.isin(new_ids)
    tc.loc[m_new, "hist_purchase_rate"] = float(train[target].mean()); tc.loc[m_new, "hist_n_offers_log"] = 0.0
    tc.loc[m_new, "tenure_days_log"] = 0.0
    tc["p_logreg_cold"] = ranker.score(models["logreg"], tc)
    summary["cold_start"] = {
        "simulated_new_share": 0.2,
        "ndcg_full_history": res["p_logreg"][f"ndcg@{K}"],
        "ndcg_with_20pct_cold": evaluation.ranking_metrics(tc, "p_logreg_cold", target, K)[f"ndcg@{K}"],
        "cold_exposure_ratio": float(
            (evaluation.order_by_score(tc, "p_logreg_cold").groupby("lead_id").head(K).provider_id.isin(new_ids).mean())
            / m_new.mean()),
        "policy": "new providers score with the prior purchase rate (empirical-Bayes shrinkage), so they are neither buried nor over-promoted; "
                  "an explicit exploration slot (1 of K for providers with <10 offers) is recommended in docs/methodology.md",
    }

    # 7. fairness of exposure
    fair = {}
    for grp in ["tier", "gender"]:
        t = fairness.exposure_by_group(test, ins, "p_logreg", grp, K)
        t.to_csv(f"{out}/fairness_{grp}.csv")
        fair[grp] = t.round(3).to_dict(orient="index")
    summary["fairness"] = fair

    # 8. operational scenarios
    sc_rows = []
    for s in ["s_shown_rank", "s_rule", "p_logreg"]:
        for k in cfg["scenarios"]["ks"]:
            sc_rows.append(scenarios.simulate(test, ins, s, k=k))
    for cs in cfg["scenarios"]["capacity_scales"]:
        for ds in cfg["scenarios"]["demand_scales"]:
            sc_rows.append(scenarios.simulate(test, ins, "p_logreg", k=K, capacity_scale=cs, demand_scale=ds))
    sc = pd.DataFrame(sc_rows); sc.to_csv(f"{out}/scenarios.csv", index=False)
    summary["scenarios"] = sc.round(3).to_dict(orient="records")
    log.info("scenarios:\n%s", sc.round(3).to_string())

    # 9. monitoring baseline: feature + score drift train->test
    summary["monitoring_psi"] = {f: round(psi(train[f].values, test[f].values), 4) for f in features.FEATURES}
    summary["monitoring_psi"]["score_logreg"] = round(psi(ranker.score(models["logreg"], train), test.p_logreg.values), 4)

    import joblib; joblib.dump(models["logreg"], f"{out}/model_logreg.joblib")   # consumed by Project 3
    save_json(summary, f"{out}/summary.json")
    test.to_csv(f"{out}/test_scored.csv", index=False)
    return summary


if __name__ == "__main__":
    run(load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/project1.yaml"))
