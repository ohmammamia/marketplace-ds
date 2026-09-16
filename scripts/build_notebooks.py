"""Builds (and optionally executes) the narrative notebooks. Notebooks contain
no business logic — they call the src modules and show results with the
Analytical question / Evidence / Why / Implementation / Result /
Interpretation / Decision structure."""
import os, sys
import nbformat as nbf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SETUP = "import sys, os\nsys.path.insert(0, os.path.abspath('../..'))\nos.chdir(os.path.abspath('../..'))\nimport pandas as pd, numpy as np, matplotlib.pyplot as plt\npd.set_option('display.width', 160)\n%matplotlib inline"


def nb(cells):
    n = nbf.v4.new_notebook(); n.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return n


def q(question, evidence, why):
    return f"### Analytical question\n{question}\n\n### Evidence so far\n{evidence}\n\n### Why this approach?\n{why}"


P1 = {
"01_data_discovery_and_quality.ipynb": [
("md", "# Project 1 — Intelligent Matching and Ranking\n## 01 · Data discovery and quality\n\n> All data here is **synthetic demo data** (`common/synthetic.py`). It reproduces the schema and structure of the marketplace's lead flow; no row is a real customer or provider.\n\n" + q(
"What do we know about customers, providers and historical offers, and is the data fit for modelling?",
"Nothing yet — this is the first look.",
"Validation runs *before* any analysis so that later decisions cite counted facts (duplicates, orphans, missingness), not impressions. The schema checks live in `src/data.py` and are reused by every downstream pipeline.")),
("code", SETUP + "\nfrom project1_matching.src import data\nins_raw, leads_raw, offers_raw = data.load('data/synthetic')\nins, leads, offers, reports = data.clean(ins_raw, leads_raw, offers_raw)\npd.concat([r.to_frame().assign(dataset=n) for n, r in reports.items()]).query('~passed')"),
("md", "### Result / Interpretation\nRaw provider data fails the unique-key check (duplicate registrations) and has inconsistent job_type casing; the offers log has orphan provider ids. All three are the kind of thing that silently corrupts a join. After cleaning, the gates pass.\n\n### Decision\nProceed with the cleaned frames; the de-duplication rule (exact match on stable attributes) is recorded in the decision log (D1)."),
("code", "print(ins.shape, leads.shape, offers.shape)\nfig, ax = plt.subplots(1, 3, figsize=(14, 3.5))\nleads.created_at.dt.to_period('W').value_counts().sort_index().plot(ax=ax[0], title='Leads per week')\nins.price_per_hour.plot.hist(ax=ax[1], bins=20, title='Provider £/hour'); leads.budget_max.plot.hist(ax=ax[1], bins=20, alpha=.5)\noffers.groupby('rank_shown')[['purchased','converted']].mean().plot.bar(ax=ax[2], title='Outcome by notification order'); plt.tight_layout()"),
("md", "### Interpretation\nDemand grows over the window (marketing ramp). Provider prices and customer budgets overlap but not perfectly — price gap will matter. Purchase probability falls with notification order: a **position effect** is present in the logs. That has a direct consequence for how the ranker must be trained (notebook 02)."),
("code", "from project1_matching.src import candidates\ncov = candidates.coverage_report(leads, ins)\ncov"),
("md", "### Interpretation\nEvery lead has at least one eligible provider, but ~10% only after relaxing the availability filter. Candidate sets are wide (median ~35), so ranking — not eligibility — is the binding problem.\n\n### Decision\nKeep the strict → relaxed candidate policy; measure ranking within candidate sets."),
],
"02_baseline_and_ranking.ipynb": [
("md", "## 02 · Baseline and ranking model\n\n" + q(
"Can a learnt ranker beat a transparent rule at putting the providers most likely to buy a lead at the top?",
"Notebook 01: outcomes depend on distance, availability, price gap *and* on notification order.",
"Two baselines (nearest-first = legacy behaviour; a hand-written rule score) are the bar. A logistic regression on pairwise features is the production candidate because it is explainable and cheap; gradient boosting is a ceiling check. `rank_shown` is used in training and fixed to 1 at inference to de-bias the position effect. Split is temporal.")),
("code", SETUP + "\nfrom project1_matching.src import data, features, baseline, ranker, evaluation\nins, leads, offers, _ = data.clean(*data.load('data/synthetic'))\ndf = features.build_training_set(leads, ins, offers)\ntrain, test = ranker.temporal_split(df, 0.25)\ntest = test.copy(); test['s_nearest'] = baseline.nearest_score(test); test['s_rule'] = baseline.rule_score(test)\nmodels = ranker.fit(ranker.make_models(), train)\nfor n, m in models.items(): test[f'p_{n}'] = ranker.score(m, test)\nres = {s: evaluation.ranking_metrics(test, s, k=3) | {'coverage': evaluation.coverage(test, s, 3)} for s in ['s_nearest','s_rule','p_logreg','p_hgb']}\npd.DataFrame(res).T.round(3)"),
("md", "### Result / Interpretation\nThe learnt ranker improves NDCG@3 over the legacy order by a few points and over the rule by ~1 point; gradient boosting adds nothing over logistic regression at this data size. The uplift is modest and honest: within a candidate set of near-equivalent local providers there is limited headroom.\n\n### Decision\nShip logistic regression (D5). The measured gain must be confirmed with an interleaving test — logged data alone cannot fully remove the position confound (limitations.md)."),
("code", "contrib = ranker.logreg_contributions(models['logreg'], test)\ncontrib.abs().mean().sort_values().plot.barh(figsize=(6,5), title='Mean |contribution| (log-odds)'); plt.tight_layout()"),
("code", "lid = test.lead_id.iloc[0]\nex = test[test.lead_id==lid].sort_values('p_logreg', ascending=False).head(3)\nexpl = contrib.loc[ex.index].round(2); expl.insert(0,'provider',ex.provider_id.values); expl.insert(1,'p',ex.p_logreg.round(3).values)\nexpl.T"),
("md", "### Interpretation\nFor one lead, the explanation table says *why* provider A outranks B: the columns are additive log-odds contributions, so an ops person can read 'closer, available at the right times, price within budget'. This is the explainability layer that ships with every recommendation."),
],
"03_evaluation_fairness_scenarios.ipynb": [
("md", "## 03 · Calibration, cold start, fairness, operational scenarios\n\n" + q(
"Is the ranker safe to operate: calibrated, fair to provider groups, robust to new providers, and does it hold up under capacity/demand stress?",
"Notebook 02: logistic regression is the chosen ranker.",
"Ranking metrics alone do not tell us whether the model is safe to run every day. Each check below corresponds to a way the system could fail quietly.")),
("code", SETUP + "\nimport json\nS = json.load(open('outputs/project1/summary.json'))\ncal = pd.read_csv('outputs/project1/calibration_logreg.csv', index_col=0)\nax = cal.plot(x='predicted', y='observed', style='o-', figsize=(4,4)); ax.plot([0,1],[0,1],'--',c='grey'); ax.set_title(f\"Calibration (ECE={S['ranking']['p_logreg']['ece']:.3f})\")"),
("md", "### Interpretation\nProbabilities are usable as probabilities (they feed Project 3's optimiser as expected purchases). Note they are calibrated *conditional on being notified first*."),
("code", "pd.DataFrame(S['cold_start']).T if False else S['cold_start']"),
("code", "pd.concat({g: pd.DataFrame(S['fairness'][g]).T for g in S['fairness']})"),
("md", "### Interpretation\nSimulated cold start (20% of providers with history reset) costs nothing in NDCG and new providers keep ~0.9 of their pool-share exposure: shrinkage to the prior works. Exposure ratios by tier/gender are close to 1 — no group is buried — and PDIs are, if anything, slightly over-exposed because they buy leads more eagerly."),
("code", "sc = pd.DataFrame(S['scenarios'])\nsc[sc.strategy=='p_logreg'].pivot_table(index='capacity_scale', columns='demand_scale', values='leads_unfilled')"),
("md", "### Interpretation / Decision\nUnder demand +25% and capacity −25% the number of leads that receive no offer rises steeply: ranking alone cannot solve a capacity problem. That is the hand-over point to **Project 3**, which allocates capacity across leads jointly instead of lead-by-lead."),
],
}

P2 = {
"01_quality_privacy_discovery.ipynb": [
("md", "# Project 2 — From Unstructured Service Feedback to Actionable Intelligence\n## 01 · Text quality, privacy, unsupervised discovery\n\n> Synthetic demo text (`common/synthetic.py`). No message is a real customer's or provider's.\n\n" + q(
"What is actually in the feedback corpus, and what must be removed before any model sees it?",
"Nothing yet.",
"Free text is where privacy failures happen. Quality filtering and PII redaction run first and are *counted*; discovery (TF-IDF + NMF) then tells us which themes exist before we commit to a taxonomy.")),
("code", SETUP + "\nfrom project2_feedback_nlp.src import quality, privacy, discovery, taxonomy\nfb = pd.read_csv('data/synthetic/feedback.csv', parse_dates=['created_at']).drop(columns='true_theme')\nfb, qrep = quality.apply(fb); kept = fb[fb.exclusion_reason=='']\nkept, prep = privacy.apply(kept)\nqrep, prep"),
("md", "### Interpretation\n~15% of records are excluded (duplicates, empty, too short, non-English) and ~20% of the remainder contain at least one identifier. Redacted text (`text_redacted`) is the only column any later step touches. Exclusion counts are reported so that volume monitoring is not distorted."),
("code", "vec, X = discovery.vectorise(kept.text_redacted); vocab = vec.get_feature_names_out()\nkdf = discovery.choose_k(X, vocab, [4,6,8,10,12]); kdf.plot(x='k', subplots=True, layout=(1,2), figsize=(9,3)); kdf"),
("code", "_, W, terms = discovery.fit_topics(X, vocab, 8)\ntaxonomy.map_topics_to_taxonomy(terms)"),
("md", "### Interpretation / Decision\nCoherence and stability are both reasonable around k=8; topics map onto recognisable operational issues (refunds, no-shows, app problems, lead quality). One or two topics are lexically driven (e.g. 'clean' appearing in praise) — that is exactly why topics are evidence for a taxonomy, not the taxonomy itself (D2)."),
],
"02_taxonomy_and_categorisation.ipynb": [
("md", "## 02 · Taxonomy, weak supervision, human-in-the-loop\n\n" + q(
"Can feedback be categorised automatically at a quality the ops team would trust, and what is the value of a human-reviewed sample?",
"Notebook 01: 8 discovered themes; a taxonomy of 8 operational categories each with an owner and an action.",
"Three stages evaluated on the same held-out reviewed set: keyword rules → classifier on rule (weak) labels → classifier on weak + reviewed labels. Predictions below a confidence threshold are routed to a review queue rather than forced.")),
("code", SETUP + "\nimport json\nS = json.load(open('outputs/project2/summary.json'))\npd.DataFrame({k: {m: S['categorisation'][k][m] for m in ['macro_f1','coverage','agreement']} for k in ['rule_baseline','model_weak_labels_only','model_weak_plus_reviewed']}).T.round(3)"),
("md", "### Result / Interpretation\nA classifier trained on weak labels alone is *worse* than the rules that produced them (it generalises their mistakes). Adding ~250 human-reviewed labels lifts macro-F1 from ~0.65 to ~0.74 at higher coverage. Human review is not a fallback; it is the highest-leverage input in the pipeline.\n\n### Decision\nBudget a standing review of low-confidence items; retrain monthly on accumulated reviewed labels (D4)."),
("code", "pd.Series(S['categorisation']['model_weak_plus_reviewed']['per_class_f1']).sort_values().plot.barh(title='Per-class F1 (held-out reviewed set)', figsize=(6,4)); plt.tight_layout()"),
("code", "pd.read_csv('outputs/project2/confusion_model.csv', index_col=0)"),
("md", "### Interpretation\n`safety_issue` is the weak class: small, lexically diverse and overlapping with praise about tidy, well-finished work. Confusion sits mostly between it and `workmanship_quality`. This is the first target for the next review batch — a concrete, evidence-backed next step rather than 'improve the model'."),
],
"03_emerging_issues_and_priorities.ipynb": [
("md", "## 03 · Emerging issues, prioritisation, monitoring\n\n" + q(
"Which issues are *growing*, and which should the ops lead look at first this week?",
"Notebook 02: every message has a category or sits in the review queue.",
"Emergence is measured relative to total volume (a marketing push lifts every category) with a Poisson test on the recent count. Priority = share × severity × growth × confidence, with the severity rubric written down and versioned.")),
("code", SETUP + "\ncounts = pd.read_csv('outputs/project2/weekly_counts.csv', index_col=0, parse_dates=True)\n(counts.div(counts.sum(1), axis=0)).plot(figsize=(11,4), title='Weekly category share'); plt.legend(bbox_to_anchor=(1,1)); plt.tight_layout()"),
("code", "pd.read_csv('outputs/project2/emerging_issues.csv')"),
("code", "pd.read_csv('outputs/project2/priorities.csv', index_col=0)"),
("md", "### Interpretation\n`credits_refund` is flagged as emerging (rate ratio ≈ 2.7, p < 0.001) and rises to the top of the priority list despite not being the most frequent category — severity and growth outweigh raw volume. The output names the owner (finance ops) and the action.\n\n### Decision\nThis table, plus the redacted examples and the review queue, is the weekly operational deliverable. Monitoring watches category-share drift and the size of the review queue (both in `summary.json`)."),
],
}

P3 = {
"01_problem_and_baseline.ipynb": [
("md", "# Project 3 — Capacity and Resource Allocation Optimisation\n## 01 · Formulation and greedy baseline\n\n" + q(
"Given a week's leads, provider capacity and match quality, how should notifications be allocated?",
"Project 1 showed that lead-by-lead ranking leaves leads unfilled once capacity tightens.",
"Formulate as a MILP (see `src/formulation.py` docstring): binary notify decisions, per-lead and per-provider limits, soft penalties for unfilled leads and for providers who receive nothing. The greedy baseline is the current behaviour formalised.")),
("code", SETUP + "\nfrom project3_allocation.src import instance, analysis\ninst = instance.build('data/synthetic', '2026-01-19', 'outputs/project1/model_logreg.joblib')\nprint(len(inst.leads), 'leads', inst.capacity.sum(), 'capacity', len(inst.pairs), 'feasible pairs')\nanalysis.compare_solvers(inst).set_index('solver').T"),
("md", "### Result / Interpretation\nThe optimiser lifts expected purchases by ~11% over greedy, fills every lead and leaves no credit-holding provider without a lead — all on the same capacity. Greedy is myopic: early leads consume providers a later priority lead needed.\n\n### Decision\nMILP with CBC (D2); weekly batch with a re-solve when new leads arrive."),
],
"02_tradeoffs_constraints_scenarios.ipynb": [
("md", "## 02 · Trade-offs, binding constraints, scenarios, robustness\n\n" + q(
"What does fairness cost, which constraint actually limits the business, and how does the allocation behave under stress?",
"Notebook 01: the MILP dominates greedy on this instance.",
"Instead of asserting weights, sweep the fairness weight and show the frontier; measure the marginal value of relaxing each constraint family; replay demand/capacity scenarios; perturb match quality to see how stable the allocation is.")),
("code", SETUP + "\ntr = pd.read_csv('outputs/project3/fairness_tradeoff.csv')\ntr.plot(x='lambda_fairness', y=['expected_purchases','providers_with_zero'], subplots=True, layout=(1,2), figsize=(9,3)); tr[['lambda_fairness','expected_purchases','providers_with_zero','gini_provider_load']]"),
("md", "### Interpretation\nFairness is nearly free here: λ=0.3 removes all zero-lead providers at a cost of <0.5 expected purchases. That is the recommendation — and the chart is what makes it defensible."),
("code", "pd.read_csv('outputs/project3/constraint_analysis.csv')"),
("md", "### Interpretation\nThe binding constraint is **offers per lead (M)**, then provider capacity; the priority minimum costs nothing. Raising M is a product decision with a revenue upside and a lead-quality downside (Project 2 shows providers already complain about wasted credits) — the optimiser quantifies one side of that trade."),
("code", "pd.read_csv('outputs/project3/scenarios.csv')"),
("code", "pd.read_csv('outputs/project3/robustness.csv')"),
("md", "### Interpretation / Decision\nUnder demand +25% the optimiser still fills every lead where greedy leaves 18 unfilled. With noise on match quality the *specific* assignment changes a lot (Jaccard 0.4–0.75) but the regret is small (<3% at σ=0.05): many near-equivalent allocations exist, so the recommendation is robust in value even when it is not unique. Output = recommended allocation + reasons + unfilled list + scenario table (`outputs/project3/`)."),
],
}


def strip_local_paths(n):
    """Executed cells capture tracebacks and warnings containing the absolute
    path of the machine that ran them. Make those paths repo-relative so the
    committed notebooks do not carry anyone's home directory."""
    for cell in n.cells:
        for out in cell.get("outputs", []):
            for key in ("text", "evalue"):
                if isinstance(out.get(key), str):
                    out[key] = out[key].replace(ROOT + os.sep, "")
            out["traceback"] = [t.replace(ROOT + os.sep, "") for t in out.get("traceback", [])]
    return n


def build(execute: bool):
    for proj, spec in [("project1_matching", P1), ("project2_feedback_nlp", P2), ("project3_allocation", P3)]:
        d = os.path.join(ROOT, proj, "notebooks"); os.makedirs(d, exist_ok=True)
        for name, cells in spec.items():
            path = os.path.join(d, name)
            nbf.write(nb(cells), path)
            if execute:
                from nbconvert.preprocessors import ExecutePreprocessor
                n = nbf.read(path, as_version=4)
                ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(n, {"metadata": {"path": d}})
                nbf.write(strip_local_paths(n), path)
            print("built", path)


if __name__ == "__main__":
    build(execute="--execute" in sys.argv)
