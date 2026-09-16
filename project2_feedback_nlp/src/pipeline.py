"""End-to-end run for Project 2. `python -m project2_feedback_nlp.src.pipeline configs/project2.yaml`

raw text → validation → quality → privacy → discovery → taxonomy →
categorisation → emerging issues → prioritisation → operational output."""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from common.validation import Schema, validate
from common.monitoring import get_logger, save_json, load_config, category_drift
from . import quality, privacy, discovery, taxonomy, classifier, emerging, priority

log = get_logger("p2.pipeline")
FEEDBACK_SCHEMA = Schema(required={"feedback_id": "O", "created_at": "M", "author_type": "O", "channel": "O", "text": "O"},
                         key="feedback_id", categories={"author_type": {"customer", "provider"}, "channel": {"review", "support", "survey"}},
                         max_missing={"text": 0.05})


def run(cfg: dict) -> dict:
    out = cfg["output_dir"]; os.makedirs(out, exist_ok=True)
    S = {}
    fb = pd.read_csv(f"{cfg['data_dir']}/feedback.csv", parse_dates=["created_at"])
    hidden = fb.pop("true_theme")           # evaluation-only labels, removed before processing
    rep = validate(fb, FEEDBACK_SCHEMA, "feedback_raw"); rep.to_frame().to_csv(f"{out}/validation_report.csv", index=False)
    S["validation"] = {"errors": len(rep.errors), "warnings": len(rep.warnings)}
    log.info("validation: %s", S["validation"])

    # quality
    fb, S["quality"] = quality.apply(fb, cfg["min_tokens"]); log.info("quality: %s", S["quality"])
    kept = fb[fb.exclusion_reason == ""].copy()
    # privacy
    kept, S["privacy"] = privacy.apply(kept); log.info("privacy: %s", S["privacy"])
    assert not kept.text_redacted.str.contains(r"\d{10}|@", regex=True).any(), "PII leaked past redaction"
    texts = kept.text_redacted

    # exploratory
    S["exploration"] = {"n_docs": len(kept), "by_author": kept.author_type.value_counts().to_dict(),
                        "by_channel": kept.channel.value_counts().to_dict(),
                        "tokens_p50_p90": [float(kept.n_tokens.quantile(.5)), float(kept.n_tokens.quantile(.9))],
                        "weekly_volume": {str(k): int(v) for k, v in kept.created_at.dt.to_period("W").value_counts().sort_index().items()}}

    # discovery
    vec, X = discovery.vectorise(texts); vocab = vec.get_feature_names_out()
    kdf = discovery.choose_k(X, vocab, cfg["topic_ks"]); kdf.to_csv(f"{out}/topic_k_selection.csv", index=False)
    _, W, terms = discovery.fit_topics(X, vocab, cfg["n_topics"])
    tmap = taxonomy.map_topics_to_taxonomy(terms); tmap["doc_share"] = np.bincount(W.argmax(1), minlength=len(terms)) / len(W)
    tmap.to_csv(f"{out}/topics.csv", index=False)
    S["discovery"] = {"k_selection": kdf.round(3).to_dict(orient="records"), "topics": tmap.round(3).to_dict(orient="records")}
    log.info("topics:\n%s", tmap.to_string())

    # categorisation. Three stages, each evaluated on the SAME held-out
    # human-reviewed set so the marginal value of human labels is visible:
    #   (a) keyword rules (transparent baseline)
    #   (b) classifier trained on weak (rule) labels only
    #   (c) classifier trained on weak labels + a human-reviewed training sample
    weak = classifier.weak_labels(texts)
    S["rule_baseline_unassigned_share"] = float((weak == "unassigned").mean())
    rng = np.random.default_rng(0)
    reviewed = rng.choice(kept.index, min(cfg["review_sample_size"] + cfg["holdout_size"], len(kept)), replace=False)
    hold_idx, rev_train_idx = reviewed[:cfg["holdout_size"]], reviewed[cfg["holdout_size"]:]
    unlab_idx = kept.index.difference(reviewed)
    truth = hidden.loc[hold_idx]
    # (a)
    ev_rule = classifier.evaluate(truth, weak.loc[hold_idx].replace("unassigned", "needs_review"))
    # (b)
    m_weak = classifier.fit(texts.loc[unlab_idx], weak.loc[unlab_idx])
    ev_weak = classifier.evaluate(truth, classifier.predict(m_weak, texts.loc[hold_idx], cfg["abstain_below"]).category)
    # (c) human labels override weak labels where reviewed
    lab = pd.concat([weak.loc[unlab_idx], hidden.loc[rev_train_idx]])
    model = classifier.fit(texts.loc[lab.index], lab)
    pred = classifier.predict(model, texts, cfg["abstain_below"])
    ev_model = classifier.evaluate(truth, pred.loc[hold_idx, "category"])
    ev_model["confusion"].to_csv(f"{out}/confusion_model.csv")
    S["categorisation"] = {"holdout_size": int(len(hold_idx)), "reviewed_train_size": int(len(rev_train_idx)),
                           "rule_baseline": {k: v for k, v in ev_rule.items() if k != "confusion"},
                           "model_weak_labels_only": {k: v for k, v in ev_weak.items() if k != "confusion"},
                           "model_weak_plus_reviewed": {k: v for k, v in ev_model.items() if k != "confusion"},
                           "needs_review_share_all": float((pred.category == "needs_review").mean())}
    for k in ["rule_baseline", "model_weak_labels_only", "model_weak_plus_reviewed"]:
        log.info("%-26s macro_f1=%.3f coverage=%.3f agreement=%.3f", k, *[S["categorisation"][k][m] for m in ["macro_f1", "coverage", "agreement"]])
    kept = kept.join(pred)

    # emerging issues
    counts = emerging.weekly_counts(kept[kept.category != "needs_review"]); counts.to_csv(f"{out}/weekly_counts.csv")
    em = emerging.detect(counts, cfg["recent_weeks"], cfg["baseline_weeks"], cfg["ratio_threshold"], cfg["alpha"])
    em.to_csv(f"{out}/emerging_issues.csv", index=False); S["emerging"] = em.to_dict(orient="records")
    log.info("emerging:\n%s", em.to_string())

    # prioritisation
    pr = priority.compute(kept, em, cfg["recent_weeks"]); pr.to_csv(f"{out}/priorities.csv")
    pr["owner"] = pr.index.map(lambda c: taxonomy.TAXONOMY[c][0]); pr["action"] = pr.index.map(lambda c: taxonomy.TAXONOMY[c][1])
    S["priorities"] = pr.reset_index().rename(columns={"index": "category"}).to_dict(orient="records")
    log.info("priorities:\n%s", pr.to_string())

    # explainability: per category — key terms, representative (redacted) examples, confidence
    lr, tf = model.named_steps["lr"], model.named_steps["tfidf"]; names = tf.get_feature_names_out()
    expl = {}
    for i, c in enumerate(lr.classes_):
        top_terms = [names[j] for j in np.argsort(lr.coef_[i])[::-1][:8]]
        ex = kept[kept.category == c].nlargest(3, "confidence").text_redacted.tolist()
        expl[c] = {"key_terms": top_terms, "examples_redacted": ex,
                   "mean_confidence": float(kept[kept.category == c].confidence.mean())}
    S["explanations"] = expl
    # human-in-the-loop queue
    queue = kept[kept.category == "needs_review"][["feedback_id", "created_at", "channel", "text_redacted", "confidence"]]
    queue.to_csv(f"{out}/review_queue.csv", index=False)
    # monitoring: category-share drift, first half vs second half
    mid = kept.created_at.median()
    S["monitoring_category_drift"] = category_drift(kept[kept.created_at <= mid].category, kept[kept.created_at > mid].category).round(3).to_dict(orient="index")
    save_json(S, f"{out}/summary.json")
    kept.drop(columns=["text", "text_clean"]).to_csv(f"{out}/feedback_categorised.csv", index=False)
    return S


if __name__ == "__main__":
    run(load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/project2.yaml"))
