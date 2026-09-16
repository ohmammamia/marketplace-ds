"""Automated categorisation.

Bootstrapping: rule labels (taxonomy seeds) → TF-IDF + logistic regression.
The learnt model generalises beyond the seed phrases; its confidence gates a
human-review queue. Evaluation uses a held-out *human-reviewed* set — in the
demo this is simulated by the generator's hidden theme labels, restricted
to a sample of the size a real review would produce."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from .taxonomy import CATEGORIES, rule_label


def make() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True)),
        ("lr", LogisticRegression(C=3.0, max_iter=3000, class_weight="balanced")),
    ])


def weak_labels(texts: pd.Series) -> pd.Series:
    return texts.map(rule_label)


def fit(texts: pd.Series, labels: pd.Series) -> Pipeline:
    m = texts.index[labels != "unassigned"]
    return make().fit(texts.loc[m], labels.loc[m])


def predict(model: Pipeline, texts: pd.Series, abstain_below: float = 0.5) -> pd.DataFrame:
    P = model.predict_proba(texts)
    idx = P.argmax(1)
    out = pd.DataFrame({"category": model.classes_[idx], "confidence": P.max(1)}, index=texts.index)
    out.loc[out.confidence < abstain_below, "category"] = "needs_review"
    return out


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> dict:
    m = y_pred != "needs_review"
    labels = [c for c in CATEGORIES if c in set(y_true)]
    rep = classification_report(y_true[m], y_pred[m], labels=labels, output_dict=True, zero_division=0)
    return {"macro_f1": float(f1_score(y_true[m], y_pred[m], labels=labels, average="macro", zero_division=0)),
            "coverage": float(m.mean()), "agreement": float((y_true[m] == y_pred[m]).mean()),
            "per_class_f1": {c: round(rep[c]["f1-score"], 3) for c in labels},
            "confusion": pd.DataFrame(confusion_matrix(y_true[m], y_pred[m], labels=labels), index=labels, columns=labels)}
