"""Ranking model. Two customers are compared:
  - logistic regression on standardised pairwise features (explainable,
    calibrated-ish, cheap) — the production candidate;
  - histogram gradient boosting (non-linear ceiling check).
Selection is by ranking metrics on a strictly later time period, not by
accuracy. Deep learning / pairwise LTR are not used: with ~10k offers and
17 features they would add cost without evidence of benefit (decision log D5)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from .features import FEATURES

# Position de-biasing: notification order (`rank_shown`) affects purchase in
# the logs but is *chosen* by the ranker, not a property of the pair. It is
# included as a training feature and fixed to 1 at inference so the learnt
# relevance is separated from the historical position effect.
TRAIN_FEATURES = FEATURES + ["rank_shown"]


def temporal_split(df: pd.DataFrame, test_frac: float = 0.25):
    cut = df.created_at.quantile(1 - test_frac)
    return df[df.created_at < cut], df[df.created_at >= cut]


def make_models() -> dict:
    return {
        "logreg": Pipeline([("sc", StandardScaler()),
                            ("lr", LogisticRegression(C=0.5, max_iter=2000, class_weight=None))]),
        "hgb": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                              l2_regularization=1.0, random_state=0),
    }


def fit(models: dict, train: pd.DataFrame, target: str = "purchased") -> dict:
    for m in models.values():
        m.fit(train[TRAIN_FEATURES], train[target])
    return models


def score(model, df: pd.DataFrame) -> np.ndarray:
    x = df[FEATURES].copy(); x["rank_shown"] = 1
    return model.predict_proba(x[TRAIN_FEATURES])[:, 1]


def logreg_contributions(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Per-row feature contributions in log-odds (standardised x * coef)."""
    sc, lr = model.named_steps["sc"], model.named_steps["lr"]
    x = df[FEATURES].copy(); x["rank_shown"] = 1
    z = sc.transform(x[TRAIN_FEATURES]) * lr.coef_[0]
    return pd.DataFrame(z, columns=TRAIN_FEATURES, index=df.index).drop(columns="rank_shown")
