"""Lightweight monitoring primitives shared by all three projects."""
from __future__ import annotations

import json
import logging
import os
import numpy as np
import pandas as pd


def get_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s", "%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. Rule of thumb: <0.1 stable, 0.1–0.25 watch, >0.25 drift."""
    ref, cur = np.asarray(reference, float), np.asarray(current, float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    r, _ = np.histogram(ref, edges)
    c, _ = np.histogram(cur, edges)
    r = (r + 0.5) / (r.sum() + 0.5 * len(r))
    c = (c + 0.5) / (c.sum() + 0.5 * len(c))
    return float(np.sum((c - r) * np.log(c / r)))


def category_drift(reference: pd.Series, current: pd.Series) -> pd.DataFrame:
    """Share change per category with a PSI-style contribution."""
    r = reference.value_counts(normalize=True)
    c = current.value_counts(normalize=True)
    idx = r.index.union(c.index)
    r, c = r.reindex(idx, fill_value=0) + 1e-4, c.reindex(idx, fill_value=0) + 1e-4
    out = pd.DataFrame({"reference_share": r, "current_share": c})
    out["contribution"] = (c - r) * np.log(c / r)
    return out.sort_values("contribution", ascending=False)


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)
