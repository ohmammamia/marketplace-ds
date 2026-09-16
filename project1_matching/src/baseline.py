"""Rule-based baseline: the transparent score a product manager could write
on a whiteboard. Everything learnt later must beat this, or it is not worth
the maintenance cost."""
import numpy as np
import pandas as pd


def rule_score(f: pd.DataFrame) -> np.ndarray:
    return (-0.5 * f.distance_km + 1.0 * f.availability_overlap - 1.0 * f.price_gap_pos
            + 1.0 * f.has_credit + 0.5 * f.gender_pref_met).values


def nearest_score(f: pd.DataFrame) -> np.ndarray:
    """What the legacy platform effectively does: nearest first."""
    return (-f.distance_km).values
