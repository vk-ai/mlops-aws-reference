"""Feature engineering (pandas). ``spark_features.py`` implements the same transform in PySpark."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FEATURES, TARGET

ENGINEERED = ["amount_per_month", "log_credit_amount", "young_applicant"]
MODEL_FEATURES = FEATURES + ENGINEERED


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["amount_per_month"] = out["credit_amount"] / out["duration_months"]
    out["log_credit_amount"] = np.log1p(out["credit_amount"])
    out["young_applicant"] = (out["age_years"] < 25).astype("float64")
    return out


def split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split without scikit-learn."""
    rng = np.random.default_rng(seed)
    test_idx = []
    for _, group in df.groupby(TARGET):
        n = int(round(len(group) * test_size))
        test_idx.extend(rng.choice(group.index.to_numpy(), size=n, replace=False).tolist())
    test = df.loc[sorted(test_idx)]
    train = df.drop(index=test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def simulate_current_batch(test: pd.DataFrame, seed: int = 0, strength: float = 1.0) -> pd.DataFrame:
    """A *simulated* production batch: larger, longer loans from younger applicants.

    strength=0 returns the held-out test rows unchanged (no intended drift).
    """
    rng = np.random.default_rng(seed)
    cur = test.copy().reset_index(drop=True)
    if strength:
        cur["credit_amount"] = (cur["credit_amount"] * (1 + 0.8 * strength)).round()
        cur["duration_months"] = (cur["duration_months"] + rng.integers(6, 18, len(cur)) * strength).round()
        cur["age_years"] = (cur["age_years"] - 8 * strength).clip(lower=19).round()
    return add_features(cur.drop(columns=[c for c in ENGINEERED if c in cur.columns]))
