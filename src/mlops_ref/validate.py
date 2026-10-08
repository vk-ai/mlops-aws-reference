"""Lightweight data-contract checks run before training (a Great Expectations-style gate, in pandas)."""
from __future__ import annotations

import pandas as pd

from .data import FEATURES, TARGET


class ValidationError(ValueError):
    pass


def validate(df: pd.DataFrame, min_rows: int = 500) -> dict:
    problems = []
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        problems.append(f"missing columns {missing}")
    else:
        if len(df) < min_rows:
            problems.append(f"only {len(df)} rows (< {min_rows})")
        nulls = df[FEATURES + [TARGET]].isna().sum()
        if nulls.any():
            problems.append(f"nulls in {nulls[nulls > 0].to_dict()}")
        if not set(df[TARGET].unique()) <= {0, 1}:
            problems.append("target must be 0/1")
        if (df["duration_months"] <= 0).any() or (df["credit_amount"] <= 0).any():
            problems.append("duration_months and credit_amount must be > 0")
        if not df["age_years"].between(18, 100).all():
            problems.append("age_years outside 18..100")
        rate = float(df[TARGET].mean()) if len(df) else 0.0
        if not 0.05 <= rate <= 0.6:
            problems.append(f"default rate {rate:.2f} outside 0.05..0.6")
    if problems:
        raise ValidationError("; ".join(problems))
    return {"rows": int(len(df)), "default_rate": round(float(df[TARGET].mean()), 4)}
