import itertools

import numpy as np
import pandas as pd
import pytest

from mlops_ref.data import TARGET
from mlops_ref.features import ENGINEERED, add_features, simulate_current_batch, split
from mlops_ref.metrics import classification_metrics, roc_auc


def test_add_features_values(raw):
    f = add_features(raw.head(3))
    row = f.iloc[0]
    assert row["amount_per_month"] == pytest.approx(row["credit_amount"] / row["duration_months"])
    assert row["log_credit_amount"] == pytest.approx(np.log1p(row["credit_amount"]))
    assert set(f["young_applicant"].unique()) <= {0.0, 1.0}
    assert "amount_per_month" not in raw.columns  # input not mutated


def test_split_is_stratified_disjoint_and_deterministic(raw):
    tr, te = split(raw, 0.2, 42)
    tr2, te2 = split(raw, 0.2, 42)
    assert len(tr) == 800 and len(te) == 200
    assert te[TARGET].mean() == pytest.approx(0.3) and tr[TARGET].mean() == pytest.approx(0.3)
    assert te.equals(te2) and tr.equals(tr2)
    both = pd.concat([tr, te]).sort_values(list(raw.columns)).reset_index(drop=True)
    assert both.equals(raw.sort_values(list(raw.columns)).reset_index(drop=True))  # a partition of the rows
    assert not split(raw, 0.2, 1)[1].equals(te)


def test_simulated_batch(featured):
    _, te = featured
    same = simulate_current_batch(te, strength=0)
    assert same[list(te.columns)].equals(te)
    shifted = simulate_current_batch(te, strength=1)
    assert shifted["credit_amount"].mean() > 1.7 * te["credit_amount"].mean()
    assert shifted["age_years"].mean() < te["age_years"].mean()
    assert shifted["amount_per_month"].equals(shifted["credit_amount"] / shifted["duration_months"])
    assert set(ENGINEERED) <= set(shifted.columns)


def brute_auc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a, b in itertools.product(pos, neg))
    return wins / (len(pos) * len(neg))


def test_auc_matches_pairwise_definition_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.round(rng.random(200) + 0.3 * y, 1)  # coarse rounding -> many ties
    assert roc_auc(y, p) == pytest.approx(brute_auc(y, p))
    assert roc_auc(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5
    with pytest.raises(ValueError):
        roc_auc(np.array([1, 1]), np.array([0.1, 0.2]))


def test_classification_metrics_by_hand():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.6, 0.2, 0.7, 0.1, 0.1, 0.3, 0.4])
    m = classification_metrics(y, p)
    # tp=2 fp=1 fn=1 tn=4
    assert (m["precision"], m["recall"], m["accuracy"]) == (pytest.approx(0.6667), pytest.approx(0.6667), 0.75)
    assert m["expected_cost"] == pytest.approx((5 * 1 + 1 * 1) / 8, abs=1e-4)
    assert m["positive_rate"] == 0.375
