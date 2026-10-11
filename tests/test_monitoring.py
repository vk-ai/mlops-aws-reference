import numpy as np
import pytest

from mlops_ref.monitoring import DAY, MonitorRules, PerformanceMonitor, replay, scenarios
from mlops_ref.train import evaluate, train

REF = {"auc": 0.80, "expected_cost": 0.60}


def _synthetic(n, auc_signal, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    s = 1 / (1 + np.exp(-(auc_signal * (2 * y - 1) + rng.normal(0, 1, n))))
    return s, y


def test_labels_join_late_and_unknown_ids_are_rejected():
    m = PerformanceMonitor()
    m.log_prediction("a", 0.9, "v1", scored_at=0)
    m.log_prediction("b", 0.1, "v1", scored_at=1)
    assert m.record_label("a", 1, labelled_at=90 * DAY)
    assert not m.record_label("zzz", 0)
    r = m.report(REF)
    assert (r["n_scored"], r["n_labelled"], r["label_share"]) == (2, 1, 0.5)
    assert r["label_lag_days"]["median"] == pytest.approx(90.0, abs=0.1)
    assert r["status"] == "insufficient_labels"


def test_drift_alone_is_an_early_warning_not_a_verdict():
    m = PerformanceMonitor()
    s, y = _synthetic(300, 1.5)
    replay(m, s, y, "v1", 0.0, arrived_share=0.1)
    r = m.report(REF, drift={"alert": True})
    assert r["status"] == "insufficient_labels" and r["early_warning"] is True
    assert "early warning only" in r["reason"]


def test_healthy_model_with_drift_is_ok_and_closes_the_drift_alert():
    m = PerformanceMonitor()
    s, y = _synthetic(400, 1.5)
    replay(m, s, y, "v1", 0.0)
    r = m.report(REF, drift={"alert": True})
    assert r["status"] == "ok" and r["drift_without_degradation"] is True
    lo, hi = r["current"]["auc_ci95"]
    assert lo <= r["current"]["auc"] <= hi


def test_degradation_without_drift_is_caught():
    m = PerformanceMonitor()
    s, y = _synthetic(400, 0.2)  # weak signal: AUC ~0.6
    replay(m, s, y, "v1", 0.0)
    r = m.report(REF, drift={"alert": False})
    assert r["status"] == "degraded" and r["degradation_without_drift"] is True
    assert r["current"]["auc_ci95"][1] < REF["auc"]


def test_small_drop_inside_the_ci_is_not_called_degraded():
    m = PerformanceMonitor()
    s, y = _synthetic(150, 1.1, seed=3)
    replay(m, s, y, "v1", 0.0)
    auc = m.report({"auc": 0.5})["current"]["auc"]
    r = m.report({"auc": auc + 0.01}, rules=MonitorRules(max_auc_drop=0.0))
    assert r["current"]["auc_ci95"][1] > auc + 0.01  # reference is inside the CI
    assert r["status"] == "ok"


def test_window_and_version_filters():
    m = PerformanceMonitor()
    s, y = _synthetic(200, 1.5)
    replay(m, s, y, "v1", 0.0, prefix="a")
    replay(m, s, y, "v2", 1000 * DAY, prefix="b")
    assert m.report(REF, version="v2")["n_scored"] == 200
    assert m.report(REF, start=0, end=10 * DAY)["n_scored"] == 40


def test_scenarios_on_real_german_credit_rows(featured, fast_params):
    tr, te = featured
    t = fast_params["train"]
    model = train(tr, t["hidden"], 40, t["lr"], t["weight_decay"], t["batch_size"], t["seed"]).model
    out = scenarios(model, tr, te, evaluate(model, te))
    assert out["concept_drift_no_feature_drift"]["status"] == "degraded"
    assert out["concept_drift_no_feature_drift"]["drift_alert"] is False
    assert out["synthetic_feature_shift"]["drift_alert"] is True
    assert out["synthetic_feature_shift"]["status"] == "ok"
    assert out["labels_not_arrived"]["status"] == "insufficient_labels"
    assert out["labels_not_arrived"]["early_warning"] is True
