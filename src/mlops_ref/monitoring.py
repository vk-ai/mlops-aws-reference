"""Performance monitoring once delayed labels arrive; data drift is only an early warning.

Credit outcomes (did the borrower default?) arrive months after the score. Until
then the only signal is input drift, and drift is a poor proxy for performance:
features can shift while the model stays accurate, and performance can drop with
no feature drift at all (concept drift). So this module

1. logs every scored prediction (SQLite: id, time, model version, score),
2. joins labels when they arrive (``record_label``),
3. computes AUC / expected cost / accuracy on the *labelled* part of a window,
   with a bootstrap CI, and compares them with the reference metrics from
   training, and
4. decides a status:

   * ``insufficient_labels``  too few labels (or too low a label share) to judge;
     if drift fired, it's reported as an early warning but nothing pages.
   * ``degraded``  AUC is below the reference by more than ``max_auc_drop`` *and*
     the CI's upper bound is below the reference (or expected cost is clearly up).
   * ``ok``  labels say performance holds, even if drift fired
     (``drift_without_degradation`` is set so the drift alert can be closed).
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .metrics import classification_metrics, roc_auc

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
  id TEXT PRIMARY KEY,
  scored_at REAL NOT NULL,
  version TEXT NOT NULL,
  score REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS labels (
  id TEXT PRIMARY KEY REFERENCES predictions(id),
  label INTEGER NOT NULL CHECK (label IN (0, 1)),
  labelled_at REAL NOT NULL
);
"""


@dataclass
class MonitorRules:
    min_labels: int = 100          # below this, don't judge performance
    min_label_share: float = 0.3   # ... or when under 30% of the window is labelled
    max_auc_drop: float = 0.05     # vs the reference (training-time test) AUC
    max_cost_increase: float = 0.15
    n_bootstrap: int = 1000
    seed: int = 0


def _auc_ci(y: np.ndarray, p: np.ndarray, n: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue  # single-class resample: AUC undefined
        vals.append(roc_auc(y[idx], p[idx]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def _cost_ci(y: np.ndarray, p: np.ndarray, n: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed + 1)
    pred = (p >= 0.5).astype(int)
    per_row = 5.0 * ((pred == 0) & (y == 1)) + 1.0 * ((pred == 1) & (y == 0))
    means = [per_row[rng.integers(0, len(y), len(y))].mean() for _ in range(n)]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


class PerformanceMonitor:
    def __init__(self, path: str | Path = ":memory:", clock=time.time):
        self.db = sqlite3.connect(str(path))
        self.db.executescript(SCHEMA)
        self.clock = clock

    def log_prediction(self, pred_id: str, score: float, version: str, scored_at: float | None = None) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO predictions(id, scored_at, version, score) VALUES (?,?,?,?)",
            (pred_id, self.clock() if scored_at is None else scored_at, version, float(score)),
        )
        self.db.commit()

    def record_label(self, pred_id: str, label: int, labelled_at: float | None = None) -> bool:
        """Join a late-arriving label. Returns False for unknown prediction ids (logged nowhere)."""
        if self.db.execute("SELECT 1 FROM predictions WHERE id = ?", (pred_id,)).fetchone() is None:
            return False
        self.db.execute(
            "INSERT OR REPLACE INTO labels(id, label, labelled_at) VALUES (?,?,?)",
            (pred_id, int(label), self.clock() if labelled_at is None else labelled_at),
        )
        self.db.commit()
        return True

    def report(self, reference: dict, start: float = float("-inf"), end: float = float("inf"),
               version: str | None = None, drift: dict | None = None, rules: MonitorRules | None = None) -> dict:
        """Performance for predictions scored in [start, end), judged against ``reference`` metrics."""
        rules = rules or MonitorRules()
        q = "SELECT p.score, l.label, p.scored_at, l.labelled_at FROM predictions p LEFT JOIN labels l ON l.id = p.id WHERE p.scored_at >= ? AND p.scored_at < ?"
        args: list = [start, end]
        if version:
            q += " AND p.version = ?"
            args.append(version)
        rows = self.db.execute(q, args).fetchall()
        n_scored = len(rows)
        lab = [(s, y, sa, la) for s, y, sa, la in rows if y is not None]
        n_lab = len(lab)
        share = n_lab / n_scored if n_scored else 0.0
        drift_alert = bool(drift and drift.get("alert"))
        out: dict = {
            "version": version,
            "n_scored": n_scored,
            "n_labelled": n_lab,
            "label_share": round(share, 4),
            "reference": {k: reference[k] for k in ("auc", "expected_cost") if k in reference},
            "drift_alert": drift_alert,
            "rules": rules.__dict__,
        }
        if lab:
            lags = np.array([la - sa for _, _, sa, la in lab])
            out["label_lag_days"] = {"median": round(float(np.median(lags)) / 86400, 1), "max": round(float(lags.max()) / 86400, 1)}
        enough = n_lab >= rules.min_labels and share >= rules.min_label_share
        if not enough:
            out["status"] = "insufficient_labels"
            out["early_warning"] = drift_alert
            out["reason"] = (
                f"{n_lab} labels ({share:.0%} of {n_scored}); need >= {rules.min_labels} and >= {rules.min_label_share:.0%}. "
                + ("Drift fired: early warning only, not a performance verdict." if drift_alert else "No drift alert.")
            )
            return out
        y = np.array([r[1] for r in lab], dtype=int)
        p = np.array([r[0] for r in lab], dtype=float)
        if y.min() == y.max():
            out["status"] = "insufficient_labels"
            out["early_warning"] = drift_alert
            out["reason"] = "labels are all one class; AUC undefined"
            return out
        m = classification_metrics(y, p)
        lo, hi = _auc_ci(y, p, rules.n_bootstrap, rules.seed)
        clo, chi = _cost_ci(y, p, rules.n_bootstrap, rules.seed)
        out["current"] = {**m, "auc_ci95": [round(lo, 4), round(hi, 4)], "expected_cost_ci95": [round(clo, 4), round(chi, 4)]}
        ref_auc = reference["auc"]
        auc_bad = m["auc"] < ref_auc - rules.max_auc_drop and hi < ref_auc
        ref_cost = reference.get("expected_cost")
        cost_bad = ref_cost is not None and m["expected_cost"] > ref_cost + rules.max_cost_increase and clo > ref_cost
        reasons = []
        if auc_bad:
            reasons.append(f"AUC {m['auc']:.3f} (95% CI [{lo:.3f}, {hi:.3f}]) below reference {ref_auc:.3f} - {rules.max_auc_drop}")
        if cost_bad:
            reasons.append(f"expected cost {m['expected_cost']:.3f} (CI low {clo:.3f}) above reference {ref_cost:.3f} + {rules.max_cost_increase}")
        out["status"] = "degraded" if reasons else "ok"
        out["early_warning"] = False
        out["drift_without_degradation"] = drift_alert and not reasons
        out["degradation_without_drift"] = bool(reasons) and not drift_alert
        out["reason"] = "; ".join(reasons) or f"AUC {m['auc']:.3f} [{lo:.3f}, {hi:.3f}] vs reference {ref_auc:.3f}: within tolerance"
        return out


DAY = 86400.0


def replay(monitor: PerformanceMonitor, scores: np.ndarray, labels: np.ndarray, version: str, start: float,
           label_delay_days: float = 60.0, arrived_share: float = 1.0, seed: int = 0, prefix: str = "r") -> None:
    """Log one scored batch (one row per day-ish) and the labels that have *arrived so far*."""
    rng = np.random.default_rng(seed)
    for i, (s, y) in enumerate(zip(scores, labels)):
        pid = f"{prefix}-{i}"
        t = start + i * (DAY / 4)
        monitor.log_prediction(pid, float(s), version, scored_at=t)
        if rng.random() < arrived_share:
            monitor.record_label(pid, int(y), labelled_at=t + label_delay_days * DAY * (0.5 + rng.random()))


def scenarios(model, train_df, test_df, reference: dict, share_threshold: float = 0.2, seed: int = 0) -> dict:
    """Four replays on the real held-out German-credit rows. What is simulated is stated per scenario."""
    from .data import TARGET
    from .drift import drift_report
    from .features import MODEL_FEATURES, simulate_current_batch
    from .train import predict_proba

    rng = np.random.default_rng(seed)
    out = {}

    def run(name, df, labels, note, arrived_share=1.0):
        mon = PerformanceMonitor()
        replay(mon, predict_proba(model, df), labels, "v1", 0.0, arrived_share=arrived_share, seed=seed, prefix=name)
        drift = drift_report(train_df, df, MODEL_FEATURES, share_threshold)
        rep = mon.report(reference, drift=drift)
        rep["drift_share"] = drift["drift_share"]
        rep["scenario"] = note
        out[name] = rep

    # 1. Real rows, real labels, selected toward older applicants with larger loans: genuine covariate shift.
    w = np.exp(0.15 * (test_df.age_years - test_df.age_years.mean()) + 0.0008 * (test_df.credit_amount - test_df.credit_amount.mean()))
    sel = test_df.iloc[np.sort(rng.choice(len(test_df), 120, replace=False, p=(w / w.sum()).to_numpy()))].reset_index(drop=True)
    run("covariate_shift_real_rows", sel, sel[TARGET].to_numpy(), "real rows + real labels, re-weighted sample (older, larger loans)")
    # 2. The repo's synthetic shifted batch (features edited), real labels kept.
    b = simulate_current_batch(test_df, seed=seed, strength=1.0)
    run("synthetic_feature_shift", b, b[TARGET].to_numpy(), "features edited by simulate_current_batch; labels are the original ones")
    # 3. Concept drift: features untouched, 35% of labels flipped (SIMULATED).
    y = test_df[TARGET].to_numpy().copy()
    flip = rng.random(len(y)) < 0.35
    y[flip] = 1 - y[flip]
    run("concept_drift_no_feature_drift", test_df, y, "features unchanged; 35% of labels flipped (simulated concept drift)")
    # 4. Same shifted batch, but only 20% of labels have arrived yet.
    run("labels_not_arrived", b, b[TARGET].to_numpy(), "synthetic shifted batch; only ~20% of labels arrived", arrived_share=0.2)
    return out
