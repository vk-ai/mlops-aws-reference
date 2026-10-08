"""Pipeline steps as plain functions.

The Airflow DAG, the DVC stages and the tests all call these. Inputs and outputs
are file paths and small JSON-able dicts, so they pass through XCom cleanly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from . import tracking
from .data import RAW_PATH, SHA256, TARGET, load_raw, verify
from .drift import drift_report, send_alert
from .features import MODEL_FEATURES, add_features, simulate_current_batch, split
from .model import CreditRiskMLP
from .train import evaluate, train
from .validate import validate

ROOT = Path(__file__).resolve().parents[2]


def load_params(path: str | Path = ROOT / "params.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def _write(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return str(path)


def ingest(raw_path: str | Path = RAW_PATH) -> dict:
    return {"raw": str(raw_path), "sha256": verify(Path(raw_path), SHA256)}


def validate_raw(raw: str) -> dict:
    return validate(load_raw(Path(raw)))


def prepare(raw: str, workdir: str | Path, test_size: float, seed: int) -> dict:
    train_df, test_df = split(load_raw(Path(raw)), test_size, seed)
    w = Path(workdir) / "data" / "interim"
    return {"train": _write(train_df, w / "train.parquet"), "test": _write(test_df, w / "test.parquet")}


def featurize(train_path: str, test_path: str, workdir: str | Path, engine: str = "pandas") -> dict:
    w = Path(workdir) / "data" / "processed"
    out = {}
    for name, path in (("train", train_path), ("test", test_path)):
        df = pd.read_parquet(path)
        if engine == "spark":
            from .spark_features import featurize_pandas_via_spark

            feats = featurize_pandas_via_spark(df)[list(df.columns) + ["amount_per_month", "log_credit_amount", "young_applicant"]]
        else:
            feats = add_features(df)
        out[name] = _write(feats, w / f"{name}.parquet")
    out["engine"] = engine
    return out


def save_model(model: CreditRiskMLP, hidden: list[int], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_features": len(MODEL_FEATURES), "hidden": hidden}, path)
    return str(path)


def load_model(path: str | Path) -> CreditRiskMLP:
    blob = torch.load(path, weights_only=True)
    n, hidden = blob["n_features"], blob["hidden"]
    import numpy as np

    model = CreditRiskMLP(n, hidden, np.zeros(n), np.ones(n))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def train_model(train_path: str, workdir: str | Path, p: dict) -> dict:
    res = train(pd.read_parquet(train_path), p["hidden"], p["epochs"], p["lr"], p["weight_decay"], p["batch_size"], p["seed"])
    path = save_model(res.model, p["hidden"], Path(workdir) / "models" / "model.pt")
    return {"model_path": path, "train_metrics": res.train_metrics, "final_loss": round(res.history[-1], 5)}


def evaluate_model(model_path: str, test_path: str, workdir: str | Path | None = None) -> dict:
    metrics = evaluate(load_model(model_path), pd.read_parquet(test_path))
    if workdir:
        out = Path(workdir) / "metrics" / "test_metrics.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def register(model_path: str, test_path: str, metrics: dict, train_params: dict, data_sha256: str, tracking_uri: str | None = None) -> dict:
    tracking.configure(tracking_uri)
    model = load_model(model_path)
    sample = pd.read_parquet(test_path)
    tags = {"data_sha256": data_sha256, "framework": f"torch {torch.__version__}"}
    params = {k: (json.dumps(v) if isinstance(v, list) else v) for k, v in train_params.items()}
    return tracking.log_and_register(model, params, {f"test_{k}": v for k, v in metrics.items()}, sample, tags)


def gate(version: str, min_auc: float, max_auc_drop: float, tracking_uri: str | None = None) -> dict:
    tracking.configure(tracking_uri)
    return tracking.promote_if_better(version, min_auc, max_auc_drop)


def make_current_batch(test_path: str, workdir: str | Path, strength: float, seed: int = 0) -> str:
    cur = simulate_current_batch(pd.read_parquet(test_path), seed=seed, strength=strength)
    return _write(cur, Path(workdir) / "data" / "current" / "batch.parquet")


def drift_check(reference_path: str, current_path: str, workdir: str | Path, share_threshold: float) -> dict:
    ref, cur = pd.read_parquet(reference_path), pd.read_parquet(current_path)
    result = drift_report(ref, cur, MODEL_FEATURES, share_threshold, out_dir=Path(workdir) / "reports")
    result["alert_sent"] = send_alert(result)
    result.pop("columns")
    return result


def deploy_canary(version: str, gate_result: dict, workdir: str | Path, initial_weight: int, tracking_uri: str | None = None) -> dict:
    """Write the deployment spec the router/infra consumes and move registry aliases.

    First model ever: becomes champion directly. Otherwise a passing challenger
    becomes ``canary`` at ``initial_weight`` percent; a failing one is not deployed.
    """
    tracking.configure(tracking_uri)
    champion = tracking.alias_version("champion")
    if not gate_result["passed"]:
        spec = {"action": "rejected", "stable": champion, "canary": None, "weight": 0, "reasons": gate_result["reasons"]}
    elif champion is None:
        tracking.set_alias("champion", version)
        spec = {"action": "initial_release", "stable": version, "canary": None, "weight": 0}
    else:
        tracking.set_alias("canary", version)
        spec = {"action": "canary", "stable": champion, "canary": version, "weight": initial_weight}
    out = Path(workdir) / "deploy" / "canary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2) + "\n")
    return spec


def run_all(workdir: str | Path, params: dict | None = None, tracking_uri: str | None = None, drift_strength: float = 1.0, engine: str | None = None) -> dict:
    """The whole DAG in one process (what `dag.test()` exercises task by task)."""
    p = params or load_params()
    out: dict = {"ingest": ingest()}
    out["validate"] = validate_raw(out["ingest"]["raw"])
    out["prepare"] = prepare(out["ingest"]["raw"], workdir, p["data"]["test_size"], p["data"]["seed"])
    out["features"] = featurize(out["prepare"]["train"], out["prepare"]["test"], workdir, engine or p["features"]["engine"])
    out["train"] = train_model(out["features"]["train"], workdir, p["train"])
    out["evaluate"] = evaluate_model(out["train"]["model_path"], out["features"]["test"], workdir)
    out["register"] = register(out["train"]["model_path"], out["features"]["test"], out["evaluate"], p["train"], out["ingest"]["sha256"], tracking_uri)
    out["gate"] = gate(out["register"]["version"], p["gate"]["min_auc"], p["gate"]["max_auc_drop"], tracking_uri)
    current = make_current_batch(out["features"]["test"], workdir, drift_strength)
    out["drift"] = drift_check(out["features"]["train"], current, workdir, p["drift"]["share_threshold"])
    out["deploy"] = deploy_canary(out["register"]["version"], out["gate"], workdir, p["canary"]["initial_weight"], tracking_uri)
    return out


__all__ = ["TARGET"]
