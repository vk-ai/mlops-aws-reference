"""MLflow tracking and model registry (local SQLite + artifact dir by default).

Aliases (MLflow >= 2.3) replace stages: ``challenger`` is the newest registered
version, ``champion`` is what production serves. ``promote_if_better`` moves
``champion`` only when the challenger passes the gate.
"""
from __future__ import annotations

import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

from .features import MODEL_FEATURES

MODEL_NAME = "credit-risk"


def default_uri(root: str | Path = ".") -> str:
    return f"sqlite:///{(Path(root) / 'mlflow.db').resolve()}"


def configure(tracking_uri: str | None = None, experiment: str = "credit-risk", artifact_root: str | Path | None = None) -> str:
    """Local by default: SQLite backend store + artifacts on local disk.

    MLflow 3.x put the plain ``./mlruns`` file store into maintenance mode (it raises
    unless MLFLOW_ALLOW_FILE_STORE=true), so the "local file store" here is a SQLite
    file plus a local artifact directory. Point MLFLOW_TRACKING_URI at a server for
    anything shared.
    """
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or default_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        root = artifact_root
        if root is None and uri.startswith("sqlite:///"):
            root = Path(uri[len("sqlite:///"):]).parent / "mlartifacts"
        mlflow.create_experiment(experiment, artifact_location=Path(root).resolve().as_uri() if root else None)
    mlflow.set_experiment(experiment)
    return uri


def log_and_register(model, params: dict, metrics: dict, sample: pd.DataFrame, tags: dict | None = None, name: str = MODEL_NAME) -> dict:
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.set_tags(tags or {})
        x = sample[MODEL_FEATURES].to_numpy(np.float32)[:5]
        mlflow.log_dict({"features": MODEL_FEATURES}, "feature_order.json")
        signature = infer_signature(x, np.zeros(len(x), dtype=np.float32))
        # MLflow 3 defaults to the traced 'pt2' format, which needs a numpy/tensor input example.
        info = mlflow.pytorch.log_model(model, name="model", signature=signature, input_example=x, registered_model_name=name)
    client = MlflowClient()
    version = info.registered_model_version
    client.set_registered_model_alias(name, "challenger", version)
    return {"run_id": run.info.run_id, "model_uri": info.model_uri, "version": str(version)}


def alias_version(alias: str, name: str = MODEL_NAME) -> str | None:
    try:
        return str(MlflowClient().get_model_version_by_alias(name, alias).version)
    except Exception:  # alias not set yet
        return None


def version_metric(version: str, metric: str, name: str = MODEL_NAME) -> float:
    client = MlflowClient()
    run_id = client.get_model_version(name, version).run_id
    return float(client.get_run(run_id).data.metrics[metric])


def promote_if_better(version: str, min_auc: float, max_auc_drop: float, name: str = MODEL_NAME) -> dict:
    """Gate: absolute floor, and not worse than the current champion by more than ``max_auc_drop``."""
    auc = version_metric(version, "test_auc", name)
    champion = alias_version("champion", name)
    reasons = []
    if auc < min_auc:
        reasons.append(f"test_auc {auc:.4f} < floor {min_auc}")
    champ_auc = None
    if champion and champion != version:
        champ_auc = version_metric(champion, "test_auc", name)
        if auc < champ_auc - max_auc_drop:
            reasons.append(f"test_auc {auc:.4f} < champion v{champion} {champ_auc:.4f} - {max_auc_drop}")
    passed = not reasons
    return {"version": version, "test_auc": auc, "champion": champion, "champion_auc": champ_auc, "passed": passed, "reasons": reasons}


def set_alias(alias: str, version: str, name: str = MODEL_NAME) -> None:
    MlflowClient().set_registered_model_alias(name, alias, version)


def load(alias_or_version: str, name: str = MODEL_NAME):
    uri = f"models:/{name}@{alias_or_version}" if not alias_or_version.isdigit() else f"models:/{name}/{alias_or_version}"
    return mlflow.pytorch.load_model(uri)
