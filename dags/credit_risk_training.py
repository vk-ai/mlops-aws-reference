"""Weekly credit-risk training DAG.

ingest -> validate -> prepare -> featurize (pandas or local Spark) -> train (PyTorch)
-> evaluate -> register (MLflow) -> gate -> deploy canary
                          prepare/featurize -> drift check (Evidently, alert)

Every task is a thin wrapper over a plain function in ``mlops_ref.pipeline``, so the
logic is unit-tested without Airflow. ``python dags/credit_risk_training.py`` runs the
whole DAG in-process with ``dag.test()``.

Config (env): MLOPS_WORKDIR (default /tmp/mlops-ref), MLFLOW_TRACKING_URI (default
SQLite in the workdir), DRIFT_WEBHOOK_URL (optional Slack-compatible webhook).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

try:  # Airflow 3
    from airflow.sdk import dag, task
except ImportError:  # Airflow 2.x
    from airflow.decorators import dag, task

WORKDIR = os.environ.get("MLOPS_WORKDIR", "/tmp/mlops-ref")


def _tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{WORKDIR}/mlflow.db"


@dag(
    dag_id="credit_risk_training",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "pytorch", "mlflow", "evidently"],
    params={"feature_engine": "pandas", "drift_strength": 1.0},
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
)
def credit_risk_training():
    @task
    def ingest() -> dict:
        from mlops_ref import pipeline

        return pipeline.ingest()

    @task
    def validate(info: dict) -> dict:
        from mlops_ref import pipeline

        return {**info, "stats": pipeline.validate_raw(info["raw"])}

    @task
    def prepare(info: dict) -> dict:
        from mlops_ref import pipeline

        p = pipeline.load_params()["data"]
        return pipeline.prepare(info["raw"], WORKDIR, p["test_size"], p["seed"])

    @task
    def featurize(paths: dict, params: dict | None = None) -> dict:
        from mlops_ref import pipeline

        engine = (params or {}).get("feature_engine", "pandas")
        return pipeline.featurize(paths["train"], paths["test"], WORKDIR, engine)

    @task
    def train(feats: dict) -> dict:
        from mlops_ref import pipeline

        return pipeline.train_model(feats["train"], WORKDIR, pipeline.load_params()["train"])

    @task
    def evaluate(trained: dict, feats: dict) -> dict:
        from mlops_ref import pipeline

        return pipeline.evaluate_model(trained["model_path"], feats["test"], WORKDIR)

    @task
    def register(trained: dict, feats: dict, metrics: dict, info: dict) -> dict:
        from mlops_ref import pipeline

        return pipeline.register(trained["model_path"], feats["test"], metrics, pipeline.load_params()["train"], info["sha256"], _tracking_uri())

    @task
    def gate(reg: dict) -> dict:
        from mlops_ref import pipeline

        g = pipeline.load_params()["gate"]
        return pipeline.gate(reg["version"], g["min_auc"], g["max_auc_drop"], _tracking_uri())

    @task
    def drift_check(feats: dict, params: dict | None = None) -> dict:
        from mlops_ref import pipeline

        strength = float((params or {}).get("drift_strength", 1.0))
        current = pipeline.make_current_batch(feats["test"], WORKDIR, strength)  # simulated production batch
        return pipeline.drift_check(feats["train"], current, WORKDIR, pipeline.load_params()["drift"]["share_threshold"])

    @task
    def deploy_canary(reg: dict, gate_result: dict) -> dict:
        from mlops_ref import pipeline

        return pipeline.deploy_canary(reg["version"], gate_result, WORKDIR, pipeline.load_params()["canary"]["initial_weight"], _tracking_uri())

    info = validate(ingest())
    feats = featurize(prepare(info))
    trained = train(feats)
    metrics = evaluate(trained, feats)
    reg = register(trained, feats, metrics, info)
    deploy_canary(reg, gate(reg))
    drift_check(feats)


dag_object = credit_risk_training()

if __name__ == "__main__":
    dag_object.test()
