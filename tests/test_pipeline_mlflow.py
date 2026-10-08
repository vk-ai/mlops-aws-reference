"""End-to-end: the DAG's task functions against a throwaway SQLite MLflow registry."""
import copy
import json

import pytest
from fastapi.testclient import TestClient

from mlops_ref import pipeline, tracking
from mlops_ref.serving import create_app
from test_serving import FEATURES


@pytest.fixture(scope="module")
def runs(tmp_path_factory, fast_params):
    work = tmp_path_factory.mktemp("work")
    uri = f"sqlite:///{work / 'mlflow.db'}"
    first = pipeline.run_all(work, fast_params, uri, drift_strength=1.0)
    second = pipeline.run_all(work, fast_params, uri, drift_strength=0.0)
    strict = copy.deepcopy(fast_params)
    strict["gate"]["min_auc"] = 0.99
    third = pipeline.run_all(work, strict, uri, drift_strength=0.0)
    return work, uri, first, second, third


def test_first_run_becomes_champion(runs):
    work, uri, first, *_ = runs
    assert first["register"]["version"] == "1"
    assert first["gate"]["passed"] and first["gate"]["champion"] is None
    assert first["deploy"] == {"action": "initial_release", "stable": "1", "canary": None, "weight": 0}
    assert first["drift"]["alert"] is True and first["drift"]["alert_sent"] is True
    assert (work / "reports" / "drift_report.html").exists()


def test_second_run_goes_to_canary(runs):
    work, uri, _, second, _ = runs
    assert second["gate"]["passed"] and second["gate"]["champion"] == "1"
    assert second["deploy"] == {"action": "canary", "stable": "1", "canary": "2", "weight": 10}
    assert second["drift"]["alert"] is False
    tracking.configure(uri)
    assert (tracking.alias_version("champion"), tracking.alias_version("canary")) == ("1", "2")


def test_gate_failure_is_not_deployed(runs):
    work, uri, *_, third = runs
    assert third["gate"]["passed"] is False and "floor 0.99" in third["gate"]["reasons"][0]
    assert third["deploy"]["action"] == "rejected"
    assert json.loads((work / "deploy" / "canary.json").read_text())["action"] == "rejected"
    tracking.configure(uri)
    assert tracking.alias_version("challenger") == "3" and tracking.alias_version("champion") == "1"


def test_registry_metadata(runs):
    _, uri, first, *_ = runs
    tracking.configure(uri)
    from mlflow import MlflowClient

    run = MlflowClient().get_run(first["register"]["run_id"])
    assert run.data.tags["data_sha256"] == first["ingest"]["sha256"]
    assert run.data.params["hidden"] == "[32, 16]"
    assert run.data.metrics["test_auc"] == pytest.approx(first["evaluate"]["auc"])


def test_serve_alias_pins_one_model(runs, monkeypatch):
    _, uri, *_ = runs
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("SERVE_ALIAS", "canary")
    c = TestClient(create_app())
    assert {c.post("/predict", json={"request_id": f"r{i}", "features": FEATURES}).json()["version"] for i in range(20)} == {"v2"}
    assert c.get("/canary").json()["decision"] == "no_canary"
    monkeypatch.setenv("SERVE_ALIAS", "nope")
    assert TestClient(create_app()).post("/predict", json={"request_id": "x", "features": FEATURES}).status_code == 503


def test_scoring_service_loads_champion_and_canary_from_registry(runs, monkeypatch):
    _, uri, *_ = runs
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("CANARY_WEIGHT", "50")
    c = TestClient(create_app())
    seen = {c.post("/predict", json={"request_id": f"r{i}", "features": FEATURES}).json()["version"] for i in range(40)}
    assert seen == {"v1", "v2"}
    status = c.get("/canary").json()
    assert (status["stable"], status["canary"], status["decision"]) == ("v1", "v2", "continue")
    assert c.post("/canary/promote").json() == {"stable": "v2"}
    tracking.configure(uri)
    assert tracking.alias_version("champion") == "2"  # on_promote moved the alias
