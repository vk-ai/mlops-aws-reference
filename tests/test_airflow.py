"""DAG structure. Skipped unless Airflow is installed (`pip install '.[airflow]'` with the constraints file).

The full in-process run is `python dags/credit_risk_training.py` (dag.test()), which CI runs
in the `airflow` job after `airflow db migrate`.
"""
import importlib.util

import pytest

from conftest import ROOT

pytest.importorskip("airflow")
pytestmark = pytest.mark.airflow


@pytest.fixture(scope="module")
def dag():
    spec = importlib.util.spec_from_file_location("credit_risk_training", ROOT / "dags" / "credit_risk_training.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.dag_object


def test_dag_tasks_and_dependencies(dag):
    assert dag.dag_id == "credit_risk_training"
    assert set(dag.task_ids) == {"ingest", "validate", "prepare", "featurize", "train", "evaluate", "register", "gate", "drift_check", "deploy_canary"}
    up = {t.task_id: set(t.upstream_task_ids) for t in dag.tasks}
    assert up["validate"] == {"ingest"}
    assert up["register"] == {"train", "featurize", "evaluate", "validate"}
    assert up["gate"] == {"register"}
    assert up["deploy_canary"] == {"register", "gate"}
    assert up["drift_check"] == {"featurize"}
    assert dag.params["feature_engine"] == "pandas"
    assert all(t.retries == 1 for t in dag.tasks)
