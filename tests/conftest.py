import copy
from pathlib import Path

import pandas as pd
import pytest

from mlops_ref import pipeline
from mlops_ref.data import load_raw
from mlops_ref.features import add_features, split

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def params():
    return pipeline.load_params(ROOT / "params.yaml")


@pytest.fixture(scope="session")
def fast_params(params):
    """Same pipeline, fewer epochs, so MLflow tests stay quick."""
    p = copy.deepcopy(params)
    p["train"]["epochs"] = 15
    return p


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    return load_raw()


@pytest.fixture(scope="session")
def featured(raw, params):
    tr, te = split(raw, params["data"]["test_size"], params["data"]["seed"])
    return add_features(tr), add_features(te)
