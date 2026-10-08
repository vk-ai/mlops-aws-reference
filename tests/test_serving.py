from fastapi.testclient import TestClient

from mlops_ref.canary import CanaryRouter
from mlops_ref.serving import as_scorer, create_app

FEATURES = {"duration_months": 24, "credit_amount": 40, "age_years": 30, **{f"a{i:02d}": 1.0 for i in range(1, 25)}}


def client():
    router = CanaryRouter(lambda f: 0.2, lambda f: 0.7, "v1", "v2", weight=50)
    return TestClient(create_app(router)), router


def test_predict_and_canary_controls():
    c, router = client()
    versions = {c.post("/predict", json={"request_id": f"r{i}", "features": FEATURES}).json()["version"] for i in range(100)}
    assert versions == {"v1", "v2"}
    assert c.get("/canary").json()["weight"] == 50
    assert c.post("/canary/weight/101").status_code == 422
    assert c.post("/canary/rollback").json() == {"weight": 0}
    assert {c.post("/predict", json={"request_id": f"r{i}", "features": FEATURES}).json()["version"] for i in range(50)} == {"v1"}
    assert c.post("/canary/promote").json() == {"stable": "v2"}
    assert c.post("/canary/promote").status_code == 409


def test_missing_features_rejected():
    c, _ = client()
    r = c.post("/predict", json={"request_id": "x", "features": {"a01": 1}})
    assert r.status_code == 422 and "age_years" in r.text


def test_as_scorer_on_trained_model(featured, fast_params):
    from mlops_ref.train import train

    t = fast_params["train"]
    m = train(featured[0], t["hidden"], 3, t["lr"], t["weight_decay"], t["batch_size"], t["seed"]).model
    p = as_scorer(m)(FEATURES)
    assert 0.0 <= p <= 1.0
