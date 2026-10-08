import torch

from mlops_ref import pipeline
from mlops_ref.train import evaluate, predict_proba, train


def _train(featured, p, **over):
    t = {**p["train"], **over}
    return train(featured[0], t["hidden"], t["epochs"], t["lr"], t["weight_decay"], t["batch_size"], t["seed"])


def test_training_is_deterministic(featured, fast_params):
    a, b = _train(featured, fast_params), _train(featured, fast_params)
    assert a.history == b.history
    for (k, v), (_, w) in zip(a.model.state_dict().items(), b.model.state_dict().items()):
        assert torch.equal(v, w), k
    c = _train(featured, fast_params, seed=8)
    assert c.history != a.history


def test_full_params_beat_the_gate_floor(featured, params):
    res = _train(featured, params)
    assert res.history[-1] < res.history[0]
    m = evaluate(res.model, featured[1])
    assert m["auc"] >= params["gate"]["min_auc"]
    assert 0.0 < m["positive_rate"] < 1.0


def test_save_load_round_trip(tmp_path, featured, fast_params):
    res = _train(featured, fast_params)
    path = pipeline.save_model(res.model, fast_params["train"]["hidden"], tmp_path / "m.pt")
    loaded = pipeline.load_model(path)
    assert (predict_proba(loaded, featured[1]) == predict_proba(res.model, featured[1])).all()
