import itertools

import pytest

from mlops_ref.canary import CanaryRouter, Guardrails


def const(p):
    return lambda features: p


def boom(features):
    raise RuntimeError("model crashed")


def ticking_clock(step_by_call):
    """Fake clock: alternating calls are start/end of a scoring call."""
    t = itertools.count()
    return lambda: next(t) * step_by_call


def ids(n):
    return [f"req-{i}" for i in range(n)]


def test_routing_is_sticky_and_roughly_weighted():
    r = CanaryRouter(const(0.1), const(0.2), "v1", "v2", weight=25)
    first = [r.choose(i) for i in ids(2000)]
    assert first == [r.choose(i) for i in ids(2000)]  # same request id -> same version
    share = first.count("v2") / len(first)
    assert 0.22 < share < 0.28


def test_weight_zero_and_no_canary_always_stable():
    assert {CanaryRouter(const(0.1), const(0.2), "v1", "v2", weight=0).choose(i) for i in ids(500)} == {"v1"}
    r = CanaryRouter(const(0.1), weight=50)
    assert {r.choose(i) for i in ids(500)} == {"stable"} and r.weight == 0
    assert r.analyse()["decision"] == "no_canary"


def test_shadow_scores_canary_without_serving_it():
    r = CanaryRouter(const(0.1), const(0.9), "v1", "v2", weight=0, shadow=True)
    out = [r.predict(i, {}) for i in ids(50)]
    assert {o["version"] for o in out} == {"v1"} and all(not o["default"] for o in out)
    assert r.stats["v2"].requests == 50 and r.stats["v2"].positives == 50


def test_failing_canary_falls_back_to_stable_and_triggers_rollback():
    r = CanaryRouter(const(0.1), boom, "v1", "v2", weight=100, guardrails=Guardrails(min_requests=20))
    out = [r.predict(i, {}) for i in ids(30)]
    assert {o["version"] for o in out} == {"v1"}  # users never see the error
    a = r.analyse()
    assert a["decision"] == "rollback" and "error rate" in a["reasons"][0]
    r.rollback(a["reasons"][0])
    assert r.weight == 0 and r.events[-1]["event"] == "rollback"


def test_guardrails_continue_then_step_up_then_promote():
    promoted = []
    r = CanaryRouter(const(0.2), const(0.25), "v1", "v2", weight=10, guardrails=Guardrails(min_requests=50), on_promote=promoted.append)
    for i in ids(200):
        r.predict(i, {})
    assert r.analyse()["decision"] == "continue"  # ~20 canary requests < 50
    for i in ids(1000)[200:]:
        r.predict(i, {})
    assert r.analyse()["decision"] == "step_up"
    assert [r.step_up() for _ in range(4)] == [25, 50, 100, 100]
    assert r.analyse()["decision"] == "promote"
    assert r.promote() == "v2" and promoted == ["v2"]
    assert r.canary_version is None and r.weight == 0 and set(r.models) == {"v2"}
    with pytest.raises(ValueError):
        r.promote()


def test_decline_rate_and_latency_guardrails():
    g = Guardrails(min_requests=20)
    r = CanaryRouter(const(0.1), const(0.9), "v1", "v2", weight=50, guardrails=g)
    for i in ids(200):
        r.predict(i, {})
    assert any("decline rate" in x for x in r.analyse()["reasons"])

    slow = CanaryRouter(const(0.1), const(0.1), "v1", "v2", weight=50, guardrails=g)
    for i in ids(200):
        slow.clock = ticking_clock(0.01 if slow.choose(i) == "v2" else 0.001)
        slow.predict(i, {})
    assert any(x.startswith("p95") for x in slow.analyse()["reasons"])


def test_labels_feed_accuracy_and_weight_validation():
    r = CanaryRouter(const(0.9), const(0.1), "v1", "v2", weight=0)
    r.predict("a", {}, label=1)
    r.predict("b", {}, label=0)
    assert r.stats["v1"].summary()["accuracy"] == 0.5
    with pytest.raises(ValueError):
        r.set_weight(101)


def test_accuracy_guardrail_needs_enough_labels():
    g = Guardrails(min_requests=10, min_labelled=50)
    r = CanaryRouter(const(0.9), const(0.1), "v1", "v2", weight=50, guardrails=g, shadow=False)
    for i in ids(80):
        r.predict(i, {}, label=1)  # stable predicts 1 (right), canary predicts 0 (wrong)
    assert not any("accuracy" in x for x in r.analyse()["reasons"])  # < 50 labels each
    for i in ids(400)[80:]:
        r.predict(i, {}, label=1)
    assert any(x.startswith("accuracy 0.0 < stable 1.0") for x in r.analyse()["reasons"])
