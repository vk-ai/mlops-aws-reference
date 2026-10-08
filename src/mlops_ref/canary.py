"""Canary release between two model versions behind a weighted router.

* Routing is sticky: ``hash(request_id) % 100 < weight`` goes to the canary, so a
  client sees one version consistently.
* Optional shadow mode: every request is also scored by the canary, but the
  stable answer is returned (no user impact).
* ``analyse`` compares canary vs stable on error rate, p95 latency, positive
  (decline) rate and, once enough delayed labels arrive, accuracy, and recommends
  continue / step_up / promote / rollback.
"""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class VersionStats:
    requests: int = 0
    errors: int = 0
    positives: int = 0
    latencies: list[float] = field(default_factory=list)
    labelled: int = 0
    correct: int = 0

    def p95(self) -> float:
        return float(np.percentile(self.latencies, 95)) if self.latencies else 0.0

    def summary(self) -> dict:
        n = max(1, self.requests)
        return {
            "requests": self.requests,
            "error_rate": round(self.errors / n, 4),
            "positive_rate": round(self.positives / max(1, self.requests - self.errors), 4),
            "p95_latency_ms": round(self.p95() * 1000, 3),
            "accuracy": round(self.correct / self.labelled, 4) if self.labelled else None,
        }


@dataclass
class Guardrails:
    min_requests: int = 200
    max_error_rate_delta: float = 0.02
    max_latency_ratio: float = 1.5
    max_positive_rate_delta: float = 0.10
    max_accuracy_drop: float = 0.03  # only checked when both versions have >= min_labelled labelled requests
    min_labelled: int = 100


class CanaryRouter:
    STEPS = (10, 25, 50, 100)

    def __init__(self, stable, canary=None, stable_version: str = "stable", canary_version: str | None = None,
                 weight: int = 0, shadow: bool = False, guardrails: Guardrails | None = None, clock=time.perf_counter,
                 on_promote=None):
        self.models = {stable_version: stable}
        self.stable_version = stable_version
        self.canary_version = canary_version
        if canary is not None and canary_version:
            self.models[canary_version] = canary
        self.weight = weight if canary is not None else 0
        self.shadow = shadow
        self.guardrails = guardrails or Guardrails()
        self.clock = clock
        self.stats: dict[str, VersionStats] = {v: VersionStats() for v in self.models}
        self.events: list[dict] = []
        self.lock = threading.Lock()
        self.on_promote = on_promote  # e.g. move the MLflow "champion" alias

    @staticmethod
    def bucket(request_id: str) -> int:
        return int.from_bytes(hashlib.sha256(request_id.encode()).digest()[:4], "big") % 100

    def choose(self, request_id: str) -> str:
        if self.canary_version and self.bucket(request_id) < self.weight:
            return self.canary_version
        return self.stable_version

    def _score(self, version: str, features) -> float | None:
        st = self.stats[version]
        t0 = self.clock()
        try:
            p = float(self.models[version](features))
        except Exception:
            with self.lock:
                st.requests += 1
                st.errors += 1
            return None
        with self.lock:
            st.requests += 1
            st.latencies.append(self.clock() - t0)
            st.positives += int(p >= 0.5)
        return p

    def predict(self, request_id: str, features, label: int | None = None) -> dict:
        version = self.choose(request_id)
        p = self._score(version, features)
        if p is None and version != self.stable_version:  # canary failed: serve stable instead
            version = self.stable_version
            p = self._score(version, features)
        if self.shadow and self.canary_version and version == self.stable_version:
            self._score(self.canary_version, features)
        if p is None:
            raise RuntimeError("stable model failed")
        if label is not None:
            with self.lock:
                self.stats[version].labelled += 1
                self.stats[version].correct += int((p >= 0.5) == bool(label))
        return {"version": version, "probability": round(p, 6), "default": p >= 0.5}

    def analyse(self) -> dict:
        if not self.canary_version:
            return {"decision": "no_canary", "reasons": []}
        g = self.guardrails
        s, c = self.stats[self.stable_version].summary(), self.stats[self.canary_version].summary()
        reasons = []
        if c["requests"] < g.min_requests:
            return {"decision": "continue", "reasons": [f"canary has {c['requests']} < {g.min_requests} requests"], "stable": s, "canary": c}
        if c["error_rate"] > s["error_rate"] + g.max_error_rate_delta:
            reasons.append(f"error rate {c['error_rate']} > stable {s['error_rate']} + {g.max_error_rate_delta}")
        if s["p95_latency_ms"] and c["p95_latency_ms"] > s["p95_latency_ms"] * g.max_latency_ratio:
            reasons.append(f"p95 {c['p95_latency_ms']}ms > {g.max_latency_ratio}x stable {s['p95_latency_ms']}ms")
        if abs(c["positive_rate"] - s["positive_rate"]) > g.max_positive_rate_delta:
            reasons.append(f"decline rate {c['positive_rate']} vs stable {s['positive_rate']} (> {g.max_positive_rate_delta})")
        if s["accuracy"] is not None and c["accuracy"] is not None and min(
                self.stats[self.stable_version].labelled, self.stats[self.canary_version].labelled) >= g.min_labelled:
            if c["accuracy"] < s["accuracy"] - g.max_accuracy_drop:
                reasons.append(f"accuracy {c['accuracy']} < stable {s['accuracy']} - {g.max_accuracy_drop}")
        decision = "rollback" if reasons else ("promote" if self.weight >= 100 else "step_up")
        return {"decision": decision, "reasons": reasons, "weight": self.weight, "stable": s, "canary": c}

    def set_weight(self, weight: int, reason: str = "") -> None:
        if not 0 <= weight <= 100:
            raise ValueError("weight must be 0..100")
        self.weight = weight
        self.events.append({"event": "weight", "weight": weight, "reason": reason})

    def step_up(self) -> int:
        nxt = next((s for s in self.STEPS if s > self.weight), 100)
        self.set_weight(nxt, "guardrails ok")
        return nxt

    def rollback(self, reason: str = "manual") -> None:
        self.set_weight(0, f"rollback: {reason}")
        self.events.append({"event": "rollback", "version": self.canary_version, "reason": reason})

    def promote(self) -> str:
        """Canary becomes stable; its stats carry over and the canary slot is cleared."""
        if not self.canary_version:
            raise ValueError("no canary to promote")
        old = self.stable_version
        self.stable_version, self.canary_version = self.canary_version, None
        self.models.pop(old, None)
        self.weight = 0
        self.events.append({"event": "promote", "version": self.stable_version, "replaced": old})
        if self.on_promote:
            self.on_promote(self.stable_version)
        return self.stable_version
