"""FastAPI scoring service with the canary router. Loads models from the MLflow registry by alias.

Two ways to run a canary:
* in-process (default): this process loads ``champion`` and ``canary`` and splits traffic itself;
* at the load balancer (terraform/): each ECS service sets ``SERVE_ALIAS`` to serve one alias,
  and the ALB splits traffic between two target groups.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .canary import CanaryRouter, Guardrails
from .features import MODEL_FEATURES, add_features


class ScoreRequest(BaseModel):
    request_id: str
    features: dict[str, float]
    label: int | None = None  # optional delayed ground truth, for canary accuracy


def as_scorer(model):
    """Wrap a model (nn.Module, or the traced module MLflow 3 loads) so it scores a dict of raw features."""
    import pandas as pd

    try:
        model.eval()
    except Exception:  # exported/traced modules don't support eval(); they were exported in eval mode
        pass

    def score(features: dict) -> float:
        df = add_features(pd.DataFrame([features]))
        x = torch.tensor(df[MODEL_FEATURES].to_numpy(np.float32))
        with torch.no_grad():
            return float(model(x).reshape(-1)[0])

    return score


def create_app(router: CanaryRouter | None = None) -> FastAPI:
    app = FastAPI(title="credit-risk scoring (canary)")
    state: dict = {"router": router}

    def r() -> CanaryRouter:
        if state["router"] is None:
            from . import tracking

            tracking.configure()
            pinned = os.environ.get("SERVE_ALIAS")  # one model per process, e.g. behind ALB weighted target groups
            stable_v = tracking.alias_version(pinned or "champion")
            canary_v = None if pinned else tracking.alias_version("canary")
            if stable_v is None:
                raise HTTPException(503, f"no '{pinned or 'champion'}' model in the registry")
            state["router"] = CanaryRouter(
                as_scorer(tracking.load(stable_v)),
                as_scorer(tracking.load(canary_v)) if canary_v and canary_v != stable_v else None,
                stable_version=f"v{stable_v}", canary_version=f"v{canary_v}" if canary_v and canary_v != stable_v else None,
                weight=int(os.environ.get("CANARY_WEIGHT", "10")), shadow=os.environ.get("CANARY_SHADOW") == "1", guardrails=Guardrails(),
                on_promote=lambda v: tracking.set_alias("champion", v.lstrip("v")),
            )
        return state["router"]

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.post("/predict")
    def predict(req: ScoreRequest):
        missing = [c for c in ("duration_months", "credit_amount", "age_years") if c not in req.features]
        if missing:
            raise HTTPException(422, f"missing features {missing}")
        return r().predict(req.request_id, req.features, req.label)

    @app.get("/canary")
    def status():
        router = r()
        a = router.analyse()
        return {"stable": router.stable_version, "canary": router.canary_version, "weight": router.weight, "shadow": router.shadow,
                "decision": a["decision"], "reasons": a["reasons"], "stats": {"stable": a.get("stable"), "canary": a.get("canary")}}

    @app.post("/canary/weight/{weight}")
    def weight(weight: int):
        try:
            r().set_weight(weight, "manual")
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        return {"weight": r().weight}

    @app.post("/canary/rollback")
    def rollback():
        r().rollback("manual")
        return {"weight": r().weight}

    @app.post("/canary/promote")
    def promote():
        try:
            return {"stable": r().promote()}
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    return app
