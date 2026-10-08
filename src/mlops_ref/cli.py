"""`mlops-ref` CLI: the DVC stages and a few operational commands."""
from __future__ import annotations

import argparse
import json
import sys

from . import pipeline
from .data import RAW_PATH, fetch


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mlops-ref")
    ap.add_argument("--workdir", default=".")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="re-download the dataset from UCI and verify its SHA-256")
    sub.add_parser("prepare", help="validate + stratified split (DVC stage)")
    f = sub.add_parser("featurize", help="feature job (DVC stage)")
    f.add_argument("--engine", choices=["pandas", "spark"])
    sub.add_parser("train", help="train the PyTorch model (DVC stage)")
    sub.add_parser("evaluate", help="test metrics -> metrics/test_metrics.json (DVC stage)")
    d = sub.add_parser("drift", help="Evidently drift report on a simulated production batch")
    d.add_argument("--strength", type=float, default=1.0)
    r = sub.add_parser("run-all", help="the whole DAG in one process, including MLflow registration")
    r.add_argument("--strength", type=float, default=1.0)
    r.add_argument("--tracking-uri")
    s = sub.add_parser("serve", help="scoring service with the canary router (models from the MLflow registry)")
    s.add_argument("--port", type=int, default=8080)
    a = ap.parse_args(argv)
    p = pipeline.load_params()
    w = a.workdir
    proc = f"{w}/data/processed"

    if a.cmd == "fetch":
        print(fetch(RAW_PATH))
    elif a.cmd == "prepare":
        info = pipeline.ingest()
        print(json.dumps({"validate": pipeline.validate_raw(info["raw"]), **pipeline.prepare(info["raw"], w, p["data"]["test_size"], p["data"]["seed"])}))
    elif a.cmd == "featurize":
        print(json.dumps(pipeline.featurize(f"{w}/data/interim/train.parquet", f"{w}/data/interim/test.parquet", w, a.engine or p["features"]["engine"])))
    elif a.cmd == "train":
        print(json.dumps(pipeline.train_model(f"{proc}/train.parquet", w, p["train"])))
    elif a.cmd == "evaluate":
        print(json.dumps(pipeline.evaluate_model(f"{w}/models/model.pt", f"{proc}/test.parquet", w)))
    elif a.cmd == "drift":
        cur = pipeline.make_current_batch(f"{proc}/test.parquet", w, a.strength)
        print(json.dumps(pipeline.drift_check(f"{proc}/train.parquet", cur, w, p["drift"]["share_threshold"])))
    elif a.cmd == "run-all":
        print(json.dumps(pipeline.run_all(w, p, a.tracking_uri, a.strength), default=str, indent=2))
    elif a.cmd == "serve":
        import uvicorn

        from .serving import create_app

        uvicorn.run(create_app(), host="0.0.0.0", port=a.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
