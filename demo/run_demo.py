"""End-to-end demo in a scratch directory: pipeline -> MLflow registry -> drift -> canary.

    python demo/run_demo.py            # ~1 minute on a laptop CPU

Everything is local: a SQLite MLflow registry under a temp dir, the bundled UCI
dataset, CPU PyTorch. The Airflow DAG runs the same functions (see README).
"""
from __future__ import annotations

import copy
import json
import logging
import os
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
logging.disable(logging.WARNING)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from mlops_ref import pipeline, tracking  # noqa: E402
from mlops_ref.canary import CanaryRouter, Guardrails  # noqa: E402
from mlops_ref.data import FEATURES, TARGET  # noqa: E402
from mlops_ref.serving import as_scorer  # noqa: E402


def show(title: str) -> None:
    print(f"\n== {title}")


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="mlops-demo-"))
    uri = f"sqlite:///{work / 'mlflow.db'}"
    params = pipeline.load_params(ROOT / "params.yaml")

    show("1. pipeline run #1 (params.yaml): ingest -> validate -> prepare -> featurize -> train -> evaluate -> register -> gate -> deploy")
    r1 = pipeline.run_all(work, params, uri, drift_strength=0.0)
    print(f"   data sha256={r1['ingest']['sha256'][:12]}… rows={r1['validate']['rows']} default_rate={r1['validate']['default_rate']}")
    print(f"   train: final_loss={r1['train']['final_loss']}  test: {json.dumps(r1['evaluate'])}")
    print(f"   registered credit-risk v{r1['register']['version']}; gate={r1['gate']['passed']}; deploy={r1['deploy']}")

    show("2. pipeline run #2 (retrain with hidden=[64, 32], epochs=120) -> challenger")
    p2 = copy.deepcopy(params)
    p2["train"].update(hidden=[64, 32], epochs=120)
    r2 = pipeline.run_all(work, p2, uri, drift_strength=0.0)
    g = r2["gate"]
    print(f"   v{g['version']} test_auc={g['test_auc']:.4f} vs champion v{g['champion']} {g['champion_auc']:.4f} (max drop {params['gate']['max_auc_drop']}) -> passed={g['passed']}")
    print(f"   deploy={r2['deploy']}")

    show("3. drift check (Evidently DataDriftPreset; reference = training features)")
    for strength in (0.0, 1.0):
        cur = pipeline.make_current_batch(r1["features"]["test"], work / f"s{strength}", strength)
        d = pipeline.drift_check(r1["features"]["train"], cur, work / f"s{strength}", params["drift"]["share_threshold"])
        print(f"   simulated shift={strength}: drift_share={d['drift_share']} threshold={d['share_threshold']} alert={d['alert']} drifted={d['drifted_columns']}")

    tracking.configure(uri)
    c = params["canary"]
    guardrails = Guardrails(c["min_requests"], c["max_error_rate_delta"], c["max_latency_ratio"], c["max_positive_rate_delta"])
    test = pd.read_parquet(r1["features"]["test"])
    traffic = pd.concat([test] * 20, ignore_index=True)  # held-out rows replayed as labelled traffic

    def release(title: str, canary_scorer=None, canary_name=None) -> None:
        show(title)
        champ, canary = tracking.alias_version("champion"), tracking.alias_version("canary")
        router = CanaryRouter(as_scorer(tracking.load(champ)), canary_scorer or as_scorer(tracking.load(canary)), f"v{champ}",
                              canary_name or f"v{canary}", weight=c["initial_weight"], guardrails=guardrails,
                              on_promote=lambda v: tracking.set_alias("champion", v.lstrip("v")))
        i = 0
        while router.canary_version and i < len(traffic):
            for _ in range(500):
                row = traffic.iloc[i]
                router.predict(f"{title[:2]}-{i}", row[FEATURES].to_dict(), int(row[TARGET]))
                i += 1
            a = router.analyse()
            cs, ss = a.get("canary", {}), a.get("stable", {})
            print(f"   weight={router.weight:>3}% canary n={cs.get('requests', 0):>4} decline={cs.get('positive_rate')} acc={cs.get('accuracy')} | "
                  f"stable decline={ss.get('positive_rate')} acc={ss.get('accuracy')} -> {a['decision']} {a['reasons'] or ''}")
            if a["decision"] == "step_up":
                router.step_up()
            elif a["decision"] == "promote":
                router.promote()
            elif a["decision"] == "rollback":
                router.rollback("; ".join(a["reasons"]))
                break
        print(f"   result: weight={router.weight}, stable={router.stable_version}, MLflow champion -> v{tracking.alias_version('champion')}")

    release("4. canary v1 (champion) vs v2: replay held-out rows through the weighted router")

    show("5. pipeline run #3 (params.yaml architecture, train.seed=11) -> challenger")
    p3 = copy.deepcopy(params)
    p3["train"]["seed"] = 11
    r3 = pipeline.run_all(work, p3, uri, drift_strength=0.0)
    g = r3["gate"]
    print(f"   v{g['version']} test_auc={g['test_auc']:.4f} vs champion v{g['champion']} {g['champion_auc']:.4f} -> passed={g['passed']}; deploy={r3['deploy']}")
    release("6. canary v3 vs champion v2")
    release("7. a deliberately broken canary (scores every applicant as a default)", lambda f: 0.97, "v-broken")
    print(f"\n   artifacts in {work}: mlflow.db, mlartifacts/, reports/drift_report.html, deploy/canary.json")


if __name__ == "__main__":
    main()
