# mlops-aws-reference

A reference MLOps pipeline that is small enough to run on a laptop. An Airflow
DAG trains a PyTorch credit-risk model on a public dataset. DVC versions the
data and pipeline, MLflow tracks runs and holds the model registry, Evidently
raises data-drift alerts, an optional PySpark job builds the features, and a
canary router shifts traffic between two model versions with automatic
promote and rollback. A Terraform module shows the same canary on AWS (S3, ECR,
ECS Fargate behind an ALB with weighted target groups).

> **Learning project.** I built this to practise the MLOps lifecycle end to end.
> It is not used in production anywhere and has never been deployed to AWS. The
> pipeline, registry, drift check and canary logic run locally and in CI, and
> the Airflow DAG has been run with `dag.test()`. The AWS Terraform is
> `validate`d and tested against a mocked provider only, never applied. See
> [What is real vs stubbed](#what-is-real-vs-stubbed).

## What it does

```
data/raw/german.data-numeric (UCI Statlog, bundled, SHA-256 pinned)
   │
   │  Airflow DAG credit_risk_training (@weekly)          ── or ──  dvc repro (prepare → featurize → train → evaluate)
   ▼
 ingest ─► validate ─► prepare ─► featurize ─► train ─► evaluate ─► register ─► gate ─► deploy_canary
 (sha256)  (contract)  (strat.   (pandas or   (PyTorch  (AUC, F1,   (MLflow     (floor +   (champion / canary
                        split)    PySpark)     MLP, CPU) cost)       run+model)  vs champ)  alias, canary.json)
                                      └────────► drift_check (Evidently DataDriftPreset → alert / webhook)

 scoring service (FastAPI): loads champion + canary by MLflow alias ─► CanaryRouter (sticky weighted split,
   shadow mode, guardrails: errors, p95 latency, decline rate, accuracy) ─► step 10→25→50→100 ─► promote | rollback
 AWS (terraform/): S3 (MLflow artifacts + DVC remote) · ECR · ECS Fargate stable + canary services ·
   ALB weighted target groups (canary_weight) · CloudWatch alarms on the canary target group
```

| Capability | How |
|---|---|
| Data | Statlog (German Credit Data), numeric version, from the UCI ML Repository (CC BY 4.0): 1,000 applicants, 24 numeric attributes, 30% bad credit risk. Bundled (102 KB) with its SHA-256 pinned; `mlops-ref fetch` re-downloads it and refuses a mismatch. A pandas data contract (columns, nulls, ranges, base rate) runs before training. |
| Orchestration | Airflow 3 TaskFlow DAG (`dags/credit_risk_training.py`, falls back to Airflow 2 imports). Each task is a thin wrapper over a plain function in `mlops_ref.pipeline`, so the logic is unit-tested without Airflow, and `python dags/credit_risk_training.py` runs the whole DAG in-process with `dag.test()`. DAG params pick the feature engine and the simulated drift strength. |
| Model | A 2-hidden-layer MLP in PyTorch (CPU), with input standardisation stored as buffers so the saved model is self-contained, `pos_weight` for the 70/30 class imbalance and seeded, deterministic training. Metrics (AUC, F1 and expected cost on the dataset's 5:1 cost matrix) are written in numpy with no scikit-learn. |
| Versioning | DVC stages `prepare → featurize → train → evaluate` with `params.yaml`, `dvc.lock`, a `metrics/test_metrics.json` metric and a local remote. CI posts `dvc metrics diff` to the job summary. |
| Tracking & registry | MLflow 3: params, metrics, data hash and torch version tags, the model with a signature and input example, and registered versions with aliases (`challenger` = newest, `champion` = serving, `canary` = under test). The gate requires AUC ≥ 0.70 and no more than 0.02 below the champion. |
| Drift | Evidently 0.7 `DataDriftPreset` over the 27 model features, with reference = training data. It alerts when more than 20% of columns drift, and posts to a Slack-compatible webhook if `DRIFT_WEBHOOK_URL` is set. The HTML report and a JSON summary are saved. |
| Feature job | `spark_features.py` is the same transform as `features.add_features` in PySpark (local mode). A test checks the outputs match exactly. |
| Canary | `CanaryRouter`: sticky hash-bucket routing, optional shadow scoring, per-version stats, automatic fallback to stable if the canary throws, guardrails (min requests, error-rate delta, p95 latency ratio, decline-rate delta, accuracy drop once labels arrive) → `continue / step_up / promote / rollback`; promotion moves the MLflow `champion` alias. The FastAPI service exposes `/predict`, `/canary`, `/canary/weight/{w}`, `/canary/rollback`, `/canary/promote`. With `SERVE_ALIAS` set it serves a single alias, which is how the ALB variant runs. |
| AWS | `terraform/`: versioned, KMS-encrypted, TLS-only S3 bucket with public access blocked; immutable, scan-on-push ECR; ECS Fargate stable and canary services (the canary scales to 0 when its weight is 0) with a least-privilege task role; ALB listener forwarding to two target groups by `canary_weight`; CloudWatch alarms on canary 5xx and p95 latency. |

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -e ".[dev]"                 # extras: dvc, spark (needs Java 17+), airflow (see below)
pytest -q                               # Spark/Airflow tests skip without their extras
python demo/run_demo.py                 # the run captured below (~1 min)

dvc repro && dvc metrics show           # pipeline via DVC  (pip install -e ".[dvc]")
mlops-ref run-all                       # whole DAG in one process → ./mlflow.db, ./mlartifacts
mlops-ref serve --port 8080             # scoring service with the canary router
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Airflow (use the official constraints file):

```bash
pip install "apache-airflow==3.3.2" --constraint \
  https://raw.githubusercontent.com/apache/airflow/constraints-3.3.2/constraints-3.12.txt
export AIRFLOW_HOME=$PWD/.airflow MLOPS_WORKDIR=/tmp/mlops-ref
airflow db migrate && python dags/credit_risk_training.py      # dag.test()
```

AWS (**not applied by me**; `example.tfvars` has placeholder IDs):

```bash
cd terraform && terraform init && terraform validate && terraform test   # mocked provider, no credentials
terraform plan -var-file=example.tfvars                                  # needs an AWS account
```

## Demo run

`python demo/run_demo.py` uses a temporary directory and a SQLite MLflow
registry. Every step is deterministic, so a rerun on the same machine gives the
same numbers. Output from 2026-10-08 (stdout; MLflow's registration messages on
stderr omitted):

```
== 1. pipeline run #1 (params.yaml): ingest -> validate -> prepare -> featurize -> train -> evaluate -> register -> gate -> deploy
   data sha256=2752b0443949… rows=1000 default_rate=0.3
   train: final_loss=0.3276  test: {"auc": 0.8121, "accuracy": 0.755, "precision": 0.5733, "recall": 0.7167, "f1": 0.637, "expected_cost": 0.585, "positive_rate": 0.375}
   registered credit-risk v1; gate=True; deploy={'action': 'initial_release', 'stable': '1', 'canary': None, 'weight': 0}

== 2. pipeline run #2 (retrain with hidden=[64, 32], epochs=120) -> challenger
   v2 test_auc=0.8038 vs champion v1 0.8121 (max drop 0.02) -> passed=True
   deploy={'action': 'canary', 'stable': '1', 'canary': '2', 'weight': 10}

== 3. drift check (Evidently DataDriftPreset; reference = training features)
   simulated shift=0.0: drift_share=0.0741 threshold=0.2 alert=False drifted=['a12', 'a17']
   simulated shift=1.0: drift_share=0.2593 threshold=0.2 alert=True drifted=['a12', 'a17', 'age_years', 'credit_amount', 'duration_months', 'log_credit_amount', 'young_applicant']

== 4. canary v1 (champion) vs v2: replay held-out rows through the weighted router
   weight= 10% canary n=  47 decline=0.2979 acc=0.6596 | stable decline=0.3775 acc=0.7528 -> continue ['canary has 47 < 200 requests']
   weight= 10% canary n= 101 decline=0.3168 acc=0.7228 | stable decline=0.376 acc=0.7553 -> continue ['canary has 101 < 200 requests']
   weight= 10% canary n= 154 decline=0.3247 acc=0.7273 | stable decline=0.3744 acc=0.7541 -> continue ['canary has 154 < 200 requests']
   weight= 10% canary n= 202 decline=0.3218 acc=0.7723 | stable decline=0.3737 acc=0.7536 -> step_up 
   weight= 25% canary n= 317 decline=0.3091 acc=0.7539 | stable decline=0.3743 acc=0.7536 -> step_up 
   weight= 50% canary n= 564 decline=0.3174 acc=0.7571 | stable decline=0.374 acc=0.7574 -> step_up 
   weight=100% canary n=1064 decline=0.312 acc=0.7566 | stable decline=0.374 acc=0.7574 -> promote 
   result: weight=0, stable=v2, MLflow champion -> v2

== 5. pipeline run #3 (params.yaml architecture, train.seed=11) -> challenger
   v3 test_auc=0.8045 vs champion v2 0.8038 -> passed=True; deploy={'action': 'canary', 'stable': '2', 'canary': '3', 'weight': 10}

== 6. canary v3 vs champion v2
   weight= 10% canary n=  40 decline=0.4 acc=0.675 | stable decline=0.3043 acc=0.7478 -> continue ['canary has 40 < 200 requests']
   weight= 10% canary n=  93 decline=0.3656 acc=0.6882 | stable decline=0.3142 acc=0.7596 -> continue ['canary has 93 < 200 requests']
   weight= 10% canary n= 137 decline=0.365 acc=0.6861 | stable decline=0.3147 acc=0.7586 -> continue ['canary has 137 < 200 requests']
   weight= 10% canary n= 191 decline=0.3403 acc=0.6911 | stable decline=0.3167 acc=0.7651 -> continue ['canary has 191 < 200 requests']
   weight= 10% canary n= 234 decline=0.312 acc=0.7222 | stable decline=0.3173 acc=0.7617 -> rollback ['accuracy 0.7222 < stable 0.7617 - 0.03']
   result: weight=0, stable=v2, MLflow champion -> v2

== 7. a deliberately broken canary (scores every applicant as a default)
   weight= 10% canary n=  50 decline=1.0 acc=0.2 | stable decline=0.3044 acc=0.76 -> continue ['canary has 50 < 200 requests']
   weight= 10% canary n= 106 decline=1.0 acc=0.1981 | stable decline=0.3132 acc=0.7729 -> continue ['canary has 106 < 200 requests']
   weight= 10% canary n= 159 decline=1.0 acc=0.2453 | stable decline=0.3087 acc=0.7673 -> continue ['canary has 159 < 200 requests']
   weight= 10% canary n= 211 decline=1.0 acc=0.2749 | stable decline=0.3108 acc=0.7697 -> rollback ['decline rate 1.0 vs stable 0.3108 (> 0.1)', 'accuracy 0.2749 < stable 0.7697 - 0.03']
   result: weight=0, stable=v2, MLflow champion -> v2

   artifacts in /tmp/mlops-demo-…: mlflow.db, mlartifacts/, reports/drift_report.html, deploy/canary.json
```

How to read it:
- **The gate and the canary disagree, and that is the point.** v2 has a lower
  test AUC than v1 (0.8038 vs 0.8121) but is within the 0.02 tolerance, and it
  won its canary. v3 is close to v2 on test AUC but lost its canary on
  accuracy. The canary compares the versions on *different* request subsets
  (sticky hash buckets) of only about 200 labelled requests, so a 3-point
  accuracy gap is within noise. With replayed held-out rows rather than real
  traffic, these decisions show the mechanics, not which model is better.
- **Drift:** train vs held-out test already differs on 2 of 27 columns (7%, sampling noise). The simulated shift (larger, longer loans, younger applicants) moves 7 of 27 (26%). I set the 20% threshold between the two after seeing both numbers, so it is tuned to this simulation.

**Airflow.** I ran the DAG with Airflow 3.3.2 (`dag.test()`, SQLite metadata
DB, Python 3.12). All 10 tasks succeeded in 23 s. The run logged
`Data drift alert: 26% of columns drifted (threshold 20%)` and wrote
`deploy/canary.json` = `{"action": "initial_release", "stable": "1", ...}`.
The CI `airflow` job repeats this run.

## Tests

```
pytest -q                     44 passed, 1 skipped (airflow)   ~1.5 min, 92% line coverage (Python 3.12, with dvc + pyspark + Java 21)
pytest -q -m airflow          1 passed                         (in an Airflow 3.3.2 environment)
terraform test                3 passed                         (mocked AWS provider)
```

Python 3.11 without the extras: 42 passed, 2 skipped (spark, airflow).

Covered: checksum and fetch (including a tampered download), the data
contract, the stratified split, AUC against a brute-force pairwise definition
with ties, deterministic training, model save/load, drift with and without the
simulated shift, the webhook POST, every canary decision path, the scoring API
(in-process router and `SERVE_ALIAS`), and an end-to-end run against a scratch
MLflow registry: initial release, canary, a rejected challenger and promotion
moving the `champion` alias. Also covered: Spark/pandas parity, DAG structure,
a DVC repro with push/pull through a local remote, and the Terraform canary
wiring (traffic split, canary scale-to-zero, alias per service, weight
validation).

CI (`.github/workflows/ci.yml`) runs five jobs, all without cloud credentials:
- `test`: Python 3.11 and 3.12, plus the demo.
- `spark`: Temurin 17 plus PySpark.
- `airflow`: the constraints install, the DAG structure test and a full `dag.test()`.
- `dvc`: round-trip test, then `dvc repro`, with `dvc metrics diff` in the job summary and an AUC floor check.
- `terraform`: `fmt -check`, `validate` and `terraform test`.

## What is real vs stubbed

**Real (runs and is tested here):**
- Training, evaluation, the MLflow registry and aliases, the gate, Evidently drift reports, the canary router and the scoring API, all on a real public dataset.
- The Airflow DAG, run end to end with Airflow 3.3.2 `dag.test()`.
- The PySpark job (local mode, Java 21), DVC pipeline and remote round trip.
- `terraform validate` and `terraform test` with a mocked AWS provider.

**Stubbed, simulated or not verified:**
- **Nothing has been deployed to AWS.** There is no `terraform plan` against a real account, and there is no VPC, MLflow server or CI/CD-to-AWS. The module expects an existing VPC and an MLflow tracking server. The mocked-provider tests check the wiring, not that AWS would accept every setting.
- **The Dockerfile and docker-compose.yml are untested,** because the build machine has no Docker.
- **"Production" traffic is simulated.** Drift is checked on a synthetic shifted copy of the test set. The canary replays held-out rows with their labels, which in practice would arrive weeks later. The broken canary in step 7 is a constant function.
- **MLflow is local SQLite plus an artifact directory,** not a file store. MLflow 3.x put the plain `./mlruns` file store into maintenance mode, so this is the local default. The ALB canary variant needs a shared tracking server, which isn't included.
- **Column names:** the numeric UCI file doesn't name its 24 columns. I named three (`duration_months`, `credit_amount`, `age_years`) after checking them against the original categorical file. The rest are `a01..a24`.
- **The model is deliberately small** (test AUC about 0.80 on 200 held-out rows). Tuning was not a goal, and with 1,000 rows the metrics have wide error bars.
- **The data contract is hand-written pandas checks,** not Great Expectations. The DVC remote is a local directory; an S3 remote is a `dvc remote add` away but untested.

## Layout

```
src/mlops_ref/   data.py · validate.py · features.py · spark_features.py · model.py · train.py · metrics.py
                 tracking.py · drift.py · canary.py · serving.py · pipeline.py · cli.py
dags/credit_risk_training.py
dvc.yaml, dvc.lock, params.yaml, metrics/test_metrics.json
data/raw/        german.data-numeric + german.doc (UCI, CC BY 4.0)
terraform/       s3 · ecr · iam · ecs · alb · cloudwatch · tests/canary.tftest.hcl · example.tfvars
tests/           45 pytest tests (Spark/Airflow/DVC ones skip without their extras)
demo/run_demo.py, Dockerfile, docker-compose.yml
```

## License

MIT. The dataset is by Hans Hofmann, from the UCI Machine Learning Repository,
licensed CC BY 4.0.
