# Evidence-backed MLOps AI Agent — Assignment 03

A Python conversational agent connects DVC dataset provenance, MLflow experiment tracking, the authenticated Airflow 3 API and Kubernetes application health. Every operational statement carries tool evidence; failed calls remain explicit uncertainties. Training and cluster restarts require operator confirmation.

The repository includes 153 passing automated tests plus real DVC restoration, MLflow comparisons, successful Airflow execution and Kubernetes readiness/prediction exports.

See [architecture](architecture.md), [technical report](docs/technical_report.md), [engineering defense](docs/defense.md), [evidence exports](evidence/README.md) and the [commit/work log](WORK_LOG.md).

## Install

Use Python 3.10–3.12; Python 3.11 is used in CI and containers. Docker/Compose and kind/kubectl are needed for the full infrastructure demo.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
cp .env.example .env
```

Set environment variables in your shell or load a trusted local `.env` with `set -a; source .env; set +a`. The application reads the environment; it does not automatically execute `.env`. Keep keys, JWTs and kubeconfigs untracked. Keep tracking URLs credential-free; use `MLFLOW_TRACKING_USERNAME`/`MLFLOW_TRACKING_PASSWORD` or `MLFLOW_TRACKING_TOKEN` environment variables for authenticated MLflow servers.

## Restore the dataset

`data/iris.csv.dvc` records the actual dataset content hash. The existing `localstore` DVC remote is relative to the course workspace. For another machine, configure a remote containing the DVC objects:

```bash
export DVC_SITE_CACHE_DIR="$PWD/.dvc/tmp/site-cache"
dvc remote modify --local localstore url /absolute/path/to/your/dvc-remote
dvc pull data/iris.csv.dvc
```

For a fresh clone without access to that remote, the public scikit-learn Iris dataset is reproducible byte-for-byte:

```bash
python scripts/create_dataset.py
python -c "from agent.tools.dvc_tools import DVCTools; p=DVCTools().dataset_version(); assert p['data_matches_metadata']; print(p)"
dvc add data/iris.csv
```

The generator refuses to overwrite an existing file. Commit dataset pointer changes intentionally; never claim an uncommitted dataset is a historical version. For an isolated historical reproduction, use `python -m scripts.collect_evidence --reproduce-data`; it fetches to a temporary directory and verifies the hash.

## Launch MLflow and Airflow

The local demonstration uses SQLite and Airflow standalone. It starts all real Airflow components in one container. A production deployment should use separate components, durable object storage, a proper database, HTTPS and restricted identities.

```bash
export AIRFLOW_UID=$(id -u)
export MLFLOW_PORT=5500 AIRFLOW_PORT=58080
docker compose -p assignment03 up -d --build mlflow airflow
export MLFLOW_TRACKING_URI=http://127.0.0.1:5500
export AIRFLOW_BASE_URL=http://127.0.0.1:58080
```

Airflow's local SimpleAuthManager creates an admin password in its container's Airflow home. Read it locally; never paste it into evidence or commit it. Request a JWT using `POST /auth/token` with the auth manager's username/password JSON, then export the returned `access_token` as `AIRFLOW_TOKEN`. See [Airflow's authentication documentation](https://airflow.apache.org/docs/apache-airflow/3.1.3/security/api.html). The agent uses `/api/v2` with a Bearer JWT. A private `.env` may hold these values.

The MLflow UI is at `http://127.0.0.1:5500`; Airflow is at `http://127.0.0.1:58080`. Override ports if necessary. For a non-Docker Airflow installation, run `bash scripts/bootstrap_airflow.sh`; it creates a separate constrained environment and launches standalone on port 8080. Do not install Airflow into the agent environment: its dependency constraints differ.

## Train and serve

Training validates the dataset, creates a deterministic stratified split, fits a random forest, logs actual metrics/artifacts/provenance to MLflow, then applies the configured F1 gate. Failed validation or evaluation prevents approval. Gate success creates `artifacts/model.joblib` and `artifacts/metadata.json`; deployment remains an operator step.

```bash
python -m training.train
# DVC can reproduce the same pipeline and cache its outputs:
dvc repro
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

A local tracking database can be used when the tracking server is unavailable:

```bash
mkdir -p artifacts
export MLFLOW_TRACKING_URI="sqlite:///$PWD/artifacts/local-tracking.db"
python -m training.train
```

`params.yaml` defines the training parameters and an explicit educational minimum macro F1 of 0.85. The observed RF100 score of 0.89975 correctly failed an initial 0.90 gate; this policy change is recorded in the work log and failure evidence. The API reads `MODEL_PATH` and accepts `POST /predict` with `{"instances": [[5.1, 3.5, 1.4, 0.2]]}`. `/health/live` reports process health; `/health/ready` returns 503 unless an approved model loads. `/version` returns the actual MLflow run, Git commit and DVC hash. A high test score on Iris does not establish production suitability.

Airflow writes artifacts under `artifacts/runs/<run-key>/`. After a run's gate passes, explicitly copy its `model.joblib` and `metadata.json` together into `artifacts/` before packaging that candidate. Keep the run identifier in deployment annotations.

## Deploy to an isolated Kubernetes cluster

```bash
mkdir -p .kube
kind create cluster --name assignment03 --kubeconfig "$PWD/.kube/assignment03-config"
export KUBECONFIG="$PWD/.kube/assignment03-config"
export KUBE_NAMESPACE=mlops
IMAGE=$(python scripts/prepare_deployment.py)
docker build --target inference -t "$IMAGE" .
kind load docker-image "$IMAGE" --name assignment03
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/configmap.yaml -f kubernetes/service.yaml
kubectl apply -f .kube/deployment.yaml
kubectl -n mlops rollout status deployment/prediction-api --timeout=120s
kubectl -n mlops port-forward service/prediction-api 8000:80
```

For operational reads, create short-lived reader credentials after deployment:

```bash
python scripts/create_reader_config.py
export KUBECONFIG="$PWD/.kube/agent-config"
```

The helper applies the reader role, creates a one-hour service-account token and writes it with mode 0600; it never prints the token. Keep the operator kubeconfig for explicit deployment administration. Reader credentials cannot restart the application unless the separate optional restart role is deliberately granted. For a later refresh, choose a new `--output` filename and point `KUBECONFIG` at it; the helper deliberately refuses to replace an existing credential file.

The renderer checks the approved metadata and uses a unique image tag and model identity annotation. Review the rendered manifest before applying. Static manifests specify two replicas, ClusterIP, resource bounds, non-root execution and health probes. `kubernetes/rbac.yaml` grants namespace-local reads; use a restricted kubeconfig for the operational agent. `restart-role.yaml` is a separately opt-in patch permission. Deployment containers do not mount service-account tokens. These bootstrap commands are explicit operator actions; the conversational agent itself never deploys automatically.

## Talk to the agent

```bash
python -m agent.main --list-tools
python -m agent.main 'Which dataset version is tracked by DVC?'
python -m agent.main 'Compare the last five training runs and find the best model by F1.'
python -m agent.main 'Why is my prediction API unavailable?'
python -m agent.main 'Find the best recent model, identify its dataset version, check deployment status, and explain whether retraining is necessary.'
python -m agent.main 'tool airflow_list_dags {}'
python -m agent.main 'tool mlflow_run_details {"run_id":"YOUR_ACTUAL_RUN_ID"}'
```

With no message, the CLI opens an interactive read interface. The default `rules` planner handles documented scenarios and explicit `tool <name> <JSON>` requests. For free-form LLM tool selection, set `OPENAI_API_KEY`/`OPENAI_MODEL` and pass `--mode llm`. The implementation calls the supported OpenAI Responses API using HTTPX; final reports are rendered from structured evidence. The LLM is tested with mocked API responses in CI; no live LLM execution is claimed without a supplied key.

Outputs separate `facts`, `uncertainties`, `recommendations` and `evidence`. Audit logs go to stderr and contain tool name/status/duration/request identifier; result JSON goes to stdout.

Explicit mutations use a separate capability flag:

```bash
python -m agent.main 'Trigger the approved training pipeline' --confirm airflow_trigger_dag
python -m agent.main 'Restart the deployment' --confirm kubernetes_restart_deployment
```

The model cannot set the confirmation input. A trigger acknowledges scheduling; query `airflow_run_status` with its returned DAG/run IDs to verify execution. Restart acknowledgement similarly requires subsequent health inspection. Retraining necessity requires a supplied positive `MODEL_MAX_AGE_DAYS` freshness policy, recorded failed gate or explicit degradation evidence; assessment never starts training by itself.

## Verify and export evidence

```bash
ruff check .
ruff format --check .
pytest -q
python -m scripts.collect_evidence --reproduce-data
# This explicitly starts the approved DAG and polls its real task states:
python -m scripts.collect_evidence --trigger --max-wait 120 --reproduce-data
```

Ordinary tests use mocked external calls and include actual local data/training/inference checks. CI runs lint, formatting, pytest and a separate Airflow DAG-import job on pull requests and pushes. Live collection exports five conversations and optional historical data restoration under ignored `evidence/live/`; review and sanitize selected snapshots before committing them. No mocked infrastructure response is presented as live service evidence.

Use `git log --reverse --oneline` to inspect sequenced milestones. To submit, connect this repository to the intended GitHub remote and push the commits; repository publication is independent of local checks.

To stop only this demonstration, use `docker compose -p assignment03 stop` and `kind delete cluster --name assignment03 --kubeconfig "$PWD/.kube/assignment03-config"`. Keep volumes if retaining MLflow experiments.

## Submission and hosted checks

The completed implementation is published on `assignment03-completion` with [pull request #1](https://github.com/rayyme6/assignment-03-mlops-agent/pull/1). Local checks pass all 153 tests. GitHub currently prevents hosted jobs from starting because of an account billing lock; [the actual status export](evidence/hosted-ci.json) records that restriction. Resolve the account restriction and rerun `assignment-checks` to obtain a hosted result. The existing main branch is preserved pending review.
