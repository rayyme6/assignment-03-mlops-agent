# Assignment 03: MLOps AI Agent

A Python agent integrating **DVC, MLflow, Airflow 3 and Kubernetes** with structured
planning, evidence-based answers and explicit write authorization.

**Due: 15 November 2026.** Commits record actual development milestones. See
[technical report](docs/technical-report.md) for measured results and remaining
live verification requirements; do not present unavailable infrastructure as tested.

Repository: https://github.com/rayyme6/assignment-03-mlops-agent

The original evidence refers to development commit IDs. After cloning, run
`git fetch history/development.bundle main:development-history` to restore those
objects. See [history notes](history/README.md).

## 1. Install

Use Python 3.12, Git and a POSIX shell. Docker, kind and kubectl are needed only
for the Kubernetes demonstration. Commands below run from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
pip install --no-deps -e .
cp .env.example .env
set -a; source .env; set +a
pytest -q
ruff check .
ruff format --check .
```

Dependencies are pinned in `requirements.lock`, generated from `pyproject.toml`. `pathspec==0.12.1` and `sqlalchemy==2.0.44` prevent
compatibility errors found during real DVC/MLflow execution. Airflow uses a separate
environment with its official constraints. Never commit `.env` or credentials.

## 2. Reconstruct and inspect DVC data

The CSV is the scikit-learn bundled Iris dataset: 150 rows, four measurements,
three classes. It contains no private data. On a fresh Git clone, bootstrap the
local demonstration remote deterministically:

```bash
python -m training.train generate
# Verify the generated data matches the committed pointer:
python -c 'from training.train import provenance; print(provenance())'
dvc add data/iris.csv
dvc push data/iris.csv.dvc
dvc status
mlops-agent 'Which dataset version is current?'
```

The committed remote is `.local-dvc-remote`, a local teaching remote. For a shared
remote, configure its URL and credentials locally (`dvc remote modify --local`).
To demonstrate recovery without altering your current work:

```bash
revision=$(git rev-parse HEAD)
remote_path="$PWD/.local-dvc-remote"
git worktree add --detach ../dataset-reproduction "$revision"
(cd ../dataset-reproduction && dvc remote modify --local local url "$remote_path" && dvc pull data/iris.csv.dvc)
```

Compare pointers with `DVCTools.execute(ToolInput(operation='diff', reference=...,
other_reference=...))`. Do not claim a version exists unless it is in Git/DVC.

## 3. Start MLflow and train

In terminal 1 (activated project venv):

```bash
mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlruns
```

In terminal 2:

```bash
source .venv/bin/activate
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export MLFLOW_EXPERIMENT=iris-training
dvc repro
python -m training.train all --c 0.1
python -m training.train all --c 2.0
mlops-agent 'Compare the best recent model'
```

`dvc repro` runs validation/preprocessing, training, evaluation and the quality
gate. Every run checks dataset provenance before logging metrics and artifacts.
A model with the highest F1 is only a candidate, not automatically production-ready.
Only FINISHED runs with finite metric values are ranked. `limit=5` means latest
five runs in the configured experiment, not an arbitrary time-based definition.

## 4. Airflow 3

Install Airflow independently to avoid dependency conflicts:

```bash
python3.12 -m venv .airflow-venv
.airflow-venv/bin/pip install 'apache-airflow==3.1.1' \
  -c https://raw.githubusercontent.com/apache/airflow/constraints-3.1.1/constraints-3.12.txt
export AIRFLOW_HOME="$PWD/.airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=false
export MLOPS_ROOT="$PWD"
export MLOPS_PYTHON="$PWD/.venv/bin/python"
export PATH="$PWD/.airflow-venv/bin:$PATH"
airflow standalone
```

Airflow creates local admin credentials under `.airflow`; keep them private. Use
the UI at localhost:8080 to unpause `iris_training`. Obtain an API JWT using the
`/auth/token` endpoint and export it as `AIRFLOW_TOKEN`. The helper below reads
local generated credentials without printing them, triggers a real DAG run,
waits for completion and exports sanitized task evidence.

```bash
# Stop manually launched services first: the helper owns ports 5000 and 8080.
AIRFLOW_PYTHON="$PWD/.airflow-venv/bin/python" \
  .venv/bin/python scripts/run_local_evidence.py
```

For an already running Airflow service:

```bash
mlops-agent 'Trigger the training pipeline' --approve airflow.trigger:iris_training
# Type CONFIRM when prompted. A queued run is not a successful run.
mlops-agent 'Report pipeline status'
```

## 5. Inference and Kubernetes

Test inference locally using a model that passed the gate:

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"features":[5.1,3.5,1.4,0.2]}'
```

Render deployment manifests from the approved artifact. The renderer verifies the model SHA-256,
quality gate and run ID, then replaces the sample placeholders in both the ConfigMap
and Deployment. Use a new image tag for each model version.

```bash
kind create cluster --name mlops-assignment
docker build -t iris-api:local-v1 .
kind load docker-image iris-api:local-v1 --name mlops-assignment
python scripts/render_deployment.py --image iris-api:local-v1 > artifacts/deploy.yaml
kubectl apply -f artifacts/deploy.yaml
kubectl -n mlops-demo rollout status deployment/iris-api --timeout=120s
kubectl -n mlops-demo get pods,deployments,services
kubectl -n mlops-demo port-forward service/iris-api 8000:80
```

The sample image is local and uses `IfNotPresent`; production should use immutable
image digests and a registry. The agent reads your configured kubeconfig or an
in-cluster service account. The supplied reader RBAC does not grant restart rights.

```bash
mlops-agent 'Why is my prediction API unavailable?'
mlops-agent 'Find the best recent model, identify the dataset version used to train it, check deployment status, and explain whether retraining is necessary'
```

Capture actual cluster evidence after deployment:

```bash
mlops-agent 'Why is my prediction API unavailable?' > evidence/04-deployment-conversation.json
mlops-agent 'Find the best recent model and explain whether retraining is necessary' > evidence/05-multitool-conversation.json
```

## 6. Natural-language LLM mode

Set `OPENAI_API_KEY` privately and optionally `OPENAI_MODEL`, then add `--llm`:

```bash
mlops-agent 'Find the best recent model and check its deployed version' --llm
```

The model returns a Pydantic-validated plan of structured tool calls. Plans are
bounded to 20 calls and cannot approve their own mutations. Without `--llm`, a
small documented rule-based router supports the demonstration phrases. No live
LLM result is claimed without an actual API response.

Use the Python tool interface for precise run IDs, metric direction, comparisons,
registry queries, task logs and DVC references. See [architecture](architecture.md)
for all tool operations and permissions.

## 7. Tests, evidence and failure exercises

`pytest -q` runs isolated tests: real model training/inference plus mocked external
services. CI executes lint, format checks and tests on pushes and pull requests.
`evidence/` contains exported records; `docs/technical-report.md` explains scope.
The local evidence helper deliberately injects a missing data value and a failed
quality gate into separate artifact directories. It never corrupts the tracked CSV.

To exercise a cluster failure after a successful deployment, intentionally set
`MODEL_PATH` to a nonexistent file in the ConfigMap, restart the demo Deployment,
and ask the agent to diagnose it. Restore the original path and restart to recover.
This is a manual experiment in the dedicated namespace, not a claimed executed test.

## 8. Repository and remaining verification

The project is published at https://github.com/rayyme6/assignment-03-mlops-agent.
The connected API publishes each development snapshot as a milestone commit;
`history/development.bundle` retains the original commit objects and timestamps.
See [history notes](history/README.md) for restoring original provenance references.

The remaining live checks are listed in `docs/submission-checklist.md`.
