# Technical report: evidence-first MLOps AI agent

Assignment 03, Fall 2026. Submission deadline: **15 November 2026**.

## Implementation and verification status

The source implements all four integrations, structured LLM planning, an offline
router, authorization, training, inference, deployment manifests and CI. **The
submission is not fully verified yet:** A live Kubernetes rollout and a live LLM request still require external access;
the GitHub repository has now been published. Hosted CI was triggered but the
runner did not start because of an account-level restriction. No cluster result,
remote CI success or LLM response has been fabricated.

Verified locally on 9 November 2026: **28 pytest cases passed** in the isolated
environment, including the new Kubernetes demonstration authorization guards.
Ruff lint and format checks passed. On 7 November, DVC data recovery matched the recorded hash, real MLflow
training runs were recorded, real authenticated Airflow API calls triggered a DAG,
and all five tasks completed successfully. FastAPI served a real model prediction.
The deployment manifest renderer was executed successfully; Docker image build and
Kubernetes server-side validation remain unexecuted. A manually triggered
[Live integration evidence workflow](../.github/workflows/integration.yml) now
automates the remaining infrastructure demonstration once hosted jobs can start.
It has not executed; its presence is not deployment evidence.

## Architecture and rationale

See [architecture.md](../architecture.md) for the system diagram, tool catalogue,
permissions and trust boundaries. Each integration exposes at least five bounded
operations. DVC uses approved argument arrays with no shell. MLflow uses
`MlflowClient`; Airflow uses bearer-authenticated `/api/v2`; Kubernetes uses its
Python client with an approved namespace. The CLI is the conversational interface.

The LLM's role is structured planning. The dispatcher validates every argument and
requires human approval for state changes. Evidence records carry request IDs,
statuses and elapsed times. Deterministic synthesis separates facts, uncertainties
and actions. An unavailable service produces an error record, never a substituted
success. Offline conversations below exercise the documented rule-based router;
the structured LLM path has a mocked test, not a claimed live API call.

## Dataset and model evidence

Iris contains 150 observations, four continuous features and three target classes.
The dataset was generated from scikit-learn's bundled copy and tracked with DVC.
A separate detached Git worktree pulled the CSV from the local DVC remote. Its MD5
matched **4224576f0267bf88902f87f0f6200967**. Exact Git reference, hash and recovery
method are recorded in [dataset-recovery.json](../evidence/dataset-recovery.json).
The local remote must be populated again on a fresh clone, or replaced by a shared
remote. Instructions are in the README.

A stratified 75/25 split uses seed 42. Standardization is fitted only on training
rows. Logistic regression candidates are compared on macro F1, with a gate of 0.90.
These are demonstration measurements on a small fixed holdout, not a claim of
production generalization or statistically significant model superiority.

| C | Macro F1 | Accuracy | Gate | Run ID |
|---|---:|---:|---|---|
| 0.1 | 0.845238 | 0.842105 | failed | `e80c30c012864faea6bb9d6349cb3ca8` |
| 2.0 | 0.948718 | 0.947368 | passed | `9f7be1b766e84e8e90efe885eb95a5f8` |

Source: [candidate-evaluations.json](../evidence/candidate-evaluations.json).
The best candidate in the recorded five-run window is `9f7be1b766e84e8e90efe885eb95a5f8`.
Ties follow the returned recent-run order. Ranking does not override the gate or
establish deployment readiness. Every run records the dataset commit and hash;
`code.dirty` is retained truthfully when the tracked source differed from HEAD.

## Airflow execution

Real run: `agent__2122c502-9af6-4827-a2d1-536588684477`.
Observed state: **success**.
The DAG executes validation → preprocessing → training → evaluation → quality_gate,
with two retries and success-dependent downstream tasks. Per-run artifact folders
prevent concurrent overwrites. Training runs in a separate Python interpreter to
avoid conflicting Airflow and MLflow dependency constraints.

The API trigger returned queued initially; success was reported only after polling
returned success. [airflow-run.json](../evidence/airflow-run.json) records the final
run, and the pipeline conversation includes actual task-instance states.

## Five tested conversations

| Scenario and prompt | Actual evidence | Result / limit |
|---|---|---|
| Dataset: “Which dataset version is current?” | [01](../evidence/01-dataset-conversation.json) | DVC pointer, commit, status and recovery instructions returned |
| Selection: “Compare best recent model” | [02](../evidence/02-model-conversation.json) | Real MLflow metrics and provenance; cluster status unknown |
| Pipeline: “Trigger training pipeline”, then “Report pipeline status” | [trigger](../evidence/03-pipeline-trigger.json), [03](../evidence/03-pipeline-conversation.json) | Authorized real trigger, followed by successful run and task evidence |
| Diagnosis: “Why is the prediction API unavailable?” | [04](../evidence/04-deployment-conversation.json) | No configured cluster; errors disclosed, no invented Pod diagnosis |
| Integration: “Find the best recent model, identify its dataset version, check deployment status, and explain whether retraining is necessary” | [05](../evidence/05-multitool-conversation.json) | Real DVC/MLflow/Airflow evidence correlated; Kubernetes unknown; no retraining triggered |

Scenarios 4 and 5 still need a successful live Kubernetes demonstration before
full-credit submission. Exported JSON is used instead of screenshots, as allowed
by the assignment. [inference.json](../evidence/inference.json) verifies a local
prediction, **not** a Kubernetes deployment.

## Two deliberately injected failures

1. **Missing dataset value.** An isolated copy of Iris had one measurement removed.
   Validation raised a null/non-finite-data error. This validates the first DAG
   stage's failure mechanism; no invalid copy was promoted. Root cause: malformed
   input. Resolution: regenerate or recover the clean version and validate before
   rerunning. This injection ran locally, not as a separate failed Airflow DAG run.
2. **Failed model quality gate.** An isolated metrics artifact with F1=0.20 was
   rejected at threshold 0.90. No approved artifact remained. Root cause: metric
   below the explicit policy. Resolution: reject promotion and investigate data,
   split and model behavior before authorizing another training run. Separately,
   the actual C=0.1 experiment also failed the gate with the measured F1 above.

Recorded errors and blocked approval are in
[failure-investigations.json](../evidence/failure-investigations.json). Tests also
inject service timeouts, HTTP 503, unavailable MLflow and CrashLoopBackOff responses.
Those latter service responses are explicitly mocks, not live infrastructure.

## Reliability and security

Tests cover multi-tool selection, exact provenance reference propagation, service
failure without hallucination, malformed input, shell/option injection, metric
ranking, namespace enforcement, exact-target approval, secret redaction, actual
inference, stale gate approvals and model/artifact mismatch. The LLM test proves
a generated write plan cannot grant itself permission. Results:
[pytest-results.xml](../evidence/pytest-results.xml) contains the original 26-test run;
[local-verification-2026-11-09.txt](../evidence/local-verification-2026-11-09.txt)
records the subsequent 28-test verification.

The default Kubernetes role is read-only. Deployment restart still needs exact
human approval and separately granted narrow cluster permissions. Secrets stay in
environment variables or ignored local files. API errors omit remote bodies;
structured logs omit tool payloads. Credentials and raw Airflow logs are excluded
from the deliverable. Only trusted pipeline-generated joblib files may be loaded.

## Engineering defense and limitations

- Why use DVC plus Git? Git records the dataset pointer and code revision; DVC
  stores and recovers the dataset content by hash.
- Why separate liveness and readiness? A responding process may have no usable
  model; readiness prevents it receiving prediction traffic in that state.
- Why not retrain the best model automatically? A high historical F1 is not
  evidence of freshness or degradation. The assessment request provides neither.
- What happens if validation fails? Downstream tasks require upstream success;
  retries do not bypass validation or the gate.
- What failed unexpectedly during setup? DVC was incompatible with a newer
  pathspec API, and MLflow with SQLAlchemy 2.1. Verified compatibility pins were
  added, followed by successful real execution.
- Remaining scope: live Kubernetes, Docker build, live OpenAI plan and a successful
  remote CI run. The user created the repository, and the connector published the
  milestone snapshots plus an original-history Git bundle.
  [Initial CI run](https://github.com/rayyme6/assignment-03-mlops-agent/actions/runs/42385050220)
  did not start its job because of an account-level restriction. The restriction
  must be resolved by the account owner before rerunning CI.
- The demo uses SQLite and Airflow standalone, limited response windows, basic
  redaction and a fixed dataset. Production needs TLS/auth, durable shared storage,
  signed artifacts, drift/freshness policies, pagination and stronger user identity.

## References

- [Airflow 3.1.1 REST API](https://airflow.apache.org/docs/apache-airflow/3.1.1/stable-rest-api-ref.html)
- [MLflow Python client](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html)
- [DVC documentation](https://dvc.org/doc)
- [Kubernetes Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)

Use [submission-checklist.md](submission-checklist.md) to complete the remaining
live verification before the deadline. The original milestone history is preserved
in [development.bundle](../history/development.bundle); restore it using the
[history instructions](../history/README.md) before reproducing old evidence.
