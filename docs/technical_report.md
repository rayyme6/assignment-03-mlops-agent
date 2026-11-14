# Technical report: MLOps AI Agent

## Purpose and implementation

This system provides one conversational interface to inspect dataset provenance, compare recorded experiments, operate an approved training workflow and diagnose an inference application. It uses Python, DVC, MLflow, Airflow 3, FastAPI, Docker, Kubernetes and GitHub Actions. The [architecture document](../architecture.md) explains the component boundaries and data flow. The [README](../README.md) gives installation, configuration and service launch commands.

The CLI offers a deterministic rules planner and an OpenAI Responses function-calling planner. Both use the same strict tool catalogue and evidence envelopes. The application generates the final facts and uncertainties from tool results; unsupported model prose cannot become an operational assertion.

## Rubric coverage

This is a traceability map for the assignment's 100-mark rubric, not a claim about marks awarded. Live behavior is established only by the exported observations linked in [the evidence index](../evidence/README.md).

| Component | Marks | Implementation and validation |
| --- | ---: | --- |
| DVC integration and provenance | 15 | Five tools list tracked data, inspect current/historical pointers, compare revisions, detect changes and explain isolated reproduction. Training records Git commit and actual DVC content hash. DVC tests cover path/revision validation and changed-data behavior. |
| MLflow experiment intelligence | 15 | Six tools inspect experiments/runs, metrics, parameters, artifacts, best recent candidates, comparisons and optional model registry entries. Candidate selection requires a completed run and finite metric; it carries a readiness caveat. |
| Airflow DAG and operations | 15 | Six tools use authenticated Airflow 3 `/api/v2`; the DAG has validation, preprocessing, training, evaluation/MLflow and a quality gate. Retries, timeouts, dependencies and task diagnostics are included. |
| Kubernetes deployment and diagnostics | 15 | Seven tools inspect application state and provide a confirmed namespace-scoped restart. Manifests specify two replicas, ClusterIP, probes, resource budgets, ConfigMap and reader/optional restart RBAC. |
| Multi-tool reasoning | 15 | The mandatory request queries MLflow, verifies recorded DVC provenance, inspects Kubernetes and reads Airflow runs. Historical and recorded hashes are correlated; missing data remains uncertain. Assessment does not trigger retraining. |
| Testing, reliability and safety | 10 | Pytest covers routing, all four integrations, timeouts, invalid requests, outages, no invented evidence, strict confirmation, namespace limits, structured audit logs, training gates and serving readiness. Failure exercises are described below. |
| Repository and CI | 5 | Reproducible project metadata, isolated Airflow setup, environment example, infrastructure files and a PR lint/test workflow. External calls are mocked in ordinary CI. Git history records sequential implementation milestones. |
| Demonstration and engineering defense | 10 | Five-scenario collector, timestamped exports, architecture, this report, failure investigations and a [defense guide](defense.md). Real Airflow task execution and Kubernetes readiness/prediction are retained in service-specific exports. |
| **Total** | **100** | |

## Tool definitions and permissions

Run `python -m agent.main --list-tools` for exact JSON schemas, defaults and required arguments. All inputs are validated before dispatch. Read tools may read their integration's recorded state but cannot execute arbitrary shell commands or cluster operations.

| Tool | Principal inputs | Permission and output |
| --- | --- | --- |
| `dvc_list_datasets` | none | Read tracked datasets and current Git commit. |
| `dvc_dataset_version` | `path`, optional `rev` | Read an approved relative `.dvc` pointer and its content identity. |
| `dvc_compare_versions` | `old_rev`, `new_rev` | Read DVC differences between resolved Git commits. |
| `dvc_data_status` | none | Read local tracked-data status. |
| `dvc_reproduction_instructions` | `rev`, `path` | Return commands/reference for reproducing data in isolation; does not switch the working checkout. |
| `mlflow_list_experiments` | none | Read active experiments. |
| `mlflow_list_runs` | optional experiment IDs, `limit` | Read bounded recent runs and recorded provenance. |
| `mlflow_run_details` | `run_id` | Read metrics, parameters, tags and root artifact listings. |
| `mlflow_best_run` | `metric`, experiment IDs, `limit`, `maximize` | Rank eligible recent runs; return comparison evidence and readiness caveat. |
| `mlflow_compare_runs` | `run_ids` | Read two or more runs and compare recorded metric definitions and provenance. |
| `mlflow_model_versions` | optional `name` | Read registry information when configured. |
| `airflow_list_dags` | none | Authenticated DAG read. |
| `airflow_list_runs` | `dag_id`, `limit` | Authenticated recent-run read. |
| `airflow_run_status` | `dag_id`, `run_id` | Read actual run and task-instance states. |
| `airflow_trigger_dag` | approved `dag_id`, optional `conf` | **Write:** trusted operator confirmation required; planner cannot supply approval. |
| `airflow_failed_tasks` | `dag_id`, `run_id` | Read failed and upstream-failed tasks. |
| `airflow_task_logs` | DAG/run/task IDs, `try_number` | Read available task diagnostics with common secret assignments redacted. |
| `kubernetes_list_pods` | optional approved namespace | Read Pod conditions, containers and restart counts. |
| `kubernetes_list_deployments` | optional approved namespace | Read replica/rollout state, images and application version. |
| `kubernetes_list_services` | optional approved namespace | Read Service type, selectors and ports. |
| `kubernetes_pod_logs` | `name`, `tail_lines`, namespace | Read bounded application logs. |
| `kubernetes_events` | optional approved namespace | Read namespace event evidence. |
| `kubernetes_deployment_health` | `name`, namespace | Correlate Deployment rollout and Pod readiness; return unhealthy Pods and events. |
| `kubernetes_restart_deployment` | `name`, namespace | **Write:** explicit confirmation and approved namespace required; optional RBAC limits patch to `prediction-api`. |

Secrets are supplied through environment/configuration, excluded from Git and omitted from audit records. API errors expose exception types rather than raw messages that could contain credentials. Common secret patterns are redacted from exported evidence. Audit events retain tool, status, duration, request and evidence IDs so a reviewer can follow a request without seeing input secrets.

## Five evaluation conversations

The collector creates these exact requests against configured real services. It does not fabricate healthy service states. The current execution results, timestamps, commit references and evidence IDs belong in the exported JSON referenced by [the evidence index](../evidence/README.md). Live exports now establish a successful real Airflow run with all five tasks successful and a Kubernetes deployment with two ready replicas, actual model provenance and a successful prediction. The five-conversation snapshot is indexed alongside those service-specific exports; mocked unit tests remain separately labelled.

| Scenario | Request | Evidence needed to defend the result |
| --- | --- | --- |
| Dataset provenance | “Which dataset version is tracked by DVC, and has the data changed?” | Actual pointer hash, Git commit, change status; isolated restoration with matching content hash when `--reproduce-data` is used. |
| Model selection | “Compare the last five training runs and find the best recent model by F1 score.” | Recorded candidate metrics, completed status, run IDs and provenance; explicit readiness caveat. |
| Pipeline execution/status | “Report the recent training pipeline workflow status.” With `--trigger`, collection instead explicitly requests training. | Real Airflow run ID and terminal state plus the five task states. A queued response alone is insufficient to claim execution success. |
| Deployment diagnosis | “Why is my prediction API unavailable? Inspect Kubernetes deployment health.” | Deployment replica/rollout state, unhealthy Pods, relevant events, Service selectors and available logs. The question's premise is not treated as an observed outage. |
| Multi-tool reasoning | “Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary.” | All four tool families, run-to-Git/DVC hash correlation, current application state, workflow state, uncertainties and a policy-based recommendation. No training side effect. |

To refresh actual observations:

```bash
python -m scripts.collect_evidence --reproduce-data
# Explicitly authorizes one approved training run and waits for observations:
python -m scripts.collect_evidence --trigger --reproduce-data --max-wait 120
```

The JSON distinguishes `ok` tool access from failed, denied, timed-out or unavailable operations. “Integration access observed” does not mean the deployed model is ready or the workflow completed. A polling deadline may leave a real run in progress; retain that uncertainty and inspect again.

## Failure investigations

The following investigations describe the implemented fault paths and how their evidence should be interpreted. Actual exercise output is indexed separately in [the evidence directory](../evidence/README.md); commands and expected behavior are not presented as a substitute for observed logs.

### Malformed dataset prevents training

**Injection:** Supply a temporary CSV with an incorrect feature schema, missing/nonfinite values or invalid class coverage. Keep the original DVC-tracked file intact.

**Root cause:** `validate_dataset` rejects input that cannot support the defined Iris model and stratified evaluation. It raises before preprocessing or training. Under Airflow, `validate` fails and its dependent tasks cannot proceed successfully.

**Diagnosis:** Inspect the validation exception in the local exercise or query `airflow_failed_tasks` followed by `airflow_task_logs`. Distinguish the failed validation task from `upstream_failed` tasks that never executed. Confirm that no new approved artifact was produced.

**Resolution:** Restore the correct schema and values. If the data intentionally changed, update the DVC pointer and commit the new version, then retrain with explicit operator authorization. Re-run validation before asserting that the issue is resolved.

### Quality gate prevents candidate approval

**Executed local injection:** A real isolated MLflow run was deliberately assigned F1=0.4 and tested against the explicit 0.85 gate. The candidate was an unloaded sentinel file; this score is an injected failure fixture, not a measured model score. The exporter verified the failed MLflow tag and gate artifact and absence of approved serving files. Temporary storage was removed afterward. A real trained-model gate failure is also observed separately in the initial baseline described below.

**Root cause:** A candidate's recorded evaluation is below the configured `min_f1` policy. Training can finish and an MLflow run can be `FINISHED` while the separate quality gate fails. A run's completed status is therefore insufficient to infer promotion eligibility.

**Diagnosis:** Inspect `quality_gate.json`, the MLflow `quality_gate` tag and recorded F1/threshold. In Airflow, inspect the `quality_gate` task and its logs. The agent reports the evidence; it must not infer that changing infrastructure will improve model quality.

**Resolution:** Investigate data quality, split representativeness and candidate configuration. Compare an improved candidate against an explicitly justified threshold. Re-run the gate and verify that it passes before building an image with that candidate. Lowering a threshold merely to make the demonstration pass is not evidence of model improvement.

### Service outage and unavailable credentials

Automated tests inject MLflow connection failures, Airflow authentication/server errors, Kubernetes connection failures, tool timeouts and failed LLM responses. Results remain `unavailable` or `timeout`, contain no invented model/Pod state and surface uncertainties. The mixed assessment continues with independent available reads. Missing a JWT or kubeconfig requires fixing access; it does not imply that the underlying workload failed.

## Validation and limitations

All 153 automated tests pass, with Ruff lint and formatting checks passing; exact commands and source fingerprints are retained in `evidence/verification.json` and JUnit XML. The automated test suite verifies strict schema validation, success routing/execution, provenance correlation, no fabricated operational output, all integration adapters, error classification, audit secrecy, confirmation enforcement, namespace restrictions, gate behavior, inference validation and readiness. A mocked HTTPX Responses loop verifies function-call exchange and proves that model-supplied `confirmed=true` cannot authorize a mutation. Run the current lint and test commands from the README; retain their actual output in the evidence index.

Iris is small and static. Its held-out F1 is not a demonstration of real-world production generalization, fairness or drift handling. The rules planner recognizes a limited set of English intents; free-form requests need the configured LLM or explicit tool syntax. There is no automatic degradation detector, deployment promotion, rollback controller, canary analysis or experiment-comparability guarantee. A highest-score candidate may have a failed gate or use an incompatible evaluation definition, which must be examined before approval.

Kubernetes readiness and a completed rollout describe the configured health checks; they do not guarantee prediction correctness or client-to-Service connectivity. Validate `/predict` separately. Airflow retry behavior can create additional MLflow evaluations after partial failure, so use run provenance and gate state when comparing them. Exported observations are snapshots rather than continuous monitoring. Local development credentials and SQLite-backed demonstrations require stronger authentication, managed storage, network policy and secret handling before a real shared production deployment.

Proposed improvements include tool-specific output schemas, end-to-end prediction probes, signed artifact verification, evaluation-dataset compatibility checks, drift monitoring, persistent approval records, workload identities and an explicit release workflow with rollback evidence.

## Observed initial baseline and educational policy

The actual random forest with 100 trees, a fixed 42 seed and 30-row holdout produced accuracy 0.90 and macro F1 0.899749373433584. The initial 0.90 gate correctly rejected it and produced no approved artifact. The demo configuration was then explicitly changed to 0.85, and the same measured result passed. This is a disclosed educational acceptance policy change, not a model improvement or evidence of production quality. A production threshold should be justified independently using representative evaluation data; the small Iris demo cannot support that conclusion.

The public Iris CSV was recreated independently from scikit-learn and matched the committed DVC MD5 `4d301abed5efe50eccda350cde38e0eb` byte-for-byte. The historical restoration exporter independently verifies content retrieval using DVC and keeps the working tree intact.

## Verified live infrastructure and diagnosis

The confirmed successful Airflow run is `agent__c0a48e7ac9154efd9d92f3dfc831c971`. Its five task states are successful, and its MLflow artifact records source commit `95f992e`, `git_dirty=false`, the actual dataset hash, and a passing 0.85 educational quality gate. API authentication was independently checked: missing JWT returned 401, invalid JWT returned 403 and valid JWT returned 200. The server runs Airflow 3.1.3 with its isolated constrained MLflow 3.2.0 client; the tracking server and agent use MLflow 3.18.0. The supported run, metric, tag and artifact APIs were verified across those versions.

The served Kubernetes model reports run `0390ac27a2e2499ba9cd76017c9794b6`, application version 1.0.0 and dataset hash `4d301abed5efe50eccda350cde38e0eb`. Actual Service requests returned successful liveness, readiness, version and prediction responses; the sample Iris input returned class 0 (`setosa`). The best later candidate, RF5 run `3151fb2312d5466e8caf9ff8ce53d1e5`, records macro F1 0.9333333333 on the same holdout. The agent reports that its run identity differs from the Deployment template rather than assuming automatic promotion. No freshness policy or degradation study was supplied, so this metric ranking does not establish that retraining is necessary.

A deliberately incorrect ConfigMap `MODEL_PATH` created a replacement Pod with liveness 200 and readiness/version/prediction 503. The agent collected actual Pod conditions, readiness-failure events and model-unavailable logs. RollingUpdate preserved the previous ready Pods, so the Service continued returning 200: an unhealthy rollout was observed, not a Service-wide outage. The original configuration was restored, the rollout returned to two ready replicas, and prediction again succeeded. Short-lived reader credentials were verified to permit `mlops` Pod reads while denying deployment patches and reads in `default`.

See `evidence/kubernetes-success.json`, `evidence/kubernetes-failure.json`, `evidence/airflow-auth.json`, `evidence/airflow-success.json` and `evidence/model-candidates.json` for the original measurements. These are dated observations of the isolated assignment environment rather than a guarantee of future service uptime. Live LLM execution remains unverified without a supplied API key; its function-calling protocol and permission boundaries are tested with explicit mock responses.

The deliberately stricter Airflow run `agent__3628b71bcb2d4704a042cc09cf48e48e` reached terminal `failed` after three quality-gate attempts; its four upstream stages succeeded. Logs identify measured F1 0.8997493734 below the injected 0.99 threshold. MLflow records `quality_gate=failed`, and the run directory contains no approved model or metadata. The agent's retained diagnosis quotes that decisive error and recommends inspecting the metric and threshold before authorizing another run. This failure candidate truthfully records `git_dirty=true` because the workspace changed during the exercise; only the successful clean run is presented as clean-source provenance. The log-summary regression discovered by this exercise is covered by three additional tests, bringing the final passing suite to 153.
