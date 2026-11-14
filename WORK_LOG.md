# Assignment implementation log

The supplied PDF is the requirements source. The user's instruction is to complete the assignment with sequenced, timely commits. Existing dataset pointers and the initial training commit are preserved; the existing untracked MLflow database is kept local.

## Sequence

1. Establish the Python project, configuration, requirements checklist and safe ignore rules.
2. Implement DVC provenance and MLflow experiment tools, then verify them.
3. Implement authenticated Airflow 3 and namespace-restricted Kubernetes tools, then verify them.
4. Implement the staged training pipeline, quality gate, inference API and deployment configuration.
5. Integrate the conversational planner, validated tool execution, confirmation controls and audit logs.
6. Add CI, run integration checks and capture reproducible evidence/failure investigations.
7. Complete installation, architecture and technical-report documentation and review the submission.

Commits are made when each milestone is verified. Commit timestamps reflect when the work is performed.

## Initial environment inspection

- Repository starts at `6adb87d`, with DVC-tracked `data/iris.csv` and an initial training script.
- No GitHub remote is configured. A repository URL has been requested while implementation continues.
- Python 3.10 and the existing DVC/MLflow/ML libraries are available.
- Airflow and the Kubernetes Python distribution need installation; ordinary Docker access is denied.
- No LLM key, Airflow token or cluster credentials are supplied. Live evidence will only claim observations actually collected. Mocked evidence is explicitly labelled.

## Requirements acceptance

- Four integrations, at least three callable tools each; typed inputs and structured outputs.
- DVC versions/diffs/status and isolated historical checkout instructions.
- MLflow recent runs, metrics/params/artifacts, candidate comparisons and registry metadata.
- Airflow 3 JWT API, staged DAG, retries, diagnostics and approved triggers.
- Kubernetes health/logs/events/services/version, two replicas, probes/resources/configuration, namespace boundary and explicit mutation confirmation.
- Multi-tool assessment correlates run provenance, deployment and workflow state; retraining requires evidence and never starts implicitly.
- At least eight meaningful pytest cases, service errors/timeouts, no fabricated results, rejected unauthorized mutations and secret-free structured audit records.
- CI lint/tests on pull requests, installation and service launch docs, technical report, five conversations and two diagnosed failures.

## Data and experiment tools verified

- Implemented five DVC and six MLflow operations using bounded subprocess calls and supported Python clients.
- Added revision/path validation, provenance checks, recent-run metric comparison and registry inspection.
- `python3 -m pytest -q tests/test_dvc_tools.py tests/test_mlflow_tools.py`: 32 passed.
- DVC's protected global cache was resolved by selecting a repository-local `DVC_SITE_CACHE_DIR`.

## Workflow and cluster tools verified

- Implemented six JWT-authenticated Airflow 3 API tools and seven Kubernetes operations.
- Triggering is restricted to an approved DAG; namespace boundaries and explicit restart confirmation are enforced before connecting.
- `python3 -m pytest -q tests/test_airflow_tools.py tests/test_kubernetes_tools.py`: 22 passed, including timeout and authorization failures.

## Training, serving and deployment verified

- Added the five-stage Airflow 3 DAG, deterministic training CLI, recorded provenance and a blocking F1 quality gate.
- Inference loads only approved artifacts, validates feature input, and exposes separate liveness/readiness and version endpoints.
- Kubernetes configuration includes two replicas, ClusterIP, probes, bounded resources and namespace-scoped RBAC.
- Added container builds, isolated local Compose services, DVC pipeline and dataset bootstrap instructions.
- All 12 training/inference/DAG tests passed. In this environment FastAPI tests require sandbox escalation because cross-thread event-loop signaling is restricted.

## Conversational integration verified

- Added a typed request/evidence API, validated tool registry and JSON audit events containing tool/status/duration/request IDs only.
- Added the supported OpenAI Responses function-calling planner and an explicit offline planner with a structured tool escape hatch.
- Combined assessment follows selected-run provenance through historical DVC metadata and checks live cluster/workflow state. Reports separate facts, uncertainties and recommended actions.
- Mutation confirmation is supplied by the operator outside model-generated arguments. No assessment starts retraining implicitly.
- Added 29 orchestration tests, including planner/tool failures, provenance mismatch, strict authorization and no fabricated results.

## Integration verification and real-service bootstrap

- Full integrated test suite: 109 passed in 42.14 seconds; lint/format CI checks added.
- Built a fresh real local MLflow experiment and captured a quality-gate rejection: macro F1 0.8997493734 was below the initial 0.90 threshold. No approved artifact was published for that run.
- Set the educational Iris demo policy explicitly to 0.85 in training configuration and the DAG. A subsequent real run passed with the same recorded metric. This demo policy is not evidence of production readiness.
- Installed official kind/kubectl locally (ignored), created only the isolated `assignment03` cluster, and began packaging the approved model with a run-specific image tag.
- Added exact run annotations to rendered Deployments and candidate/deployment identity checks to the agent.

## CI, regression and failure evidence verified

- Final integration/regression suite at this milestone: 116 passed in 73.23 seconds; `ruff check .` and `ruff format --check .` passed.
- Added PR CI with dataset reconstruction, lint/format/tests and a separately constrained real Airflow DAG import check.
- Added live five-conversation collector, isolated DVC restoration, deployment renderer and two deliberately injected local failure exports.
- Executed both failures: invalid numeric data blocked every downstream artifact; injected F1 0.4 failed the real MLflow-backed 0.85 quality gate without promoting a model. Results are retained in `evidence/failures.json`.
- Added regressions for deployed-versus-selected run identity, absent identity, failed-gate recommendations, nonstandard DVC pointer names, and one-use confirmation capabilities.
- Connected the user-supplied GitHub repository and created `assignment03-completion`; the existing divergent `origin/main` is preserved for review.

## Final review hardening

- Rejected non-object LLM arguments (including null/false/empty arrays) and wired the bounded timeout configuration into every integration.
- Clarified that a Deployment template's run annotation describes desired state, rather than proving what every serving Pod has loaded during rollout.
- Removed the incompatible Airflow optional extra; the constrained Airflow environment remains separate from the agent environment.
- Added a helper for one-hour namespace-scoped reader credentials, stored only in ignored mode-0600 kubeconfig files.
- Added 19 targeted review regression cases; 53 planner/review cases pass with lint/format checks.

## Submission documentation and resumed verification

- Added complete installation/service launch instructions, architecture diagrams, all tool definitions/permissions, rubric traceability, engineering defense and known limitations.
- Retained actual local failure exports and machine-readable test evidence; no live LLM execution is claimed without an API key.
- On resuming, the isolated Kubernetes deployment was 2/2 ready. The demonstration MLflow/Airflow containers had stopped during an external runtime interruption; only these assignment containers are being restarted.
- Full resumed checks: 135 pytest cases passed, Ruff lint and formatting passed. Further live workflow and controlled rollout observations will be added separately.

## Final policy and credential review verified

- Corrected contradictory retraining uncertainty when recorded failed-gate evidence already supports a recommendation.
- Rejected nonfinite/nonpositive freshness policies and retained explicit uncertainty for missing run timestamps.
- Rejected credential-bearing HTTP tracking endpoints before evaluation reads or logging; authentication belongs in separate MLflow environment variables.
- Full suite: 150 tests passed in 27.82 seconds. Ruff lint, formatting and whitespace checks passed; the verification export records the exact tested source diff.

## Real data, experiment and workflow evidence

- Reproduced the final source through DVC and refreshed `dvc.lock` with real approved artifacts and measured metrics.
- Real MLflow comparison: RF5 macro F1 0.9333333333; RF50/RF100 macro F1 0.8997493734 on the same fixed holdout. These are candidates, not production-readiness conclusions.
- Authenticated Airflow 3 API verification observed missing-token401, invalid-token403 and valid-token200 without exporting credentials.
- Confirmed training DAG run `agent__c0a48e7ac9154efd9d92f3dfc831c971` finished successfully with all five task states successful. Its model metadata records source95f992e and `git_dirty=false`.

## Live diagnosis regression corrected

- Real Airflow logs placed the decisive quality-gate exception beyond a long startup preamble. The summary now prioritizes structured error/warning messages and searches the full sanitized log before producing concise diagnostics.
- Three regression cases cover late decisive errors and structured log events; 63 targeted planner/review/regression tests passed with Ruff checks.
- The controlled Airflow run failed only its quality gate after three attempts, with all four upstream stages successful. Diagnosis is refreshed from that existing run without retriggering training.
