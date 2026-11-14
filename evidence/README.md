# Exported evidence

All service snapshots below are actual observations from the isolated assignment environment on 14 November 2026. JSON files retain timestamps, tool status, request/evidence IDs and structured responses. Local test doubles are identified separately; they are not proof of running infrastructure.

| Export | Observed evidence |
| --- | --- |
| [verification.json](verification.json), [tests.xml](tests.xml) | 153 passing local tests; lint/format checks; exact source fingerprint and scope. |
| [model-candidates.json](model-candidates.json) | Actual MLflow-backed RF5/RF50 comparison on the same fixed holdout. |
| [airflow-auth.json](airflow-auth.json) | Missing-token401, invalid-token403 and authenticated200; no credentials exported. |
| [airflow-success.json](airflow-success.json) | Confirmed real training run; all five stages successful; timestamped task transitions. |
| [airflow-failure.json](airflow-failure.json) | Deliberately stricter quality gate, retry/task evidence and the agent's actual diagnosis. |
| [kubernetes-success.json](kubernetes-success.json) | Two ready replicas, real model version/provenance and successful Service prediction. |
| [kubernetes-failure.json](kubernetes-failure.json) | Missing-model replacement Pod returns503; old Pods preserve Service availability; configuration restored and prediction recovered. |
| [failures.json](failures.json) | Two isolated local fault exercises: missing numeric data and explicitly injected metric below the real MLflow-backed gate. |
| [conversations.json](conversations.json) | Five live agent conversations across all four services plus isolated historical DVC restoration. |

The successful Airflow model metadata records source95f992e with `git_dirty=false`. The deployed model and the highest-scoring later candidate have different run IDs; the agent reports this difference and does not promote or retrain automatically. LLM protocol tests use mocked responses because no live key was supplied. Hosted CI status is recorded separately from local verification.

Refresh actual observations with:

```bash
python -m scripts.collect_evidence --reproduce-data
# Explicitly authorizes the approved training DAG and records real task statuses:
python -m scripts.collect_evidence --trigger --max-wait 120 --reproduce-data
```

Fresh exports go under ignored `evidence/live/`. Review and sanitize them before retaining a submission snapshot. DVC restoration fetches to a temporary directory and verifies the historical content hash without modifying the original checkout. Failed calls remain failures in every export. Credentials, kubeconfig contents and tokens are excluded.
