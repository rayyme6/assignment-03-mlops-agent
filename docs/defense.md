# Demonstration and engineering defense

Run the installation and service setup in [the README](../README.md), then use this sequence. Keep the [actual evidence index](../evidence/README.md) open beside the terminal. The JSON exports, rather than the prompt or an expected response, establish what happened.

## Suggested sequence

1. Show `git log --oneline --reverse` and explain the implementation milestones. Run `python -m agent.main --list-tools` to display the 24 structured functions.
2. Run lint and pytest using the README commands. Explain why ordinary CI mocks network services and why these tests do not prove that a real scheduler or cluster is running.
3. Ask for dataset provenance. Show the DVC hash, Git commit, tracked-data status and an isolated restoration whose hash matches. Explain the distinction between a data content ID and a Git revision.
4. Compare recent MLflow candidates. Show actual recorded F1, parameters, run status and provenance. Explain why a completed run or best metric does not establish quality-gate approval or production readiness.
5. Explicitly authorize the training DAG. Poll the real run ID and show the five task states. Retain a `running` result as uncertain if it has not completed.
6. Inspect the Kubernetes deployment, Pods, Services, events and relevant logs. Show two desired replicas, resource budgets and distinct liveness/readiness probes. Check `/predict` separately before claiming inference succeeds.
7. Run the mandatory mixed assessment and trace its evidence IDs across all four integrations. Show the historical pointer/run hash comparison and explain any missing evidence or mismatch. Confirm that the request did not initiate retraining.
8. Show the malformed-data and failed-quality-gate investigations. Explain which task failed, which downstream work was prevented and what observation verifies resolution.

## Agent commands

```bash
python -m agent.main "Which dataset version is tracked by DVC, and has the data changed?"
python -m agent.main "Compare the last five training runs and find the best recent model by F1 score."
python -m agent.main "Trigger the model training pipeline" --confirm airflow_trigger_dag
python -m agent.main 'tool airflow_list_runs {"dag_id":"training_pipeline","limit":5}'
python -m agent.main "Why is my prediction API unavailable?"
python -m agent.main "Find the best recent model, identify the dataset version used to train it, check deployment status, and explain whether retraining is necessary."
```

Use a run ID returned by the real service in the next command; do not copy an illustrative ID into the evidence:

```text
python -m agent.main 'tool airflow_run_status {"dag_id":"training_pipeline","run_id":"ACTUAL_RUN_ID"}'
python -m agent.main 'tool airflow_failed_tasks {"dag_id":"training_pipeline","run_id":"ACTUAL_RUN_ID"}'
python -m agent.main 'tool airflow_task_logs {"dag_id":"training_pipeline","run_id":"ACTUAL_RUN_ID","task_id":"quality_gate","try_number":1}'
```

`--confirm` is a separate operator input. The interactive loop supports reads; a single approved request makes write authorization reviewable. Run a write without `--confirm` to demonstrate denial before a service call. An explicit tool request containing `"confirmed": true` is invalid because the field is hidden from the tool schema.

## Discussion prompts

| Question | Defensible explanation |
| --- | --- |
| Where is the AI component? | `--mode llm` uses Responses function calls to select registered tools. Rules mode supplies a transparent fallback. Both share execution validation and deterministic evidence reporting. |
| Why avoid letting an LLM write the final status report? | The tool may be unavailable or return incomplete evidence. Application-generated facts retain evidence IDs and keep unsupported prose from becoming a status assertion. |
| How do you identify the dataset used by a run? | Read recorded Git/DVC tags from MLflow, load the pointer at that Git commit, then compare its recorded content hash. Workspace state alone is insufficient to establish historical provenance. |
| What makes training reproducible? | Dataset content identity, source reference, split seed, parameters and evaluation metrics are recorded. Accessible DVC storage and the environment are also required. A dirty source state reduces reproducibility and must remain visible. |
| What does “best” mean? | It means the highest or lowest finite requested metric among the bounded recent eligible runs. It is not a deployment decision or a guarantee of compatible evaluation conditions. |
| Why Airflow 3 API v2? | It is the public authenticated operation boundary for the selected Airflow version; the adapter uses JWT headers rather than scraping UI pages. |
| What does the DAG gate prevent? | Invalid data prevents training, and a failed metric gate prevents approval of that candidate. Neither training completion nor a `FINISHED` MLflow run is sufficient evidence of approval. |
| What does a timeout mean for a triggered run? | Acceptance may have occurred before the response was lost. Inspect runs using their IDs before retrying; the adapter avoids automatic duplicate POST retries. |
| Can the model authorize a restart? | No. Runtime schemas exclude `confirmed`, trusted request confirmation is checked outside the planner, the module checks the namespace and Kubernetes RBAC provides an independent boundary. |
| Why two replicas? | They allow continued serving when one Pod is unavailable, subject to resource capacity and readiness. Replica count alone does not prove availability. |
| What distinguishes the probes? | Liveness checks the responsive process; readiness checks that an approved model is loaded. A missing model should make readiness fail without creating an unnecessary process restart loop. |
| When is retraining justified? | An explicit exceeded freshness policy, a failed quality gate requiring a new candidate or supplied degradation evidence can support a recommendation. A deployment outage alone does not establish model degradation. |
| How are secrets kept out of evidence? | Audit records omit arguments, raw outputs and exception text. Returned evidence uses redaction and the collector also removes known secret environment values. Credentials are configured outside tracked files. |
| What remains beyond the assignment prototype? | Real drift detection, stronger evaluation policy, signed artifacts, persistent approval records, end-to-end probes and an explicit deployment/rollback workflow. |

## Unexpected failure drill

Start with evidence rather than guessing. Determine whether the failure concerns tool access, data validation, model quality, workflow execution or serving infrastructure. Inspect the narrowest relevant adapter and preserve the request/evidence IDs.

For an unavailable API, first distinguish lack of credentials or cluster access from an actual Pod problem. If cluster access works, inspect desired and ready replicas, observed rollout generation, Pod conditions, restart counts, waiting/termination reasons, Service selectors and events. Use logs from the observed unhealthy Pod. `ImagePullBackOff`, `OOMKilled` and a readiness probe failure have different possible causes; only report the one supported by the returned evidence.

For a failed workflow, inspect actual task instances and distinguish the task that failed from tasks blocked upstream. Read the failed task's current try log, fix the evidenced cause and explicitly authorize a retry when needed. A successful subsequent read establishes recovery only within that read's scope.

If a required service cannot be reached during evaluation, show its `unavailable`/`timeout` envelope, explain what remains unknown and demonstrate the corresponding mocked test. Do not present mocked success as a live deployment or completed DAG run.
