"""Correlated identity, recorded gate policy, and one-use confirmation regressions."""

import httpx

from agent.executor import ToolExecutor
from agent.orchestrator import Orchestrator
from agent.schemas import AgentRequest

COMBINED = "Find the best recent model, identify its dataset, check deployment, and explain whether retraining is needed."


def evidence_executor(gate="passed", deployed="older-run"):
    executor = ToolExecutor()
    calls = []

    def best():
        return {
            "best_run": {
                "run_id": "selected-run",
                "metrics": {"f1_score": 0.9},
                "provenance": {
                    "git_commit": "a" * 40,
                    "dvc_hash": "content-hash",
                    "dataset_path": "data/iris.csv",
                    "dvc_file": "data/custom-pointer.dvc",
                    "quality_gate": gate,
                },
            }
        }

    def version(path: str, rev: str):
        calls.append({"path": path, "rev": rev})
        return {"dvc_hash": "content-hash", "git_commit": rev}

    def data_status():
        return {"changed": False}

    def health():
        return {"healthy": True, "deployment": {"model_run_id": deployed}}

    def runs():
        return {"dag_runs": [{"dag_run_id": "completed", "state": "success"}]}

    for name, function in {
        "mlflow_best_run": best,
        "dvc_dataset_version": version,
        "dvc_data_status": data_status,
        "kubernetes_deployment_health": health,
        "airflow_list_runs": runs,
    }.items():
        executor.register(name, function)
    return executor, calls


def test_selected_candidate_and_deployed_model_mismatch_is_reported(monkeypatch):
    monkeypatch.delenv("MODEL_MAX_AGE_DAYS", raising=False)
    executor, _ = evidence_executor(deployed="older-run")
    response = Orchestrator(executor).chat(COMBINED)
    assert any(
        "older-run" in fact and "selected-run" in fact and "match=False" in fact
        for fact in response.facts
    )
    assert all("trigger" not in result.tool for result in response.evidence)


def test_missing_deployed_run_annotation_remains_uncertain():
    executor, _ = evidence_executor(deployed=None)
    response = Orchestrator(executor).chat(COMBINED)
    assert any("identity cannot be verified" in item for item in response.uncertainties)


def test_failed_recorded_gate_supports_retraining_recommendation_without_execution():
    executor, _ = evidence_executor(gate="failed")
    response = Orchestrator(executor).chat(COMBINED)
    assert any("recorded failed quality gate" in item for item in response.recommendations)
    assert not any(result.tool == "airflow_trigger_dag" for result in response.evidence)


def test_recorded_dvc_metadata_path_is_used_instead_of_inferred_filename():
    executor, calls = evidence_executor()
    response = Orchestrator(executor).chat(COMBINED)
    assert calls == [{"path": "data/custom-pointer.dvc", "rev": "a" * 40}]
    assert any("Historical DVC hash matches" in fact for fact in response.facts)


def test_one_confirmation_allows_only_one_llm_mutation_attempt():
    executor = ToolExecutor()
    writes = []

    def trigger(confirmed: bool = False):
        assert confirmed is True
        writes.append("triggered")
        return {"dag_run_id": "one-authorized-run", "state": "queued"}

    executor.register("airflow_trigger_dag", trigger)
    rounds = 0

    def handler(request):
        nonlocal rounds
        rounds += 1
        output = []
        if rounds <= 2:
            output = [
                {
                    "type": "function_call",
                    "call_id": f"call-{rounds}",
                    "name": "airflow_trigger_dag",
                    "arguments": "{}",
                }
            ]
        return httpx.Response(200, json={"output": output})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    response = Orchestrator(executor, mode="llm", llm_client=client).chat(
        AgentRequest(
            message="Trigger the approved workflow once.",
            confirmed_tools=["airflow_trigger_dag"],
        )
    )
    assert writes == ["triggered"]
    assert [result.status for result in response.evidence] == ["ok", "denied"]
