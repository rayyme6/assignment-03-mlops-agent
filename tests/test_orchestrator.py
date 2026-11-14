"""Tool selection, evidence fidelity, authentication failures and trust boundaries."""

import json
import logging
import subprocess

import httpx
import pytest
from pydantic import ValidationError

from agent.executor import ToolExecutor
from agent.orchestrator import Orchestrator, default_executor
from agent.schemas import AgentRequest


COMBINED = (
    "Find the best recent model, identify the dataset version used to train it, "
    "check deployment status, and explain whether retraining is necessary."
)


@pytest.fixture(autouse=True)
def no_optional_policy(monkeypatch):
    monkeypatch.delenv("MODEL_MAX_AGE_DAYS", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def fake_executor(hash_value="dataset-v1", fail_best=False):
    executor = ToolExecutor()
    calls = []

    def best_run(limit: int = 20):
        calls.append(("mlflow_best_run", {"limit": limit}))
        if fail_best:
            raise ConnectionError("MLflow unavailable token=should-never-leak")
        return {"best_run": {"run_id": "a" * 32, "metrics": {"f1_score": 0.93},
                             "start_time": 1_600_000_000_000,
                             "tags": {"git_commit": "b" * 40, "dataset_path": "data/iris.csv",
                                      "dvc_hash": "dataset-v1"}}, "metric": "f1_score"}

    def dataset_version(path: str = "data/iris.csv.dvc", rev: str | None = None):
        calls.append(("dvc_dataset_version", {"path": path, "rev": rev}))
        return {"dvc_hash": hash_value, "dataset_path": "data/iris.csv", "git_commit": rev}

    def data_status():
        calls.append(("dvc_data_status", {}))
        return {"changed": False}

    def deployment_health():
        calls.append(("kubernetes_deployment_health", {}))
        return {"healthy": False, "unhealthy_pods": [{"name": "prediction-api-123"}],
                "events": [{"reason": "Unhealthy", "message": "readiness probe failed"}]}

    def list_services():
        calls.append(("kubernetes_list_services", {}))
        return {"services": [{"name": "prediction-api", "type": "ClusterIP"}]}

    def pod_logs(name: str):
        calls.append(("kubernetes_pod_logs", {"name": name}))
        return {"pod": name, "logs": "model artifact is missing"}

    def list_runs():
        calls.append(("airflow_list_runs", {}))
        return {"dag_runs": [{"dag_run_id": "training-run-1", "state": "failed"}]}

    def trigger_dag(confirmed: bool = False):
        assert confirmed is True
        calls.append(("airflow_trigger_dag", {"confirmed": confirmed}))
        return {"state": "queued", "dag_run_id": "requested-run"}

    def restart_deployment(confirmed: bool = False):
        assert confirmed is True
        calls.append(("kubernetes_restart_deployment", {"confirmed": confirmed}))
        return {"status": "requested"}

    for name, function in {
        "mlflow_best_run": best_run, "dvc_dataset_version": dataset_version,
        "dvc_data_status": data_status, "kubernetes_deployment_health": deployment_health,
        "kubernetes_list_services": list_services, "kubernetes_pod_logs": pod_logs,
        "airflow_list_runs": list_runs, "airflow_trigger_dag": trigger_dag,
        "kubernetes_restart_deployment": restart_deployment,
    }.items():
        executor.register(name, function)
    return executor, calls


def test_default_catalog_registers_all_four_services_without_connecting():
    executor = default_executor()
    assert len(executor.tools) >= 24
    assert {name.split("_")[0] for name in executor.tools} == {"dvc", "mlflow", "airflow", "kubernetes"}
    definitions = {item["name"]: item for item in executor.definitions()}
    assert "confirmed" not in definitions["airflow_trigger_dag"]["parameters"]["properties"]
    assert "confirmed" not in definitions["kubernetes_restart_deployment"]["parameters"]["properties"]


@pytest.mark.parametrize("message,expected", [
    ("Which dataset version is current?", ["dvc_dataset_version", "dvc_data_status"]),
    ("Compare the last five experiments based on F1", ["mlflow_best_run"]),
    ("Check recent Airflow workflow runs", ["airflow_list_runs"]),
    ("Why is the prediction API unavailable?", ["kubernetes_deployment_health", "kubernetes_list_services", "kubernetes_pod_logs"]),
])
def test_rules_select_and_execute_actual_tools(message, expected):
    executor, calls = fake_executor()
    response = Orchestrator(executor).chat(message)
    assert [name for name, _ in calls] == expected
    assert all(result.status == "ok" for result in response.evidence)
    assert all(result.request_id == response.request_id for result in response.evidence)


def test_combined_scenario_correlates_all_four_services_and_never_triggers_training():
    executor, calls = fake_executor()
    response = Orchestrator(executor).chat(COMBINED)
    assert [name for name, _ in calls] == ["mlflow_best_run", "dvc_dataset_version", "dvc_data_status",
                                         "kubernetes_deployment_health", "airflow_list_runs"]
    assert calls[1][1] == {"rev": "b" * 40, "path": "data/iris.csv.dvc"}
    assert any("Historical DVC hash matches" in fact for fact in response.facts)
    assert any("undetermined" in uncertainty for uncertainty in response.uncertainties)
    assert all("Retraining recommended" not in item for item in response.recommendations)
    assert any("metric rank alone is insufficient" in item for item in response.recommendations)


def test_provenance_mismatch_is_reported_as_uncertainty():
    executor, _ = fake_executor(hash_value="different-dataset")
    response = Orchestrator(executor).chat(COMBINED)
    assert any("differs from historical Git/DVC pointer" in item for item in response.uncertainties)
    assert not any("Historical DVC hash matches" in item for item in response.facts)


def test_unavailable_mlflow_does_not_invent_candidate_or_provenance():
    executor, calls = fake_executor(fail_best=True)
    response = Orchestrator(executor).chat(COMBINED)
    assert response.evidence[0].status == "unavailable"
    assert response.evidence[0].data is None
    assert not any(name == "dvc_dataset_version" for name, _ in calls)
    assert not any("Candidate " in fact for fact in response.facts)
    assert "should-never-leak" not in response.model_dump_json()
    assert any("No selected model provenance" in item for item in response.uncertainties)
    assert {name for name, _ in calls} >= {"kubernetes_deployment_health", "airflow_list_runs"}


def test_explicit_freshness_policy_can_support_retraining_recommendation(monkeypatch):
    monkeypatch.setenv("MODEL_MAX_AGE_DAYS", "30")
    executor, calls = fake_executor()
    response = Orchestrator(executor).chat(COMBINED)
    assert any("Retraining recommended" in item and "30 days" in item for item in response.recommendations)
    assert not any(name == "airflow_trigger_dag" for name, _ in calls)


@pytest.mark.parametrize("message,tool", [
    ("Trigger the training pipeline", "airflow_trigger_dag"),
    ("Restart the deployment", "kubernetes_restart_deployment"),
])
def test_mutations_require_trusted_explicit_confirmation(message, tool):
    executor, calls = fake_executor()
    denied = Orchestrator(executor).chat(message)
    assert denied.evidence[0].status == "denied"
    assert calls == []
    approved = Orchestrator(executor).chat(AgentRequest(message=message, confirmed_tools=[tool]))
    assert approved.evidence[0].status == "ok"
    assert calls == [(tool, {"confirmed": True})]


def test_planner_cannot_self_confirm_through_tool_arguments():
    executor, calls = fake_executor()
    response = Orchestrator(executor).chat('tool kubernetes_restart_deployment {"confirmed": true}')
    assert response.evidence[0].status == "invalid"
    assert calls == []


@pytest.mark.parametrize("exception,status", [
    (TimeoutError("secret timeout context"), "timeout"),
    (subprocess.TimeoutExpired("dvc", 1), "timeout"),
    (httpx.ReadTimeout("secret request URL"), "timeout"),
    (ConnectionError("token=secret"), "unavailable"),
    (ValueError("bad arguments with secret"), "invalid"),
    (PermissionError("not authorized"), "denied"),
])
def test_executor_has_distinct_failure_envelopes(exception, status):
    executor = ToolExecutor()

    def failing_tool():
        raise exception

    executor.register("failing_tool", failing_tool)
    response = Orchestrator(executor).chat("tool failing_tool {}")
    assert response.evidence[0].status == status
    assert response.evidence[0].data is None
    assert response.facts == []
    assert response.uncertainties
    assert "secret" not in response.model_dump_json()


def test_audit_logs_are_structured_and_exclude_inputs_outputs_and_secrets(caplog):
    executor = ToolExecutor()

    def sensitive_tool(api_key: str):
        return {"password": "output-password", "note": "token=output-token"}

    executor.register("sensitive_tool", sensitive_tool)
    with caplog.at_level(logging.INFO, logger="mlops.audit"):
        result = executor.execute("sensitive_tool", {"api_key": "input-secret-key"}, "request-123")
    logs = [json.loads(record.message) for record in caplog.records if record.name == "mlops.audit"]
    assert logs[0]["tool"] == "sensitive_tool"
    assert logs[0]["status"] == "ok"
    assert logs[0]["request_id"] == "request-123"
    assert logs[0]["duration_ms"] >= 0
    assert logs[0]["evidence_id"] == result.evidence_id
    assert "input-secret-key" not in caplog.text
    assert "output-password" not in caplog.text
    assert "output-token" not in result.model_dump_json()
    assert result.data["password"] == "[REDACTED]"


def test_tool_input_validation_is_strict_and_rejects_unknown_fields():
    executor, calls = fake_executor()
    for args in ({"limit": "5"}, {"limit": True}, {"limit": 5, "shell": "rm -rf /"}):
        result = executor.execute("mlflow_best_run", args, "request-1")
        assert result.status == "invalid"
    assert calls == []


def test_unknown_tool_and_malformed_json_do_not_execute():
    executor, calls = fake_executor()
    response = Orchestrator(executor).chat('tool not_real {}')
    assert response.evidence[0].status == "invalid"
    malformed = Orchestrator(executor).chat('tool mlflow_best_run {bad json}')
    assert malformed.evidence == []
    assert malformed.uncertainties
    assert calls == []


@pytest.mark.parametrize("message", ["", "x" * 8001])
def test_empty_and_oversized_requests_are_rejected(message):
    with pytest.raises(ValidationError):
        AgentRequest(message=message)


def test_unrecognized_request_does_not_fabricate_facts():
    executor, calls = fake_executor()
    response = Orchestrator(executor).chat("hello there")
    assert response.facts == []
    assert response.uncertainties
    assert calls == []


def test_llm_tool_loop_sends_recorded_evidence_and_ignores_unsupported_prose():
    executor, calls = fake_executor()
    seen_requests = []

    def handler(request):
        body = json.loads(request.content)
        seen_requests.append(body)
        assert body["store"] is False
        if len(seen_requests) == 1:
            assert body["input"] == [{"role": "user", "content": "Inspect recent experiment candidates"}]
            return httpx.Response(200, json={"output": [{"type": "function_call", "call_id": "call1",
                "name": "mlflow_best_run", "arguments": '{"limit": 5}'}]})
        evidence = json.loads(body["input"][-1]["output"])
        assert evidence["data"]["best_run"]["metrics"]["f1_score"] == 0.93
        assert evidence["status"] == "ok"
        return httpx.Response(200, json={"output": [{"type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "I invented a perfect metric of 1.0 and all Pods are healthy"}]}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    response = Orchestrator(executor, mode="llm", llm_client=client).chat("Inspect recent experiment candidates")
    assert len(seen_requests) == 2
    assert calls == [("mlflow_best_run", {"limit": 5})]
    assert "invented a perfect" not in response.model_dump_json()
    assert response.evidence[0].data["best_run"]["metrics"]["f1_score"] == 0.93


def test_llm_cannot_authorize_its_own_mutation():
    executor, calls = fake_executor()
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        if count == 1:
            return httpx.Response(200, json={"output": [{"type": "function_call", "call_id": "call1",
                "name": "kubernetes_restart_deployment", "arguments": '{"confirmed": true}'}]})
        return httpx.Response(200, json={"output": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    response = Orchestrator(executor, mode="llm", llm_client=client).chat("Inspect cluster health")
    assert response.evidence[0].status == "invalid"
    assert calls == []


def test_failed_llm_does_not_invent_operational_output():
    executor, calls = fake_executor()
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(503, json={"error": "unavailable"})))
    response = Orchestrator(executor, mode="llm", llm_client=client).chat("Inspect the system")
    assert response.evidence == []
    assert response.facts == []
    assert any("LLM planning unavailable" in item for item in response.uncertainties)
    assert calls == []


def test_llm_without_key_reports_unavailable_without_network():
    executor, calls = fake_executor()
    response = Orchestrator(executor, mode="llm").chat("Inspect the system")
    assert response.evidence == []
    assert response.facts == []
    assert any("OPENAI_API_KEY" in item for item in response.uncertainties)
    assert calls == []


def test_llm_step_limit_preserves_collected_evidence():
    executor, calls = fake_executor()
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"output": [
        {"type": "function_call", "call_id": "call1", "name": "mlflow_best_run", "arguments": "{}"}
    ]})))
    response = Orchestrator(executor, mode="llm", llm_client=client, max_steps=1).chat("Inspect the system")
    assert len(calls) == 1
    assert response.evidence[0].status == "ok"
    assert any("step limit" in item for item in response.uncertainties)
