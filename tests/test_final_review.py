"""Regression checks for malformed planner input, timeout config and evidence scope."""

import json
import math

import httpx
import pytest

from agent.executor import ToolExecutor
from agent.orchestrator import Orchestrator, default_executor


@pytest.mark.parametrize("arguments", ["[]", "false", "0", "null", '""', '"text"'])
def test_non_object_llm_arguments_do_not_execute(arguments):
    executor = ToolExecutor()
    calls = []

    def inspect_state():
        calls.append("executed")
        return {"healthy": True}

    executor.register("inspect_state", inspect_state)
    rounds = 0

    def handler(request):
        nonlocal rounds
        rounds += 1
        if rounds == 1:
            return httpx.Response(
                200,
                json={
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "invalid-arguments",
                            "name": "inspect_state",
                            "arguments": arguments,
                        }
                    ]
                },
            )
        output = json.loads(request.content)["input"][-1]
        assert json.loads(output["output"])["status"] == "invalid"
        return httpx.Response(200, json={"output": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    response = Orchestrator(executor, mode="llm", llm_client=client).chat("Inspect state")
    assert [result.status for result in response.evidence] == ["invalid"]
    assert response.facts == []
    assert calls == []


@pytest.mark.parametrize("timeout", ["0", "-1", "121", "NaN", "inf", "-inf", "abc", ""])
def test_invalid_tool_timeout_configuration_is_rejected(monkeypatch, timeout):
    monkeypatch.setenv("TOOL_TIMEOUT_SECONDS", timeout)
    with pytest.raises(ValueError, match="TOOL_TIMEOUT_SECONDS"):
        default_executor()


@pytest.mark.parametrize("timeout", ["0.25", "7.5", "120"])
def test_timeout_configuration_reaches_all_network_adapters(monkeypatch, timeout):
    monkeypatch.setenv("TOOL_TIMEOUT_SECONDS", timeout)
    monkeypatch.delenv("MLFLOW_HTTP_REQUEST_TIMEOUT", raising=False)
    monkeypatch.delenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", raising=False)
    executor = default_executor()
    for tool in ("dvc_data_status", "airflow_list_dags", "kubernetes_list_pods"):
        assert executor.tools[tool].function.__self__.timeout == float(timeout)
    import os

    assert os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] == str(math.ceil(float(timeout)))
    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "0"


def test_explicit_mlflow_timeout_override_is_preserved(monkeypatch):
    monkeypatch.setenv("TOOL_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "42")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
    default_executor()
    import os

    assert os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] == "42"
    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "2"


def test_model_annotation_is_labeled_as_deployment_template_evidence(monkeypatch):
    from agent.schemas import AgentResponse, ToolResult

    monkeypatch.delenv("MODEL_MAX_AGE_DAYS", raising=False)
    response = AgentResponse(
        request_id="request",
        planner="rules",
        evidence=[
            ToolResult(
                tool="mlflow_best_run",
                status="ok",
                duration_ms=1,
                request_id="request",
                data={
                    "best_run": {
                        "run_id": "new-run",
                        "metrics": {"f1_score": 0.9},
                        "tags": {"quality_gate": "passed"},
                    }
                },
            ),
            ToolResult(
                tool="kubernetes_deployment_health",
                status="ok",
                duration_ms=1,
                request_id="request",
                data={
                    "healthy": False,
                    "deployment": {"model_run_id": "new-run"},
                    "unhealthy_pods": [],
                },
            ),
        ],
    )
    Orchestrator._summarize(response)
    assert any("Deployment template model run new-run" in fact for fact in response.facts)
    assert not any("Deployed model run" in fact for fact in response.facts)
    assert any("health=False" in fact for fact in response.facts)
