import json
import subprocess
from types import SimpleNamespace as NS
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient

from agent.orchestrator import Agent, redact
from agent.schemas import ToolInput
from agent.tools.airflow_tools import AirflowTools
from agent.tools.dvc_tools import DVCTools
from agent.tools.kubernetes_tools import KubernetesTools
from agent.tools.mlflow_tools import MLflowTools
from api.main import app
from training.train import generate, preprocess, quality_gate, train, validate


@pytest.fixture
def toolset():
    result = {}
    for name, cls in [
        ("dvc", DVCTools),
        ("mlflow", MLflowTools),
        ("airflow", AirflowTools),
        ("kubernetes", KubernetesTools),
    ]:
        result[name] = Mock(operations=cls.operations)
    result["mlflow"].execute.return_value = {
        "best": {
            "run_id": "run-1",
            "metrics": {"f1": 0.96},
            "tags": {"dataset.git_commit": "abc123", "quality_gate": "passed"},
        },
        "metric": "f1",
        "scope": "latest five runs",
    }
    result["dvc"].execute.return_value = {}
    result["kubernetes"].execute.return_value = []
    result["airflow"].execute.return_value = {"dag_runs": []}
    return result


def test_multitool_reads_provenance_without_triggering(toolset):
    answer = Agent(toolset).ask("Find best model and explain whether retraining is necessary")
    assert {e.tool for e in answer.evidence} == set(toolset)
    assert not any(e.operation in {"trigger", "restart"} for e in answer.evidence)
    assert any("0.96" in fact for fact in answer.facts)
    assert toolset["dvc"].execute.call_args_list[0].args[0].reference == "abc123"


def test_no_fabrication_when_tracking_fails(toolset):
    toolset["mlflow"].execute.side_effect = ConnectionError("private-server-detail")
    answer = Agent(toolset).ask("best model")
    assert not any("Candidate" in f for f in answer.facts)
    assert any("mlflow.best" in u for u in answer.uncertainties)
    assert "private-server-detail" not in answer.model_dump_json()


@pytest.mark.parametrize(
    "exc", [TimeoutError(), httpx.ReadTimeout("secret"), subprocess.TimeoutExpired("dvc", 30)]
)
def test_timeouts_become_evidence_errors(toolset, exc):
    toolset["dvc"].execute.side_effect = exc
    e = Agent(toolset).dispatch("dvc", {"operation": "status"}, "req")
    assert e.status == "error" and e.data is None


@pytest.mark.parametrize(
    "args",
    [
        {"operation": "status", "limit": 0},
        {"operation": "status", "confirmed": True},
        {"operation": "shell"},
    ],
)
def test_invalid_requests_never_reach_tools(toolset, args):
    e = Agent(toolset).dispatch("dvc", args, "req")
    assert e.status == "error"
    toolset["dvc"].execute.assert_not_called()


def test_cluster_mutation_requires_exact_approval(toolset):
    agent = Agent(toolset)
    a = {"operation": "restart"}
    assert agent.dispatch("kubernetes", a, "req").status == "denied"
    assert (
        agent.dispatch("kubernetes", a, "req", {"kubernetes.restart:other/api"}).status == "denied"
    )
    toolset["kubernetes"].execute.assert_not_called()
    assert (
        agent.dispatch("kubernetes", a, "req", {"kubernetes.restart:mlops-demo/iris-api"}).status
        == "ok"
    )


def test_namespace_rejected_before_connecting():
    tool = KubernetesTools(core=Mock(), apps=Mock())
    with pytest.raises(PermissionError):
        tool.execute(ToolInput(operation="restart", namespace="production"))
    tool.apps.patch_namespaced_deployment.assert_not_called()


def test_airflow_dag_allowlist():
    client = Mock()
    with pytest.raises(PermissionError):
        AirflowTools(client).execute(ToolInput(operation="trigger", dag_id="dangerous"))
    client.post.assert_not_called()


def test_airflow_v3_path_and_payload():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"dag_run_id": "real-returned-id", "state": "queued"})

    client = httpx.Client(
        base_url="http://airflow",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer fake-test-only"},
    )
    result = AirflowTools(client).execute(ToolInput(operation="trigger"))
    assert seen[0].url.path == "/api/v2/dags/iris_training/dagRuns"
    assert "logical_date" in json.loads(seen[0].content)
    assert result["state"] == "queued"  # Do not claim queued means successful.


def test_airflow_service_failure():
    client = httpx.Client(
        base_url="http://airflow",
        transport=httpx.MockTransport(
            lambda r: httpx.Response(503, json={"detail": "unavailable"})
        ),
    )
    e = Agent({"airflow": AirflowTools(client)}).dispatch("airflow", {"operation": "runs"}, "req")
    assert e.status == "error"


def run(run_id, f1, status="FINISHED"):
    return NS(
        info=NS(run_id=run_id, status=status, start_time=1),
        data=NS(metrics={"f1": f1}, params={}, tags={}),
    )


def test_best_model_filters_failed_and_nonfinite_runs():
    client = Mock()
    client.get_experiment_by_name.return_value = NS(experiment_id="1", name="iris-training")
    client.search_runs.return_value = [
        run("failed", 1.0, "FAILED"),
        run("a", 0.8),
        run("b", 0.95),
        run("nan", float("nan")),
    ]
    result = MLflowTools(client).execute(ToolInput(operation="best"))
    assert result["best"]["run_id"] == "b"
    assert "production" in result["caution"]


def test_metric_minimization():
    client = Mock()
    client.get_experiment_by_name.return_value = NS(experiment_id="1", name="iris-training")
    client.search_runs.return_value = [run("a", 0.8), run("b", 0.95)]
    assert (
        MLflowTools(client).execute(ToolInput(operation="best", direction="min"))["best"]["run_id"]
        == "a"
    )


def test_dvc_rejects_shell_and_option_injection(tmp_path):
    tool = DVCTools(tmp_path)
    for ref in ["--help", "HEAD; touch /tmp/pwned", "$(whoami)"]:
        with pytest.raises(ValueError):
            tool.execute(ToolInput(operation="version", reference=ref))


def test_logs_exclude_secrets_and_include_trace_fields(toolset, caplog, monkeypatch):
    monkeypatch.setenv("AIRFLOW_TOKEN", "super-private-token")
    toolset["airflow"].execute.return_value = {"logs": "Bearer super-private-token"}
    with caplog.at_level("INFO", logger="mlops.agent"):
        e = Agent(toolset).dispatch("airflow", {"operation": "runs"}, "trace123")
    assert "super-private-token" not in caplog.text + e.model_dump_json()
    record = json.loads(caplog.records[0].message)
    assert record["request_id"] == "trace123"
    assert all(k in record for k in ["tool", "status", "duration_ms"])


def test_recursive_secret_redaction():
    assert redact({"password": "abc", "x": ["token=abcd"]}) == {
        "password": "[REDACTED]",
        "x": ["token=[REDACTED]"],
    }


def test_validation_failure_blocks_bad_data(tmp_path):
    path = tmp_path / "iris.csv"
    generate(path)
    path.write_text(path.read_text().replace("5.1", "", 1))
    with pytest.raises(ValueError, match="nulls"):
        validate(path)


def test_quality_gate_failure_does_not_approve(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({"metrics": {"f1": 0.2}}))
    with pytest.raises(ValueError, match="promotion blocked"):
        quality_gate(tmp_path)
    assert not (tmp_path / "approved.json").exists()


def test_api_not_ready_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing"))
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert client.post("/predict", json={"features": [1, 2, 3, 4]}).status_code == 503


def test_actual_training_and_prediction(tmp_path, monkeypatch):
    path = generate(tmp_path / "iris.csv")
    preprocess(path, tmp_path)
    model = train(tmp_path)
    monkeypatch.setenv("MODEL_PATH", model)
    with TestClient(app) as client:
        assert client.get("/health/ready").status_code == 200
        r = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
        assert r.status_code == 200 and r.json()["class"] == 0
        assert client.post("/predict", json={"features": [1, 2]}).status_code == 422
        assert client.post("/predict", json={"features": [-1, 2, 3, 4]}).status_code == 422


def test_failed_gate_removes_stale_approval(tmp_path):
    (tmp_path / "approved.json").write_text('{"old": true}')
    (tmp_path / "metrics.json").write_text(json.dumps({"metrics": {"f1": 0.2}}))
    with pytest.raises(ValueError):
        quality_gate(tmp_path)
    assert not (tmp_path / "approved.json").exists()


def test_llm_plan_is_executed_but_cannot_authorize_writes(toolset, monkeypatch):
    from agent.orchestrator import Call, Plan

    client = Mock()
    client.beta.chat.completions.parse.return_value = NS(
        choices=[
            NS(
                message=NS(
                    parsed=Plan(
                        calls=[Call(tool="kubernetes", arguments=ToolInput(operation="restart"))]
                    )
                )
            )
        ]
    )
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: client)
    answer = Agent(toolset).ask("Restart deployment", use_llm=True)
    assert answer.evidence[0].status == "denied"
    toolset["kubernetes"].execute.assert_not_called()


def test_unhealthy_pod_diagnosis_uses_returned_logs(toolset):
    def execute(a):
        if a.operation == "pods":
            return [
                {
                    "name": "iris-api-abc",
                    "phase": "Running",
                    "containers": [
                        {
                            "name": "api",
                            "ready": False,
                            "restart_count": 3,
                            "state": {"waiting": {"reason": "CrashLoopBackOff"}},
                        }
                    ],
                }
            ]
        if a.operation == "logs":
            return {"logs": "Model file not found"}
        return []

    toolset["kubernetes"].execute.side_effect = execute
    answer = Agent(toolset).ask("Why is the API unavailable?")
    assert any("CrashLoopBackOff" in a for a in answer.recommendations)
    assert any("restarts=3" in f for f in answer.facts)
    assert any(
        e.operation == "logs" and e.data["logs"] == "Model file not found" for e in answer.evidence
    )
