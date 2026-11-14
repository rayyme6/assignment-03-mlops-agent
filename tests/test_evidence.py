import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from agent.schemas import AgentResponse, ToolResult
from scripts.collect_evidence import collect, reproduce_dataset, sanitize


class ObservedAgent:
    """Injected service evidence for collector tests; never used by the live CLI."""

    def __init__(self, pipeline_available=False, states=None):
        self.requests = []
        self.pipeline_available = pipeline_available
        self.states = iter(states or ["success"])

    def chat(self, request):
        self.requests.append(request)
        message = request.message
        tool, data, status = "dvc_dataset_version", {"dvc_hash": "actual-test-pointer"}, "ok"
        if "Trigger" in message:
            tool, data = (
                "airflow_trigger_dag",
                {"dag_run_id": "manual__fixture", "dag_id": "training_pipeline"},
            )
            if not self.pipeline_available:
                status, data = "unavailable", None
        elif "tool airflow_run_status" in message:
            tool, data = (
                "airflow_run_status",
                {
                    "run": {"state": next(self.states)},
                    "task_instances": [{"task_id": "validate", "state": "success"}],
                },
            )
        elif "tool airflow_failed_tasks" in message:
            tool, data = (
                "airflow_failed_tasks",
                {"failed_tasks": [{"task_id": "quality_gate", "state": "failed"}]},
            )
        elif "workflow status" in message:
            tool = "airflow_list_runs"
            data = {"dag_runs": [{"dag_run_id": "recent__fixture", "dag_id": "training_pipeline"}]}
            if not self.pipeline_available:
                status, data = "unavailable", None
        elif "unavailable" in message:
            tool, data, status = "kubernetes_deployment_health", None, "unavailable"
        elif "best" in message:
            tool, data = (
                "mlflow_best_run",
                {"best_run": {"run_id": "fixture", "metrics": {"f1_score": 0.9}}},
            )
        result = ToolResult(
            tool=tool,
            status=status,
            data=data,
            duration_ms=1.0,
            request_id="fixture-request",
            error="Observed service unavailable" if status != "ok" else None,
        )
        return AgentResponse(request_id="fixture-request", planner="rules", evidence=[result])


def test_collector_exports_five_conversations_and_keeps_service_errors(tmp_path):
    agent = ObservedAgent()
    path, report = collect(tmp_path, agent=agent)
    assert path.exists()
    assert json.loads(path.read_text()) == report
    assert report["kind"] == "live-observation"
    assert len(report["conversations"]) == 5
    services = report["environment"]["integrations"]
    assert services["dvc"]["available"] is True
    assert services["mlflow"]["available"] is True
    assert services["airflow"]["available"] is False
    assert services["kubernetes"]["available"] is False
    assert report["conversations"][2]["response"]["evidence"][0]["data"] is None
    assert all(request.confirmed_tools == [] for request in agent.requests)
    assert not any("Trigger" in request.message for request in agent.requests)


def test_explicit_trigger_polls_real_run_and_captures_task_statuses(tmp_path, monkeypatch):
    agent = ObservedAgent(pipeline_available=True, states=["running", "success"])
    monkeypatch.setattr("scripts.collect_evidence.time.sleep", lambda _: None)
    _, report = collect(tmp_path, agent=agent, trigger=True, max_wait=5)
    trigger = [request for request in agent.requests if "Trigger" in request.message]
    assert len(trigger) == 1
    assert trigger[0].confirmed_tools == ["airflow_trigger_dag"]
    assert len(report["pipeline_follow_up"]) == 2
    final = report["pipeline_follow_up"][-1]["response"]["evidence"][0]
    assert final["data"]["run"]["state"] == "success"
    assert final["data"]["task_instances"][0]["task_id"] == "validate"


def test_read_only_pipeline_collection_inspects_one_existing_run(tmp_path):
    agent = ObservedAgent(pipeline_available=True, states=["running"])
    _, report = collect(tmp_path, agent=agent)
    assert len(report["pipeline_follow_up"]) == 1
    assert report["trigger_authorized"] is False
    assert all(not request.confirmed_tools for request in agent.requests)


def test_failed_pipeline_collects_failed_tasks_without_retrying_training(tmp_path):
    agent = ObservedAgent(pipeline_available=True, states=["failed"])
    _, report = collect(tmp_path, agent=agent, trigger=True, max_wait=0)
    assert len(report["pipeline_follow_up"]) == 2
    assert report["pipeline_follow_up"][-1]["scenario"] == "pipeline_failure_diagnostics"
    assert sum("Trigger" in request.message for request in agent.requests) == 1


def test_failed_trigger_is_not_retried_and_does_not_invent_run(tmp_path):
    agent = ObservedAgent(pipeline_available=False)
    _, report = collect(tmp_path, agent=agent, trigger=True)
    assert report["pipeline_follow_up"] == []
    assert sum("Trigger" in request.message for request in agent.requests) == 1


def test_pending_run_at_deadline_is_exported_as_pending(tmp_path):
    agent = ObservedAgent(pipeline_available=True, states=["queued"])
    _, report = collect(tmp_path, agent=agent, trigger=True, max_wait=0)
    last = report["pipeline_follow_up"][-1]["response"]["evidence"][0]
    assert last["data"]["run"]["state"] == "queued"


def test_collector_sanitizes_nested_keys_bearers_and_environment_secret(monkeypatch):
    monkeypatch.setenv("TEST_API_KEY", "hidden-value-from-environment")
    value = {
        "api_key": "abc",
        "logs": ["Bearer credentialvalue", "echo hidden-value-from-environment"],
    }
    result = sanitize(value)
    serialized = json.dumps(result)
    assert "credentialvalue" not in serialized
    assert "hidden-value-from-environment" not in serialized
    assert result["api_key"] == "[REDACTED]"


@pytest.mark.parametrize("wait,poll", [(121, 1), (-1, 1), (2, 0), (2, 11)])
def test_collector_rejects_unbounded_polling(tmp_path, wait, poll):
    with pytest.raises(ValueError, match="max_wait"):
        collect(tmp_path, agent=ObservedAgent(), max_wait=wait, poll_interval=poll)
    assert list(tmp_path.iterdir()) == []


def test_isolated_dvc_reproduction_verifies_bytes_and_removes_copy(tmp_path, monkeypatch):
    content = b"feature,target\n1,0\n"
    checksum = hashlib.md5(content).hexdigest()
    monkeypatch.setattr(
        "scripts.collect_evidence.DVCTools.dataset_version",
        lambda self, rev: {
            "git_commit": "a" * 40,
            "dataset_path": "data/iris.csv",
            "dvc_hash": checksum,
            "hash_algorithm": "md5",
        },
    )
    destinations = []

    def restore(argv, **kwargs):
        assert argv[:2] == ["dvc", "get"]
        assert argv[argv.index("--rev") + 1] == "a" * 40
        destination = Path(argv[argv.index("--out") + 1])
        assert destination.is_relative_to(Path(kwargs["cwd"]))
        assert kwargs["timeout"] == 45
        destinations.append(destination)
        destination.write_bytes(content)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", restore)
    result = reproduce_dataset(tmp_path)
    assert result["status"] == "ok"
    assert result["restored_md5"] == checksum
    assert result["original_checkout_modified"] is False
    assert result["temporary_copy_removed"] is True
    assert all(not path.exists() for path in destinations)


def test_reproduction_failure_preserves_uncertainty_and_hides_error_details(tmp_path, monkeypatch):
    def failure(*args, **kwargs):
        raise ConnectionError("secret credential in arbitrary upstream error")

    monkeypatch.setattr("scripts.collect_evidence.DVCTools.dataset_version", failure)
    result = reproduce_dataset(tmp_path)
    assert result["status"] == "unavailable"
    assert "secret credential" not in json.dumps(result)
    assert "hash_matches" not in result
