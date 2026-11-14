from unittest.mock import Mock

import httpx
import pytest

from agent.tools.airflow_tools import AirflowTools


def make_tools(handler):
    return AirflowTools(
        base_url="http://airflow:8080",
        token="test-jwt",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_airflow3_authentication_and_run_order():
    def handler(request):
        assert request.url.path == "/api/v2/dags/training_pipeline/dagRuns"
        assert request.headers["authorization"] == "Bearer test-jwt"
        assert request.url.params["order_by"] == "-run_after"
        return httpx.Response(200, json={"dag_runs": [{"dag_run_id": "run-1", "state": "success"}]})

    result = make_tools(handler).list_runs()
    assert result["dag_runs"][0]["state"] == "success"


def test_missing_airflow_token_never_sends_request(monkeypatch):
    monkeypatch.delenv("AIRFLOW_TOKEN", raising=False)
    client = Mock()
    with pytest.raises(RuntimeError, match="AIRFLOW_TOKEN"):
        AirflowTools(client=client).list_dags()
    client.request.assert_not_called()


def test_trigger_requires_allowlist_and_confirmation():
    client = Mock()
    tools = AirflowTools(token="test", client=client)
    with pytest.raises(PermissionError, match="confirmation"):
        tools.trigger_dag()
    with pytest.raises(PermissionError, match="allowlist"):
        tools.trigger_dag("delete_everything", confirmed=True)
    client.request.assert_not_called()


def test_trigger_has_unique_auditable_id():
    import json

    ids = []

    def handler(request):
        assert request.method == "POST"
        payload = json.loads(request.content)
        assert payload["conf"] == {"n_estimators": 20}
        assert payload["logical_date"]
        ids.append(payload["dag_run_id"])
        return httpx.Response(200, json={"dag_run_id": payload["dag_run_id"], "state": "queued"})

    tools = make_tools(handler)
    assert tools.trigger_dag(conf={"n_estimators": 20}, confirmed=True)["state"] == "queued"
    tools.trigger_dag(conf={"n_estimators": 20}, confirmed=True)
    assert ids[0] != ids[1]


def test_failed_tasks_distinguishes_skipped_upstream():
    tools = make_tools(
        lambda request: httpx.Response(
            200,
            json={
                "task_instances": [
                    {"task_id": "validate", "state": "failed"},
                    {"task_id": "train", "state": "upstream_failed"},
                    {"task_id": "evaluate", "state": "success"},
                ]
            },
        )
    )
    result = tools.failed_tasks("training_pipeline", "run:one")
    assert [task["task_id"] for task in result["failed_tasks"]] == ["validate", "train"]


@pytest.mark.parametrize("status", [401, 403, 500, 503])
def test_service_failure_is_not_reported_as_success(status):
    tools = make_tools(lambda request: httpx.Response(status, json={"detail": "unavailable"}))
    with pytest.raises(httpx.HTTPStatusError):
        tools.list_dags()


def test_timeout_propagates_without_duplicate_trigger():
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        raise httpx.ReadTimeout("scheduler unavailable", request=request)

    with pytest.raises(httpx.ReadTimeout):
        make_tools(handler).trigger_dag(confirmed=True)
    assert count == 1


def test_invalid_id_and_limits_rejected_before_call():
    tools = AirflowTools(token="test", client=Mock())
    with pytest.raises(ValueError):
        tools.list_runs("../other")
    with pytest.raises(ValueError):
        tools.list_runs(limit=0)
    tools.client.request.assert_not_called()


def test_run_status_collects_actual_task_states():
    def handler(request):
        if request.url.path.endswith("taskInstances"):
            return httpx.Response(
                200, json={"task_instances": [{"task_id": "validate", "state": "success"}]}
            )
        return httpx.Response(200, json={"dag_run_id": "run1", "state": "running"})

    result = make_tools(handler).run_status("training_pipeline", "run1")
    assert result["run"]["state"] == "running"
    assert result["task_instances"][0]["task_id"] == "validate"


def test_logs_redact_common_secret_assignments():
    tools = make_tools(
        lambda request: httpx.Response(200, json={"content": "password=private validation failed"})
    )
    result = tools.task_logs("training_pipeline", "run1", "validate")
    assert "private" not in result["logs"]["content"]
    assert "validation failed" in result["logs"]["content"]
