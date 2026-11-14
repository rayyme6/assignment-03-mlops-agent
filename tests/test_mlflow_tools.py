from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.tools.mlflow_tools import MLflowTools


def run(run_id, score=0.9, status="FINISHED", metric="f1_score"):
    return SimpleNamespace(
        info=SimpleNamespace(
            run_id=run_id, experiment_id="1", status=status, start_time=100, end_time=200
        ),
        data=SimpleNamespace(
            metrics={metric: score},
            params={"n_estimators": "100"},
            tags={
                "git_commit": "abc123",
                "dvc_hash": "recorded_hash",
                "dataset_path": "data/iris.csv",
            },
        ),
    )


@pytest.fixture
def client():
    client = Mock()
    client.search_experiments.return_value = [
        SimpleNamespace(experiment_id="1", name="Iris", lifecycle_stage="active")
    ]
    client.list_artifacts.return_value = [
        SimpleNamespace(path="model", is_dir=True, file_size=None)
    ]
    return client


def test_best_recent_run_excludes_failed_missing_and_nonfinite_metrics(client):
    client.search_runs.return_value = [
        run("winner", 0.93),
        run("low", 0.8),
        run("failed", 1.0, "FAILED"),
        run("missing", 1.0, metric="accuracy"),
        run("nan", float("nan")),
    ]
    result = MLflowTools(client=client).best_run()
    assert result["best_run"]["run_id"] == "winner"
    assert result["best_run"]["provenance"]["dvc_hash"] == "recorded_hash"
    assert result["excluded_runs"] == 3
    assert "production readiness" in result["caveat"]
    assert client.search_runs.call_args.kwargs["order_by"] == ["attributes.start_time DESC"]


def test_no_finished_candidate_returns_no_fabricated_winner(client):
    client.search_runs.return_value = [run("failed", 1.0, "FAILED")]
    result = MLflowTools(client=client).best_run()
    assert result["best_run"] is None
    assert result["candidates"] == []


def test_minimize_metric_selection(client):
    client.search_runs.return_value = [
        run("high", 2.0, metric="loss"),
        run("low", 0.2, metric="loss"),
    ]
    result = MLflowTools(client=client).best_run(metric="loss", maximize=False)
    assert result["best_run"]["run_id"] == "low"


def test_run_details_include_metrics_parameters_and_artifact_evidence(client):
    client.get_run.return_value = run("run-123")
    details = MLflowTools(client=client).run_details("run-123")
    assert details["metrics"]["f1_score"] == 0.9
    assert details["params"]["n_estimators"] == "100"
    assert details["artifacts"][0]["path"] == "model"
    client.get_run.assert_called_once_with("run-123")


def test_compare_runs_preserves_separate_experiment_evidence(client):
    client.get_run.side_effect = [run("a", 0.8), run("b", 0.9)]
    comparison = MLflowTools(client=client).compare_runs(["a", "b"])
    assert [item["run_id"] for item in comparison["runs"]] == ["a", "b"]
    assert comparison["metrics"] == ["f1_score"]


def test_service_failure_propagates_without_synthesizing_results(client):
    client.search_experiments.side_effect = ConnectionError("tracking unavailable")
    with pytest.raises(ConnectionError, match="unavailable"):
        MLflowTools(client=client).best_run()


@pytest.mark.parametrize("limit", [0, -1, 101, True, "5"])
def test_invalid_run_limits_do_not_call_service(client, limit):
    with pytest.raises(ValueError, match="limit"):
        MLflowTools(client=client).list_runs(limit=limit)
    client.search_experiments.assert_not_called()


def test_registry_uses_supported_api_and_returns_actual_versions(client):
    client.search_model_versions.return_value = [
        SimpleNamespace(
            name="iris",
            version="2",
            run_id="run-2",
            status="READY",
            current_stage="None",
            aliases=["candidate"],
        )
    ]
    result = MLflowTools(client=client).model_versions("iris")
    assert result["versions"][0]["run_id"] == "run-2"
    assert result["versions"][0]["aliases"] == ["candidate"]
    client.search_model_versions.assert_called_once_with(
        max_results=100, filter_string="name = 'iris'"
    )


def test_registry_filter_rejects_injection(client):
    with pytest.raises(ValueError, match="model name"):
        MLflowTools(client=client).model_versions("iris' OR name LIKE '%")
    client.search_model_versions.assert_not_called()


def test_empty_experiment_store_does_not_query_invalid_scope(client):
    client.search_experiments.return_value = []
    assert MLflowTools(client=client).list_runs()["runs"] == []
    client.search_runs.assert_not_called()
