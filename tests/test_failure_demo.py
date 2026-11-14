import json

from scripts.inject_failures import ROOT, _validation_failure, export_failures


def test_actual_missing_value_failure_stops_before_training(tmp_path):
    report = _validation_failure(ROOT / "data/iris.csv", tmp_path)
    assert report["expected_failure_observed"] is True
    assert report["observed_exception"]["type"] == "ValueError"
    assert report["evidence"]["missing_feature_cells_in_first_row"] == 1
    assert report["evidence"]["downstream_training_blocked"] is True
    assert not report["evidence"]["output_directory_created"]


def test_actual_failure_export_uses_local_mlflow_and_prevents_promotion(tmp_path):
    path = tmp_path / "evidence/failures.json"
    report = export_failures(path)
    assert json.loads(path.read_text()) == report
    assert report["kind"] == "live-observation"
    assert report["all_expected_failures_observed"] is True
    assert report["source_dataset_unchanged"] is True
    assert report["temporary_workspaces_removed"] is True
    assert report["tracking_store_retained"] is False
    gate = report["investigations"][1]
    assert gate["observed_exception"]["type"] == "ValueError"
    evidence = gate["evidence"]
    assert evidence["database_created"] is True
    assert evidence["recorded_metrics"]["f1_score"] == 0.4
    assert evidence["recorded_metrics"]["min_f1"] == 0.85
    assert evidence["recorded_tags"]["quality_gate"] == "failed"
    assert "quality_gate.json" in evidence["recorded_artifacts"]
    assert evidence["quality_gate_file"]["state"] == "failed"
    assert evidence["candidate_bytes_unchanged"] is True
    assert evidence["approved_model_exists"] is False
    assert evidence["approved_metadata_exists"] is False
