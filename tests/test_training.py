import json
from pathlib import Path
from unittest.mock import Mock

import joblib
import pandas as pd
import pytest

from training.train import (
    ROOT,
    dataset_provenance,
    preprocess,
    quality_gate,
    train_model,
    validate_dataset,
)


def test_valid_iris_dataset_has_real_rows_and_features():
    result = validate_dataset(ROOT / "data/iris.csv")
    assert result["rows"] == 150
    assert len(result["features"]) == 4


def test_missing_and_invalid_data_prevent_training(tmp_path):
    frame = pd.read_csv(ROOT / "data/iris.csv")
    frame.iloc[0, 0] = float("nan")
    bad = tmp_path / "bad.csv"
    frame.to_csv(bad, index=False)
    with pytest.raises(ValueError, match="non-finite"):
        validate_dataset(bad)
    frame.drop(columns="target").to_csv(bad, index=False)
    with pytest.raises(ValueError, match="columns"):
        validate_dataset(bad)


def test_dvc_hash_matches_actual_dataset():
    result = dataset_provenance(ROOT / "data/iris.csv")
    assert result["dvc_hash"] == "4d301abed5efe50eccda350cde38e0eb"
    assert len(result["git_commit"]) == 40


def test_dirty_dataset_hash_is_rejected_before_provenance_recording(tmp_path):
    data = tmp_path / "iris.csv"
    data.write_text("altered")
    Path(str(data) + ".dvc").write_text("outs:\n- md5: original\n")
    with pytest.raises(ValueError, match="differs"):
        dataset_provenance(data)


def test_split_and_training_are_reproducible(tmp_path):
    first = preprocess(ROOT / "data/iris.csv", tmp_path / "first")
    second = preprocess(ROOT / "data/iris.csv", tmp_path / "second")
    a, b = joblib.load(first), joblib.load(second)
    pd.testing.assert_frame_equal(a["x_train"], b["x_train"])
    model = joblib.load(train_model(first, tmp_path / "first", n_estimators=5))
    assert model.n_features_in_ == 4
    assert len(model.predict(a["x_test"])) == 30


def test_failed_quality_gate_never_produces_approved_model(tmp_path, monkeypatch):
    import mlflow

    client = Mock()
    monkeypatch.setattr(mlflow, "MlflowClient", lambda **kwargs: client)
    (tmp_path / "metrics.json").write_text(json.dumps({"f1_score": 0.4}))
    (tmp_path / "candidate_metadata.json").write_text(json.dumps(
        {"run_id": "test-run", "tracking_uri": "sqlite:///not-used.db"}
    ))
    (tmp_path / "candidate.joblib").write_text("not promoted")
    with pytest.raises(ValueError, match="Quality gate failed"):
        quality_gate(tmp_path / "metrics.json", tmp_path, min_f1=0.9)
    assert not (tmp_path / "model.joblib").exists()
    assert not (tmp_path / "metadata.json").exists()
    assert json.loads((tmp_path / "quality_gate.json").read_text())["state"] == "failed"
    client.set_tag.assert_called_once_with("test-run", "quality_gate", "failed")


def test_passed_quality_gate_publishes_matching_provenance(tmp_path, monkeypatch):
    import mlflow

    monkeypatch.setattr(mlflow, "MlflowClient", lambda **kwargs: Mock())
    (tmp_path / "metrics.json").write_text(json.dumps({"f1_score": 0.95}))
    (tmp_path / "candidate_metadata.json").write_text(json.dumps(
        {"run_id": "test-run", "tracking_uri": "sqlite:///not-used.db"}
    ))
    (tmp_path / "candidate.joblib").write_bytes(b"candidate")
    result = quality_gate(tmp_path / "metrics.json", tmp_path)
    assert result["state"] == "passed"
    assert (tmp_path / "model.joblib").read_bytes() == b"candidate"
    assert json.loads((tmp_path / "metadata.json").read_text())["quality_gate"] == "passed"
