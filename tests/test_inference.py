import json

import joblib
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from api.main import create_app


def test_missing_model_keeps_liveness_but_fails_readiness(tmp_path):
    with TestClient(create_app(tmp_path / "missing.joblib")) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert client.post("/predict", json={"instances": [[5, 3, 1, 0.2]]}).status_code == 503


def test_unapproved_model_is_not_served(tmp_path):
    (tmp_path / "metadata.json").write_text('{"quality_gate": "failed"}')
    with TestClient(create_app(tmp_path / "model.joblib")) as client:
        assert client.get("/health/ready").status_code == 503


def test_predictions_use_real_model_and_return_run_version(tmp_path):
    iris = load_iris(as_frame=True)
    model = RandomForestClassifier(n_estimators=5, random_state=42).fit(iris.data, iris.target)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "quality_gate": "passed",
                "run_id": "real-local-model",
                "model_version": "real-local-model",
                "features": list(iris.data.columns),
            }
        )
    )
    with TestClient(create_app(path)) as client:
        assert client.get("/health/ready").status_code == 200
        result = client.post("/predict", json={"instances": [[5.1, 3.5, 1.4, 0.2]]})
        assert result.status_code == 200
        assert result.json()["predictions"] == [0]
        assert result.json()["model_version"] == "real-local-model"
        assert client.get("/version").json()["run_id"] == "real-local-model"
        for invalid in [[], [[1, 2]], [[-1, 2, 3, 4]], [[1, 2, 3, "NaN"]]]:
            assert client.post("/predict", json={"instances": invalid}).status_code == 422
