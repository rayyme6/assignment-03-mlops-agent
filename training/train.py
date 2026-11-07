"""Reproducible stages shared by DVC, Airflow and the local integration test."""

import argparse
import hashlib
import json
import logging
import os
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger(__name__)
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def generate(path="data/iris.csv"):
    iris = load_iris()
    frame = pd.DataFrame(iris.data, columns=FEATURES)
    frame["target"] = iris.target
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def validate(data="data/iris.csv"):
    frame = pd.read_csv(data)
    if list(frame.columns) != FEATURES + ["target"]:
        raise ValueError("Dataset schema mismatch")
    if len(frame) < 30 or frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Dataset has insufficient rows, nulls or non-finite values")
    if set(frame.target.unique()) != {0, 1, 2} or frame.target.value_counts().min() < 5:
        raise ValueError("Expected three sufficiently populated Iris classes")
    if (frame[FEATURES] <= 0).any().any():
        raise ValueError("Measurements must be positive")
    LOG.info("Validated %s rows", len(frame))
    return {"rows": len(frame), "sha256": hashlib.sha256(Path(data).read_bytes()).hexdigest()}


def preprocess(data="data/iris.csv", out="artifacts"):
    validate(data)
    frame = pd.read_csv(data)
    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURES], frame.target, test_size=0.25, stratify=frame.target, random_state=42
    )
    Path(out).mkdir(parents=True, exist_ok=True)
    joblib.dump((x_train, x_test, y_train, y_test), Path(out) / "split.joblib")
    return str(out)


def train(out="artifacts", c=1.0):
    x_train, _, y_train, _ = joblib.load(Path(out) / "split.joblib")
    # Fit preprocessing only on the training split; no test leakage.
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=500, random_state=42))
    model.fit(x_train, y_train)
    joblib.dump(model, Path(out) / "model.joblib")
    return str(Path(out) / "model.joblib")


def provenance(data="data/iris.csv"):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    meta = yaml.safe_load(Path(data + ".dvc").read_text())
    committed = yaml.safe_load(
        subprocess.check_output(["git", "show", f"HEAD:{data}.dvc"], text=True)
    )
    actual = hashlib.md5(Path(data).read_bytes()).hexdigest()
    if meta != committed or actual != meta["outs"][0]["md5"]:
        raise ValueError("Dataset differs from committed DVC metadata; dvc add and commit first")
    dirty = subprocess.check_output(
        [
            "git",
            "diff",
            "--name-only",
            "HEAD",
            "--",
            "agent",
            "api",
            "training",
            "dags",
            "pyproject.toml",
            "dvc.yaml",
        ],
        text=True,
    ).strip()
    return {
        "dataset.git_commit": commit,
        "dataset.dvc_md5": actual,
        "dataset.path": data,
        "code.dirty": str(bool(dirty)).lower(),
    }


def evaluate(out="artifacts", data="data/iris.csv", threshold=0.9):
    import mlflow

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "iris-training"))
    _, x_test, _, y_test = joblib.load(Path(out) / "split.joblib")
    model = joblib.load(Path(out) / "model.joblib")
    predicted = model.predict(x_test)
    metrics = {
        "f1": float(f1_score(y_test, predicted, average="macro")),
        "accuracy": float(accuracy_score(y_test, predicted)),
    }
    tags = provenance(data)
    tags["quality_gate"] = "passed" if metrics["f1"] >= threshold else "failed"
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "C": model[-1].C,
                "random_state": 42,
                "test_size": 0.25,
                "quality_threshold": threshold,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags(tags)
        mlflow.log_artifact(str(Path(out) / "model.joblib"), artifact_path="model")
        result = {
            "run_id": run.info.run_id,
            "metrics": metrics,
            "tags": tags,
            "model_sha256": hashlib.sha256(Path(out, "model.joblib").read_bytes()).hexdigest(),
        }
        Path(out, "metrics.json").write_text(json.dumps(result, indent=2))
        mlflow.log_artifact(str(Path(out) / "metrics.json"))
    return result


def quality_gate(out="artifacts", threshold=0.9):
    result = json.loads(Path(out, "metrics.json").read_text())
    if result["metrics"]["f1"] < threshold:
        Path(out, "approved.json").unlink(missing_ok=True)
        raise ValueError(f"Quality gate failed: F1 below {threshold}; promotion blocked")
    Path(out, "approved.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["generate", "validate", "preprocess", "train", "evaluate", "gate", "all"]
    )
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--c", type=float, default=1.0)
    a = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if a.stage == "generate":
        generate()
    if a.stage in {"validate", "all"}:
        validate()
    if a.stage in {"preprocess", "all"}:
        preprocess(out=a.out)
    if a.stage in {"train", "all"}:
        train(a.out, a.c)
    if a.stage in {"evaluate", "all"}:
        evaluate(a.out, threshold=a.threshold)
    if a.stage in {"gate", "all"}:
        quality_gate(a.out, a.threshold)


if __name__ == "__main__":
    main()
