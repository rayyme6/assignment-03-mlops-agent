"""Reproducible Iris training stages shared by the CLI, DVC and Airflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
LOG = logging.getLogger(__name__)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_dataset(data_path: str | Path) -> dict:
    """Fail before training when the schema or numeric/label values are invalid."""
    path = Path(data_path).resolve()
    frame = pd.read_csv(path)
    if list(frame.columns) != FEATURES + ["target"]:
        raise ValueError("Dataset columns must be the four Iris features followed by target")
    if len(frame) < 30 or not all(pd.api.types.is_numeric_dtype(t) for t in frame.dtypes):
        raise ValueError("Dataset needs at least 30 rows of numeric values")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Dataset contains missing or non-finite values")
    if not frame["target"].isin([0, 1, 2]).all() or frame["target"].nunique() != 3:
        raise ValueError("Dataset must contain all three Iris classes: 0, 1, 2")
    if frame["target"].value_counts().min() < 5:
        raise ValueError("Each class needs at least five rows for a stratified split")
    LOG.info("Dataset validated: rows=%s", len(frame))
    return {"data_path": str(path), "rows": len(frame), "features": FEATURES}


def dataset_provenance(data_path: str | Path) -> dict:
    path = Path(data_path).resolve()
    metadata = yaml.safe_load(path.with_suffix(path.suffix + ".dvc").read_text())
    recorded = metadata["outs"][0]["md5"]
    actual = hashlib.md5(path.read_bytes()).hexdigest()  # DVC content ID, not a security hash.
    if actual != recorded:
        raise ValueError(
            "Dataset content differs from its DVC pointer; run dvc add and commit first"
        )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            text=True,
            timeout=10,
        ).strip()
    )
    return {
        "git_commit": commit,
        "dvc_hash": actual,
        "dataset_path": str(path.relative_to(ROOT)),
        "git_dirty": dirty,
    }


def preprocess(
    data_path: str | Path, output_dir: str | Path, test_size: float = 0.2, random_state: int = 42
) -> str:
    validate_dataset(data_path)
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size must lie between 0.1 and 0.5")
    frame = pd.read_csv(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURES],
        frame["target"],
        test_size=test_size,
        random_state=random_state,
        stratify=frame["target"],
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prepared = output / "prepared.joblib"
    joblib.dump(
        {
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
            "test_size": test_size,
            "random_state": random_state,
            "provenance": dataset_provenance(data_path),
        },
        prepared,
    )
    LOG.info("Prepared deterministic stratified train/test split")
    return str(prepared)


def train_model(prepared_path: str | Path, output_dir: str | Path, n_estimators: int = 100) -> str:
    if not isinstance(n_estimators, int) or not 1 <= n_estimators <= 1000:
        raise ValueError("n_estimators must be an integer between 1 and 1000")
    prepared = joblib.load(prepared_path)
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=prepared["random_state"], n_jobs=1
    )
    model.fit(prepared["x_train"], prepared["y_train"])
    target = Path(output_dir) / "candidate.joblib"
    joblib.dump(model, target)
    LOG.info("Candidate trained: trees=%s", n_estimators)
    return str(target)


def evaluate_and_log(
    model_path: str | Path,
    prepared_path: str | Path,
    output_dir: str | Path,
    tracking_uri: str | None = None,
) -> dict:
    import mlflow

    output = Path(output_dir)
    prepared = joblib.load(prepared_path)
    model = joblib.load(model_path)
    predictions = model.predict(prepared["x_test"])
    metrics = {
        "accuracy": float(accuracy_score(prepared["y_test"], predictions)),
        "f1_score": float(f1_score(prepared["y_test"], predictions, average="macro")),
    }
    tracking = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_tracking_uri(tracking)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "Iris-Classification"))
    with mlflow.start_run(run_name=f"RF-{model.n_estimators}") as run:
        mlflow.log_params(
            {
                "n_estimators": model.n_estimators,
                "random_state": prepared["random_state"],
                "test_size": prepared["test_size"],
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags({**prepared["provenance"], "quality_gate": "pending"})
        metadata = {
            **prepared["provenance"],
            "run_id": run.info.run_id,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "model_version": run.info.run_id,
            "features": FEATURES,
            "quality_gate": "pending",
            "tracking_uri": tracking,
        }
        _write_json(output / "metrics.json", metrics)
        _write_json(output / "candidate_metadata.json", metadata)
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(output / "candidate_metadata.json"), artifact_path="provenance")
    LOG.info(
        "MLflow evaluation recorded: run_id=%s f1_score=%.4f",
        metadata["run_id"],
        metrics["f1_score"],
    )
    return {
        "metrics_path": str(output / "metrics.json"),
        "output_dir": str(output),
        "run_id": metadata["run_id"],
        "metrics": metrics,
    }


def quality_gate(metrics_path: str | Path, output_dir: str | Path, min_f1: float = 0.85) -> dict:
    import mlflow

    if not np.isfinite(min_f1) or not 0 <= min_f1 <= 1:
        raise ValueError("min_f1 must lie between 0 and 1")
    output = Path(output_dir)
    metrics = json.loads(Path(metrics_path).read_text())
    metadata = json.loads((output / "candidate_metadata.json").read_text())
    passed = bool(np.isfinite(metrics["f1_score"]) and metrics["f1_score"] >= min_f1)
    state = "passed" if passed else "failed"
    metadata.update(quality_gate=state, min_f1=min_f1)
    _write_json(
        output / "quality_gate.json",
        {"state": state, "min_f1": min_f1, "f1_score": metrics["f1_score"]},
    )
    client = mlflow.MlflowClient(tracking_uri=metadata["tracking_uri"])
    client.set_tag(metadata["run_id"], "quality_gate", state)
    client.log_metric(metadata["run_id"], "min_f1", min_f1)
    client.log_artifact(metadata["run_id"], str(output / "quality_gate.json"))
    if not passed:
        LOG.error("Quality gate failed: f1_score=%s min_f1=%s", metrics["f1_score"], min_f1)
        raise ValueError(f"Quality gate failed: F1 {metrics['f1_score']:.4f} < {min_f1:.4f}")
    # Only the approved artifact is available at the serving path.
    shutil.copyfile(output / "candidate.joblib", output / "model.joblib")
    _write_json(output / "metadata.json", metadata)
    LOG.info("Quality gate passed; candidate approved locally, deployment remains explicit")
    return {
        "state": state,
        "run_id": metadata["run_id"],
        "metrics": metrics,
        "model_path": str(output / "model.joblib"),
    }


def run_pipeline(
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    min_f1: float | None = None,
    n_estimators: int | None = None,
    tracking_uri: str | None = None,
) -> dict:
    config = yaml.safe_load((ROOT / "params.yaml").read_text())["training"]
    data = Path(data_path) if data_path else ROOT / "data/iris.csv"
    output = Path(output_dir) if output_dir else ROOT / "artifacts"
    validate_dataset(data)
    prepared = preprocess(data, output, config["test_size"], config["random_state"])
    candidate = train_model(
        prepared, output, config["n_estimators"] if n_estimators is None else n_estimators
    )
    evaluation = evaluate_and_log(candidate, prepared, output, tracking_uri)
    return quality_gate(
        evaluation["metrics_path"], output, config["min_f1"] if min_f1 is None else min_f1
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-f1", type=float)
    parser.add_argument("--n-estimators", type=int)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    print(
        json.dumps(
            run_pipeline(args.data, args.output_dir, args.min_f1, args.n_estimators), indent=2
        )
    )


if __name__ == "__main__":
    main()
