"""Export two deliberately injected failures using real local training and MLflow code.

The F1 score is deliberately injected for the gate demonstration, not measured
model degradation. Every fixture and tracking store lives in a temporary directory;
approved serving artifacts are never used as an output destination.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from agent.executor import redact

ROOT = Path(__file__).resolve().parents[1]


def _validation_failure(source: Path, workspace: Path) -> dict:
    from training.train import run_pipeline

    with source.open(newline="") as stream:
        rows = list(csv.reader(stream))
    rows[1][0] = ""  # Keep the valid Iris schema and row count; inject one missing feature.
    malformed = workspace / "missing-feature.csv"
    with malformed.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    output = workspace / "validation-artifacts"
    exception = None
    try:
        run_pipeline(data_path=malformed, output_dir=output)
    except ValueError as exc:
        exception = {"type": type(exc).__name__, "message": str(exc)}
    files = {
        name: (output / name).exists()
        for name in (
            "prepared.joblib",
            "candidate.joblib",
            "model.joblib",
            "metadata.json",
        )
    }
    expected = exception is not None and "missing or non-finite" in exception["message"]
    return {
        "name": "missing_dataset_feature",
        "injected_change": "One numeric feature in an isolated copy of Iris was replaced with an empty cell.",
        "expected_exception": "ValueError",
        "observed_exception": exception,
        "expected_failure_observed": expected,
        "evidence": {
            "input_rows": len(rows) - 1,
            "input_columns": rows[0],
            "missing_feature_cells_in_first_row": sum(item == "" for item in rows[1][:-1]),
            "output_directory_created": output.exists(),
            "downstream_files_exist": files,
            "downstream_training_blocked": expected and not any(files.values()),
        },
        "diagnosis": {
            "root_cause": "A required numeric feature is missing, so finite-value validation rejects the dataset.",
            "resolution": "Repair the source data, validate it, update its DVC pointer and commit provenance before retraining.",
            "observed_containment": "No prepared dataset, trained candidate, or approved serving artifact was created.",
        },
    }


def _quality_gate_failure(workspace: Path) -> dict:
    # Local SQLite and a temporary artifact URI exercise the real supported SDK.
    from mlflow import MlflowClient

    from training.train import quality_gate

    database = workspace / "tracking.db"
    tracking_uri = f"sqlite:///{database}"
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.create_experiment(
        f"Injected-gate-failure-{uuid4().hex[:12]}",
        artifact_location=(workspace / "mlflow-artifacts").as_uri(),
    )
    run = client.create_run(
        experiment,
        tags={
            "purpose": "deliberately_injected_failure",
            "quality_gate": "pending",
            "metric_source": "injected_for_gate_demonstration",
        },
    )
    run_id = run.info.run_id
    client.log_metric(run_id, "f1_score", 0.4)
    client.set_terminated(run_id, status="FINISHED")
    output = workspace / "gate-artifacts"
    output.mkdir()
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps({"f1_score": 0.4}))
    (output / "candidate_metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "tracking_uri": tracking_uri,
                "metric_source": "injected_for_gate_demonstration",
            }
        )
    )
    # The rejecting gate never reads candidate bytes. This is explicitly a sentinel,
    # not a claim that an evaluated model achieved the deliberately injected score.
    candidate = output / "candidate.joblib"
    candidate.write_bytes(b"failure-demonstration sentinel; must never be promoted")
    before_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
    exception = None
    try:
        quality_gate(metrics_path, output, min_f1=0.85)
    except ValueError as exc:
        exception = {"type": type(exc).__name__, "message": str(exc)}
    recorded = client.get_run(run_id)
    gate_file = json.loads((output / "quality_gate.json").read_text())
    artifacts = [artifact.path for artifact in client.list_artifacts(run_id)]
    expected = exception is not None and "Quality gate failed" in exception["message"]
    return {
        "name": "below_threshold_quality_gate",
        "injected_change": "F1=0.4 was deliberately recorded in a real isolated MLflow run; the configured gate is 0.85.",
        "expected_exception": "ValueError",
        "observed_exception": exception,
        "expected_failure_observed": expected,
        "evidence": {
            "tracking_backend": "real local MLflow SQLite",
            "database_created": database.is_file(),
            "run_id": recorded.info.run_id,
            "run_status": recorded.info.status,
            "recorded_metrics": dict(recorded.data.metrics),
            "recorded_tags": dict(recorded.data.tags),
            "recorded_artifacts": artifacts,
            "quality_gate_file": gate_file,
            "candidate_kind": "unloaded sentinel bytes; injected metric is not a measured model result",
            "candidate_bytes_unchanged": hashlib.sha256(candidate.read_bytes()).hexdigest()
            == before_hash,
            "approved_model_exists": (output / "model.joblib").exists(),
            "approved_metadata_exists": (output / "metadata.json").exists(),
        },
        "diagnosis": {
            "root_cause": "The deliberately supplied F1 score 0.4 is lower than the explicit acceptance threshold 0.85.",
            "resolution": "Inspect real evaluation and data evidence, then improve/retrain a candidate and rerun the gate. Revise policy only with an explicit justified decision.",
            "observed_containment": "The MLflow gate tag is failed and no serving model or approved metadata was published.",
        },
    }


def export_failures(
    output: str | Path = ROOT / "evidence/failures.json",
    source: str | Path = ROOT / "data/iris.csv",
) -> dict:
    source = Path(source).resolve()
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="mlops-validation-failure-") as validation_directory:
        validation = _validation_failure(source, Path(validation_directory))
    with tempfile.TemporaryDirectory(prefix="mlops-gate-failure-") as gate_directory:
        gate = _quality_gate_failure(Path(gate_directory))
    report = redact(
        {
            "schema_version": 1,
            "kind": "live-observation",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "scope": "Two deliberately injected local failures executed with real validation, quality-gate, and MLflow SDK code; no external deployment state is inferred.",
            "source_dataset_unchanged": hashlib.sha256(source.read_bytes()).hexdigest()
            == original_hash,
            "temporary_workspaces_removed": not Path(validation_directory).exists()
            and not Path(gate_directory).exists(),
            "tracking_store_retained": False,
            "all_expected_failures_observed": validation["expected_failure_observed"]
            and gate["expected_failure_observed"],
            "investigations": [validation, gate],
        }
    )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/failures.json")
    args = parser.parse_args()
    report = export_failures(args.output)
    print(
        json.dumps(
            {
                "evidence": str(args.output),
                "kind": report["kind"],
                "all_expected_failures_observed": report["all_expected_failures_observed"],
                "source_dataset_unchanged": report["source_dataset_unchanged"],
            }
        )
    )


if __name__ == "__main__":
    main()
