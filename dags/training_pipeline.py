"""Airflow 3 TaskFlow DAG; isolated training interpreter, per-run artifact paths."""

import hashlib
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow.sdk import dag, get_current_context, task


def run_stage(stage, out):
    root = Path(os.environ["MLOPS_ROOT"]).resolve()
    python = os.environ.get("MLOPS_PYTHON", str(root / ".venv/bin/python"))
    # Administrator-configured executable and fixed stages; no user-controlled shell.
    subprocess.run(
        [python, "-m", "training.train", stage, "--out", out], cwd=root, check=True, timeout=300
    )
    return out


@dag(
    dag_id="iris_training",
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(seconds=15)},
)
def training_pipeline():
    @task
    def validation():
        run_id = get_current_context()["run_id"]
        out = "artifacts/" + hashlib.sha256(run_id.encode()).hexdigest()[:24]
        return run_stage("validate", out)

    @task
    def preprocessing(out):
        return run_stage("preprocess", out)

    @task
    def training(out):
        return run_stage("train", out)

    @task
    def evaluation(out):
        return run_stage("evaluate", out)

    @task
    def quality_gate(out):
        return run_stage("gate", out)

    quality_gate(evaluation(training(preprocessing(validation()))))


training_pipeline()
