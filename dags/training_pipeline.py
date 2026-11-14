"""Airflow 3 Task SDK workflow; no task runs at DAG import time."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow.sdk import dag, get_current_context, task


@dag(
    dag_id="training_pipeline",
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    is_paused_upon_creation=False,
    max_active_runs=1,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(seconds=30),
        "retry_exponential_backoff": True,
        "execution_timeout": timedelta(minutes=10),
    },
    params={"min_f1": 0.85, "n_estimators": 100},
    tags=["mlops", "training"],
    doc_md="Validate → preprocess → train → evaluate/MLflow → quality gate. No deployment side effects.",
)
def training_pipeline():
    @task
    def validate() -> dict:
        import hashlib
        import os

        from training.train import validate_dataset

        root = Path(os.getenv("APP_ROOT", str(Path(__file__).resolve().parents[1])))
        context = get_current_context()
        run_key = hashlib.sha256(str(context["run_id"]).encode()).hexdigest()[:16]
        validation = validate_dataset(root / "data/iris.csv")
        validation["output_dir"] = str(root / "artifacts/runs" / run_key)
        return validation

    @task
    def preprocess(validated: dict) -> dict:
        from training.train import preprocess as prepare

        prepared = prepare(validated["data_path"], validated["output_dir"])
        return {**validated, "prepared_path": prepared}

    @task
    def train(prepared: dict) -> dict:
        from training.train import train_model

        params = get_current_context()["params"]
        candidate = train_model(
            prepared["prepared_path"], prepared["output_dir"], n_estimators=params["n_estimators"]
        )
        return {**prepared, "model_path": candidate}

    @task
    def evaluate_and_log(trained: dict) -> dict:
        from training.train import evaluate_and_log as evaluate

        return evaluate(trained["model_path"], trained["prepared_path"], trained["output_dir"])

    @task
    def quality_gate(evaluated: dict) -> dict:
        from training.train import quality_gate as gate

        return gate(
            evaluated["metrics_path"],
            evaluated["output_dir"],
            min_f1=get_current_context()["params"]["min_f1"],
        )

    quality_gate(evaluate_and_log(train(preprocess(validate()))))


training_pipeline()
