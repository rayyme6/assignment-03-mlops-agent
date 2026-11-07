"""Airflow 3 TaskFlow DAG. Shared run-specific artifacts for the local executor demo."""
from datetime import datetime, timedelta, timezone

from airflow.sdk import dag, get_current_context, task


@dag(dag_id="iris_training", schedule=None, start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
     catchup=False, max_active_runs=1,
     default_args={"retries": 2, "retry_delay": timedelta(seconds=15)})
def training_pipeline():
    @task
    def validation():
        from training.train import validate
        return validate()

    @task
    def preprocessing(_validation):
        import hashlib
        from training.train import preprocess
        run_id = get_current_context()["run_id"]
        # The scheduler-generated run ID cannot escape the artifact directory.
        out = "artifacts/" + hashlib.sha256(run_id.encode()).hexdigest()[:24]
        return preprocess(out=out)

    @task
    def training(out):
        from training.train import train
        train(out)
        return out

    @task
    def evaluation(out):
        from training.train import evaluate
        evaluate(out)
        return out

    @task
    def quality_gate(out):
        from training.train import quality_gate as check
        return check(out)

    quality_gate(evaluation(training(preprocessing(validation()))))


training_pipeline()
