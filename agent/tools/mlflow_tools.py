"""Use the supported client; retain provenance and ranking evidence."""

import math
import os


class MLflowTools:
    operations = {"experiments", "runs", "run", "best", "compare", "registry"}

    def __init__(self, client=None):
        if client is None:
            os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "15")
            os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
            from mlflow import MlflowClient

            client = MlflowClient(
                tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            )
        self.client = client

    @staticmethod
    def record(run):
        return {
            "run_id": run.info.run_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "metrics": dict(run.data.metrics),
            "params": dict(run.data.params),
            "tags": dict(run.data.tags),
        }

    def execute(self, a):
        if a.operation == "experiments":
            return [
                {"id": x.experiment_id, "name": x.name}
                for x in self.client.search_experiments(max_results=a.limit)
            ]
        if a.operation == "run":
            result = self.record(self.client.get_run(a.run_id))
            result["artifacts"] = [x.path for x in self.client.list_artifacts(a.run_id)]
            return result
        if a.operation == "compare":
            if len(a.run_ids) < 2:
                raise ValueError("At least two run IDs are required")
            return {"runs": [self.record(self.client.get_run(x)) for x in a.run_ids]}
        if a.operation == "registry":
            return [
                {"name": m.name, "version": m.version, "run_id": m.run_id, "status": m.status}
                for m in self.client.search_model_versions(max_results=a.limit)
            ]
        experiment = self.client.get_experiment_by_name(
            os.getenv("MLFLOW_EXPERIMENT", "iris-training")
        )
        if experiment is None:
            return {"runs": [], "best": None, "reason": "Configured experiment does not exist"}
        runs = [
            self.record(r)
            for r in self.client.search_runs(
                [experiment.experiment_id],
                max_results=a.limit,
                order_by=["attributes.start_time DESC"],
            )
        ]
        if a.operation == "runs":
            return {"runs": runs}
        if a.operation == "best":
            eligible = [
                r
                for r in runs
                if r["status"] == "FINISHED"
                and isinstance(r["metrics"].get(a.metric), (int, float))
                and math.isfinite(r["metrics"][a.metric])
            ]
            ranked = sorted(
                eligible, key=lambda r: r["metrics"][a.metric], reverse=a.direction == "max"
            )
            return {
                "runs": runs,
                "best": ranked[0] if ranked else None,
                "metric": a.metric,
                "direction": a.direction,
                "scope": f"Latest {a.limit} runs in {experiment.name}",
                "caution": "Metric ranking alone does not establish production readiness.",
            }
        raise ValueError("Unsupported MLflow operation")
