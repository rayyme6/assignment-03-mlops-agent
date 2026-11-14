"""Evidence-preserving experiment queries through the supported MLflow client."""

from __future__ import annotations

import math
import os
import re
from typing import Any


class MLflowTools:
    CAVEAT = (
        "The selected run is a candidate based on recorded evaluation metrics. "
        "A high score alone does not establish production readiness; inspect quality gates, "
        "data provenance, deployment health, and any supplied drift/freshness policy."
    )

    def __init__(self, tracking_uri: str | None = None, client: Any = None):
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"
        )
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from mlflow.tracking import MlflowClient

            self._client = MlflowClient(tracking_uri=self.tracking_uri)
        return self._client

    @staticmethod
    def _limit(limit: int) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        return limit

    @staticmethod
    def _identifier(value: str, label: str) -> str:
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value):
            raise ValueError(f"Invalid {label}")
        return value

    @staticmethod
    def _run_dict(run: Any) -> dict[str, Any]:
        tags = dict(run.data.tags)
        return {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "metrics": {
                key: value if math.isfinite(value) else None
                for key, value in run.data.metrics.items()
            },
            "params": dict(run.data.params),
            "tags": tags,
            "provenance": {
                key: tags.get(key)
                for key in ("git_commit", "dvc_hash", "dataset_path", "dvc_file", "quality_gate")
            },
        }

    def list_experiments(self) -> dict[str, Any]:
        experiments = []
        page_token = None
        while True:
            kwargs = {"max_results": 100}
            if page_token:
                kwargs["page_token"] = page_token
            page = self.client.search_experiments(**kwargs)
            experiments.extend(
                {
                    "experiment_id": item.experiment_id,
                    "name": item.name,
                    "lifecycle_stage": item.lifecycle_stage,
                }
                for item in page
            )
            page_token = getattr(page, "token", None)
            if not page_token:
                break
        return {"experiments": experiments}

    def _experiment_ids(self, experiment_ids: list[str] | None) -> list[str]:
        if experiment_ids is None:
            return [item["experiment_id"] for item in self.list_experiments()["experiments"]]
        if not isinstance(experiment_ids, list) or len(experiment_ids) > 100:
            raise ValueError("experiment_ids must be a list of at most 100 IDs")
        return [self._identifier(item, "experiment ID") for item in experiment_ids]

    def list_runs(self, experiment_ids: list[str] | None = None, limit: int = 5) -> dict[str, Any]:
        limit = self._limit(limit)
        ids = self._experiment_ids(experiment_ids)
        runs = (
            self.client.search_runs(
                experiment_ids=ids,
                max_results=limit,
                order_by=["attributes.start_time DESC"],
            )
            if ids
            else []
        )
        return {"runs": [self._run_dict(run) for run in runs], "limit": limit}

    def run_details(self, run_id: str) -> dict[str, Any]:
        run_id = self._identifier(run_id, "run ID")
        run = self._run_dict(self.client.get_run(run_id))
        run["artifacts"] = [
            {"path": artifact.path, "is_dir": artifact.is_dir, "file_size": artifact.file_size}
            for artifact in self.client.list_artifacts(run_id)
        ]
        run["artifact_scope"] = (
            "root artifact directory; nested artifacts remain in listed directories"
        )
        return run

    def best_run(
        self,
        metric: str = "f1_score",
        experiment_ids: list[str] | None = None,
        limit: int = 20,
        maximize: bool = True,
    ) -> dict[str, Any]:
        self._identifier(metric, "metric name")
        if not isinstance(maximize, bool):
            raise ValueError("maximize must be boolean")
        recent = self.list_runs(experiment_ids=experiment_ids, limit=limit)["runs"]
        eligible = [
            run
            for run in recent
            if run["status"] == "FINISHED"
            and run["metrics"].get(metric) is not None
            and math.isfinite(run["metrics"][metric])
        ]
        eligible.sort(key=lambda run: run["metrics"][metric], reverse=maximize)
        return {
            "best_run": eligible[0] if eligible else None,
            "candidates": eligible,
            "excluded_runs": len(recent) - len(eligible),
            "metric": metric,
            "maximize": maximize,
            "selection_scope": f"Latest {limit} runs by start time across the selected active experiments",
            "caveat": self.CAVEAT,
        }

    def compare_runs(self, run_ids: list[str]) -> dict[str, Any]:
        if not isinstance(run_ids, list) or not 2 <= len(run_ids) <= 20:
            raise ValueError("Provide between 2 and 20 run IDs")
        runs = [self.run_details(run_id) for run_id in run_ids]
        return {
            "runs": runs,
            "metrics": sorted({name for run in runs for name in run["metrics"]}),
            "caveat": "Compare only results with compatible datasets, splits, and evaluation definitions.",
        }

    def model_versions(self, name: str | None = None) -> dict[str, Any]:
        if name is not None:
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_. /-]{1,200}", name):
                raise ValueError("Invalid registered model name")
        kwargs = {"max_results": 100}
        if name:
            kwargs["filter_string"] = f"name = '{name}'"
        versions = []
        while True:
            page = self.client.search_model_versions(**kwargs)
            versions.extend(
                {
                    "name": item.name,
                    "version": item.version,
                    "run_id": item.run_id,
                    "status": item.status,
                    "current_stage": item.current_stage,
                    "aliases": list(getattr(item, "aliases", [])),
                }
                for item in page
            )
            token = getattr(page, "token", None)
            if not token:
                break
            kwargs["page_token"] = token
        return {"versions": versions, "registry_available": True}
