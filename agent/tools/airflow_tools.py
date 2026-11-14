"""Bounded, authenticated operations against the Airflow 3 public API."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from urllib.parse import quote, urlparse
from uuid import uuid4

import httpx


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 250:
        raise ValueError(f"Invalid {label}")
    if any(ord(char) < 32 for char in value) or "/" in value or "\\" in value:
        raise ValueError(f"Invalid {label}")
    if value in {".", ".."}:
        raise ValueError(f"Invalid {label}")
    return quote(value, safe="")


def _limit(value: int, maximum: int = 100) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"Limit must be an integer between 1 and {maximum}")
    return value


class AirflowTools:
    """Only named API operations are exposed; mutations need explicit approval."""

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float = 15,
        client: httpx.Client | None = None,
        approved_dags: tuple[str, ...] = ("training_pipeline",),
    ):
        endpoint = (base_url or os.getenv("AIRFLOW_BASE_URL", "http://localhost:8080")).rstrip("/")
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("AIRFLOW_BASE_URL must be an HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("AIRFLOW_BASE_URL must not contain credentials, query, or fragment")
        self.base_url = endpoint.removesuffix("/api/v2") + "/api/v2"
        self.token = token or os.getenv("AIRFLOW_TOKEN")
        if timeout <= 0:
            raise ValueError("Timeout must be positive")
        self.timeout = timeout
        self.client = client
        self.approved_dags = frozenset(approved_dags)

    def _request(self, method: str, path: str, **kwargs):
        if not self.token:
            raise RuntimeError("Airflow unavailable: set AIRFLOW_TOKEN to a valid JWT")
        headers = {"Authorization": f"Bearer {self.token}", "Accept": "application/json"}
        # No automatic POST retries: the scheduler might already have accepted a run.
        if self.client is not None:
            response = self.client.request(
                method, self.base_url + path, headers=headers, timeout=self.timeout, **kwargs
            )
        else:
            with httpx.Client(follow_redirects=False) as client:
                response = client.request(
                    method, self.base_url + path, headers=headers, timeout=self.timeout, **kwargs
                )
        response.raise_for_status()
        return response.json()

    def list_dags(self) -> dict:
        return self._request("GET", "/dags", params={"limit": 100, "only_active": True})

    def list_runs(self, dag_id: str = "training_pipeline", limit: int = 5) -> dict:
        dag = _identifier(dag_id, "DAG ID")
        return self._request(
            "GET", f"/dags/{dag}/dagRuns", params={"limit": _limit(limit), "order_by": "-run_after"}
        )

    def run_status(self, dag_id: str, run_id: str) -> dict:
        dag, run = _identifier(dag_id, "DAG ID"), _identifier(run_id, "run ID")
        run_data = self._request("GET", f"/dags/{dag}/dagRuns/{run}")
        tasks = self._request("GET", f"/dags/{dag}/dagRuns/{run}/taskInstances")
        return {
            "run": run_data,
            "task_instances": tasks.get("task_instances", []),
            "total_entries": tasks.get("total_entries"),
        }

    def trigger_dag(
        self,
        dag_id: str = "training_pipeline",
        conf: dict | None = None,
        confirmed: bool = False,
    ) -> dict:
        dag = _identifier(dag_id, "DAG ID")
        if dag_id not in self.approved_dags:
            raise PermissionError("DAG is outside the approved training DAG allowlist")
        if confirmed is not True:
            raise PermissionError("Triggering a training DAG requires explicit user confirmation")
        if conf is not None and not isinstance(conf, dict):
            raise ValueError("DAG configuration must be a JSON object")
        # A unique client-generated run ID makes any uncertain acceptance auditable.
        now = datetime.now(timezone.utc).isoformat()
        return self._request(
            "POST",
            f"/dags/{dag}/dagRuns",
            json={"dag_run_id": f"agent__{uuid4().hex}", "logical_date": now, "conf": conf or {}},
        )

    def failed_tasks(self, dag_id: str, run_id: str) -> dict:
        dag, run = _identifier(dag_id, "DAG ID"), _identifier(run_id, "run ID")
        data = self._request(
            "GET", f"/dags/{dag}/dagRuns/{run}/taskInstances", params={"limit": 100}
        )
        tasks = data.get("task_instances", [])
        return {
            "dag_id": dag_id,
            "run_id": run_id,
            "failed_tasks": [
                task for task in tasks if task.get("state") in {"failed", "upstream_failed"}
            ],
            "total_entries": data.get("total_entries"),
            "diagnostic": "Inspect logs for failed tasks; upstream_failed tasks did not execute.",
        }

    def task_logs(self, dag_id: str, run_id: str, task_id: str, try_number: int = 1) -> dict:
        dag, run = _identifier(dag_id, "DAG ID"), _identifier(run_id, "run ID")
        task = _identifier(task_id, "task ID")
        attempt = _limit(try_number, maximum=1000)
        data = self._request(
            "GET",
            f"/dags/{dag}/dagRuns/{run}/taskInstances/{task}/logs/{attempt}",
            params={"full_content": True},
        )

        # Credentials occasionally appear in task output; redact common assignments.
        def redact(value):
            if isinstance(value, str):
                return re.sub(
                    r"(?i)(token|password|api[_-]?key|secret)(\s*[:=]\s*)[^\s,;]+",
                    r"\1\2[REDACTED]",
                    value,
                )
            if isinstance(value, list):
                return [redact(item) for item in value]
            if isinstance(value, dict):
                return {key: redact(item) for key, item in value.items()}
            return value

        return {
            "dag_id": dag_id,
            "run_id": run_id,
            "task_id": task_id,
            "try_number": attempt,
            "logs": redact(data),
        }
