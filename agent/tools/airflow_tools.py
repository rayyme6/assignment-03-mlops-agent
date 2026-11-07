"""Authenticated Airflow 3 /api/v2 requests with bounded I/O."""

import os
from datetime import datetime, timezone
from urllib.parse import quote
from uuid import uuid4

import httpx


class AirflowTools:
    operations = {"dags", "runs", "tasks", "logs", "trigger"}

    def __init__(self, client=None):
        self.client = client or httpx.Client(
            base_url=os.getenv("AIRFLOW_URL", "http://localhost:8080").rstrip("/"),
            headers={"Authorization": "Bearer " + os.getenv("AIRFLOW_TOKEN", "")},
            timeout=15,
        )

    def execute(self, a):
        if a.dag_id != "iris_training":
            raise PermissionError("DAG is not approved")
        base = f"/api/v2/dags/{quote(a.dag_id, safe='')}/dagRuns"
        if a.operation == "trigger":
            r = self.client.post(
                base,
                json={
                    "dag_run_id": f"agent__{uuid4()}",
                    "logical_date": datetime.now(timezone.utc).isoformat(),
                    "conf": {},
                },
            )
        elif a.operation == "dags":
            r = self.client.get("/api/v2/dags", params={"limit": a.limit})
        elif a.operation == "runs":
            r = self.client.get(base, params={"limit": a.limit, "order_by": "-logical_date"})
        elif a.operation in {"tasks", "logs"}:
            if not a.run_id:
                raise ValueError("run_id is required")
            path = base + "/" + quote(a.run_id, safe="") + "/taskInstances"
            if a.operation == "logs":
                if not a.task_id:
                    raise ValueError("task_id is required")
                path += "/" + quote(a.task_id, safe="") + f"/logs/{a.try_number}"
            r = self.client.get(path)
        else:
            raise ValueError("Unsupported Airflow operation")
        r.raise_for_status()
        return r.json()
