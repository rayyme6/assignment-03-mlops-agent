"""Export five agent conversations backed by live tool observations.

Run ``python -m scripts.collect_evidence`` from the repository. ``--trigger`` is
explicit authorization to start the allowlisted training DAG. Ordinary collection
only reads services. Unavailable services remain visible as unavailable evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.executor import redact
from agent.orchestrator import Orchestrator, default_executor
from agent.schemas import AgentRequest
from agent.tools.dvc_tools import DVCTools

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = (
    ("dataset_provenance", "Which dataset version is tracked by DVC, and has the data changed?"),
    (
        "model_selection",
        "Compare the last five training runs and find the best recent model by F1 score.",
    ),
    ("pipeline_status", "Report the recent training pipeline workflow status."),
    (
        "deployment_diagnosis",
        "Why is my prediction API unavailable? Inspect Kubernetes deployment health.",
    ),
    (
        "multi_tool_reasoning",
        "Find the best recent model, identify the dataset version used to train it, "
        "check the deployment status, and explain whether retraining is necessary.",
    ),
)


def sanitize(value: Any) -> Any:
    """Redact both common secret formats and values from known secret variables."""
    value = redact(value)
    secrets = [
        item
        for key, item in os.environ.items()
        if re.search(r"(?i)token|password|secret|api_?key", key) and len(item) >= 6
    ]

    def visit(item):
        if isinstance(item, dict):
            return {key: visit(content) for key, content in item.items()}
        if isinstance(item, list):
            return [visit(content) for content in item]
        if isinstance(item, str):
            for secret in secrets:
                item = item.replace(secret, "[REDACTED]")
        return item

    return visit(value)


def reproduce_dataset(repo: Path = ROOT, rev: str = "HEAD", timeout: float = 45) -> dict:
    """Fetch one historical DVC output to a temporary directory and verify its hash."""
    started = time.monotonic()
    result = {"kind": "live-observation", "method": "dvc get --rev into temporary directory"}
    try:
        tools = DVCTools(repo, timeout=min(timeout, 15))
        provenance = tools.dataset_version(rev=rev)
        if provenance["hash_algorithm"] != "md5" or provenance["dvc_hash"].endswith(".dir"):
            raise ValueError("Collector reproduction supports a single MD5-tracked file")
        with tempfile.TemporaryDirectory(prefix="mlops-data-reproduction-") as directory:
            restored = Path(directory) / "restored.csv"
            argv = [
                "dvc",
                "get",
                str(repo.resolve()),
                provenance["dataset_path"],
                "--rev",
                provenance["git_commit"],
                "--out",
                str(restored),
            ]
            env = os.environ.copy()
            env["DVC_SITE_CACHE_DIR"] = str(Path(directory) / "site-cache")
            env["DVC_NO_ANALYTICS"] = "1"
            subprocess.run(
                argv,
                cwd=directory,
                env=env,
                timeout=timeout,
                check=True,
                capture_output=True,
                text=True,
            )
            checksum = hashlib.md5(restored.read_bytes()).hexdigest()  # noqa: S324
            result.update(
                {
                    "status": "ok" if checksum == provenance["dvc_hash"] else "hash_mismatch",
                    "git_commit": provenance["git_commit"],
                    "dataset_path": provenance["dataset_path"],
                    "expected_dvc_hash": provenance["dvc_hash"],
                    "restored_md5": checksum,
                    "restored_bytes": restored.stat().st_size,
                    "hash_matches": checksum == provenance["dvc_hash"],
                    "original_checkout_modified": False,
                    "temporary_copy_removed": True,
                }
            )
    except (TimeoutError, subprocess.TimeoutExpired):
        result.update({"status": "timeout", "error": "Reproduction timed out; no success inferred"})
    except Exception as exc:
        result.update(
            {
                "status": "unavailable",
                "error": f"Reproduction unavailable ({type(exc).__name__}); no success inferred",
            }
        )
    result["duration_ms"] = round((time.monotonic() - started) * 1000, 3)
    return sanitize(result)


def _entry(agent, scenario: str, request: AgentRequest) -> dict:
    response = agent.chat(request)
    return {
        "scenario": scenario,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "request": request.model_dump(),
        "response": response.model_dump(),
    }


def _tool_request(tool: str, arguments: dict) -> AgentRequest:
    return AgentRequest(message=f"tool {tool} {json.dumps(arguments)}")


def _pipeline_details(
    agent, entry: dict, trigger: bool, max_wait: float, poll_interval: float
) -> list[dict]:
    observations = []
    successful = [item for item in entry["response"]["evidence"] if item["status"] == "ok"]
    run = None
    if trigger:
        for item in successful:
            if item["tool"] == "airflow_trigger_dag":
                run = item.get("data")
                break
    else:
        for item in successful:
            if item["tool"] == "airflow_list_runs":
                runs = (item.get("data") or {}).get("dag_runs", [])
                run = runs[0] if runs else None
                break
    if not isinstance(run, dict):
        return observations
    run_id = run.get("dag_run_id") or run.get("run_id")
    if not run_id:
        return observations
    args = {"dag_id": run.get("dag_id", "training_pipeline"), "run_id": run_id}
    deadline = time.monotonic() + max_wait
    while True:
        observation = _entry(
            agent, "pipeline_task_status", _tool_request("airflow_run_status", args)
        )
        observations.append(observation)
        evidence = observation["response"]["evidence"]
        status = next((item for item in evidence if item["tool"] == "airflow_run_status"), None)
        if not status or status["status"] != "ok":
            break
        state = (status.get("data") or {}).get("run", {}).get("state")
        if state == "failed":
            failed = _entry(
                agent, "pipeline_failure_diagnostics", _tool_request("airflow_failed_tasks", args)
            )
            observations.append(failed)
            break
        if state == "success" or not trigger or time.monotonic() >= deadline:
            break
        time.sleep(min(poll_interval, max(0, deadline - time.monotonic())))
    return observations


def collect(
    output_root: str | Path = ROOT / "evidence/live",
    *,
    agent=None,
    repo: Path = ROOT,
    trigger: bool = False,
    max_wait: float = 120,
    poll_interval: float = 2,
    reproduce_data: bool = False,
) -> tuple[Path, dict]:
    """Persist observed responses. An 'ok' tool call is evidence of service access only."""
    if not 0 <= max_wait <= 120 or not 0 < poll_interval <= 10:
        raise ValueError("max_wait must be 0..120 seconds and poll_interval must be 0..10 seconds")
    # Set before constructing a lazy MlflowClient; preserve operator overrides.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    agent = agent or Orchestrator(executor=default_executor(repo), mode="rules")
    conversations, follow_up = [], []
    for scenario, message in SCENARIOS:
        request = AgentRequest(message=message)
        if scenario == "pipeline_status" and trigger:
            request = AgentRequest(
                message="Trigger the approved model training pipeline.",
                confirmed_tools=["airflow_trigger_dag"],
            )
        entry = _entry(agent, scenario, request)
        conversations.append(entry)
        if scenario == "pipeline_status":
            follow_up = _pipeline_details(agent, entry, trigger, max_wait, poll_interval)
    statuses = {name: [] for name in ("dvc", "mlflow", "airflow", "kubernetes")}
    for entry in conversations + follow_up:
        for item in entry["response"]["evidence"]:
            prefix = item["tool"].split("_", 1)[0]
            if prefix in statuses:
                statuses[prefix].append(
                    {
                        "tool": item["tool"],
                        "status": item["status"],
                        "evidence_id": item["evidence_id"],
                    }
                )
    collected_at = datetime.now(timezone.utc)
    report = {
        "schema_version": 1,
        "kind": "live-observation",
        "collected_at": collected_at.isoformat(),
        "scope": "Actual tool observations; unavailable or unobserved states are not simulated successes.",
        "trigger_authorized": trigger,
        "environment": {
            "python": platform.python_version(),
            "integrations": {
                name: {
                    "available": any(item["status"] == "ok" for item in items),
                    "all_observed_calls_succeeded": bool(items)
                    and all(item["status"] == "ok" for item in items),
                    "observations": items,
                }
                for name, items in statuses.items()
            },
        },
        "conversations": conversations,
        "pipeline_follow_up": follow_up,
        "dataset_reproduction": reproduce_dataset(repo)
        if reproduce_data
        else {
            "status": "not_requested",
            "instruction": "Pass --reproduce-data for an isolated historical DVC checkout.",
        },
    }
    report = sanitize(report)
    destination = Path(output_root) / collected_at.strftime("%Y%m%dT%H%M%S.%fZ")
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / "conversations.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return path, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/live")
    parser.add_argument(
        "--trigger", action="store_true", help="Explicitly authorize the approved training DAG"
    )
    parser.add_argument(
        "--max-wait", type=float, default=120, help="Training poll deadline in seconds (0..120)"
    )
    parser.add_argument(
        "--reproduce-data",
        action="store_true",
        help="Verify historical DVC data in a temporary directory",
    )
    args = parser.parse_args()
    path, report = collect(
        args.output,
        trigger=args.trigger,
        max_wait=args.max_wait,
        reproduce_data=args.reproduce_data,
    )
    availability = {
        name: item["available"] for name, item in report["environment"]["integrations"].items()
    }
    print(
        json.dumps(
            {
                "evidence": str(path),
                "kind": report["kind"],
                "integration_access_observed": availability,
            }
        )
    )


if __name__ == "__main__":
    main()
