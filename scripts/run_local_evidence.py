"""Start real local services, exercise them, export sanitized evidence, then shut down.

Airflow and training run in separate virtualenvs. Set AIRFLOW_PYTHON to the Airflow
venv's interpreter. Kubernetes is intentionally never simulated in these exports.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.orchestrator import Agent, redact  # noqa: E402


def wait_http(url, processes, seconds=120):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for p in processes:
            if p.poll() is not None:
                raise RuntimeError("A service exited; inspect local service logs")
        try:
            r = httpx.get(url, timeout=2, trust_env=False)
            if r.status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise TimeoutError(f"Service did not become reachable: {url}")


def main():
    os.chdir(ROOT)
    os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"
    os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "1"
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "10"
    os.environ["DVC_NO_ANALYTICS"] = "true"
    # Local demonstration traffic must not leave the host through a proxy.
    for key in list(os.environ):
        if key.lower().endswith("_proxy"):
            os.environ.pop(key)
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    evidence = ROOT / "evidence"
    evidence.mkdir(exist_ok=True)
    processes, logs = [], []

    def start(command, log):
        handle = open(ROOT / "artifacts" / log, "w")
        logs.append(handle)
        p = subprocess.Popen(
            command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True
        )
        processes.append(p)
        return p

    def save(name, value):
        (evidence / name).write_text(json.dumps(redact(value), indent=2, default=str))

    try:
        Path("artifacts").mkdir(exist_ok=True)
        start(
            [
                sys.executable,
                "-m",
                "mlflow",
                "server",
                "--host",
                "127.0.0.1",
                "--port",
                "5000",
                "--backend-store-uri",
                "sqlite:///mlflow.db",
                "--artifacts-destination",
                "./mlruns",
                "--workers",
                "1",
            ],
            "mlflow.log",
        )
        wait_http("http://127.0.0.1:5000/health", processes)
        print("MLflow server ready", flush=True)
        with open(evidence / "dvc-repro.txt", "w") as output:
            subprocess.run(
                ["dvc", "repro", "--force"],
                check=True,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=180,
            )
        candidates = []
        for c in [0.1, 2.0]:
            for stage in ["preprocess", "train", "evaluate"]:
                subprocess.run(
                    [sys.executable, "-m", "training.train", stage, "--c", str(c)],
                    check=True,
                    timeout=120,
                )
            gate = subprocess.run(
                [sys.executable, "-m", "training.train", "gate"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            candidates.append(
                {
                    "C": c,
                    "gate_exit_code": gate.returncode,
                    "evaluation": json.loads(Path("artifacts/metrics.json").read_text()),
                }
            )
        save("candidate-evaluations.json", candidates)
        print("Three actual MLflow training runs completed", flush=True)
        agent = Agent()
        save(
            "01-dataset-conversation.json",
            agent.ask("Which dataset version is current?").model_dump(),
        )
        # Deliberately injected data and quality-gate failures, using isolated paths.
        from fastapi.testclient import TestClient

        from api.main import app
        from training.train import generate, quality_gate, validate

        bad = Path("artifacts/bad-data.csv")
        generate(bad)
        bad.write_text(bad.read_text().replace("5.1", "", 1))
        failures = []
        try:
            validate(bad)
        except ValueError as exc:
            failures.append(
                {
                    "scenario": "missing dataset value",
                    "observed_error": str(exc),
                    "resolution": "Regenerate clean data; keep invalid data out of training.",
                }
            )
        gate = Path("artifacts/failed-gate")
        gate.mkdir(exist_ok=True)
        (gate / "metrics.json").write_text(json.dumps({"metrics": {"f1": 0.2}}))
        try:
            quality_gate(gate)
        except ValueError as exc:
            failures.append(
                {
                    "scenario": "F1 below quality threshold",
                    "observed_error": str(exc),
                    "approved_file_exists": (gate / "approved.json").exists(),
                    "resolution": "Reject promotion and investigate the evaluation errors.",
                }
            )
        save("failure-investigations.json", failures)
        with TestClient(app) as client:
            save(
                "inference.json",
                {
                    "ready": client.get("/health/ready").json(),
                    "prediction": client.post(
                        "/predict", json={"features": [5.1, 3.5, 1.4, 0.2]}
                    ).json(),
                },
            )
        airflow_python = os.getenv("AIRFLOW_PYTHON")
        if airflow_python:
            os.environ["AIRFLOW_HOME"] = str(ROOT / ".airflow")
            os.environ["AIRFLOW__CORE__DAGS_FOLDER"] = str(ROOT / "dags")
            os.environ["AIRFLOW__CORE__LOAD_EXAMPLES"] = "false"
            os.environ["AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_USERS"] = "admin:admin"
            os.environ["MLOPS_ROOT"] = str(ROOT)
            os.environ["MLOPS_PYTHON"] = sys.executable
            os.environ["PATH"] = str(Path(airflow_python).parent) + os.pathsep + os.environ["PATH"]
            start([airflow_python, "-m", "airflow", "standalone"], "airflow.log")
            wait_http("http://127.0.0.1:8080/auth/token", processes, seconds=180)
            password_file = ROOT / ".airflow/simple_auth_manager_passwords.json.generated"
            passwords = json.loads(password_file.read_text())
            r = httpx.post(
                "http://127.0.0.1:8080/auth/token",
                json={"username": "admin", "password": passwords["admin"]},
                timeout=10,
            )
            r.raise_for_status()
            os.environ["AIRFLOW_TOKEN"] = r.json()["access_token"]
            os.environ["AIRFLOW_URL"] = "http://127.0.0.1:8080"
            agent = Agent()
            client = httpx.Client(
                base_url=os.environ["AIRFLOW_URL"],
                headers={"Authorization": "Bearer " + os.environ["AIRFLOW_TOKEN"]},
            )
            for _ in range(120):
                r = client.get("/api/v2/dags/iris_training")
                if r.status_code == 200:
                    break
                time.sleep(1)
            r.raise_for_status()
            client.patch("/api/v2/dags/iris_training", json={"is_paused": False}).raise_for_status()
            # This script is explicitly invoked to run the training demonstration.
            triggered = agent.ask(
                "Trigger training pipeline", approvals={"airflow.trigger:iris_training"}
            )
            save("03-pipeline-trigger.json", triggered.model_dump())
            accepted = [
                e for e in triggered.evidence if e.operation == "trigger" and e.status == "ok"
            ]
            if not accepted:
                raise RuntimeError("Airflow did not accept the trigger; inspect exported evidence")
            run_id = accepted[0].data["dag_run_id"]
            from urllib.parse import quote

            for _ in range(180):
                r = client.get("/api/v2/dags/iris_training/dagRuns/" + quote(run_id, safe=""))
                r.raise_for_status()
                if r.json()["state"] in {"success", "failed"}:
                    break
                time.sleep(1)
            save("airflow-run.json", r.json())
            save("03-pipeline-conversation.json", agent.ask("Report pipeline status").model_dump())
            print("Airflow run state: " + r.json()["state"], flush=True)
        save("02-model-conversation.json", agent.ask("Compare best recent model").model_dump())
        save(
            "04-deployment-conversation.json",
            agent.ask("Why is the prediction API unavailable?").model_dump(),
        )
        save(
            "05-multitool-conversation.json",
            agent.ask(
                "Find the best recent model, identify its dataset version, check deployment status, "
                "and explain whether retraining is necessary"
            ).model_dump(),
        )
        print("Evidence exported; unavailable systems remain explicitly unknown", flush=True)
    finally:
        import signal

        for p in reversed(processes):
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
        for handle in logs:
            handle.close()


if __name__ == "__main__":
    main()
