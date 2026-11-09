"""Explicitly authorized deployment to the disposable kind demonstration cluster."""

import json
import subprocess
import time
from pathlib import Path

import httpx

from scripts.render_deployment import render

CONTEXT = "kind-mlops-assignment"


def deploy_demo(*, confirmed=False):
    if not confirmed:
        raise PermissionError("Explicit confirmation is required for the Kubernetes demonstration")
    context = subprocess.check_output(
        ["kubectl", "config", "current-context"], text=True, timeout=10
    ).strip()
    if context != CONTEXT:
        raise PermissionError(f"Refusing to change cluster: expected context {CONTEXT}")
    image = "iris-api:evidence"
    manifest = Path("artifacts/deploy.yaml")
    manifest.write_text(render("artifacts", image))
    for command, timeout in [
        (["docker", "build", "-t", image, "."], 900),
        (["kind", "load", "docker-image", image, "--name", "mlops-assignment"], 180),
        (["kubectl", "--context", CONTEXT, "apply", "-f", str(manifest)], 60),
        (
            [
                "kubectl",
                "--context",
                CONTEXT,
                "-n",
                "mlops-demo",
                "rollout",
                "status",
                "deployment/iris-api",
                "--timeout=180s",
            ],
            200,
        ),
    ]:
        subprocess.run(command, check=True, timeout=timeout)
    forward = subprocess.Popen(
        [
            "kubectl",
            "--context",
            CONTEXT,
            "-n",
            "mlops-demo",
            "port-forward",
            "service/iris-api",
            "18000:80",
            "--address=127.0.0.1",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        with httpx.Client(base_url="http://127.0.0.1:18000", timeout=5, trust_env=False) as client:
            for _ in range(30):
                if forward.poll() is not None:
                    raise RuntimeError("Kubernetes port-forward exited")
                try:
                    ready = client.get("/health/ready")
                    if ready.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("Deployed inference service did not become ready")
            prediction = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
            prediction.raise_for_status()
            approved = json.loads(Path("artifacts/approved.json").read_text())
            if ready.json()["run_id"] != approved["run_id"]:
                raise ValueError("Deployed service reports a different model run")
            if prediction.json()["class"] != 0:
                raise ValueError("Deployed model failed the known Iris prediction smoke test")
            return {
                "context": CONTEXT,
                "namespace": "mlops-demo",
                "image": image,
                "approved_run_id": approved["run_id"],
                "ready": ready.json(),
                "prediction": prediction.json(),
                "rollout": "completed",
            }
    finally:
        forward.terminate()
        try:
            forward.wait(timeout=10)
        except subprocess.TimeoutExpired:
            forward.kill()
            forward.wait(timeout=5)
