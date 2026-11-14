"""Create a short-lived, namespace-restricted agent kubeconfig without printing credentials."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import yaml


def create_reader_config(kubectl: str, operator_config: Path, output: Path) -> None:
    def run(*arguments):
        return subprocess.run(
            [kubectl, "--kubeconfig", str(operator_config), *arguments],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout

    run("apply", "-f", "kubernetes/rbac.yaml")
    config = json.loads(run("config", "view", "--raw", "--minify", "-o", "json"))
    token = run("-n", "mlops", "create", "token", "mlops-agent", "--duration=1h").strip()
    reader = {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": config["clusters"],
        "users": [{"name": "mlops-reader", "user": {"token": token}}],
        "contexts": [
            {
                "name": "mlops-reader",
                "context": {
                    "cluster": config["clusters"][0]["name"],
                    "user": "mlops-reader",
                    "namespace": "mlops",
                },
            }
        ],
        "current-context": "mlops-reader",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Atomic creation prevents accidentally replacing existing operator credentials.
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        yaml.safe_dump(reader, stream)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kubectl", default="kubectl")
    parser.add_argument("--operator-config", type=Path, default=Path(".kube/assignment03-config"))
    parser.add_argument("--output", type=Path, default=Path(".kube/agent-config"))
    args = parser.parse_args()
    create_reader_config(args.kubectl, args.operator_config, args.output)
    print(f"Created restricted kubeconfig at {args.output}; expires in one hour")
