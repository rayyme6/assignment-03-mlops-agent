"""Render a model-specific Deployment for operator review before applying it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def render(metadata_path: Path, output: Path) -> str:
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("quality_gate") != "passed":
        raise ValueError("Only a quality-approved artifact may be packaged for serving")
    run_id = metadata["run_id"]
    if (
        not isinstance(run_id, str)
        or len(run_id) != 32
        or any(c not in "0123456789abcdef" for c in run_id)
    ):
        raise ValueError("Expected an MLflow hexadecimal run identifier")
    deployment = yaml.safe_load(Path("kubernetes/deployment.yaml").read_text())
    pod = deployment["spec"]["template"]
    pod["metadata"]["annotations"] = {
        "mlops/run-id": run_id,
        "mlops/dvc-hash": metadata["dvc_hash"],
        "mlops/git-commit": metadata["git_commit"],
    }
    image = f"mlops-prediction:{run_id}"
    pod["spec"]["containers"][0]["image"] = image
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(deployment, sort_keys=False))
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("artifacts/metadata.json"))
    parser.add_argument("--output", type=Path, default=Path(".kube/deployment.yaml"))
    args = parser.parse_args()
    print(render(args.metadata, args.output))
