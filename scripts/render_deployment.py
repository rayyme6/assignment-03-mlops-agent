"""Render reviewed manifests; this command does not modify a cluster."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def render(artifact_dir, image, manifest_dir="kubernetes"):
    artifact_dir = Path(artifact_dir)
    approved = json.loads((artifact_dir / "approved.json").read_text())
    digest = hashlib.sha256((artifact_dir / "model.joblib").read_bytes()).hexdigest()
    if approved.get("model_sha256") != digest or approved["tags"].get("quality_gate") != "passed":
        raise ValueError("Model does not match an approved evaluation")
    documents = []
    for name in ["namespace", "configmap", "deployment", "service", "rbac"]:
        documents.extend(yaml.safe_load_all(Path(manifest_dir, name + ".yaml").read_text()))
    for document in documents:
        if document["kind"] == "ConfigMap":
            document["data"].update(MODEL_RUN_ID=approved["run_id"], MODEL_VERSION=digest[:12])
        if document["kind"] == "Deployment":
            template = document["spec"]["template"]
            template["metadata"]["annotations"]["mlops/run-id"] = approved["run_id"]
            template["spec"]["containers"][0]["image"] = image
    return yaml.safe_dump_all(documents, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    print(render(args.artifacts, args.image))
