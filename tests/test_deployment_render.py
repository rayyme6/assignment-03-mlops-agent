import json

import pytest
import yaml

from scripts.prepare_deployment import render


def test_render_requires_gate_and_uses_exact_model_identity(tmp_path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"quality_gate": "failed"}))
    with pytest.raises(ValueError, match="quality-approved"):
        render(metadata, tmp_path / "deploy.yaml")
    metadata.write_text(
        json.dumps(
            {
                "quality_gate": "passed",
                "run_id": "a" * 32,
                "dvc_hash": "d" * 32,
                "git_commit": "c" * 40,
            }
        )
    )
    assert render(metadata, tmp_path / "deploy.yaml") == "mlops-prediction:" + "a" * 32
    manifest = yaml.safe_load((tmp_path / "deploy.yaml").read_text())
    assert manifest["spec"]["template"]["metadata"]["annotations"]["mlops/run-id"] == "a" * 32
    assert manifest["spec"]["replicas"] >= 2
