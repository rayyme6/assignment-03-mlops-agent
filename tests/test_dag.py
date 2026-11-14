"""Check workflow semantics without requiring an Airflow scheduler in ordinary CI."""

import ast
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_dag_has_required_dependencies_and_retry_policy():
    tree = ast.parse((ROOT / "dags/training_pipeline.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    stages = {node.name for node in function.body if isinstance(node, ast.FunctionDef)}
    assert stages == {"validate", "preprocess", "train", "evaluate_and_log", "quality_gate"}
    final_call = function.body[-1].value
    chain = []
    while isinstance(final_call, ast.Call):
        chain.append(final_call.func.id)
        final_call = final_call.args[0] if final_call.args else None
    assert chain == ["quality_gate", "evaluate_and_log", "train", "preprocess", "validate"]
    decorator = function.decorator_list[0]
    args = {kw.arg: kw.value for kw in decorator.keywords}
    retries = ast.literal_eval(args["default_args"].keys[0])
    assert retries == "retries"
    assert ast.literal_eval(args["default_args"].values[0]) == 2
    assert ast.literal_eval(args["catchup"]) is False


def test_kubernetes_manifests_have_health_resources_and_two_replicas():
    deployment = yaml.safe_load((ROOT / "kubernetes/deployment.yaml").read_text())
    assert deployment["metadata"]["namespace"] == "mlops"
    assert deployment["spec"]["replicas"] >= 2
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert container["readinessProbe"]["httpGet"]["path"] == "/health/ready"
    assert container["livenessProbe"]["httpGet"]["path"] == "/health/live"
    assert set(container["resources"]) == {"limits", "requests"}
    service = yaml.safe_load((ROOT / "kubernetes/service.yaml").read_text())
    assert service["spec"]["type"] == "ClusterIP"
