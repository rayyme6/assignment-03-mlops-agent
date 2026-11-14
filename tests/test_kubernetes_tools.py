from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.tools.kubernetes_tools import KubernetesTools


def item(**kwargs):
    return SimpleNamespace(**kwargs)


def pod(ready=True, reason=None, restarts=0):
    return item(
        metadata=item(
            name="prediction-api-abc", namespace="mlops", labels={"app": "prediction-api"}
        ),
        status=item(
            phase="Running",
            conditions=[item(type="Ready", status="True" if ready else "False")],
            container_statuses=[
                item(
                    name="api",
                    ready=ready,
                    restart_count=restarts,
                    image="mlops-agent:1.0.0",
                    state=item(waiting=item(reason=reason)),
                    last_state=item(terminated=item(reason="OOMKilled") if restarts else None),
                )
            ],
        ),
    )


def deployment(ready=2):
    return item(
        metadata=item(name="prediction-api", namespace="mlops", generation=2),
        spec=item(
            replicas=2,
            selector=item(match_labels={"app": "prediction-api"}),
            template=item(
                metadata=item(labels={"app.kubernetes.io/version": "1.0.0"}),
                spec=item(containers=[item(image="mlops-agent:1.0.0")]),
            ),
        ),
        status=item(
            ready_replicas=ready,
            available_replicas=ready,
            updated_replicas=2,
            observed_generation=2,
            conditions=[],
        ),
    )


def tools_with_cluster():
    core, apps = Mock(), Mock()
    core.list_namespaced_pod.return_value = item(items=[pod(), pod()])
    core.list_namespaced_event.return_value = item(items=[])
    apps.read_namespaced_deployment.return_value = deployment()
    return KubernetesTools(namespace="mlops", core_api=core, apps_api=apps), core, apps


def test_kubernetes_modification_rejected_without_explicit_confirmation():
    tools, _, apps = tools_with_cluster()
    for confirmation in (False, "yes", 1):
        with pytest.raises(PermissionError, match="confirmation"):
            tools.restart_deployment(confirmed=confirmation)
    apps.patch_namespaced_deployment.assert_not_called()


def test_confirmed_restart_rejected_outside_approved_namespace():
    tools, _, apps = tools_with_cluster()
    with pytest.raises(PermissionError, match="approved namespace"):
        tools.restart_deployment(namespace="kube-system", confirmed=True)
    apps.patch_namespaced_deployment.assert_not_called()


def test_restart_is_only_request_not_claim_of_rollout_success():
    tools, _, apps = tools_with_cluster()
    result = tools.restart_deployment(confirmed=True)
    assert result["status"] == "requested"
    args = apps.patch_namespaced_deployment.call_args
    assert args.args[:2] == ("prediction-api", "mlops")
    assert args.args[2]["spec"]["template"]["metadata"]["annotations"][
        "kubectl.kubernetes.io/restartedAt"
    ]


def test_unhealthy_pod_diagnosis_retains_crash_and_restart_evidence():
    tools, core, apps = tools_with_cluster()
    core.list_namespaced_pod.return_value = item(items=[pod(False, "CrashLoopBackOff", 4), pod()])
    core.list_namespaced_event.return_value = item(
        items=[
            item(
                type="Warning",
                reason="Unhealthy",
                message="Readiness probe failed: connection refused",
                count=3,
                involved_object=item(name="prediction-api-abc", kind="Pod"),
                last_timestamp=None,
            )
        ]
    )
    apps.read_namespaced_deployment.return_value = deployment(ready=1)
    result = tools.deployment_health()
    assert result["healthy"] is False
    assert result["restart_count"] == 4
    assert result["unhealthy_pods"][0]["containers"][0]["waiting_reason"] == "CrashLoopBackOff"
    assert result["unhealthy_pods"][0]["containers"][0]["last_termination_reason"] == "OOMKilled"
    assert result["events"][0]["reason"] == "Unhealthy"
    assert result["deployment"]["version"] == "1.0.0"


def test_healthy_deployment_requires_observed_rollout_and_ready_pods():
    tools, core, apps = tools_with_cluster()
    assert tools.deployment_health()["healthy"] is True
    stale = deployment()
    stale.status.observed_generation = 1
    apps.read_namespaced_deployment.return_value = stale
    assert tools.deployment_health()["healthy"] is False
    apps.read_namespaced_deployment.return_value = deployment()
    core.list_namespaced_pod.return_value = item(items=[pod()])
    assert tools.deployment_health()["healthy"] is False


def test_namespace_guard_applies_to_reads_as_well_as_writes():
    tools, core, _ = tools_with_cluster()
    with pytest.raises(PermissionError):
        tools.list_pods(namespace="default")
    core.list_namespaced_pod.assert_not_called()


def test_api_failure_propagates_instead_of_inventing_health():
    tools, _, apps = tools_with_cluster()
    apps.read_namespaced_deployment.side_effect = ConnectionError("cluster offline")
    with pytest.raises(ConnectionError, match="cluster offline"):
        tools.deployment_health()


def test_bounded_log_retrieval_and_redaction():
    tools, core, _ = tools_with_cluster()
    core.read_namespaced_pod_log.return_value = "token=private connection refused"
    result = tools.pod_logs("prediction-api-abc", tail_lines=40)
    assert result["logs"] == "token=[REDACTED] connection refused"
    assert core.read_namespaced_pod_log.call_args.kwargs["limit_bytes"] == 65536
    with pytest.raises(ValueError):
        tools.pod_logs("../../other")
    with pytest.raises(ValueError):
        tools.pod_logs("prediction-api-abc", tail_lines=1001)


def test_services_preserve_selector_and_port():
    tools, core, _ = tools_with_cluster()
    core.list_namespaced_service.return_value = item(
        items=[
            item(
                metadata=item(name="prediction-api"),
                spec=item(
                    type="ClusterIP",
                    cluster_ip="10.96.0.7",
                    selector={"app": "prediction-api"},
                    ports=[item(port=80, target_port=8000)],
                ),
            )
        ]
    )
    service = tools.list_services()["services"][0]
    assert service["type"] == "ClusterIP"
    assert service["selector"] == {"app": "prediction-api"}
    assert service["ports"] == [{"port": 80, "target_port": 8000}]
