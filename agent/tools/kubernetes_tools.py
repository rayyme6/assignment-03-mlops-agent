"""Namespace-scoped Kubernetes inspection with an explicitly confirmed restart."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone


def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _name(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?", value
    ):
        raise ValueError("Invalid Kubernetes resource name")
    return value


class KubernetesTools:
    def __init__(self, namespace=None, core_api=None, apps_api=None, timeout=15):
        self.namespace = _name(namespace or os.getenv("KUBE_NAMESPACE", "mlops"))
        self.core_api = core_api
        self.apps_api = apps_api
        if timeout <= 0:
            raise ValueError("Timeout must be positive")
        self.timeout = timeout

    def _namespace(self, namespace=None):
        requested = self.namespace if namespace is None else _name(namespace)
        if requested != self.namespace:
            raise PermissionError(f"Namespace is outside the approved namespace: {self.namespace}")
        return requested

    def _connect(self):
        if self.core_api is not None and self.apps_api is not None:
            return
        from kubernetes.config.config_exception import ConfigException

        from kubernetes import client, config

        try:
            config.load_incluster_config()
        except ConfigException:
            config.load_kube_config()
        self.core_api = self.core_api or client.CoreV1Api()
        self.apps_api = self.apps_api or client.AppsV1Api()

    @staticmethod
    def _pod(pod):
        status, metadata = _get(pod, "status"), _get(pod, "metadata")
        containers = []
        for container in _get(status, "container_statuses", []) or []:
            state = _get(container, "state")
            waiting, terminated = _get(state, "waiting"), _get(state, "terminated")
            previous = _get(_get(container, "last_state"), "terminated")
            containers.append(
                {
                    "name": _get(container, "name"),
                    "ready": _get(container, "ready", False),
                    "restart_count": _get(container, "restart_count", 0),
                    "waiting_reason": _get(waiting, "reason"),
                    "waiting_message": _get(waiting, "message"),
                    "terminated_reason": _get(terminated, "reason"),
                    "exit_code": _get(terminated, "exit_code"),
                    "last_termination_reason": _get(previous, "reason"),
                    "image": _get(container, "image"),
                }
            )
        conditions = [
            {
                "type": _get(c, "type"),
                "status": _get(c, "status"),
                "reason": _get(c, "reason"),
                "message": _get(c, "message"),
            }
            for c in _get(status, "conditions", []) or []
        ]
        ready = any(c["type"] == "Ready" and c["status"] == "True" for c in conditions)
        return {
            "name": _get(metadata, "name"),
            "namespace": _get(metadata, "namespace"),
            "labels": _get(metadata, "labels", {}) or {},
            "phase": _get(status, "phase"),
            "reason": _get(status, "reason"),
            "message": _get(status, "message"),
            "ready": ready,
            "containers": containers,
            "conditions": conditions,
            "restart_count": sum(c["restart_count"] for c in containers),
        }

    @staticmethod
    def _deployment(deployment):
        metadata, spec = _get(deployment, "metadata"), _get(deployment, "spec")
        status = _get(deployment, "status")
        template = _get(spec, "template")
        labels = _get(_get(template, "metadata"), "labels", {}) or {}
        return {
            "name": _get(metadata, "name"),
            "namespace": _get(metadata, "namespace"),
            "generation": _get(metadata, "generation"),
            "observed_generation": _get(status, "observed_generation"),
            "desired_replicas": _get(spec, "replicas", 0) or 0,
            "ready_replicas": _get(status, "ready_replicas", 0) or 0,
            "available_replicas": _get(status, "available_replicas", 0) or 0,
            "updated_replicas": _get(status, "updated_replicas", 0) or 0,
            "version": labels.get("app.kubernetes.io/version", labels.get("version")),
            "model_run_id": (_get(_get(template, "metadata"), "annotations", {}) or {}).get(
                "mlops/run-id"
            ),
            "images": [
                _get(c, "image") for c in _get(_get(template, "spec"), "containers", []) or []
            ],
            "conditions": [
                {
                    "type": _get(c, "type"),
                    "status": _get(c, "status"),
                    "reason": _get(c, "reason"),
                    "message": _get(c, "message"),
                }
                for c in _get(status, "conditions", []) or []
            ],
        }

    def list_pods(self, namespace: str | None = None) -> dict:
        namespace = self._namespace(namespace)
        self._connect()
        data = self.core_api.list_namespaced_pod(namespace, _request_timeout=self.timeout)
        return {"namespace": namespace, "pods": [self._pod(pod) for pod in data.items]}

    def list_deployments(self, namespace: str | None = None) -> dict:
        namespace = self._namespace(namespace)
        self._connect()
        data = self.apps_api.list_namespaced_deployment(namespace, _request_timeout=self.timeout)
        return {
            "namespace": namespace,
            "deployments": [self._deployment(item) for item in data.items],
        }

    def list_services(self, namespace: str | None = None) -> dict:
        namespace = self._namespace(namespace)
        self._connect()
        data = self.core_api.list_namespaced_service(namespace, _request_timeout=self.timeout)
        services = []
        for item in data.items:
            spec = _get(item, "spec")
            services.append(
                {
                    "name": _get(_get(item, "metadata"), "name"),
                    "type": _get(spec, "type"),
                    "cluster_ip": _get(spec, "cluster_ip"),
                    "selector": _get(spec, "selector", {}),
                    "ports": [
                        {"port": _get(port, "port"), "target_port": _get(port, "target_port")}
                        for port in _get(spec, "ports", []) or []
                    ],
                }
            )
        return {"namespace": namespace, "services": services}

    def pod_logs(self, name: str, tail_lines: int = 100, namespace: str | None = None) -> dict:
        namespace = self._namespace(namespace)
        name = _name(name)
        if (
            isinstance(tail_lines, bool)
            or not isinstance(tail_lines, int)
            or not 1 <= tail_lines <= 1000
        ):
            raise ValueError("tail_lines must be an integer between 1 and 1000")
        self._connect()
        logs = self.core_api.read_namespaced_pod_log(
            name, namespace, tail_lines=tail_lines, limit_bytes=65536, _request_timeout=self.timeout
        )
        logs = re.sub(
            r"(?i)(token|password|api[_-]?key|secret)(\s*[:=]\s*)[^\s,;]+", r"\1\2[REDACTED]", logs
        )
        return {"namespace": namespace, "pod": name, "logs": logs}

    def events(self, namespace: str | None = None) -> dict:
        namespace = self._namespace(namespace)
        self._connect()
        data = self.core_api.list_namespaced_event(namespace, _request_timeout=self.timeout)
        events = []
        for event in data.items:
            involved = _get(event, "involved_object")
            events.append(
                {
                    "type": _get(event, "type"),
                    "reason": _get(event, "reason"),
                    "message": _get(event, "message"),
                    "count": _get(event, "count"),
                    "resource": _get(involved, "name"),
                    "kind": _get(involved, "kind"),
                    "last_timestamp": str(_get(event, "last_timestamp") or ""),
                }
            )
        return {"namespace": namespace, "events": events}

    def deployment_health(self, name: str = "prediction-api", namespace: str | None = None) -> dict:
        namespace = self._namespace(namespace)
        name = _name(name)
        self._connect()
        deployment = self.apps_api.read_namespaced_deployment(
            name, namespace, _request_timeout=self.timeout
        )
        result = self._deployment(deployment)
        selector = _get(_get(_get(deployment, "spec"), "selector"), "match_labels", {}) or {}
        if not selector:
            raise ValueError(
                "Deployment has no match_labels; cannot safely select its application pods"
            )
        pod_data = self.core_api.list_namespaced_pod(
            namespace,
            label_selector=",".join(f"{key}={value}" for key, value in sorted(selector.items())),
            _request_timeout=self.timeout,
        )
        pods = [self._pod(pod) for pod in pod_data.items]
        desired = result["desired_replicas"]
        current_generation = result["generation"]
        observed = result["observed_generation"]
        rollout_observed = (
            current_generation is not None
            and observed is not None
            and observed >= current_generation
        )
        healthy = (
            desired > 0
            and result["ready_replicas"] >= desired
            and result["available_replicas"] >= desired
            and result["updated_replicas"] >= desired
            and rollout_observed
            and len(pods) >= desired
            and all(pod["ready"] for pod in pods)
        )
        return {
            "namespace": namespace,
            "deployment": result,
            "healthy": healthy,
            "pods": pods,
            "unhealthy_pods": [pod for pod in pods if not pod["ready"]],
            "restart_count": sum(pod["restart_count"] for pod in pods),
            "events": self.events(namespace)["events"],
            "health_scope": "Deployment readiness and rollout; end-to-end prediction is not probed.",
        }

    def restart_deployment(
        self, name: str = "prediction-api", namespace: str | None = None, confirmed: bool = False
    ) -> dict:
        namespace = self._namespace(namespace)
        name = _name(name)
        if confirmed is not True:
            raise PermissionError("Restarting a deployment requires explicit user confirmation")
        self._connect()
        restarted_at = datetime.now(timezone.utc).isoformat()
        body = {
            "spec": {
                "template": {
                    "metadata": {"annotations": {"kubectl.kubernetes.io/restartedAt": restarted_at}}
                }
            }
        }
        self.apps_api.patch_namespaced_deployment(
            name, namespace, body, _request_timeout=self.timeout
        )
        return {
            "namespace": namespace,
            "deployment": name,
            "restart_requested_at": restarted_at,
            "status": "requested",
            "note": "Inspect deployment health to verify rollout completion.",
        }
