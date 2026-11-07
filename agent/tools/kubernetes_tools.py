"""Namespace-scoped Kubernetes diagnostics and explicit rollout restart."""

import os
from datetime import datetime, timezone


class KubernetesTools:
    operations = {"pods", "deployments", "services", "events", "logs", "restart"}

    def __init__(self, core=None, apps=None):
        self.core, self.apps = core, apps
        self.namespace = os.getenv("APP_NAMESPACE", "mlops-demo")
        self.deployment = os.getenv("APP_DEPLOYMENT", "iris-api")

    def _connect(self):
        if self.core is None:
            from kubernetes import client, config

            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
            self.core, self.apps = client.CoreV1Api(), client.AppsV1Api()

    def execute(self, a):
        if a.namespace != self.namespace:
            raise PermissionError("Namespace is not approved")
        if a.operation == "restart" and a.name != self.deployment:
            raise PermissionError("Deployment is not approved")
        self._connect()
        kw = {"namespace": a.namespace, "_request_timeout": (5, 15)}
        selector = "app=iris-api"
        if a.operation == "pods":
            pods = self.core.list_namespaced_pod(**kw, label_selector=selector).items
            return [
                {
                    "name": p.metadata.name,
                    "phase": p.status.phase,
                    "conditions": [c.to_dict() for c in p.status.conditions or []],
                    "containers": [c.to_dict() for c in p.status.container_statuses or []],
                }
                for p in pods
            ]
        if a.operation == "deployments":
            deployments = self.apps.list_namespaced_deployment(**kw, label_selector=selector).items
            return [
                {
                    "name": d.metadata.name,
                    "desired": d.spec.replicas,
                    "available": d.status.available_replicas or 0,
                    "updated": d.status.updated_replicas or 0,
                    "generation": d.metadata.generation,
                    "observed_generation": d.status.observed_generation,
                    "images": [c.image for c in d.spec.template.spec.containers],
                    "annotations": d.spec.template.metadata.annotations or {},
                }
                for d in deployments
            ]
        if a.operation == "services":
            return [
                s.to_dict()
                for s in self.core.list_namespaced_service(**kw, label_selector=selector).items
            ]
        if a.operation == "events":
            # Namespace is dedicated to this application.
            return [
                {
                    "reason": e.reason,
                    "message": e.message,
                    "count": e.count,
                    "object": e.involved_object.name,
                }
                for e in self.core.list_namespaced_event(**kw, limit=100).items
            ]
        if a.operation == "logs":
            p = self.core.read_namespaced_pod(a.name, **kw)
            if (p.metadata.labels or {}).get("app") != "iris-api":
                raise PermissionError("Pod does not belong to the approved application")
            return {
                "pod": a.name,
                "logs": self.core.read_namespaced_pod_log(
                    a.name, **kw, tail_lines=100, limit_bytes=16000
                ),
            }
        if a.operation == "restart":
            body = {
                "spec": {
                    "template": {
                        "metadata": {
                            "annotations": {
                                "kubectl.kubernetes.io/restartedAt": datetime.now(
                                    timezone.utc
                                ).isoformat()
                            }
                        }
                    }
                }
            }
            d = self.apps.patch_namespaced_deployment(a.name, body=body, **kw)
            return {
                "name": d.metadata.name,
                "requested": "restart",
                "note": "Accepted restart request; rollout completion has not been verified.",
            }
        raise ValueError("Unsupported Kubernetes operation")
