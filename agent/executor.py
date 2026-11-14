"""Validated tool dispatch, trusted confirmation boundary and secret-free audit records."""

from __future__ import annotations

import inspect
import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints

import httpx
from pydantic import ConfigDict, ValidationError, create_model

from agent.schemas import ToolResult

LOGGER = logging.getLogger("mlops.audit")
MUTATIONS = frozenset({"airflow_trigger_dag", "kubernetes_restart_deployment"})


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]"
            if re.search(r"(?i)password|secret|token|api.?key|authorization", key)
            else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)Bearer\s+[\w.\-]+", "Bearer [REDACTED]", value)
        value = re.sub(
            r"(?i)(password|token|secret|api[_-]?key)(\s*[:=]\s*)[^\s,;]+", r"\1\2[REDACTED]", value
        )
        return re.sub(r"(https?://)[^/@\s]+:[^/@\s]+@", r"\1[REDACTED]@", value)
    return value


@dataclass
class Tool:
    name: str
    function: Callable
    description: str
    model: Any


class ToolExecutor:
    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, name: str, function: Callable, description: str = ""):
        fields = {}
        hints = get_type_hints(function)
        for key, parameter in inspect.signature(function).parameters.items():
            if key == "confirmed":
                continue  # A planner cannot grant itself authorization.
            annotation = hints.get(key, Any)
            if annotation is Any:
                if key in {"namespace", "name"}:
                    annotation = str | None if parameter.default is None else str
                else:
                    raise TypeError(f"Tool {name} input {key} needs an explicit type")
            default = ... if parameter.default is inspect.Parameter.empty else parameter.default
            fields[key] = (annotation, default)
        model = create_model(
            name + "Input", __config__=ConfigDict(extra="forbid", strict=True), **fields
        )
        self.tools[name] = Tool(name, function, description or name.replace("_", " "), model)

    def definitions(self):
        definitions = []
        for tool in self.tools.values():
            schema = tool.model.model_json_schema()
            # Responses supports non-strict JSON schema; runtime validation is always strict.
            definitions.append(
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": schema,
                    "strict": False,
                }
            )
        return definitions

    def execute(
        self, name: str, arguments: dict, request_id: str, confirmed_tools=()
    ) -> ToolResult:
        started = time.monotonic()
        status, data, error = "ok", None, None
        try:
            if name not in self.tools:
                raise ValueError("Unknown tool; use the tool catalogue")
            tool = self.tools[name]
            args = tool.model.model_validate(arguments).model_dump()
            if name in MUTATIONS:
                if name not in confirmed_tools:
                    raise PermissionError("Explicit user confirmation for this tool is required")
                args["confirmed"] = True
            data = redact(tool.function(**args))
            # Do not send unserializable or non-finite values to the caller or LLM.
            json.dumps(data, allow_nan=False)
        except PermissionError:
            status, error = (
                "denied",
                "Operation denied: explicit confirmation or approved namespace required",
            )
        except (ValueError, TypeError, ValidationError):
            status, error = (
                "invalid",
                "Invalid tool inputs or response; check the tool's JSON schema",
            )
        except (TimeoutError, subprocess.TimeoutExpired, httpx.TimeoutException):
            status, error = (
                "timeout",
                "Tool timed out; operation outcome is unknown; inspect before retrying writes",
            )
        except Exception as exc:
            # Error text can include request URLs, credentials or raw service logs: never emit it.
            status, error = (
                "unavailable",
                f"Tool unavailable ({type(exc).__name__}); no operational result was inferred",
            )
        duration = round((time.monotonic() - started) * 1000, 3)
        result = ToolResult(
            tool=name,
            status=status,
            data=data if status == "ok" else None,
            error=error,
            duration_ms=duration,
            request_id=request_id,
        )
        LOGGER.info(
            json.dumps(
                {
                    "tool": name if name in self.tools else "unknown",
                    "status": status,
                    "duration_ms": duration,
                    "request_id": request_id,
                    "evidence_id": result.evidence_id,
                }
            )
        )
        return result
