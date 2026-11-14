"""Public request and evidence envelopes."""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    message: str = Field(min_length=1, max_length=8000)
    confirmed_tools: list[Literal["airflow_trigger_dag", "kubernetes_restart_deployment"]] = Field(
        default_factory=list
    )


class ToolResult(BaseModel):
    evidence_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    tool: str
    status: Literal["ok", "invalid", "denied", "timeout", "unavailable"]
    data: Any = None
    error: str | None = None
    duration_ms: float
    request_id: str


class AgentResponse(BaseModel):
    request_id: str
    planner: str
    facts: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    evidence: list[ToolResult] = Field(default_factory=list)
