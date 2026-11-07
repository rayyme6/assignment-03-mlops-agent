"""Validated, bounded tool contracts. LLM outputs never grant authority."""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: str
    reference: str = Field(default="HEAD", max_length=200)
    other_reference: str = Field(default="HEAD~1", max_length=200)
    run_id: str = Field(default="", max_length=200)
    run_ids: list[str] = Field(default_factory=list, max_length=20)
    metric: str = Field(default="f1", pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    direction: Literal["max", "min"] = "max"
    limit: int = Field(default=5, ge=1, le=100)
    dag_id: str = "iris_training"
    task_id: str = ""
    try_number: int = Field(default=1, ge=1, le=100)
    namespace: str = "mlops-demo"
    name: str = "iris-api"


class Evidence(BaseModel):
    request_id: str
    tool: str
    operation: str
    status: Literal["ok", "error", "denied"]
    data: Any = None
    error: str | None = None
    duration_ms: float


class Answer(BaseModel):
    request_id: str
    facts: list[str]
    uncertainties: list[str]
    recommendations: list[str]
    evidence: list[Evidence]
