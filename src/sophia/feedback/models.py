"""Feedback payload models for Sophia → Hermes communication."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class StepResult(BaseModel):
    """Result of a single plan step execution."""

    step_index: int
    action: str
    outcome: Literal["success", "failure", "skipped"]
    error: str | None = None
    duration_ms: int | None = None


class StateDiff(BaseModel):
    """Changes to CWM state."""

    added_nodes: list[str] = Field(default_factory=list)
    removed_nodes: list[str] = Field(default_factory=list)
    modified_nodes: list[str] = Field(default_factory=list)


class FeedbackPayload(BaseModel):
    """Feedback sent from Sophia to Hermes."""

    # Correlation (at least one required)
    correlation_id: str | None = None
    plan_id: str | None = None
    execution_id: str | None = None

    # Outcome
    feedback_type: Literal["observation", "plan", "execution", "validation"]
    outcome: Literal["accepted", "rejected", "created", "success", "failure", "partial"]
    reason: str

    # Details (optional, type-dependent)
    state_diff: StateDiff | None = None
    step_results: list[StepResult] | None = None
    node_ids_created: list[str] | None = None

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_service: str = "sophia"

    @model_validator(mode="after")
    def validate_correlation_key(self) -> "FeedbackPayload":
        """Validate at least one correlation key is present."""
        if not any([self.correlation_id, self.plan_id, self.execution_id]):
            raise ValueError(
                "At least one of correlation_id, plan_id, or execution_id required"
            )
        return self
