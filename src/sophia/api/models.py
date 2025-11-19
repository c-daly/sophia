"""Pydantic models for API requests and responses."""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class PlanRequest(BaseModel):
    """Request model for the /plan endpoint."""

    goal: Dict[str, Any] = Field(
        ...,
        description="Goal specification with description and target_state",
        json_schema_extra={
            "example": {
                "description": "red block in bin",
                "target_state": "red_block_in_bin",
            }
        },
    )


class PlanStep(BaseModel):
    """Model for a single plan step."""

    id: str = Field(..., description="Action ID from knowledge graph")
    name: str = Field(..., description="Action name")
    type: str = Field(..., description="Node type")
    action_type: str = Field(..., description="Type of action (e.g., MOVE, GRASP)")
    target: str = Field(default="", description="Target of the action")


class PlanResponse(BaseModel):
    """Response model for the /plan endpoint."""

    plan: List[PlanStep] = Field(..., description="Ordered list of plan steps")
    goal: Dict[str, Any] = Field(..., description="Original goal specification")
    plan_id: str = Field(..., description="Unique plan identifier")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of plan creation",
    )


class ImagineRequest(BaseModel):
    """Request model for the /imagine endpoint."""

    cwm_g_imagery: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="CWM-G imagery data (generative memory content)",
    )
    cwm_e_emotion_tags: Optional[List[str]] = Field(
        default=None,
        description="CWM-E emotion tags to consider",
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context for imagination",
    )
    model_version: str = Field(
        default="v1.0",
        description="Model version for imagination",
    )
    horizon: int = Field(
        default=5,
        description="Planning horizon for imagination",
        gt=0,
    )
    assumptions: Optional[List[str]] = Field(
        default=None,
        description="Assumptions for imagination",
    )


class ImaginedState(BaseModel):
    """Model for an imagined state."""

    state_id: str = Field(..., description="Unique state identifier")
    description: str = Field(..., description="Description of the imagined state")
    confidence: float = Field(..., description="Confidence score", ge=0.0, le=1.0)
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional properties of the state",
    )


class ImagineResponse(BaseModel):
    """Response model for the /imagine endpoint."""

    imagined_states: List[ImaginedState] = Field(
        ..., description="List of imagined future states"
    )
    imagination_id: str = Field(..., description="Unique imagination identifier")
    model_version: str = Field(..., description="Model version used")
    horizon: int = Field(..., description="Planning horizon used")
    assumptions: List[str] = Field(
        default_factory=list,
        description="Assumptions used in imagination",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of imagination creation",
    )


class ExecuteRequest(BaseModel):
    """Request model for the /execute endpoint."""

    plan_id: str = Field(..., description="Plan ID to execute")
    step_index: Optional[int] = Field(
        default=None,
        description="Optional step index to execute (executes all if None)",
        ge=0,
    )
    dry_run: bool = Field(
        default=False,
        description="If true, simulate execution without state changes",
    )


class ExecutionResult(BaseModel):
    """Model for a single execution result."""

    step: PlanStep = Field(..., description="Plan step that was executed")
    status: str = Field(..., description="Execution status (success, failed, skipped)")
    message: str = Field(default="", description="Optional status message")
    state_changes: Dict[str, Any] = Field(
        default_factory=dict,
        description="State changes resulting from execution",
    )


class ExecuteResponse(BaseModel):
    """Response model for the /execute endpoint."""

    plan_id: str = Field(..., description="Plan ID that was executed")
    results: List[ExecutionResult] = Field(..., description="Execution results")
    overall_status: str = Field(
        ..., description="Overall execution status (success, partial, failed)"
    )
    execution_id: str = Field(..., description="Unique execution identifier")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of execution",
    )


class HealthResponse(BaseModel):
    """Response model for the /health endpoint."""

    status: str = Field(..., description="Overall health status")
    components: Dict[str, bool] = Field(
        ..., description="Health status of individual components"
    )
    version: str = Field(default="0.1.0", description="API version")
