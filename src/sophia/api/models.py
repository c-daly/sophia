"""Pydantic models for API requests and responses."""

from typing import Dict, Any, List, Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone


# =============================================================================
# Persona API Models (CWM-E)
# =============================================================================

PersonaEntryType = Literal["belief", "decision", "observation", "reflection"]
PersonaSentiment = Literal["positive", "negative", "neutral", "mixed"]


class PersonaEntryCreate(BaseModel):
    """Request model for creating a persona entry."""

    entry_type: PersonaEntryType = Field(..., description="Type of persona entry")
    content: str = Field(
        ..., min_length=1, max_length=10000, description="Main narrative content"
    )
    summary: Optional[str] = Field(
        None, max_length=200, description="Short summary of the entry"
    )
    trigger: Optional[str] = Field(
        None, max_length=200, description="What caused this entry"
    )
    sentiment: Optional[PersonaSentiment] = Field(
        None, description="Sentiment classification"
    )
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Confidence score"
    )
    related_process_ids: List[str] = Field(
        default_factory=list, description="Linked process IDs"
    )
    related_goal_ids: List[str] = Field(
        default_factory=list, description="Linked goal IDs"
    )
    emotion_tags: List[str] = Field(default_factory=list, description="Emotion tags")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class PersonaEntryResponse(BaseModel):
    """Response after creating a persona entry."""

    entry_id: str = Field(..., description="Unique entry identifier")
    cwm_state_id: str = Field(..., description="CWM state envelope ID")
    timestamp: datetime = Field(..., description="Creation timestamp")


class PersonaEntryFull(BaseModel):
    """Complete persona entry with all fields."""

    entry_id: str = Field(..., description="Unique entry identifier")
    timestamp: datetime = Field(..., description="Entry timestamp")
    entry_type: PersonaEntryType = Field(..., description="Type of persona entry")
    content: str = Field(..., description="Main narrative content")
    summary: Optional[str] = Field(None, description="Short summary")
    trigger: Optional[str] = Field(None, description="What caused this entry")
    sentiment: Optional[PersonaSentiment] = Field(None, description="Sentiment")
    confidence: Optional[float] = Field(None, description="Confidence score")
    related_process_ids: List[str] = Field(
        default_factory=list, description="Linked process IDs"
    )
    related_goal_ids: List[str] = Field(
        default_factory=list, description="Linked goal IDs"
    )
    emotion_tags: List[str] = Field(default_factory=list, description="Emotion tags")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class PersonaEntryUpdate(BaseModel):
    """Request model for partial update of persona entry."""

    summary: Optional[str] = Field(None, max_length=200)
    sentiment: Optional[PersonaSentiment] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    emotion_tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class PersonaListResponse(BaseModel):
    """Response for listing persona entries."""

    entries: List[PersonaEntryFull] = Field(..., description="List of entries")
    total: int = Field(..., description="Total matching entries")
    limit: int = Field(..., description="Limit applied")
    offset: int = Field(..., description="Offset applied")


class SentimentResponse(BaseModel):
    """Aggregated sentiment from recent persona entries."""

    sentiment: Optional[str] = Field(None, description="Most common recent sentiment")
    confidence_avg: Optional[float] = Field(None, description="Average confidence")
    recent_sentiment_trend: Optional[Literal["rising", "falling", "stable"]] = Field(
        None, description="Sentiment trend direction"
    )
    emotion_distribution: Dict[str, int] = Field(
        default_factory=dict, description="Count of each emotion tag"
    )
    entry_count: int = Field(0, description="Number of entries aggregated")
    last_updated: Optional[datetime] = Field(
        None, description="Timestamp of most recent entry"
    )


class PlanRequest(BaseModel):
    """Request model for the /plan endpoint."""

    correlation_id: Optional[str] = Field(
        default=None,
        description="Correlation ID for request tracing and feedback",
    )
    goal: Dict[str, Any] = Field(
        ...,
        description="Goal specification as structured dict",
        json_schema_extra={
            "examples": [
                {
                    "description": "red block in bin",
                    "target_state": "red_block_in_bin",
                },
            ]
        },
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional context describing entities, constraints, or media references",
    )
    constraints: Optional[List[str]] = Field(
        default=None, description="Hard constraints the planner must honor"
    )
    priority: Optional[str] = Field(
        default=None, description="Informational priority label (e.g., P0/P1/P2)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Arbitrary metadata for downstream audits"
    )

    def goal_dict(self) -> Dict[str, Any]:
        """Return the goal as a structured dictionary."""
        if isinstance(self.goal, dict):
            return self.goal
        return {"description": self.goal, "target_state": ""}


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


class StateResponse(BaseModel):
    """Response model for GET /state endpoint."""

    state: Dict[str, Any] = Field(..., description="Current world state")
    state_id: str = Field(..., description="Unique state identifier")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of state",
    )


class StateUpdateRequest(BaseModel):
    """Request model for POST /state endpoint."""

    state: Dict[str, Any] = Field(
        ...,
        description="State updates to apply",
        json_schema_extra={
            "example": {
                "red_block": {"location": "bin", "grasped": False},
                "gripper": {"position": "bin", "holding": None},
            }
        },
    )


class StateUpdateResponse(BaseModel):
    """Response model for POST /state endpoint."""

    state_id: str = Field(..., description="New state identifier")
    cwm_state_id: Optional[str] = Field(
        default=None, description="CWM-A state envelope ID"
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of update",
    )
    validation_passed: bool = Field(
        default=True, description="Whether SHACL validation passed"
    )
    entity_diffs: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Entity changes in this update"
    )


class CWMStateResponse(BaseModel):
    """Response model for CWMState records."""

    state_id: str = Field(..., description="Globally unique identifier")
    model_type: str = Field(..., description="CWM_A, CWM_G, or CWM_E")
    source: str = Field(..., description="Subsystem that emitted the record")
    timestamp: str = Field(..., description="ISO timestamp")
    confidence: float = Field(..., description="Certainty score (0.0-1.0)")
    status: str = Field(..., description="observed, imagined, or reflected")
    links: Dict[str, Any] = Field(
        default_factory=dict, description="Related entity IDs"
    )
    tags: List[str] = Field(default_factory=list, description="Free-form labels")
    data: Dict[str, Any] = Field(..., description="Model-specific payload")


class CWMStateListResponse(BaseModel):
    """Response model for listing CWMState records."""

    states: List[CWMStateResponse] = Field(..., description="List of CWM states")
    total: int = Field(..., description="Total number of states")
    model_type: Optional[str] = Field(
        default=None, description="Filter by model type if applied"
    )


class SimulateRequest(BaseModel):
    """Request model for the /simulate endpoint."""

    entities: List[Dict[str, Any]] = Field(
        ...,
        description="List of entities in the simulation environment",
        json_schema_extra={
            "example": [
                {
                    "id": "red_block",
                    "type": "object",
                    "properties": {"mass": 0.5, "shape": "cube"},
                    "position": {"x": 0.0, "y": 0.0, "z": 0.1},
                }
            ]
        },
    )
    media_sample_id: Optional[str] = Field(
        default=None,
        description="Optional media sample ID to use as visual context for JEPA simulation",
        json_schema_extra={"example": "sample_abc123"},
    )
    sensor_refs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sensor references for perception data",
        json_schema_extra={
            "example": [
                {
                    "sensor_id": "camera_1",
                    "sensor_type": "camera",
                    "frame_id": "base_link",
                }
            ]
        },
    )
    talos_metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Talos simulator metadata",
        json_schema_extra={
            "example": {
                "simulator_version": "stub-v1.0",
                "physics_engine": "none",
                "time_step": 0.01,
                "use_hardware": False,
            }
        },
    )
    initial_state: Dict[str, Any] = Field(
        default_factory=dict, description="Initial state of the system"
    )
    actions: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Optional action sequence to simulate"
    )
    k_steps: int = Field(
        default=5, description="Number of forward prediction steps", gt=0, le=100
    )
    assumptions: Optional[List[str]] = Field(
        default=None, description="Assumptions for the simulation"
    )


class SimulateResponse(BaseModel):
    """Response model for the /simulate endpoint."""

    simulation_id: str = Field(..., description="Unique simulation identifier")
    imagined_processes: List[Dict[str, Any]] = Field(
        ..., description="Imagined processes during simulation"
    )
    imagined_states: List[Dict[str, Any]] = Field(
        ..., description="K-step rollout of imagined states"
    )
    k_steps: int = Field(..., description="Number of steps in rollout")
    model_version: str = Field(..., description="JEPA model version used")
    overall_confidence: float = Field(
        ..., description="Overall confidence of the simulation", ge=0.0, le=1.0
    )
    media_sample_id: Optional[str] = Field(
        default=None,
        description="Media sample ID if simulation used visual context",
    )
    media_embeddings: Optional[List[str]] = Field(
        default=None,
        description="Milvus embedding IDs for JEPA-generated visual representations",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of simulation creation",
    )


class HermesProposalRequest(BaseModel):
    """Request model for Hermes LLM proposal ingestion."""

    proposal_id: str = Field(..., description="Unique identifier for this proposal")
    correlation_id: Optional[str] = Field(
        default=None,
        description="Correlation ID for request tracing and feedback",
    )
    source_service: str = Field(
        default="hermes", description="Source service (typically 'hermes')"
    )
    llm_provider: str = Field(
        ...,
        description="LLM provider name (e.g., 'openai', 'anthropic', 'azure')",
        json_schema_extra={"example": "openai"},
    )
    model: str = Field(
        ...,
        description="Model identifier (e.g., 'gpt-4', 'claude-3-opus')",
        json_schema_extra={"example": "gpt-4"},
    )
    generated_at: str = Field(
        ...,
        description="ISO timestamp when proposal was generated",
        json_schema_extra={"example": "2025-11-23T12:00:00Z"},
    )
    confidence: float = Field(
        ...,
        description="Confidence score for this proposal",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.85},
    )
    raw_text: Optional[str] = Field(
        default=None,
        description="Raw LLM response text",
    )
    plan_steps: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Structured plan steps from the LLM",
        json_schema_extra={
            "example": [
                {
                    "action": "move_to_red_block",
                    "target": "red_block",
                    "parameters": {},
                }
            ]
        },
    )
    imagined_states: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Imagined future states from the LLM",
        json_schema_extra={
            "example": [
                {
                    "state_id": "state_1",
                    "entities": {"red_block": {"location": "bin"}},
                }
            ]
        },
    )
    diagnostics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Diagnostic information from the LLM",
        json_schema_extra={"example": {"reasoning": "Block needs to be moved"}},
    )
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Tool calls requested by the LLM",
        json_schema_extra={
            "example": [
                {
                    "tool": "get_object_location",
                    "parameters": {"object_id": "red_block"},
                }
            ]
        },
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata for provenance tracking",
    )
    proposed_nodes: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Structured entity proposals with embeddings",
    )
    document_embedding: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Embedding of the full document text",
    )
    proposed_edges: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Proposed relationships between entities",
    )


class HermesProposalResponse(BaseModel):
    """Response model for Hermes proposal ingestion."""

    proposal_id: str = Field(..., description="The ingested proposal identifier")
    stored_node_ids: List[str] = Field(
        ..., description="Node IDs created in Neo4j for this proposal"
    )
    stored_edge_ids: List[str] = Field(
        default_factory=list,
        description="Edge IDs created in Neo4j for this proposal",
    )
    status: str = Field(
        ...,
        description="Ingestion status",
        json_schema_extra={"example": "accepted"},
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of ingestion",
    )
    validation_results: Optional[Dict[str, Any]] = Field(
        default=None,
        description="SHACL validation results if applicable",
    )
    relevant_context: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Relevant graph context for LLM prompt",
    )


# =============================================================================
# HCG Graph API Models (for Apollo's Neo4j removal)
# =============================================================================


class HCGEntityResponse(BaseModel):
    """Response model for a single HCG entity (node)."""

    id: str = Field(..., description="Unique entity identifier (uuid)")
    type: str = Field(..., description="Entity type (e.g., 'object', 'action')")
    name: str = Field(..., description="Human-readable name")
    properties: Dict[str, Any] = Field(
        default_factory=dict, description="Entity properties"
    )
    labels: List[str] = Field(
        default_factory=list,
        description="Node labels (reserved for future use)",
    )
    created_at: Optional[str] = Field(
        default=None, description="ISO timestamp of creation"
    )


class HCGEdgeResponse(BaseModel):
    """Response model for a single HCG edge (relationship)."""

    id: str = Field(..., description="Unique edge identifier")
    source_id: str = Field(..., description="Source entity uuid")
    target_id: str = Field(..., description="Target entity uuid")
    edge_type: str = Field(..., description="Relationship type (e.g., 'enables')")
    properties: Dict[str, Any] = Field(
        default_factory=dict, description="Edge properties"
    )


class HCGGraphSnapshotResponse(BaseModel):
    """Response model for a full graph snapshot."""

    entities: List[HCGEntityResponse] = Field(
        ..., description="All entities in the graph"
    )
    edges: List[HCGEdgeResponse] = Field(..., description="All edges in the graph")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp when snapshot was taken",
    )
    entity_count: int = Field(..., description="Total number of entities")
    edge_count: int = Field(..., description="Total number of edges")
