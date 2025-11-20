"""Pydantic models for JEPA simulation contexts and results."""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class Entity(BaseModel):
    """Model for an entity in the simulation context."""

    id: str = Field(..., description="Unique entity identifier")
    type: str = Field(
        ..., description="Entity type (e.g., 'object', 'agent', 'location')"
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict, description="Entity properties and attributes"
    )
    position: Optional[Dict[str, float]] = Field(
        default=None, description="Spatial position (x, y, z coordinates)"
    )


class SensorReference(BaseModel):
    """Model for sensor references in the simulation context."""

    sensor_id: str = Field(..., description="Unique sensor identifier")
    sensor_type: str = Field(
        ...,
        description="Type of sensor (e.g., 'camera', 'lidar', 'force', 'proprioception')",
    )
    frame_id: Optional[str] = Field(
        default=None, description="Reference frame for sensor data"
    )
    last_reading: Optional[Dict[str, Any]] = Field(
        default=None, description="Most recent sensor reading"
    )


class TalosMetadata(BaseModel):
    """Model for Talos simulator metadata."""

    simulator_version: str = Field(
        default="stub-v1.0", description="Version of Talos/Gazebo simulator"
    )
    physics_engine: str = Field(
        default="none",
        description="Physics engine in use (e.g., 'ODE', 'Bullet', 'none')",
    )
    time_step: float = Field(
        default=0.01, description="Simulation time step in seconds"
    )
    use_hardware: bool = Field(
        default=False, description="Whether using hardware simulator or CPU stub"
    )
    robot_model: Optional[str] = Field(
        default=None, description="Robot model identifier (e.g., 'talos', 'ur5')"
    )


class SimulationContext(BaseModel):
    """Context for simulation requests with entities, sensors, and metadata."""

    entities: List[Entity] = Field(
        ..., description="List of entities in the simulation environment"
    )
    sensor_refs: List[SensorReference] = Field(
        default_factory=list, description="Sensor references for perception data"
    )
    talos_metadata: TalosMetadata = Field(
        default_factory=TalosMetadata, description="Talos simulator metadata"
    )
    initial_state: Dict[str, Any] = Field(
        default_factory=dict, description="Initial state of the system"
    )
    actions: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Optional action sequence to simulate"
    )


class ImaginedProcess(BaseModel):
    """Model for an imagined process during simulation."""

    process_id: str = Field(..., description="Unique process identifier")
    description: str = Field(..., description="Description of the process")
    confidence: float = Field(..., description="Confidence score", ge=0.0, le=1.0)
    model_version: str = Field(..., description="Model version used for imagination")
    horizon: int = Field(..., description="Planning horizon", gt=0)
    assumptions: List[str] = Field(
        default_factory=list, description="Assumptions made during imagination"
    )
    imagined: bool = Field(
        default=True,
        description="Flag indicating this is an imagined (not executed) process",
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict, description="Additional process properties"
    )


class ImaginedState(BaseModel):
    """Model for an imagined state resulting from simulation."""

    state_id: str = Field(..., description="Unique state identifier")
    step: int = Field(..., description="Step number in the rollout", ge=0)
    description: str = Field(..., description="Description of the imagined state")
    confidence: float = Field(..., description="Confidence score", ge=0.0, le=1.0)
    model_version: str = Field(..., description="Model version used for imagination")
    horizon: int = Field(..., description="Planning horizon", gt=0)
    assumptions: List[str] = Field(
        default_factory=list, description="Assumptions made during imagination"
    )
    imagined: bool = Field(
        default=True,
        description="Flag indicating this is an imagined (not executed) state",
    )
    state_data: Dict[str, Any] = Field(
        default_factory=dict, description="State data and properties"
    )
    entities: List[Entity] = Field(
        default_factory=list, description="Entities in this imagined state"
    )


class SimulationResult(BaseModel):
    """Result of a JEPA simulation with k-step rollout."""

    simulation_id: str = Field(..., description="Unique simulation identifier")
    context: SimulationContext = Field(..., description="Simulation context used")
    imagined_processes: List[ImaginedProcess] = Field(
        default_factory=list, description="Imagined processes during simulation"
    )
    imagined_states: List[ImaginedState] = Field(
        ..., description="K-step rollout of imagined states"
    )
    k_steps: int = Field(..., description="Number of steps in rollout", gt=0)
    model_version: str = Field(
        default="jepa-stub-v1.0", description="JEPA model version used"
    )
    overall_confidence: float = Field(
        ..., description="Overall confidence of the simulation", ge=0.0, le=1.0
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of simulation creation",
    )
