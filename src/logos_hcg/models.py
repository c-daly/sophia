"""
Type-safe data models for HCG nodes.

These models represent the four primary node types in the HCG:
- Entity: Concrete instances in the world
- Concept: Abstract categories/types
- State: Temporal snapshots of entity properties
- Process: Actions that cause state changes

Each node type includes embedding metadata for vector integration (Section 4.2):
- embedding_id: Reference to vector in Milvus
- embedding_model: Model used for embedding generation
- last_sync: Timestamp of last vector sync

See Project LOGOS spec: Section 4.1 for ontology structure.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EmbeddingMetadata(BaseModel):
    """
    Embedding metadata for vector integration (Section 4.2).

    Properties:
    - embedding_id: Reference to vector in Milvus (matches node UUID)
    - embedding_model: Model used to generate the embedding
    - last_sync: Timestamp of last synchronization with Milvus
    """

    embedding_id: str | None = None
    embedding_model: str | None = None
    last_sync: datetime | None = None

    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat() if v else None,
        }
    )


class Entity(BaseModel):
    """
    Represents a concrete instance in the HCG.

    Properties:
    - uuid: Unique identifier (required, string with 'entity-' prefix)
    - name: Human-readable name
    - description: Optional description
    - created_at: Timestamp of creation
    - Additional properties for spatial/physical entities (width, height, depth, etc.)
    - Embedding metadata (Section 4.2): embedding_id, embedding_model, last_sync
    """

    uuid: str
    name: str | None = None
    description: str | None = None
    created_at: datetime | None = None

    # Spatial properties (optional)
    width: float | None = Field(None, ge=0)
    height: float | None = Field(None, ge=0)
    depth: float | None = Field(None, ge=0)
    radius: float | None = Field(None, ge=0)
    mass: float | None = Field(None, ge=0)

    # Gripper properties (optional)
    max_grasp_width: float | None = Field(None, ge=0)
    max_force: float | None = Field(None, ge=0)

    # Joint properties (optional)
    joint_type: str | None = None  # enum: revolute|prismatic|fixed|continuous
    min_angle: float | None = None
    max_angle: float | None = None

    # Vector embedding metadata (Section 4.2)
    embedding_id: str | None = None
    embedding_model: str | None = None
    last_sync: datetime | None = None

    # Store any additional properties from Neo4j
    extra_properties: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )

    @field_validator("created_at", mode="before")
    @classmethod
    def parse_neo4j_datetime(cls, v):
        """Convert Neo4j DateTime to Python datetime."""
        if v is None:
            return None
        # Neo4j DateTime objects can be converted to Python datetime
        if hasattr(v, "to_native"):
            return v.to_native()
        return v


class Concept(BaseModel):
    """
    Represents an abstract category/type in the HCG.

    Properties:
    - uuid: Unique identifier (required, string with 'concept-' prefix)
    - name: Concept name (required, unique)
    - description: Optional description
    - Embedding metadata (Section 4.2): embedding_id, embedding_model, last_sync
    """

    uuid: str
    name: str
    description: str | None = None

    # Vector embedding metadata (Section 4.2)
    embedding_id: str | None = None
    embedding_model: str | None = None
    last_sync: datetime | None = None

    # Store any additional properties from Neo4j
    extra_properties: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )


class State(BaseModel):
    """
    Represents a temporal snapshot of entity properties.

    Properties:
    - uuid: Unique identifier (required, string with 'state-' prefix)
    - timestamp: Time of state snapshot (required)
    - name: Optional state name
    - Position: position_x, position_y, position_z
    - Orientation: orientation_roll, orientation_pitch, orientation_yaw
    - Boolean flags: is_grasped, is_closed, is_empty
    - Physical: grasp_width, applied_force
    - Embedding metadata (Section 4.2): embedding_id, embedding_model, last_sync
    """

    uuid: str
    timestamp: datetime
    name: str | None = None

    # Position properties
    position_x: float | None = None
    position_y: float | None = None
    position_z: float | None = None

    # Orientation properties
    orientation_roll: float | None = None
    orientation_pitch: float | None = None
    orientation_yaw: float | None = None

    # Boolean state flags
    is_grasped: bool | None = None
    is_closed: bool | None = None
    is_empty: bool | None = None

    # Physical properties
    grasp_width: float | None = Field(None, ge=0)
    applied_force: float | None = Field(None, ge=0)

    # Vector embedding metadata (Section 4.2)
    embedding_id: str | None = None
    embedding_model: str | None = None
    last_sync: datetime | None = None

    # Store any additional properties from Neo4j
    extra_properties: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_neo4j_datetime(cls, v):
        """Convert Neo4j DateTime to Python datetime."""
        if v is None:
            return None
        # Neo4j DateTime objects can be converted to Python datetime
        if hasattr(v, "to_native"):
            return v.to_native()
        return v


class Process(BaseModel):
    """
    Represents an action that causes state changes.

    Properties:
    - uuid: Unique identifier (required, string with 'process-' prefix)
    - start_time: Process start timestamp (required)
    - name: Optional process name
    - description: Optional description
    - duration_ms: Duration in milliseconds
    - Embedding metadata (Section 4.2): embedding_id, embedding_model, last_sync
    """

    uuid: str
    start_time: datetime
    name: str | None = None
    description: str | None = None
    duration_ms: int | None = Field(None, ge=0)

    # Vector embedding metadata (Section 4.2)
    embedding_id: str | None = None
    embedding_model: str | None = None
    last_sync: datetime | None = None

    # Store any additional properties from Neo4j
    extra_properties: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )

    @field_validator("start_time", mode="before")
    @classmethod
    def parse_neo4j_datetime(cls, v):
        """Convert Neo4j DateTime to Python datetime."""
        if v is None:
            return None
        # Neo4j DateTime objects can be converted to Python datetime
        if hasattr(v, "to_native"):
            return v.to_native()
        return v
