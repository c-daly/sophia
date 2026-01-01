"""Node representation in the knowledge graph."""

from typing import Any, Dict, List
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class Node(BaseModel):
    """Represents a node in the knowledge graph with logos-standard properties.

    Attributes:
        uuid: Unique identifier for the node
        name: Human-readable name for the node
        type: Type/category of the node
        is_type_definition: True if this node defines a type, False for instances
        ancestors: Type inheritance chain (e.g., ["parent_type", "root_type"])
        properties: Additional properties associated with the node
    """

    model_config = ConfigDict(frozen=False)

    uuid: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    type: str
    is_type_definition: bool = False
    ancestors: List[str] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)

    @property
    def id(self) -> str:
        """Backward compatibility alias for uuid."""
        return self.uuid

    def __hash__(self) -> int:
        """Make Node hashable for use in sets and as dict keys."""
        return hash(self.uuid)

    def __eq__(self, other: object) -> bool:
        """Check equality based on uuid."""
        if not isinstance(other, Node):
            return False
        return self.uuid == other.uuid
