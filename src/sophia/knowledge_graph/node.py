"""Node representation in the knowledge graph."""

from typing import Any, Dict
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class Node(BaseModel):
    """Represents a node in the knowledge graph.

    Attributes:
        id: Unique identifier for the node
        type: Type/category of the node
        properties: Additional properties associated with the node
    """

    model_config = ConfigDict(frozen=False)

    id: str = Field(default_factory=lambda: str(uuid4()))
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)

    def __hash__(self) -> int:
        """Make Node hashable for use in sets and as dict keys."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Check equality based on ID."""
        if not isinstance(other, Node):
            return False
        return self.id == other.id
