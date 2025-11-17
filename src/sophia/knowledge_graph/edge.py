"""Edge representation in the knowledge graph."""

from typing import Any, Dict
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class Edge(BaseModel):
    """Represents an edge in the knowledge graph.

    Attributes:
        id: Unique identifier for the edge
        source: ID of the source node
        target: ID of the target node
        relation: Type of relationship
        properties: Additional properties associated with the edge
    """

    model_config = ConfigDict(frozen=False)

    id: str = Field(default_factory=lambda: str(uuid4()))
    source: str
    target: str
    relation: str
    properties: Dict[str, Any] = Field(default_factory=dict)

    def __hash__(self) -> int:
        """Make Edge hashable for use in sets and as dict keys."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Check equality based on ID."""
        if not isinstance(other, Edge):
            return False
        return self.id == other.id
