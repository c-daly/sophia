"""Edge representation in the knowledge graph."""

from typing import Any, Dict
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class Edge(BaseModel):
    """Represents an edge in the knowledge graph.

    In the reified edge model, edges are stored as nodes connected to
    source and target via structural :FROM/:TO relationships.

    Attributes:
        id: Unique identifier for the edge
        source: ID of the source node
        target: ID of the target node
        relation: Type of relationship
        bidirectional: Whether the relationship is bidirectional
        properties: Additional properties associated with the edge
    """

    model_config = ConfigDict(frozen=False)

    id: str = Field(default_factory=lambda: str(uuid4()))
    source: str
    target: str
    relation: str
    bidirectional: bool = False
    properties: Dict[str, Any] = Field(default_factory=dict)

    def __hash__(self) -> int:
        """Make Edge hashable for use in sets and as dict keys."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Check equality based on ID."""
        if not isinstance(other, Edge):
            return False
        return self.id == other.id
