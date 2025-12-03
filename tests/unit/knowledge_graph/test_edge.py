"""Tests for knowledge graph Edge class."""

import pytest
from sophia.knowledge_graph.edge import Edge


pytestmark = pytest.mark.unit


def test_edge_creation() -> None:
    """Test creating a basic edge."""
    edge = Edge(
        source="node1",
        target="node2",
        relation="relates_to",
        properties={"weight": 1.0},
    )

    assert edge.id is not None
    assert edge.source == "node1"
    assert edge.target == "node2"
    assert edge.relation == "relates_to"
    assert edge.properties == {"weight": 1.0}


def test_edge_with_custom_id() -> None:
    """Test creating an edge with custom ID."""
    edge = Edge(
        id="custom-edge-id", source="node1", target="node2", relation="connects"
    )

    assert edge.id == "custom-edge-id"


def test_edge_equality() -> None:
    """Test edge equality based on ID."""
    edge1 = Edge(id="same-id", source="n1", target="n2", relation="r")
    edge2 = Edge(id="same-id", source="n1", target="n2", relation="r")
    edge3 = Edge(id="different-id", source="n1", target="n2", relation="r")

    assert edge1 == edge2
    assert edge1 != edge3


def test_edge_hashable() -> None:
    """Test that edges are hashable."""
    edge1 = Edge(id="id1", source="n1", target="n2", relation="r")
    edge2 = Edge(id="id2", source="n1", target="n2", relation="r")

    edge_set = {edge1, edge2}
    assert len(edge_set) == 2
