"""Tests for KnowledgeGraph class."""

import pytest
from sophia.knowledge_graph.graph import KnowledgeGraph
from sophia.knowledge_graph.node import Node
from sophia.knowledge_graph.edge import Edge


pytestmark = pytest.mark.unit


def test_knowledge_graph_creation() -> None:
    """Test creating an empty knowledge graph."""
    kg = KnowledgeGraph()

    assert kg.node_count() == 0
    assert kg.edge_count() == 0


def test_add_node() -> None:
    """Test adding nodes to the graph."""
    kg = KnowledgeGraph()
    node = Node(id="n1", type="concept")

    kg.add_node(node)

    assert kg.node_count() == 1
    assert kg.get_node("n1") == node


def test_add_edge() -> None:
    """Test adding edges to the graph."""
    kg = KnowledgeGraph()
    node1 = Node(id="n1", type="concept")
    node2 = Node(id="n2", type="concept")
    edge = Edge(id="e1", source="n1", target="n2", relation="relates_to")

    kg.add_node(node1)
    kg.add_node(node2)
    kg.add_edge(edge)

    assert kg.edge_count() == 1
    assert kg.get_edge("e1") == edge


def test_add_edge_missing_source() -> None:
    """Test that adding edge with missing source raises error."""
    kg = KnowledgeGraph()
    node = Node(id="n2", type="concept")
    edge = Edge(source="n1", target="n2", relation="relates_to")

    kg.add_node(node)

    with pytest.raises(ValueError, match="Source node .* not found"):
        kg.add_edge(edge)


def test_add_edge_missing_target() -> None:
    """Test that adding edge with missing target raises error."""
    kg = KnowledgeGraph()
    node = Node(id="n1", type="concept")
    edge = Edge(source="n1", target="n2", relation="relates_to")

    kg.add_node(node)

    with pytest.raises(ValueError, match="Target node .* not found"):
        kg.add_edge(edge)


def test_get_neighbors() -> None:
    """Test getting neighbors of a node."""
    kg = KnowledgeGraph()
    node1 = Node(id="n1", type="concept")
    node2 = Node(id="n2", type="concept")
    node3 = Node(id="n3", type="concept")
    edge1 = Edge(source="n1", target="n2", relation="relates_to")
    edge2 = Edge(source="n1", target="n3", relation="relates_to")

    kg.add_node(node1)
    kg.add_node(node2)
    kg.add_node(node3)
    kg.add_edge(edge1)
    kg.add_edge(edge2)

    neighbors = kg.get_neighbors("n1")
    neighbor_ids = {n.id for n in neighbors}

    assert len(neighbors) == 2
    assert neighbor_ids == {"n2", "n3"}


def test_get_edges_from() -> None:
    """Test getting outgoing edges from a node."""
    kg = KnowledgeGraph()
    node1 = Node(id="n1", type="concept")
    node2 = Node(id="n2", type="concept")
    edge = Edge(id="e1", source="n1", target="n2", relation="relates_to")

    kg.add_node(node1)
    kg.add_node(node2)
    kg.add_edge(edge)

    edges = kg.get_edges_from("n1")

    assert len(edges) == 1
    assert edges[0].id == "e1"


def test_remove_node() -> None:
    """Test removing a node from the graph."""
    kg = KnowledgeGraph()
    node1 = Node(id="n1", type="concept")
    node2 = Node(id="n2", type="concept")
    edge = Edge(source="n1", target="n2", relation="relates_to")

    kg.add_node(node1)
    kg.add_node(node2)
    kg.add_edge(edge)

    assert kg.remove_node("n1") is True
    assert kg.node_count() == 1
    assert kg.edge_count() == 0  # Edge should be removed too
    assert kg.get_node("n1") is None


def test_remove_nonexistent_node() -> None:
    """Test removing a node that doesn't exist."""
    kg = KnowledgeGraph()

    assert kg.remove_node("nonexistent") is False


def test_clear() -> None:
    """Test clearing the graph."""
    kg = KnowledgeGraph()
    node1 = Node(id="n1", type="concept")
    node2 = Node(id="n2", type="concept")
    edge = Edge(source="n1", target="n2", relation="relates_to")

    kg.add_node(node1)
    kg.add_node(node2)
    kg.add_edge(edge)

    kg.clear()

    assert kg.node_count() == 0
    assert kg.edge_count() == 0
