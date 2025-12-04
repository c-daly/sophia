"""Tests for knowledge graph Node class."""

import pytest
from sophia.knowledge_graph.node import Node


pytestmark = pytest.mark.unit


def test_node_creation() -> None:
    """Test creating a basic node."""
    node = Node(type="concept", properties={"name": "test"})

    assert node.id is not None
    assert node.type == "concept"
    assert node.properties == {"name": "test"}


def test_node_with_custom_id() -> None:
    """Test creating a node with custom ID."""
    node = Node(id="custom-id", type="entity")

    assert node.id == "custom-id"
    assert node.type == "entity"


def test_node_equality() -> None:
    """Test node equality based on ID."""
    node1 = Node(id="same-id", type="concept")
    node2 = Node(id="same-id", type="concept")
    node3 = Node(id="different-id", type="concept")

    assert node1 == node2
    assert node1 != node3


def test_node_hashable() -> None:
    """Test that nodes are hashable."""
    node1 = Node(id="id1", type="concept")
    node2 = Node(id="id2", type="concept")

    node_set = {node1, node2}
    assert len(node_set) == 2


def test_node_properties_modification() -> None:
    """Test modifying node properties."""
    node = Node(type="concept", properties={"count": 1})

    node.properties["count"] = 2
    node.properties["new_field"] = "value"

    assert node.properties["count"] == 2
    assert node.properties["new_field"] == "value"
