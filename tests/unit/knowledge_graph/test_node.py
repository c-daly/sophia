"""Tests for knowledge graph Node class."""

import pytest
from sophia.knowledge_graph.node import Node


pytestmark = pytest.mark.unit


def test_node_creation() -> None:
    """Test creating a basic node."""
    node = Node(name="Test Node", type="concept", properties={"extra": "data"})

    assert node.uuid is not None
    assert node.name == "Test Node"
    assert node.type == "concept"
    assert node.is_type_definition is False
    assert node.ancestors == []
    assert node.properties == {"extra": "data"}
    # Backward compat alias
    assert node.id == node.uuid


def test_node_with_custom_uuid() -> None:
    """Test creating a node with custom UUID."""
    node = Node(uuid="custom-id", name="Custom", type="entity")

    assert node.uuid == "custom-id"
    assert node.id == "custom-id"  # alias
    assert node.name == "Custom"
    assert node.type == "entity"


def test_node_with_ancestors() -> None:
    """Test creating a node with ancestors."""
    node = Node(
        uuid="test-1",
        name="Test",
        type="object",
        ancestors=["physical_entity", "entity"],
        is_type_definition=False,
    )

    assert node.ancestors == ["physical_entity", "entity"]
    assert node.is_type_definition is False


def test_node_type_definition() -> None:
    """Test creating a type definition node."""
    node = Node(
        uuid="type-action",
        name="action",
        type="action",
        ancestors=["process", "entity"],
        is_type_definition=True,
    )

    assert node.is_type_definition is True
    assert node.type == "action"


def test_node_equality() -> None:
    """Test node equality based on uuid."""
    node1 = Node(uuid="same-id", name="Node1", type="concept")
    node2 = Node(uuid="same-id", name="Node2", type="concept")
    node3 = Node(uuid="different-id", name="Node3", type="concept")

    assert node1 == node2
    assert node1 != node3


def test_node_hashable() -> None:
    """Test that nodes are hashable."""
    node1 = Node(uuid="id1", name="Node1", type="concept")
    node2 = Node(uuid="id2", name="Node2", type="concept")

    node_set = {node1, node2}
    assert len(node_set) == 2


def test_node_properties_modification() -> None:
    """Test modifying node properties."""
    node = Node(name="Test", type="concept", properties={"count": 1})

    node.properties["count"] = 2
    node.properties["new_field"] = "value"

    assert node.properties["count"] == 2
    assert node.properties["new_field"] == "value"
