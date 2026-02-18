"""Tests for updated Node/Edge models aligned with reified edge model."""

import pytest

from sophia.knowledge_graph.node import Node
from sophia.knowledge_graph.edge import Edge


pytestmark = pytest.mark.unit


def test_node_has_no_ancestors():
    n = Node(name="Paris", type="location")
    assert not hasattr(n, "ancestors") or "ancestors" not in n.model_fields


def test_node_has_no_is_type_definition():
    n = Node(name="Paris", type="location")
    assert not hasattr(n, "is_type_definition") or "is_type_definition" not in n.model_fields


def test_edge_has_bidirectional():
    e = Edge(source="a", target="b", relation="RELATED_TO", bidirectional=True)
    assert e.bidirectional is True


def test_edge_bidirectional_defaults_false():
    e = Edge(source="a", target="b", relation="IS_A")
    assert e.bidirectional is False
