"""Tests for the deterministic structural type-correction handler (#504)."""

from __future__ import annotations

from sophia.maintenance.config import MaintenanceConfig
from sophia.maintenance.type_correction_handler import (
    build_type_correction_handler,
)


class FakeHCG:
    """Minimal in-memory HCG: content nodes + reified edges, recording retypes."""

    def __init__(self, nodes, edges):
        # nodes: list of {uuid, type, ...}; edges: list of {id, source, target, relation}
        self.nodes = {n["uuid"]: dict(n) for n in nodes}
        self.edges = list(edges)
        self.updates = []  # (uuid, props)

    def list_all_edges(self, relation_type=None, limit=1000):
        return [
            e
            for e in self.edges
            if relation_type is None or e["relation"] == relation_type
        ]

    def get_nodes_batch(self, uuids):
        return [self.nodes[u] for u in uuids if u in self.nodes]

    def update_node(self, uuid, properties=None):
        self.nodes.setdefault(uuid, {"uuid": uuid}).update(properties or {})
        self.updates.append((uuid, dict(properties or {})))
        return uuid


def _run(hcg):
    build_type_correction_handler(config=MaintenanceConfig(), hcg=hcg)()


def test_evicts_part_from_its_same_type_whole():
    """tusk PART_OF narwhal, both in `marine_mammal` -> tusk is evicted, narwhal stays."""
    hcg = FakeHCG(
        nodes=[
            {"uuid": "narwhal", "type": "marine_mammal"},
            {"uuid": "tusk", "type": "marine_mammal"},
            {"uuid": "dolphin", "type": "marine_mammal"},  # peer, no anti-link edge
        ],
        edges=[
            {"id": "r1", "source": "tusk", "target": "narwhal", "relation": "PART_OF"}
        ],
    )
    _run(hcg)
    # the part (source of PART_OF) returns to the junk-drawer
    assert hcg.nodes["tusk"]["type"] == "entity"
    assert hcg.nodes["tusk"]["type_uuid"] == "type_entity"
    assert hcg.nodes["tusk"]["needs_reclassification"] is True
    # the whole, and an unrelated peer, are untouched
    assert hcg.nodes["narwhal"]["type"] == "marine_mammal"
    assert hcg.nodes["dolphin"]["type"] == "marine_mammal"
    assert "dolphin" not in {u for u, _ in hcg.updates}


def test_evicts_product_target_for_production_relation():
    """oak PRODUCES acorn, both `flora` -> the product (target) is evicted."""
    hcg = FakeHCG(
        nodes=[{"uuid": "oak", "type": "flora"}, {"uuid": "acorn", "type": "flora"}],
        edges=[
            {"id": "r1", "source": "oak", "target": "acorn", "relation": "PRODUCES"}
        ],
    )
    _run(hcg)
    assert hcg.nodes["acorn"]["type"] == "entity"  # the product leaves
    assert hcg.nodes["oak"]["type"] == "flora"  # the producer stays


def test_no_eviction_across_different_types():
    """A PART_OF spanning two DIFFERENT types is not an intra-type error."""
    hcg = FakeHCG(
        nodes=[
            {"uuid": "tusk", "type": "animal_part"},
            {"uuid": "narwhal", "type": "marine_mammal"},
        ],
        edges=[
            {"id": "r1", "source": "tusk", "target": "narwhal", "relation": "PART_OF"}
        ],
    )
    _run(hcg)
    assert hcg.updates == []
    assert hcg.nodes["tusk"]["type"] == "animal_part"


def test_no_eviction_for_base_types():
    """Two junk-drawer entities linked by PART_OF are skipped (base type)."""
    hcg = FakeHCG(
        nodes=[{"uuid": "a", "type": "entity"}, {"uuid": "b", "type": "entity"}],
        edges=[{"id": "r1", "source": "a", "target": "b", "relation": "PART_OF"}],
    )
    _run(hcg)
    assert hcg.updates == []


def test_taxonomic_isa_edge_is_not_a_correction():
    """IS_A is taxonomy, not meronymy -> never triggers eviction."""
    hcg = FakeHCG(
        nodes=[{"uuid": "x", "type": "mammal"}, {"uuid": "y", "type": "mammal"}],
        edges=[{"id": "r1", "source": "x", "target": "y", "relation": "IS_A"}],
    )
    _run(hcg)
    assert hcg.updates == []
