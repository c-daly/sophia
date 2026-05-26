"""Tests for the Neo4j/Milvus adapters feeding the emergence handler (#505)."""

from __future__ import annotations

from collections import Counter

from sophia.maintenance.emergence_handler import current_categories, load_type_members


class _FakeHCG:
    def __init__(self, nodes_by_type, edges, batch):
        self._nodes_by_type = nodes_by_type
        self._edges = edges
        self._batch = batch

    def list_all_nodes(self, node_type=None):
        return self._nodes_by_type.get(node_type, [])

    def query_edges_from(self, uuid):
        return self._edges.get(uuid, [])

    def get_nodes_batch(self, uuids):
        return [self._batch[u] for u in uuids if u in self._batch]


class _FakeMilvus:
    def __init__(self, embeddings):
        self._embeddings = embeddings

    def get_embedding(self, node_type, uuid):
        return self._embeddings.get(uuid)


def test_current_categories_excludes_entity_and_reserved():
    hcg = _FakeHCG(
        nodes_by_type={
            "type_definition": [
                {"name": "entity"}, {"name": "concept"},
                {"name": "reserved_state"}, {"name": "object"},
            ]
        },
        edges={},
        batch={},
    )
    assert set(current_categories(hcg)) == {"concept", "object"}


def test_load_type_members_builds_member_with_signature():
    hcg = _FakeHCG(
        nodes_by_type={
            "entity": [
                {"uuid": "u1", "name": "derivative", "type": "entity",
                 "properties": {"hermes_type_hint": "concept"}}
            ]
        },
        edges={"u1": [{"relation": "DEFINED_AS", "target": "u2"}]},
        batch={"u2": {"uuid": "u2", "name": "limit", "type": "entity"}},
    )
    milvus = _FakeMilvus({"u1": {"embedding": [0.1, 0.2], "model": "m"}})

    members = load_type_members(hcg, milvus, "type_entity")

    assert len(members) == 1
    m = members[0]
    assert m.uuid == "u1" and m.embedding == [0.1, 0.2] and m.model == "m"
    assert m.hermes_type_hint == "concept"
    assert m.signature == Counter({("DEFINED_AS", "entity"): 1})
    assert m.neighbors[0]["neighbor_name"] == "limit"


def test_load_type_members_skips_nodes_without_embedding():
    hcg = _FakeHCG(
        nodes_by_type={"entity": [{"uuid": "u1", "name": "x", "type": "entity"}]},
        edges={},
        batch={},
    )
    milvus = _FakeMilvus({})  # no embedding for u1
    assert load_type_members(hcg, milvus, "type_entity") == []
