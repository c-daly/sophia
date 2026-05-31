"""Tests for the Neo4j/Milvus adapters feeding the emergence handler (#505)."""

from __future__ import annotations

from collections import Counter

from sophia.maintenance.emergence_handler import current_categories, load_type_members


class _FakeHCG:
    def __init__(self, nodes_by_type, edges, batch, is_a_edges=None):
        self._nodes_by_type = nodes_by_type
        self._edges = edges
        self._batch = batch
        self._is_a_edges = is_a_edges or {}

    def list_all_nodes(self, node_type=None):
        return self._nodes_by_type.get(node_type, [])

    def list_all_edges(self, relation_type=None, target_uuid=None, **kw):
        return self._is_a_edges.get((relation_type, target_uuid), [])

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
                {"name": "entity"},
                {"name": "concept"},
                {"name": "reserved_state"},
                {"name": "object"},
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
                {
                    "uuid": "u1",
                    "name": "derivative",
                    "type": "entity",
                    "properties": {"hermes_type_hint": "concept"},
                }
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


def test_load_type_members_minted_type_uses_is_a_edges():
    # A minted type resolves membership via incoming IS_A edges to its uuid,
    # NOT a label scan -- so a same-label sibling type cannot bleed members in.
    type_uuid = "type_concept_abc12345"
    hcg = _FakeHCG(
        nodes_by_type={},
        edges={},
        batch={
            "u1": {
                "uuid": "u1",
                "name": "deriv",
                "type": "concept",
                "type_uuid": type_uuid,
            },
            "u2": {
                "uuid": "u2",
                "name": "integ",
                "type": "concept",
                "type_uuid": type_uuid,
            },
        },
        is_a_edges={
            ("IS_A", type_uuid): [
                {"source": "u1", "target": type_uuid},
                {"source": "u2", "target": type_uuid},
            ]
        },
    )
    milvus = _FakeMilvus(
        {
            "u1": {"embedding": [0.1, 0.2], "model": "m"},
            "u2": {"embedding": [0.3, 0.4], "model": "m"},
        }
    )
    members = load_type_members(hcg, milvus, type_uuid)
    assert {m.uuid for m in members} == {"u1", "u2"}


def test_load_type_members_tolerates_dangling_is_a_edges():
    # get_nodes_batch may not return every uuid (deleted node); those drop out.
    type_uuid = "type_concept_deadbeef"
    hcg = _FakeHCG(
        nodes_by_type={},
        edges={},
        batch={
            "u1": {"uuid": "u1", "name": "x", "type": "concept", "type_uuid": type_uuid}
        },
        is_a_edges={
            ("IS_A", type_uuid): [
                {"source": "u1", "target": type_uuid},
                {"source": "gone", "target": type_uuid},
            ]
        },
    )
    milvus = _FakeMilvus({"u1": {"embedding": [0.1], "model": "m"}})
    members = load_type_members(hcg, milvus, type_uuid)
    assert {m.uuid for m in members} == {"u1"}


def test_minted_member_embeddings_read_from_base_collection():
    """A member retyped to a slug that maps to a non-Entity collection (e.g.
    'concept' -> 'Concept') must still be read from the base 'entity' collection
    where its embedding was actually stored. Otherwise re-emergence queries the
    wrong collection, misses, and silently drops the member (greptile #149)."""
    type_uuid = "type_concept_abc12345"
    hcg = _FakeHCG(
        nodes_by_type={},
        edges={},
        batch={
            "u1": {
                "uuid": "u1",
                "name": "deriv",
                "type": "concept",
                "type_uuid": type_uuid,
            }
        },
        is_a_edges={("IS_A", type_uuid): [{"source": "u1", "target": type_uuid}]},
    )

    class _CollectionAwareMilvus:
        """Serves embeddings ONLY from the base 'Entity' collection, mirroring
        where entity-derived vectors actually live."""

        def get_embedding(self, node_type, uuid):
            if node_type != "Entity":
                return None
            return {"embedding": [0.1, 0.2], "model": "m"}

    members = load_type_members(hcg, _CollectionAwareMilvus(), type_uuid)
    # Read from Entity (base), not Concept (the slug's mapping) -> member kept.
    assert {m.uuid for m in members} == {"u1"}


def test_retyped_member_with_stale_is_a_edge_excluded_from_parent():
    """A member split out of a parent type into a child keeps a stale IS_A edge
    to the parent (the client can't delete edges), but its authoritative
    type_uuid now points at the child. Re-emergence on the parent must NOT
    re-include and re-mint it (#149 review)."""
    parent_uuid = "type_tool_parent01"
    child_uuid = "type_hammer_child02"
    hcg = _FakeHCG(
        nodes_by_type={},
        edges={},
        batch={
            # u1 still belongs to the parent.
            "u1": {"uuid": "u1", "name": "w", "type": "tool", "type_uuid": parent_uuid},
            # u2 was retyped into the child but kept its stale IS_A -> parent.
            "u2": {
                "uuid": "u2",
                "name": "h",
                "type": "hammer",
                "type_uuid": child_uuid,
            },
        },
        is_a_edges={
            ("IS_A", parent_uuid): [
                {"source": "u1", "target": parent_uuid},
                {"source": "u2", "target": parent_uuid},  # stale
            ]
        },
    )
    milvus = _FakeMilvus(
        {
            "u1": {"embedding": [0.1], "model": "m"},
            "u2": {"embedding": [0.2], "model": "m"},
        }
    )
    members = load_type_members(hcg, milvus, parent_uuid)
    # u2 retyped away -> excluded despite the stale IS_A edge.
    assert {m.uuid for m in members} == {"u1"}
