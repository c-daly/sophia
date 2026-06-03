"""Tests for the Neo4j/Milvus adapters feeding the emergence handler (#505)."""

from __future__ import annotations

from collections import Counter

from sophia.maintenance.emergence_handler import current_categories, load_type_members


class _FakeHCG:
    def __init__(self, nodes_by_type, edges, batch, nodes_by_type_uuid=None):
        self._nodes_by_type = nodes_by_type
        self._edges = edges
        self._batch = batch
        # Membership is now a pure `type_uuid` property: minted-type members are
        # served by get_nodes_by_type_uuid, NOT by IS_A-edge traversal (#505).
        self._nodes_by_type_uuid = nodes_by_type_uuid or {}

    def list_all_nodes(self, node_type=None):
        return self._nodes_by_type.get(node_type, [])

    def get_nodes_by_type_uuid(self, type_uuid):
        return list(self._nodes_by_type_uuid.get(type_uuid, []))

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


def test_load_type_members_minted_type_uses_type_uuid_property():
    # A minted type resolves membership via the authoritative `type_uuid`
    # property (get_nodes_by_type_uuid), NOT a label scan or IS_A edges -- so a
    # same-label sibling type cannot bleed members in (#505).
    type_uuid = "type_concept_abc12345"
    hcg = _FakeHCG(
        nodes_by_type={},
        edges={},
        batch={},
        nodes_by_type_uuid={
            type_uuid: [
                {
                    "uuid": "u1",
                    "name": "deriv",
                    "type": "concept",
                    "type_uuid": type_uuid,
                },
                {
                    "uuid": "u2",
                    "name": "integ",
                    "type": "concept",
                    "type_uuid": type_uuid,
                },
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


def test_load_type_members_skips_minted_member_without_embedding():
    # A type_uuid member with no Milvus embedding can't be clustered -> dropped.
    type_uuid = "type_concept_deadbeef"
    hcg = _FakeHCG(
        nodes_by_type={},
        edges={},
        batch={},
        nodes_by_type_uuid={
            type_uuid: [
                {"uuid": "u1", "name": "x", "type": "concept", "type_uuid": type_uuid},
                {"uuid": "u2", "name": "y", "type": "concept", "type_uuid": type_uuid},
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
        batch={},
        nodes_by_type_uuid={
            type_uuid: [
                {
                    "uuid": "u1",
                    "name": "deriv",
                    "type": "concept",
                    "type_uuid": type_uuid,
                }
            ]
        },
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


def test_retyped_member_excluded_from_parent_by_type_uuid():
    """A member split out of a parent type into a child has its authoritative
    `type_uuid` repointed at the child. Membership is that property, so loading
    the parent (get_nodes_by_type_uuid(parent)) simply does not return it -- no
    stale-edge cleanup or guard is needed (#505)."""
    parent_uuid = "type_tool_parent01"
    child_uuid = "type_hammer_child02"
    hcg = _FakeHCG(
        nodes_by_type={},
        edges={},
        batch={},
        nodes_by_type_uuid={
            # u1 still belongs to the parent; u2 was retyped into the child, so
            # its `type_uuid` now points there and it is absent from the parent.
            parent_uuid: [
                {"uuid": "u1", "name": "w", "type": "tool", "type_uuid": parent_uuid}
            ],
            child_uuid: [
                {"uuid": "u2", "name": "h", "type": "hammer", "type_uuid": child_uuid}
            ],
        },
    )
    milvus = _FakeMilvus(
        {
            "u1": {"embedding": [0.1], "model": "m"},
            "u2": {"embedding": [0.2], "model": "m"},
        }
    )
    members = load_type_members(hcg, milvus, parent_uuid)
    # u2 retyped away -> excluded; membership follows the `type_uuid` property.
    assert {m.uuid for m in members} == {"u1"}
