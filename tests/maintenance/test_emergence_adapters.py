"""Tests for the Neo4j/Milvus adapters feeding the emergence handler (#505)."""

from __future__ import annotations

from collections import Counter

from sophia.maintenance.emergence_handler import current_categories, load_type_members


class _FakeHCG:
    def __init__(self, members_by_type=None, edges=None, batch=None, type_defs=None):
        # Membership is the instance->type IS_A edge: get_members_of_type(uuid)
        # returns the member rows for a realm-root uuid (its drainage pool) or a
        # minted-type uuid (its members) uniformly (B2/B3, DESIGN §3).
        self._members_by_type = members_by_type or {}
        self._edges = edges or {}
        self._batch = batch or {}
        self._type_defs = type_defs or []

    def get_members_of_type(self, type_uuid):
        return list(self._members_by_type.get(type_uuid, []))

    def list_all_nodes(self, node_type=None):
        if node_type == "type_definition":
            return list(self._type_defs)
        return []

    def get_all_type_definitions(self):
        return [
            {
                "uuid": td.get("uuid", ""),
                "name": td.get("name"),
                "properties": dict(td.get("properties", {})),
            }
            for td in self._type_defs
        ]

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
        type_defs=[
            {"name": "entity"},
            {"name": "concept"},
            {"name": "reserved_state"},
            {"name": "object"},
        ]
    )
    assert set(current_categories(hcg)) == {"concept", "object"}


def test_load_type_members_builds_member_with_signature():
    # A realm-root uuid yields that realm's pool via the instance->type IS_A edge
    # query (get_members_of_type), uniform with minted types (B2/B3).
    hcg = _FakeHCG(
        members_by_type={
            "type_entity": [
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
    milvus = _FakeMilvus({"u1": {"embedding": [0.1, 0.2], "embedding_model": "m"}})

    members = load_type_members(hcg, milvus, "type_entity")

    assert len(members) == 1
    m = members[0]
    assert m.uuid == "u1" and m.embedding == [0.1, 0.2] and m.model == "m"
    assert m.hermes_type_hint == "concept"
    assert m.signature == Counter({("DEFINED_AS", "entity"): 1})
    assert m.neighbors[0]["neighbor_name"] == "limit"


def test_load_type_members_skips_nodes_without_embedding():
    hcg = _FakeHCG(
        members_by_type={
            "type_entity": [{"uuid": "u1", "name": "x", "type": "entity"}]
        },
    )
    milvus = _FakeMilvus({})  # no embedding for u1
    assert load_type_members(hcg, milvus, "type_entity") == []


def test_load_type_members_minted_type_uses_is_a_edge_query():
    # A minted type resolves membership via the instance->type IS_A edge
    # (get_members_of_type), anchored on THIS type's uuid -- so a same-label
    # sibling type cannot bleed members in (B2/B3).
    type_uuid = "type_concept_abc12345"
    hcg = _FakeHCG(
        members_by_type={
            type_uuid: [
                {"uuid": "u1", "name": "deriv", "type": "concept"},
                {"uuid": "u2", "name": "integ", "type": "concept"},
            ]
        },
    )
    milvus = _FakeMilvus(
        {
            "u1": {"embedding": [0.1, 0.2], "embedding_model": "m"},
            "u2": {"embedding": [0.3, 0.4], "embedding_model": "m"},
        }
    )
    members = load_type_members(hcg, milvus, type_uuid)
    assert {m.uuid for m in members} == {"u1", "u2"}


def test_load_type_members_skips_minted_member_without_embedding():
    # A member with no Milvus embedding can't be clustered -> dropped.
    type_uuid = "type_concept_deadbeef"
    hcg = _FakeHCG(
        members_by_type={
            type_uuid: [
                {"uuid": "u1", "name": "x", "type": "concept"},
                {"uuid": "u2", "name": "y", "type": "concept"},
            ]
        },
    )
    milvus = _FakeMilvus({"u1": {"embedding": [0.1], "embedding_model": "m"}})
    members = load_type_members(hcg, milvus, type_uuid)
    assert {m.uuid for m in members} == {"u1"}


def test_minted_member_embeddings_read_from_base_collection():
    """A member retyped to a slug that maps to a non-Entity collection (e.g.
    'concept' -> 'Concept') must still be read from the base 'entity' collection
    where its embedding was actually stored. Otherwise re-emergence queries the
    wrong collection, misses, and silently drops the member (greptile #149)."""
    type_uuid = "type_concept_abc12345"
    hcg = _FakeHCG(
        members_by_type={
            type_uuid: [{"uuid": "u1", "name": "deriv", "type": "concept"}]
        },
    )

    class _CollectionAwareMilvus:
        """Serves embeddings ONLY from the base 'Entity' collection, mirroring
        where entity-derived vectors actually live."""

        def get_embedding(self, node_type, uuid):
            if node_type != "Entity":
                return None
            return {"embedding": [0.1, 0.2], "embedding_model": "m"}

    members = load_type_members(hcg, _CollectionAwareMilvus(), type_uuid)
    # Read from Entity (base), not Concept (the slug's mapping) -> member kept.
    assert {m.uuid for m in members} == {"u1"}


def test_retyped_member_excluded_from_parent_by_is_a_edge():
    """A member re-pointed out of a parent type into a child has its single upward
    IS_A edge moved to the child. Membership is that edge, so loading the parent
    (get_members_of_type(parent)) simply does not return it -- no stale-edge
    cleanup or guard is needed (B2/B3, DESIGN §3)."""
    parent_uuid = "type_tool_parent01"
    child_uuid = "type_hammer_child02"
    hcg = _FakeHCG(
        members_by_type={
            # u1's IS_A edge still points at the parent; u2's was re-pointed at
            # the child, so the parent's edge query no longer returns it.
            parent_uuid: [{"uuid": "u1", "name": "w", "type": "entity"}],
            child_uuid: [{"uuid": "u2", "name": "h", "type": "entity"}],
        },
    )
    milvus = _FakeMilvus(
        {
            "u1": {"embedding": [0.1], "embedding_model": "m"},
            "u2": {"embedding": [0.2], "embedding_model": "m"},
        }
    )
    members = load_type_members(hcg, milvus, parent_uuid)
    # u2 re-pointed away -> excluded; membership follows the IS_A edge.
    assert {m.uuid for m in members} == {"u1"}


def test_member_read_uses_is_a_edge_query_for_realm_root_and_minted():
    """Membership is read through the instance->type IS_A edge query
    (get_members_of_type) for BOTH a realm-root uuid (its pool) and a minted-type
    uuid -- never get_nodes_by_type_uuid or a list_all_nodes scan (de-slug, B2/B3).
    """
    calls: list[tuple[str, object]] = []

    class _RecordingHCG(_FakeHCG):
        def get_members_of_type(self, type_uuid):
            calls.append(("get_members_of_type", type_uuid))
            return super().get_members_of_type(type_uuid)

        def get_nodes_by_type_uuid(self, type_uuid):  # must NOT be used
            calls.append(("get_nodes_by_type_uuid", type_uuid))
            return []

        def list_all_nodes(self, node_type=None):  # must NOT be used for members
            calls.append(("list_all_nodes", node_type))
            return super().list_all_nodes(node_type)

    hcg = _RecordingHCG(
        members_by_type={
            "type_entity": [{"uuid": "r1", "name": "r", "type": "entity"}],
            "type_car_abc12345": [{"uuid": "c1", "name": "c", "type": "entity"}],
        },
    )
    milvus = _FakeMilvus(
        {
            "r1": {"embedding": [0.1], "embedding_model": "m"},
            "c1": {"embedding": [0.2], "embedding_model": "m"},
        }
    )

    load_type_members(hcg, milvus, "type_entity")  # realm-root drainage pool
    load_type_members(hcg, milvus, "type_car_abc12345")  # minted type

    assert calls == [
        ("get_members_of_type", "type_entity"),
        ("get_members_of_type", "type_car_abc12345"),
    ]
