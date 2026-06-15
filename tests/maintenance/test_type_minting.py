"""Tests for emergent type minting (#505)."""

from __future__ import annotations

from collections import Counter

from sophia.maintenance.emergence_types import EmergentCluster, Member, NameResult
from sophia.maintenance.type_minting import mint_type


class FakeHCG:
    def __init__(self, existing_edges=None):
        self.added_nodes = []
        self.updated = []
        self.edges = []
        self.deleted = []
        self.edge_props = []
        self._existing_edges = existing_edges or {}

    def query_edges_from(self, uuid):
        return self._existing_edges.get(uuid, [])

    def delete_node(self, uuid):
        self.deleted.append(uuid)
        return True

    def delete_edge(self, edge_uuid):
        return self.delete_node(edge_uuid)

    def add_node(self, name, node_type, uuid=None, properties=None, **kw):
        self.added_nodes.append(
            {
                "name": name,
                "node_type": node_type,
                "uuid": uuid,
                "properties": properties or {},
            }
        )
        return uuid

    def update_node(self, uuid, properties=None):
        self.updated.append((uuid, properties))
        return uuid

    def add_edge(self, source_uuid, target_uuid, relation, **kw):
        self.edges.append((source_uuid, target_uuid, relation))
        self.edge_props.append(kw.get("properties"))
        return "edge"


class FakeMilvus:
    def __init__(self):
        self.centroids = {}

    def update_centroid(self, type_uuid, centroid, model):
        self.centroids[type_uuid] = (centroid, model)


def _cluster() -> EmergentCluster:
    return EmergentCluster(
        members=[
            Member(
                uuid="u1",
                name="derivative",
                embedding=[0.0, 2.0],
                signature=Counter(),
                current_type="entity",
                hermes_type_hint="concept",
                neighbors=[],
                model="all-MiniLM-L6-v2",
            ),
            Member(
                uuid="u2",
                name="integral",
                embedding=[2.0, 0.0],
                signature=Counter(),
                current_type="entity",
                hermes_type_hint="concept",
                neighbors=[],
                model="all-MiniLM-L6-v2",
            ),
        ]
    )


def test_mint_creates_type_node_and_centroid_without_touching_members():
    """mint_type creates the type-definition node + centroid + its own IS_A edge
    to the parent, but does NOT touch the members -- membership is the
    instance->type IS_A edge, re-pointed by the draining caller via
    placement.reparent (B2/B3, DESIGN §3)."""
    hcg, milvus = FakeHCG(), FakeMilvus()
    name = NameResult(label="concept", description="ideas", confidence=0.8)

    type_uuid = mint_type(
        _cluster(),
        name,
        hcg=hcg,
        milvus=milvus,
        source_cluster_id="cl1",
        parent_type_uuid="entity-uuid",
        realm="entity",
    )

    # Opaque uuid -- type identity is positional (incoming IS_A), never encoded
    # in the uuid; the name lives on the node, not the id.
    assert not type_uuid.startswith("type_")
    assert "concept" not in type_uuid

    tdef = [n for n in hcg.added_nodes if n["node_type"] == "entity"]
    assert len(tdef) == 1
    props = tdef[0]["properties"]
    assert "is_type_definition" not in props
    assert "ancestors" not in props
    assert props["name_history"][0]["name"] == "concept"
    assert props["name_history"][0]["hermes_confidence"] == 0.8

    # centroid = mean([0,2], [2,0]) = [1, 1]
    centroid, model = milvus.centroids[type_uuid]
    assert centroid == [1.0, 1.0]
    assert model == "all-MiniLM-L6-v2"

    # Members are NOT touched: no `type_uuid`/`type` stamp and no member->type
    # IS_A edge (mint owns only the type node, its centroid and its parent edge).
    assert hcg.updated == []
    assert not any(src in {"u1", "u2"} for src, _tgt, _rel in hcg.edges)

    # The minted type IS_A the parent it was given -- this taxonomy edge
    # (type-definition -> parent type-definition) IS the structure and is KEPT.
    assert (type_uuid, "entity-uuid", "IS_A") in hcg.edges


def test_mint_under_explicit_parent_creates_is_a_edge():
    """An explicit parent_type_uuid drives the IS_A edge to that parent (#505).
    That structural edge IS the typing fact -- no `ancestors` snapshot is
    stored on the node (B1 T3)."""
    hcg, milvus = FakeHCG(), FakeMilvus()
    name = NameResult(label="mammal", description="", confidence=0.8)

    type_uuid = mint_type(
        _cluster(),
        name,
        hcg=hcg,
        milvus=milvus,
        source_cluster_id="cl1",
        parent_type_uuid="type_animal",
        realm="entity",
    )

    tdef = [n for n in hcg.added_nodes if n["node_type"] == "entity"]
    assert "ancestors" not in tdef[0]["properties"]
    assert (type_uuid, "type_animal", "IS_A") in hcg.edges


def test_same_label_mints_are_distinct_no_overwrite():
    """Two clusters Hermes names identically must not collide on uuid/centroid."""
    hcg, milvus = FakeHCG(), FakeMilvus()
    name = NameResult(label="concept", description="", confidence=0.8)

    uuid_a = mint_type(
        _cluster(), name, hcg=hcg, milvus=milvus,
        source_cluster_id="a", parent_type_uuid="entity-uuid", realm="entity",
    )
    uuid_b = mint_type(
        _cluster(), name, hcg=hcg, milvus=milvus,
        source_cluster_id="b", parent_type_uuid="entity-uuid", realm="entity",
    )

    assert uuid_a != uuid_b
    assert not uuid_a.startswith("type_") and not uuid_b.startswith("type_")
    assert uuid_a in milvus.centroids and uuid_b in milvus.centroids
    # Distinct taxonomy IS_A edges (minted type -> parent); the members are never
    # touched by mint -- membership is the instance->type IS_A edge, owned by the
    # draining caller (B2/B3).
    assert (uuid_a, "entity-uuid", "IS_A") in hcg.edges
    assert (uuid_b, "entity-uuid", "IS_A") in hcg.edges
    assert hcg.updated == []
    assert not any(src in {"u1", "u2"} for src, _tgt, _rel in hcg.edges)


def test_messy_label_is_not_encoded_in_the_opaque_uuid():
    """A multi-word/punctuated Hermes label encodes nothing into the uuid: the
    uuid is opaque (uuid4), so spaces/punctuation can never leak into the id.
    The human-readable label is preserved verbatim on the node name and in
    name_history. Members are not retyped here (membership is the IS_A edge)."""
    hcg, milvus = FakeHCG(), FakeMilvus()
    name = NameResult(label="Living Thing!", description="", confidence=0.7)

    type_uuid = mint_type(
        _cluster(), name, hcg=hcg, milvus=milvus,
        source_cluster_id="cl1", parent_type_uuid="entity-uuid", realm="entity",
    )

    assert not type_uuid.startswith("type_")
    assert " " not in type_uuid and "!" not in type_uuid
    assert "living" not in type_uuid.lower()
    # Members are not touched -- no slug/type_uuid stamp.
    assert hcg.updated == []
    tdef = [n for n in hcg.added_nodes if n["node_type"] == "entity"]
    assert tdef[0]["name"] == "Living Thing!"
    assert tdef[0]["properties"]["name_history"][0]["name"] == "Living Thing!"


def test_mint_does_not_touch_member_is_a_edges():
    """Membership is the instance->type IS_A edge, owned by the draining caller.
    mint creates NO member->type IS_A edge, deletes no member edge, and stamps no
    `type_uuid`/`type` property (B2/B3, DESIGN §3)."""
    parent = "type_tool_parent01"
    hcg = FakeHCG(
        existing_edges={
            "u1": [
                {"id": "e_old1", "relation": "IS_A", "source": "u1", "target": parent}
            ],
            "u2": [
                {"id": "e_old2", "relation": "IS_A", "source": "u2", "target": parent}
            ],
        }
    )
    name = NameResult(label="hammer", description="", confidence=0.8)

    new_uuid = mint_type(
        _cluster(), name, hcg=hcg, milvus=FakeMilvus(),
        source_cluster_id="cl", parent_type_uuid=parent, realm="entity",
    )

    # mint touches no member: no edge created/deleted, no property stamped.
    assert hcg.deleted == []
    assert hcg.updated == []
    assert not any(src in {"u1", "u2"} for src, _tgt, _rel in hcg.edges)
    # Only the taxonomy IS_A (new type -> parent) is created.
    assert (new_uuid, parent, "IS_A") in hcg.edges
    # Human-readable label preserved for display/lineage.
    tdef = [n for n in hcg.added_nodes if n["node_type"] == "entity"]
    assert tdef[0]["properties"]["name_history"][0]["name"] == "hammer"


def test_mint_writes_no_ancestors_property():
    """Structure -- the IS_A edges, walked on demand -- is the membership/typing
    fact; there is NO `ancestors` property on a minted type node (DESIGN §3,
    naming-driven-typing B1 T3). The parent IS_A edge, which IS that structural
    fact, must still be created."""
    hcg, milvus = FakeHCG(), FakeMilvus()
    name = NameResult(label="concept", description="ideas", confidence=0.8)

    type_uuid = mint_type(
        _cluster(),
        name,
        hcg=hcg,
        milvus=milvus,
        source_cluster_id="cl1",
        parent_type_uuid="type_animal",
        realm="entity",
    )

    # The add_node call captured the node's properties kwarg.
    tdef = [n for n in hcg.added_nodes if n["node_type"] == "entity"]
    assert len(tdef) == 1
    props = tdef[0]["properties"]
    assert "ancestors" not in props
    # The genuine non-structural property (name_history) is still written.
    assert props["name_history"][0]["name"] == "concept"
    # The parent IS_A edge IS the structure and is still created.
    assert (type_uuid, "type_animal", "IS_A") in hcg.edges


def test_mint_carries_placed_by_on_parent_is_a_edge():
    """When placed_by is supplied, the type->parent IS_A edge carries it as the
    parent-driven traceability tag (DESIGN sec 6, B1 T4b)."""
    hcg, milvus = FakeHCG(), FakeMilvus()
    name = NameResult(label="boat", description="", confidence=0.8)

    type_uuid = mint_type(
        _cluster(),
        name,
        hcg=hcg,
        milvus=milvus,
        source_cluster_id="cl1",
        parent_type_uuid="type_vehicle",
        realm="entity",
        placed_by="parent_resolution",
    )

    assert (type_uuid, "type_vehicle", "IS_A") in hcg.edges
    idx = hcg.edges.index((type_uuid, "type_vehicle", "IS_A"))
    assert hcg.edge_props[idx] == {"placed_by": "parent_resolution"}


def test_mint_omits_placed_by_property_when_absent():
    """The gated-off rollup caller omits placed_by; the IS_A edge then carries no
    placement property (non-breaking)."""
    hcg, milvus = FakeHCG(), FakeMilvus()
    name = NameResult(label="boat", description="", confidence=0.8)

    type_uuid = mint_type(
        _cluster(), name, hcg=hcg, milvus=milvus,
        source_cluster_id="cl1", parent_type_uuid="entity-uuid", realm="entity",
    )

    idx = hcg.edges.index((type_uuid, "entity-uuid", "IS_A"))
    assert hcg.edge_props[idx] is None
