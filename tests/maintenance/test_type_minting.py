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


def test_mint_creates_type_node_centroid_and_retypes():
    hcg, milvus = FakeHCG(), FakeMilvus()
    name = NameResult(label="concept", description="ideas", confidence=0.8)

    type_uuid = mint_type(
        _cluster(), name, hcg=hcg, milvus=milvus, source_cluster_id="cl1"
    )

    # Unique suffix avoids same-label mints overwriting each other.
    assert type_uuid.startswith("type_concept_")
    assert type_uuid != "type_concept"

    tdef = [n for n in hcg.added_nodes if n["node_type"] == "type_definition"]
    assert len(tdef) == 1
    props = tdef[0]["properties"]
    # Structural typing (#171): node_type="type_definition" + the type_ uuid
    # prefix carry the type-layer signal; the legacy flag is never written.
    assert "is_type_definition" not in props
    # No `ancestors` snapshot -- structure (IS_A edges) is the typing fact (B1).
    assert "ancestors" not in props
    assert props["name_history"][0]["name"] == "concept"
    assert props["name_history"][0]["hermes_confidence"] == 0.8

    # centroid = mean([0,2], [2,0]) = [1, 1]
    centroid, model = milvus.centroids[type_uuid]
    assert centroid == [1.0, 1.0]
    assert model == "all-MiniLM-L6-v2"

    # members retyped via the authoritative `type_uuid` property (membership is
    # the property -- emergence no longer creates instance->type IS_A edges).
    assert ("u1", {"type": "concept", "type_uuid": type_uuid}) in hcg.updated
    assert ("u2", {"type": "concept", "type_uuid": type_uuid}) in hcg.updated
    # No member->type IS_A edge is created (#505).
    assert ("u1", type_uuid, "IS_A") not in hcg.edges
    assert ("u2", type_uuid, "IS_A") not in hcg.edges
    assert not any(src in {"u1", "u2"} for src, _tgt, _rel in hcg.edges)

    # The minted type IS_A its default parent (type_entity), mirroring the
    # seeder IS_A type-hierarchy chain so ancestors match the graph -- this
    # taxonomy edge (type-definition -> parent type-definition) is KEPT (#505).
    assert (type_uuid, "type_entity", "IS_A") in hcg.edges


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
    )

    tdef = [n for n in hcg.added_nodes if n["node_type"] == "type_definition"]
    assert "ancestors" not in tdef[0]["properties"]
    assert (type_uuid, "type_animal", "IS_A") in hcg.edges


def test_same_label_mints_are_distinct_no_overwrite():
    """Two clusters Hermes names identically must not collide on uuid/centroid."""
    hcg, milvus = FakeHCG(), FakeMilvus()
    name = NameResult(label="concept", description="", confidence=0.8)

    uuid_a = mint_type(_cluster(), name, hcg=hcg, milvus=milvus, source_cluster_id="a")
    uuid_b = mint_type(_cluster(), name, hcg=hcg, milvus=milvus, source_cluster_id="b")

    assert uuid_a != uuid_b
    assert uuid_a.startswith("type_concept_")
    assert uuid_b.startswith("type_concept_")
    assert uuid_a in milvus.centroids and uuid_b in milvus.centroids
    # Members are tied to each distinct mint via their `type_uuid` property
    # (overwritten on each retype), not via instance->type IS_A edges (#505).
    assert ("u1", {"type": "concept", "type_uuid": uuid_a}) in hcg.updated
    assert ("u1", {"type": "concept", "type_uuid": uuid_b}) in hcg.updated
    # Only taxonomy IS_A edges (minted type -> parent) exist; no member edges.
    assert (uuid_a, "type_entity", "IS_A") in hcg.edges
    assert (uuid_b, "type_entity", "IS_A") in hcg.edges
    assert not any(src in {"u1", "u2"} for src, _tgt, _rel in hcg.edges)


def test_messy_label_is_slugified_into_identifiers():
    """A multi-word/punctuated Hermes label must not inject spaces into the
    type_uuid or the member `type` string (greptile #149). The human-readable
    label is still preserved in name_history."""
    hcg, milvus = FakeHCG(), FakeMilvus()
    name = NameResult(label="Living Thing!", description="", confidence=0.7)

    type_uuid = mint_type(
        _cluster(), name, hcg=hcg, milvus=milvus, source_cluster_id="cl1"
    )

    assert type_uuid.startswith("type_living_thing_")
    assert " " not in type_uuid and "!" not in type_uuid
    # Members retyped with the slug, never the raw label.
    assert ("u1", {"type": "living_thing", "type_uuid": type_uuid}) in hcg.updated
    assert ("u2", {"type": "living_thing", "type_uuid": type_uuid}) in hcg.updated


def test_mint_does_not_touch_member_is_a_edges():
    """Membership is the `type_uuid` property, so mint creates NO member->type
    IS_A edge and does NOT delete a member's prior IS_A edges. A member split
    out of a parent is excluded from the parent at load time purely because its
    `type_uuid` now points at the new type (#505)."""
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
        _cluster(), name, hcg=hcg, milvus=FakeMilvus(), source_cluster_id="cl"
    )

    # No member IS_A edges are deleted (nothing to clean up) or created.
    assert hcg.deleted == []
    assert not any(src in {"u1", "u2"} for src, _tgt, _rel in hcg.edges)
    # Membership is recorded purely via the `type_uuid` property.
    assert ("u1", {"type": "hammer", "type_uuid": new_uuid}) in hcg.updated
    assert ("u2", {"type": "hammer", "type_uuid": new_uuid}) in hcg.updated
    # Only the taxonomy IS_A (new type -> parent) is created.
    assert (new_uuid, "type_entity", "IS_A") in hcg.edges
    # Human-readable label preserved for display/lineage.
    tdef = [n for n in hcg.added_nodes if n["node_type"] == "type_definition"]
    assert tdef[0]["properties"]["name_history"][0]["name"] == "hammer"


def test_mint_type_only_skips_member_retype_for_super_types():
    """An internal super-type mints with retype_members=False: the type node,
    centroid and IS_A edge are created, but no member is retyped (its members
    belong to the leaf subtypes below it) (#505)."""
    hcg, milvus = FakeHCG(), FakeMilvus()
    name = NameResult(label="mathematics", description="", confidence=0.8)

    type_uuid = mint_type(
        _cluster(),
        name,
        hcg=hcg,
        milvus=milvus,
        source_cluster_id="cl",
        retype_members=False,
    )

    # No members were retyped.
    assert hcg.updated == []
    # But the type node, its centroid and the taxonomy IS_A edge still exist.
    assert any(n["uuid"] == type_uuid for n in hcg.added_nodes)
    assert type_uuid in milvus.centroids
    assert (type_uuid, "type_entity", "IS_A") in hcg.edges


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
    )

    # The add_node call captured the node's properties kwarg.
    tdef = [n for n in hcg.added_nodes if n["node_type"] == "type_definition"]
    assert len(tdef) == 1
    props = tdef[0]["properties"]
    assert "ancestors" not in props
    # The genuine non-structural property (name_history) is still written.
    assert props["name_history"][0]["name"] == "concept"
    # The parent IS_A edge IS the structure and is still created.
    assert (type_uuid, "type_animal", "IS_A") in hcg.edges
