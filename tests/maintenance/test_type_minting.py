"""Tests for emergent type minting (#505)."""

from __future__ import annotations

from collections import Counter

from sophia.maintenance.emergence_types import EmergentCluster, Member, NameResult
from sophia.maintenance.type_minting import mint_type


class FakeHCG:
    def __init__(self):
        self.added_nodes = []
        self.updated = []
        self.edges = []

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
    assert props["is_type_definition"] is True
    assert props["ancestors"] == ["root"]
    assert props["name_history"][0]["name"] == "concept"
    assert props["name_history"][0]["hermes_confidence"] == 0.8

    # centroid = mean([0,2], [2,0]) = [1, 1]
    centroid, model = milvus.centroids[type_uuid]
    assert centroid == [1.0, 1.0]
    assert model == "all-MiniLM-L6-v2"

    # members retyped + IS_A edges to the new (unique) type uuid
    assert ("u1", {"type": "concept", "type_uuid": type_uuid}) in hcg.updated
    assert ("u2", {"type": "concept", "type_uuid": type_uuid}) in hcg.updated
    assert ("u1", type_uuid, "IS_A") in hcg.edges
    assert ("u2", type_uuid, "IS_A") in hcg.edges


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
    assert ("u1", uuid_a, "IS_A") in hcg.edges
    assert ("u1", uuid_b, "IS_A") in hcg.edges


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
    # Human-readable label preserved for display/lineage.
    tdef = [n for n in hcg.added_nodes if n["node_type"] == "type_definition"]
    assert tdef[0]["properties"]["name_history"][0]["name"] == "Living Thing!"
