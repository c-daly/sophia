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

    assert type_uuid == "type_concept"

    tdef = [n for n in hcg.added_nodes if n["node_type"] == "type_definition"]
    assert len(tdef) == 1
    props = tdef[0]["properties"]
    assert props["is_type_definition"] is True
    assert props["ancestors"] == ["root"]
    assert props["name_history"][0]["name"] == "concept"
    assert props["name_history"][0]["hermes_confidence"] == 0.8

    # centroid = mean([0,2], [2,0]) = [1, 1]
    centroid, model = milvus.centroids["type_concept"]
    assert centroid == [1.0, 1.0]
    assert model == "all-MiniLM-L6-v2"

    # members retyped + IS_A edges to the new type
    assert ("u1", {"type": "concept"}) in hcg.updated
    assert ("u2", {"type": "concept"}) in hcg.updated
    assert ("u1", "type_concept", "IS_A") in hcg.edges
    assert ("u2", "type_concept", "IS_A") in hcg.edges
