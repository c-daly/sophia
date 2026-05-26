"""Tests for emergence shared dataclasses (#505)."""

from __future__ import annotations

from collections import Counter

from sophia.maintenance.emergence_types import EmergentCluster, Member, NameResult


def _member(uuid: str = "u1") -> Member:
    return Member(
        uuid=uuid,
        name="derivative",
        embedding=[0.1, 0.2],
        signature=Counter({("DEFINED_AS", "concept"): 1}),
        current_type="entity",
        hermes_type_hint="concept",
        neighbors=[
            {"relation": "DEFINED_AS", "neighbor_name": "limit", "neighbor_type": "entity"}
        ],
    )


def test_member_and_cluster_construct():
    m = _member()
    cluster = EmergentCluster(members=[m])
    assert cluster.size == 1
    assert cluster.embeddings == [[0.1, 0.2]]
    assert cluster.members[0].hermes_type_hint == "concept"


def test_name_result():
    r = NameResult(label="concept", description="abstract idea", confidence=0.8)
    assert r.label == "concept" and r.confidence == 0.8
