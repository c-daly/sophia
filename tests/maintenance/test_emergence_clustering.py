"""Tests for dual-signal full-membership clustering (#505)."""

from __future__ import annotations

from collections import Counter

from sophia.maintenance.emergence_clustering import find_emergent_clusters
from sophia.maintenance.emergence_types import Member

VT = 1.0  # variance_threshold for these well-separated fixtures
MIN = 3
IMPROVE = 0.1


def _m(uuid: str, vec: list[float], sig_key: tuple[str, str]) -> Member:
    return Member(
        uuid=uuid,
        name=uuid,
        embedding=vec,
        signature=Counter({sig_key: 1}),
        current_type="entity",
        hermes_type_hint=None,
        neighbors=[],
    )


def test_two_coherent_groups_split():
    phys = [
        _m(f"p{i}", [0.0 + i * 0.01, 0.0], ("MOVED_TO", "location")) for i in range(4)
    ]
    concept = [
        _m(f"c{i}", [9.0 + i * 0.01, 9.0], ("DEFINED_AS", "concept")) for i in range(4)
    ]
    clusters = find_emergent_clusters(
        phys + concept,
        min_cluster_size=MIN,
        variance_threshold=VT,
        min_cohesion_improvement=IMPROVE,
    )
    assert len(clusters) == 2
    groups = {frozenset(m.uuid for m in c.members) for c in clusters}
    assert frozenset(f"p{i}" for i in range(4)) in groups
    assert frozenset(f"c{i}" for i in range(4)) in groups


def test_three_groups_recurse_to_three():
    members = []
    for (bx, by), sig in [
        ((0.0, 0.0), ("A", "t")),
        ((9.0, 9.0), ("B", "t")),
        ((0.0, 18.0), ("C", "t")),
    ]:
        members += [_m(f"{sig[0]}{i}", [bx + i * 0.01, by], sig) for i in range(3)]
    clusters = find_emergent_clusters(
        members,
        min_cluster_size=MIN,
        variance_threshold=VT,
        min_cohesion_improvement=IMPROVE,
    )
    assert len(clusters) == 3


def test_single_cohesive_group_returns_empty():
    members = [_m(f"a{i}", [0.0 + i * 0.01, 0.0], ("A", "t")) for i in range(6)]
    clusters = find_emergent_clusters(
        members,
        min_cluster_size=MIN,
        variance_threshold=VT,
        min_cohesion_improvement=IMPROVE,
    )
    assert clusters == []


def test_below_min_size_not_returned():
    members = [_m(f"p{i}", [0.0, 0.0], ("MOVED_TO", "location")) for i in range(2)]
    clusters = find_emergent_clusters(
        members,
        min_cluster_size=MIN,
        variance_threshold=VT,
        min_cohesion_improvement=IMPROVE,
    )
    assert clusters == []
