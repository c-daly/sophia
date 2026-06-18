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


def test_empty_index0_signature_does_not_break_coherent_cluster():
    # Member 0 is edge-less (empty signature). The cluster is still coherent
    # because the members that DO carry structure agree -- index 0 must not
    # disqualify the whole cluster.
    phys = [_m("p0", [0.0, 0.0], ("", ""))]
    phys[0].signature = Counter()  # edge-less
    phys += [
        _m(f"p{i}", [0.0 + i * 0.01, 0.0], ("MOVED_TO", "location"))
        for i in range(1, 4)
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
    groups = {frozenset(m.uuid for m in c.members) for c in clusters}
    assert frozenset(f"p{i}" for i in range(4)) in groups


def test_duplicate_value_embeddings_do_not_crash():
    # Two members sharing an identical embedding VALUE (and even object) must
    # not raise -- value/identity matching consumes each member once.
    shared = [0.0, 0.0]
    phys = [_m(f"p{i}", shared, ("MOVED_TO", "location")) for i in range(4)]
    concept = [_m(f"c{i}", [9.0, 9.0], ("DEFINED_AS", "concept")) for i in range(4)]
    clusters = find_emergent_clusters(
        phys + concept,
        min_cluster_size=MIN,
        variance_threshold=VT,
        min_cohesion_improvement=IMPROVE,
    )
    # No KeyError; every input member is accounted for exactly once.
    out = [m.uuid for c in clusters for m in c.members]
    assert len(out) == len(set(out))


def test_real_scale_unit_embeddings_cluster():
    """Regression for #505: near-unit high-dim embeddings still cluster.

    The prior recursive-binary-split returned 0 clusters on real OpenAI
    embeddings -- a binary split reduces absolute variance only marginally, so
    the cohesion-improvement gate was never met. Agglomerative + silhouette
    recovers the latent groups. Three orthogonal-ish unit groups -> 3 clusters.
    """
    import math

    def unit(v: list[float]) -> list[float]:
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    dim = 8
    members = []
    for g in range(3):
        for i in range(4):
            v = [0.0] * dim
            v[g] = 1.0
            v[(g + 4) % dim] = 0.12 * (i + 1)  # small within-group spread
            members.append(_m(f"g{g}_{i}", unit(v), (f"R{g}", "t")))

    clusters = find_emergent_clusters(
        members,
        min_cluster_size=MIN,
        variance_threshold=0.6,
        min_cohesion_improvement=IMPROVE,
    )
    assert len(clusters) == 3
    for c in clusters:
        groups = {m.uuid.split("_")[0] for m in c.members}
        assert len(groups) == 1  # each cluster is exactly one group


def test_threshold_hierarchy_respects_supercluster_floor():
    """The threshold selector's component floor is rollup_min_supercluster_size
    (the super-type minting control), not the entity-leaf floor. A 2-type
    component mints a super-type at floor 2 but not at floor 3. (#220 review)"""
    from sophia.maintenance.emergence_clustering import (
        find_emergent_hierarchy_threshold,
    )

    # a,b are near-identical (cosine ~1.0); c,d point elsewhere -> one 2-type
    # component above the 0.7 threshold.
    members = [
        _m("a", [1.0, 0.0], ("R", "x")),
        _m("b", [0.99, 0.02], ("R", "x")),
        _m("c", [0.0, 1.0], ("R", "y")),
        _m("d", [-1.0, 0.0], ("R", "z")),
    ]
    at_2 = find_emergent_hierarchy_threshold(
        members, sim_threshold=0.7, min_supercluster_size=2
    )
    assert len(at_2) == 1
    assert {m.uuid for m in at_2[0].members} == {"a", "b"}

    # Raising the super-type floor to 3 silently used to do nothing (it consulted
    # the leaf floor); now it is honored -> the 2-type component is too small.
    at_3 = find_emergent_hierarchy_threshold(
        members, sim_threshold=0.7, min_supercluster_size=3
    )
    assert at_3 == []
