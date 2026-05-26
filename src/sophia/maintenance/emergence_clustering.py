"""Dual-signal, full-membership clustering for emergent type discovery (#505).

Clusters the ENTIRE membership of a type (never outliers): recursively binary-
splits with type_emergence._kmeans_2 while a split (a) leaves a group still above
``variance_threshold`` (i.e. not yet cohesive), (b) improves cohesion by at least
``min_cohesion_improvement``, and (c) keeps both halves >= ``min_cluster_size``.
A group at/below ``variance_threshold`` is a cohesive leaf and is not split.

A returned cluster must also be structurally coherent (members mutually similar
on their neighbor-relation signature) -- the second of the two agreeing signals.
"""

from __future__ import annotations

from sophia.ingestion.type_emergence import _kmeans_2, _mean_vector, _variance
from sophia.maintenance.emergence_types import EmergentCluster, Member
from sophia.maintenance.structural_signature import signature_similarity

_STRUCTURAL_SIM_THRESHOLD = 0.5


def _variance_of(members: list[Member]) -> float:
    vectors = [m.embedding for m in members]
    return _variance(vectors, _mean_vector(vectors))


def _embedding_groups(members: list[Member]) -> list[list[Member]]:
    """Binary embedding split; returns the (1 or 2) non-empty sub-groups.

    _kmeans_2 returns the original embedding objects, so members are recovered
    by object identity.
    """
    if len(members) < 2:
        return [members]
    by_id = {id(m.embedding): m for m in members}
    c0, c1 = _kmeans_2([m.embedding for m in members])
    g0 = [by_id[id(v)] for v in c0]
    g1 = [by_id[id(v)] for v in c1]
    return [g for g in (g0, g1) if g]


def _structurally_coherent(group: list[Member]) -> bool:
    if len(group) < 2:
        return True
    ref = group[0].signature
    return all(
        signature_similarity(ref, m.signature) >= _STRUCTURAL_SIM_THRESHOLD
        for m in group[1:]
    )


def _cohesion_improvement(parent: list[Member], group: list[Member]) -> float:
    pv = _variance_of(parent)
    if pv <= 0:
        return 0.0
    return (pv - _variance_of(group)) / pv


def _recursive_clusters(
    members: list[Member],
    *,
    min_cluster_size: int,
    variance_threshold: float,
    min_cohesion_improvement: float,
) -> list[list[Member]]:
    """Recursively split the full set; stop when a group is cohesive or too small."""
    if _variance_of(members) <= variance_threshold:
        return [members]  # already cohesive -> leaf
    if len(members) < 2 * min_cluster_size:
        return [members]
    groups = _embedding_groups(members)
    if len(groups) < 2 or any(len(g) < min_cluster_size for g in groups):
        return [members]
    if min(_cohesion_improvement(members, g) for g in groups) < min_cohesion_improvement:
        return [members]  # split doesn't meaningfully help -> leaf
    leaves: list[list[Member]] = []
    for g in groups:
        leaves.extend(
            _recursive_clusters(
                g,
                min_cluster_size=min_cluster_size,
                variance_threshold=variance_threshold,
                min_cohesion_improvement=min_cohesion_improvement,
            )
        )
    return leaves


def find_emergent_clusters(
    members: list[Member],
    *,
    min_cluster_size: int,
    variance_threshold: float,
    min_cohesion_improvement: float,
) -> list[EmergentCluster]:
    """Cluster the full membership; return cohesive, structurally-coherent groups.

    No outlier step. Returns [] when the membership is already one cohesive group
    (nothing split out) -- which also makes the job idempotent on tidy types.
    """
    if len(members) < 2 * min_cluster_size:
        return []
    leaves = _recursive_clusters(
        members,
        min_cluster_size=min_cluster_size,
        variance_threshold=variance_threshold,
        min_cohesion_improvement=min_cohesion_improvement,
    )
    if len(leaves) < 2:
        return []  # nothing split out -> no new sub-types
    return [
        EmergentCluster(members=leaf)
        for leaf in leaves
        if len(leaf) >= min_cluster_size and _structurally_coherent(leaf)
    ]
