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


def _resolve_group(vectors: list[list[float]], pool: dict[int, Member]) -> list[Member]:
    """Map cluster vectors back to members, consuming each member at most once.

    ``_kmeans_2`` returns the original embedding objects, but we do not rely on
    object identity (k-means could copy a vector, and two members may share an
    equal embedding). Each returned vector claims the first still-unconsumed
    member whose embedding is identical (by identity first, else by value).
    """
    group: list[Member] = []
    for vec in vectors:
        match_key = next(
            (k for k, m in pool.items() if m.embedding is vec),
            None,
        )
        if match_key is None:
            match_key = next(
                (k for k, m in pool.items() if m.embedding == vec),
                None,
            )
        if match_key is None:
            continue
        group.append(pool.pop(match_key))
    return group


def _embedding_groups(members: list[Member]) -> list[list[Member]]:
    """Binary embedding split; returns the (1 or 2) non-empty sub-groups."""
    if len(members) < 2:
        return [members]
    c0, c1 = _kmeans_2([m.embedding for m in members])
    pool = dict(enumerate(members))
    g0 = _resolve_group(c0, pool)
    g1 = _resolve_group(c1, pool)
    # Any members not claimed (e.g. duplicate-value ambiguity) stay grouped with g0.
    g0.extend(pool.values())
    return [g for g in (g0, g1) if g]


def _structurally_coherent(group: list[Member]) -> bool:
    """Whether the group's members agree on their neighbor-relation signature.

    Robust to edge-less members: a node with an empty signature carries no
    structural signal, so it neither anchors the comparison nor disqualifies the
    cluster. We compare the members that *do* have a signature against each
    other; a cluster where every member is edge-less is treated as coherent
    (uniform, no contradicting structure) rather than rejected on index 0.
    """
    if len(group) < 2:
        return True
    with_sig = [m for m in group if m.signature]
    if len(with_sig) < 2:
        # 0 or 1 members carry structure -- nothing to contradict.
        return True
    ref = with_sig[0].signature
    return all(
        signature_similarity(ref, m.signature) >= _STRUCTURAL_SIM_THRESHOLD
        for m in with_sig[1:]
    )


def _split_improvement(parent: list[Member], groups: list[list[Member]]) -> float:
    """Fractional reduction in *weighted* within-group variance from the split.

    Uses the size-weighted average of the child variances (not the worst child),
    so a split that tightens the membership overall is accepted even if one
    child is only marginally more cohesive than the parent::

        (parent_var - sum(len(g) * var(g)) / len(members)) / parent_var

    Returns 0.0 when the parent has no variance (nothing to improve).
    """
    pv = _variance_of(parent)
    if pv <= 0:
        return 0.0
    n = sum(len(g) for g in groups)
    if n == 0:
        return 0.0
    weighted_child_var = sum(len(g) * _variance_of(g) for g in groups) / n
    return (pv - weighted_child_var) / pv


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
    if _split_improvement(members, groups) < min_cohesion_improvement:
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
