"""Flat agglomerative clustering for emergent type discovery (#505).

Replaces the prior recursive binary-split + absolute-variance gating, which did
not work on real high-dimensional embeddings: a single binary split reduces
absolute within-cluster variance only marginally (curse of dimensionality), so
the cohesion-improvement gate was never met and *no* clusters were ever found
(verified: 235 diverse entities -> 0 clusters). This uses average-linkage
agglomerative clustering with the cut chosen by silhouette -- scale-invariant,
so it recovers domain clusters from real embeddings.

Validated empirically via ``sophia.experiments.run_cluster_sweep``: agglomerative
scored purity 0.97 / ARI 0.45 against domain ground truth, vs the prior
algorithm's 0 clusters on the same data.

Distance is Euclidean; on unit-norm embeddings (OpenAI) this is monotonic with
cosine distance, so the clustering matches cosine geometry while remaining valid
for the small non-unit fixtures used in tests.

Structural coherence (neighbor-relation signature agreement) is now ADVISORY,
not a hard veto -- as a veto it rejected every real cluster (#505 review).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from sophia.ingestion.type_emergence import _mean_vector, _variance
from sophia.maintenance.emergence_types import EmergentCluster, Member

# Above this membership we sample (seeded) before clustering, to bound the
# O(n^2) distance matrix / O(n^3) agglomeration in this pure-Python impl.
_MAX_CLUSTER_INPUT = 800


def _variance_of(members: list[Member]) -> float:
    vectors = [m.embedding for m in members]
    return float(_variance(vectors, _mean_vector(vectors)))


def _euclidean(a: list[float], b: list[float]) -> float:
    return float(sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5)


def _distance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    n = len(vectors)
    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = _euclidean(vectors[i], vectors[j])
            d[i][j] = d[j][i] = dist
    return d


def _silhouette(dmat: list[list[float]], labels: list[int]) -> float:
    """Mean silhouette over a precomputed distance matrix; -1 if < 2 clusters."""
    uniq = sorted(set(labels))
    if len(uniq) < 2:
        return -1.0
    members_by_label = {c: [i for i, v in enumerate(labels) if v == c] for c in uniq}
    scores = []
    for i, li in enumerate(labels):
        same = [j for j in members_by_label[li] if j != i]
        if not same:
            # Singleton cluster: silhouette is defined as 0, not (b - 0)/b = 1.
            # Scoring a lone point 1.0 would bias k-selection toward partitions
            # full of singletons (which are then dropped by min_cluster_size,
            # yielding "no clusters found"). See #505 review.
            scores.append(0.0)
            continue
        a = sum(dmat[i][j] for j in same) / len(same)
        b = min(
            sum(dmat[i][j] for j in members_by_label[c]) / len(members_by_label[c])
            for c in uniq
            if c != li
        )
        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return sum(scores) / len(labels)


def _agglomerative_partitions(
    dmat: list[list[float]], k_min: int, k_max: int
) -> dict[int, list[int]]:
    """Average-linkage agglomeration; return {k: labels} for k in [k_min, k_max]."""
    n = len(dmat)
    members = {i: [i] for i in range(n)}
    sizes = {i: 1 for i in range(n)}
    dist = {(i, j): dmat[i][j] for i in range(n) for j in range(i + 1, n)}

    def key(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    active = set(range(n))
    next_id = n
    partitions: dict[int, list[int]] = {}
    while len(active) > 1:
        best_pair = None
        best_d = None
        for (a, b), d in dist.items():
            if a in active and b in active and (best_d is None or d < best_d):
                best_d, best_pair = d, (a, b)
        a, b = best_pair  # type: ignore[misc]
        new = next_id
        next_id += 1
        members[new] = members[a] + members[b]
        sizes[new] = sizes[a] + sizes[b]
        for c in active:
            if c in (a, b):
                continue
            dac = dist.get(key(a, c), 0.0)
            dbc = dist.get(key(b, c), 0.0)
            dist[key(new, c)] = (sizes[a] * dac + sizes[b] * dbc) / sizes[new]
        active.discard(a)
        active.discard(b)
        active.add(new)
        k = len(active)
        if k_min <= k <= k_max:
            labels = [0] * n
            for ci, cid in enumerate(active):
                for m in members[cid]:
                    labels[m] = ci
            partitions[k] = labels
    return partitions


def find_emergent_clusters(
    members: list[Member],
    *,
    min_cluster_size: int,
    variance_threshold: float,
    min_cohesion_improvement: float = 0.0,
) -> list[EmergentCluster]:
    """Cluster the full membership into cohesive sub-groups via agglomeration.

    Returns [] when the type is already cohesive (variance at/below
    ``variance_threshold`` -- a junk-drawer pre-filter) or when fewer than two
    sub-clusters of >= ``min_cluster_size`` emerge. ``min_cohesion_improvement``
    is accepted for backward compatibility but unused (the cut is silhouette-
    chosen, not gated on a fixed improvement).
    """
    n = len(members)
    if n < 2 * min_cluster_size:
        return []
    if _variance_of(members) <= variance_threshold:
        return []  # cohesive type -> nothing to split out

    work = members
    if n > _MAX_CLUSTER_INPUT:
        import random

        work = random.Random(0).sample(members, _MAX_CLUSTER_INPUT)

    vectors = [m.embedding for m in work]
    dmat = _distance_matrix(vectors)
    k_max = max(2, len(work) // min_cluster_size)
    partitions = _agglomerative_partitions(dmat, 2, k_max)
    if not partitions:
        return []
    _, labels = max(partitions.items(), key=lambda kv: _silhouette(dmat, kv[1]))

    groups: dict[int, list[Member]] = {}
    for m, lab in zip(work, labels):
        groups.setdefault(lab, []).append(m)
    clusters = [
        EmergentCluster(members=g)
        for g in groups.values()
        if len(g) >= min_cluster_size
    ]
    return clusters if len(clusters) >= 2 else []


@dataclass
class HierarchyNode:
    """A node in the emergent type hierarchy.

    Leaf nodes (``children == []``) are the fine clusters from
    :func:`find_emergent_clusters`; internal nodes group child nodes discovered
    by running the SAME clustering on the children's centroids (treating each
    cluster as a point). This rolls fine types up into super-types -- e.g.
    "linear algebra" + "calculus" -> a "mathematics" super-type.
    """

    members: list[Member]  # all leaf members beneath this node
    centroid: list[float]
    children: list["HierarchyNode"] = field(default_factory=list)


def _centroid(vectors: list[list[float]]) -> list[float]:
    n = len(vectors)
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            acc[i] += x
    return [x / n for x in acc]


def find_emergent_hierarchy(
    members: list[Member],
    *,
    min_cluster_size: int,
    variance_threshold: float,
    min_supercluster_size: int = 2,
    max_depth: int = 4,
) -> list[HierarchyNode]:
    """Discover a multi-level type hierarchy from the junk-drawer membership.

    Level 0 is the fine clusters from :func:`find_emergent_clusters`. Each higher
    level re-runs the clustering on the previous level's centroids -- the same
    algorithm, with clusters as points -- so related fine types roll up into
    super-types. A node that doesn't join any super-cluster carries up unchanged.
    Stops when the level no longer consolidates, there are too few nodes to form
    a super-cluster, or ``max_depth`` is reached. Returns the root nodes (``[]``
    when there's no structure to organise).
    """
    leaf_clusters = find_emergent_clusters(
        members,
        min_cluster_size=min_cluster_size,
        variance_threshold=variance_threshold,
    )
    if len(leaf_clusters) < 2:
        return []

    nodes = [
        HierarchyNode(
            members=list(c.members),
            centroid=_centroid([m.embedding for m in c.members]),
        )
        for c in leaf_clusters
    ]

    depth = 1
    while depth < max_depth and len(nodes) >= 2 * min_supercluster_size:
        synthetic = [
            Member(
                uuid=str(i),
                name=str(i),
                embedding=node.centroid,
                signature=Counter(),
                current_type="type",
                hermes_type_hint=None,
                neighbors=[],
                model=None,
            )
            for i, node in enumerate(nodes)
        ]
        # No junk-drawer pre-filter at the super level (centroids always vary).
        groups = find_emergent_clusters(
            synthetic, min_cluster_size=min_supercluster_size, variance_threshold=0.0
        )
        if len(groups) < 2:
            break
        grouped = [sorted(int(m.name) for m in g.members) for g in groups]
        used = {i for g in grouped for i in g}
        new_nodes: list[HierarchyNode] = []
        for g in grouped:
            children = [nodes[i] for i in g]
            child_members = [m for ch in children for m in ch.members]
            new_nodes.append(
                HierarchyNode(
                    members=child_members,
                    centroid=_centroid([m.embedding for m in child_members]),
                    children=children,
                )
            )
        new_nodes.extend(nodes[i] for i in range(len(nodes)) if i not in used)
        if len(new_nodes) >= len(nodes):
            break  # no consolidation this level -> done
        nodes = new_nodes
        depth += 1
    return nodes
