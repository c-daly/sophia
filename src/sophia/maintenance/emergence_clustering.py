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

from dataclasses import dataclass, field
from typing import cast

import numpy as np

from sophia.ingestion.type_emergence import _mean_vector, _variance
from sophia.maintenance.emergence_types import EmergentCluster, Member

# Above this membership we sample (seeded) before clustering, to bound the
# O(n^2) distance matrix / O(n^3) agglomeration in this pure-Python impl.
_MAX_CLUSTER_INPUT = 800


def _variance_of(members: list[Member]) -> float:
    vectors = [m.embedding for m in members]
    return float(_variance(vectors, _mean_vector(vectors)))


def _euclidean(a: list[float], b: list[float]) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(diff @ diff))


def _distance_matrix(vectors: list[list[float]]) -> "np.ndarray":
    """Pairwise euclidean distances via the Gram trick (vectorized, #177).

    Returns an (n, n) ndarray; indexes exactly like the previous
    list-of-lists. The naive broadcast (n, n, d) intermediate would need
    hundreds of GB at production scale, so: ||x-y||^2 = |x|^2+|y|^2-2x.y.
    """
    x = np.asarray(vectors, dtype=np.float64)
    sq = np.einsum("ij,ij->i", x, x)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
    np.maximum(d2, 0.0, out=d2)  # clamp float negatives on the diagonal
    return cast("np.ndarray", np.sqrt(d2))


def _silhouette(dmat: "np.ndarray", labels: list[int]) -> float:
    """Mean silhouette over a precomputed distance matrix; -1 if < 2 clusters.

    Vectorized (#177): per-cluster mean distances come from one D @ onehot
    product. Semantics preserved exactly, including the singleton rule:
    silhouette of a lone point is 0, not (b - 0)/b = 1 -- scoring it 1.0
    would bias k-selection toward partitions full of singletons (which are
    then dropped by min_cluster_size, yielding "no clusters found"). See
    #505 review.
    """
    lab = np.asarray(labels)
    uniq = np.unique(lab)
    if uniq.size < 2:
        return -1.0
    d = np.asarray(dmat, dtype=np.float64)
    n = lab.size
    onehot = (lab[:, None] == uniq[None, :]).astype(np.float64)  # (n, k)
    counts = onehot.sum(axis=0)  # (k,)
    sums = d @ onehot  # (n, k): total distance from i to each cluster
    own_idx = np.searchsorted(uniq, lab)
    own_count = counts[own_idx]
    own_sum = sums[np.arange(n), own_idx]
    # a(i): mean distance to OWN cluster, self excluded.
    with np.errstate(invalid="ignore", divide="ignore"):
        a = own_sum / np.maximum(own_count - 1.0, 1.0)
        means = sums / counts[None, :]  # mean distance to each cluster
    means[np.arange(n), own_idx] = np.inf
    b = means.min(axis=1)
    denom = np.maximum(a, b)
    s = np.where(denom > 0, (b - a) / np.where(denom > 0, denom, 1.0), 0.0)
    s = np.where(own_count <= 1, 0.0, s)  # singleton rule
    return float(s.mean())


def _agglomerative_partitions(
    dmat: "np.ndarray", k_min: int, k_max: int
) -> dict[int, list[int]]:
    """Average-linkage agglomeration; return {k: labels} for k in [k_min, k_max].

    Vectorized Lance-Williams (#177): the working distance matrix is updated
    in place; the merge argmin is one masked scan instead of an O(n^2) dict
    sweep per step. Same linkage formula, same tie behavior (first minimum
    in row-major order, matching the previous insertion-ordered dict scan).
    """
    d = np.asarray(dmat, dtype=np.float64).copy()
    n = d.shape[0]
    np.fill_diagonal(d, np.inf)
    sizes = np.ones(n)
    cluster_of = np.arange(n)  # row index -> current cluster row
    alive = np.ones(n, dtype=bool)
    partitions: dict[int, list[int]] = {}
    for _step in range(n - 1):
        masked = np.where(alive[:, None] & alive[None, :], d, np.inf)
        flat = int(np.argmin(masked))
        a, b = divmod(flat, n)
        if a > b:
            a, b = b, a
        # Lance-Williams average-linkage update onto row/col a.
        wa, wb = sizes[a], sizes[b]
        new_row = (wa * d[a] + wb * d[b]) / (wa + wb)
        d[a, :] = new_row
        d[:, a] = new_row
        d[a, a] = np.inf
        sizes[a] = wa + wb
        alive[b] = False
        cluster_of[cluster_of == b] = a
        k = int(alive.sum())
        if k_min <= k <= k_max:
            live = {row: ci for ci, row in enumerate(np.flatnonzero(alive))}
            partitions[k] = [live[row] for row in cluster_of]
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


def find_emergent_hierarchy_threshold(
    members: list[Member],
    *,
    sim_threshold: float,
    min_supercluster_size: int,
) -> list[HierarchyNode]:
    """Neighborhood-frame super-clustering for the type layer (sophia #220).

    Groups type-centroids by cosine-threshold connected components instead of
    silhouette-argmax over agglomeration, which on the type layer collapses to one
    diffuse blob and selects no families. Each connected component of
    >= ``min_supercluster_size`` types becomes ONE super-type: an internal node
    over per-type leaf nodes, so the existing reparent path mints the super-type
    and re-parents the member types under it. Sub-threshold types carry nothing
    up. Returns [] when no component reaches ``min_supercluster_size``.
    """
    n = len(members)
    if n < 2:
        return []
    mat = np.asarray([m.embedding for m in members], dtype=np.float64)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    sim = mat @ mat.T
    adj = sim >= sim_threshold
    np.fill_diagonal(adj, False)

    seen: set[int] = set()
    nodes: list[HierarchyNode] = []
    for start in range(n):
        if start in seen:
            continue
        # Mark each node seen WHEN pushed (not when popped) so a node in a dense
        # component is never queued more than once.
        seen.add(start)
        stack, comp = [start], [start]
        while stack:
            x = stack.pop()
            for j in np.flatnonzero(adj[x]):
                j = int(j)
                if j not in seen:
                    seen.add(j)
                    comp.append(j)
                    stack.append(j)
        if len(comp) < min_supercluster_size:
            continue
        comp_members = [members[j] for j in comp]
        children = [
            HierarchyNode(members=[m], centroid=m.embedding, children=[])
            for m in comp_members
        ]
        nodes.append(
            HierarchyNode(
                members=comp_members,
                centroid=_centroid([m.embedding for m in comp_members]),
                children=children,
            )
        )
    return nodes
