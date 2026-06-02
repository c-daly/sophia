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
        a = sum(dmat[i][j] for j in same) / len(same) if same else 0.0
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
