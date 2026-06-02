"""Clustering-algorithm candidates for the type-emergence experiment (#505).

Each agent implements the experiment ``AgentDefinition`` contract --
``process(embeddings) -> list[list[int]]`` (clusters as lists of member
indices) -- so candidate algorithms can be compared head-to-head on the
experiment harness. Pure numpy; cosine geometry (embeddings are unit vectors).
"""

from __future__ import annotations

import numpy as np


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def _cosine_kmeans(x: np.ndarray, k: int, seed: int, iters: int = 30) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centroids = x[rng.choice(len(x), k, replace=False)].copy()
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        new = np.argmax(x @ centroids.T, axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            pts = x[labels == c]
            if len(pts):
                centroids[c] = pts.mean(0)
                centroids[c] /= np.linalg.norm(centroids[c]) + 1e-9
    return labels


def silhouette_cosine(x: np.ndarray, labels: np.ndarray) -> float:
    """Mean cosine silhouette; -1.0 if fewer than 2 populated clusters."""
    uniq = set(int(c) for c in labels)
    if len(uniq) < 2:
        return -1.0
    dist = 1.0 - (x @ x.T)
    np.fill_diagonal(dist, 0.0)
    scores = []
    for i in range(len(x)):
        same = labels == labels[i]
        same[i] = False
        a = dist[i, same].mean() if same.any() else 0.0
        b = min(dist[i, labels == c].mean() for c in uniq if c != labels[i])
        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return float(np.mean(scores))


def _clusters_from_labels(labels: np.ndarray, min_size: int) -> list[list[int]]:
    out: list[list[int]] = []
    for c in sorted(set(int(v) for v in labels)):
        idx = [i for i in range(len(labels)) if labels[i] == c]
        if len(idx) >= min_size:
            out.append(idx)
    return out


def _best_k_labels(x: np.ndarray, min_size: int, seed: int) -> np.ndarray | None:
    """Cosine k-means over a range of k; pick k by best cosine silhouette."""
    hi = max(2, len(x) // min_size)
    best = None
    for k in range(2, hi + 1):
        labels = _cosine_kmeans(x, k, seed)
        score = silhouette_cosine(x, labels)
        if best is None or score > best[0]:
            best = (score, labels)
    return best[1] if best else None


class CosineKMeansAgent:
    """Flat cosine k-means; k auto-selected by silhouette."""

    def __init__(self, min_cluster_size: int = 3, seed: int = 0) -> None:
        self.min_cluster_size = min_cluster_size
        self.seed = seed

    def process(self, embeddings: list[list[float]]) -> list[list[int]]:
        x = _normalize(np.asarray(embeddings, dtype=float))
        labels = _best_k_labels(x, self.min_cluster_size, self.seed)
        if labels is None:
            return []
        return _clusters_from_labels(labels, self.min_cluster_size)


class ProductionAgglomerativeAgent:
    """The SHIPPED clustering: maintenance.emergence_clustering.find_emergent_clusters.

    Wraps embeddings as embedding-only Members and calls the production function,
    so the sweep benchmarks exactly what ships (agglomerative + silhouette).
    """

    def __init__(
        self, min_cluster_size: int = 3, variance_threshold: float = 0.6, **_: object
    ) -> None:
        self.min_cluster_size = min_cluster_size
        self.variance_threshold = variance_threshold

    def process(self, embeddings: list[list[float]]) -> list[list[int]]:
        from sophia.maintenance.emergence_clustering import find_emergent_clusters
        from sophia.maintenance.emergence_types import Member

        members = [
            Member(
                uuid=str(i),
                name=str(i),
                embedding=list(e),
                signature=set(),
                current_type="entity",
                hermes_type_hint=None,
                neighbors=[],
                model=None,
            )
            for i, e in enumerate(embeddings)
        ]
        clusters = find_emergent_clusters(
            members,
            min_cluster_size=self.min_cluster_size,
            variance_threshold=self.variance_threshold,
            min_cohesion_improvement=0.0,
        )
        return [[int(m.name) for m in c.members] for c in clusters]
