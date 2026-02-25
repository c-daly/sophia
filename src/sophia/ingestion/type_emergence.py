"""Type emergence detection via variance monitoring and k-means.

Monitors type centroid variance. When variance exceeds a threshold,
runs k-means(k=2) to detect whether the type should split into
two sub-types.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

MAX_KMEANS_ITERATIONS = 20


@dataclass
class SplitCandidate:
    """Result of emergence detection when a split is warranted."""

    sub_clusters: list[dict] = field(default_factory=list)
    should_split: bool = False


def _euclidean_distance_sq(a: list[float], b: list[float]) -> float:
    """Squared Euclidean distance between two vectors."""
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    """Compute the element-wise mean of a list of vectors."""
    dim = len(vectors[0])
    n = len(vectors)
    return [sum(v[d] for v in vectors) / n for d in range(dim)]


def _variance(vectors: list[list[float]], centroid: list[float]) -> float:
    """Mean squared distance from centroid."""
    if not vectors:
        return 0.0
    return sum(_euclidean_distance_sq(v, centroid) for v in vectors) / len(vectors)


def _kmeans_2(embeddings: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """Simple k-means with k=2.

    Returns two clusters as lists of embeddings.
    """
    if len(embeddings) < 2:
        return embeddings, []

    # Initialize: first point and the point farthest from it
    c0 = list(embeddings[0])
    farthest_idx = max(
        range(1, len(embeddings)),
        key=lambda i: _euclidean_distance_sq(embeddings[i], c0),
    )
    c1 = list(embeddings[farthest_idx])

    cluster_0: list[list[float]] = []
    cluster_1: list[list[float]] = []

    for _ in range(MAX_KMEANS_ITERATIONS):
        cluster_0 = []
        cluster_1 = []

        # Assign points to nearest centroid
        for emb in embeddings:
            d0 = _euclidean_distance_sq(emb, c0)
            d1 = _euclidean_distance_sq(emb, c1)
            if d0 <= d1:
                cluster_0.append(emb)
            else:
                cluster_1.append(emb)

        # Handle empty clusters
        if not cluster_0 or not cluster_1:
            return cluster_0 or cluster_1, []

        # Recompute centroids
        new_c0 = _mean_vector(cluster_0)
        new_c1 = _mean_vector(cluster_1)

        # Check convergence
        if new_c0 == c0 and new_c1 == c1:
            break

        c0 = new_c0
        c1 = new_c1

    return cluster_0, cluster_1


class TypeEmergenceDetector:
    """Detects when a type should split into sub-types."""

    def __init__(self, milvus, hcg, variance_threshold: float = 0.5):
        self._milvus = milvus
        self._hcg = hcg
        self._variance_threshold = variance_threshold

    def check_type(self, type_uuid: str) -> SplitCandidate | None:
        """Check whether a type's variance warrants a split.

        Args:
            type_uuid: The UUID of the type_definition node.

        Returns:
            SplitCandidate if a split is detected, None otherwise.
        """
        type_node = self._hcg.get_node(type_uuid)
        if not type_node:
            return None

        props = type_node.get("properties", {})
        variance = props.get("centroid_variance", 0.0)

        if variance < self._variance_threshold:
            return None

        # Variance is high — fetch all member embeddings and run k-means
        embeddings = self._milvus.get_all_embeddings(type_uuid)
        if not embeddings or len(embeddings) < 4:
            return None

        cluster_a, cluster_b = _kmeans_2(embeddings)

        if not cluster_a or not cluster_b:
            return None

        # Check that both sub-clusters have meaningfully lower variance
        centroid_a = _mean_vector(cluster_a)
        centroid_b = _mean_vector(cluster_b)
        var_a = _variance(cluster_a, centroid_a)
        var_b = _variance(cluster_b, centroid_b)

        # Both sub-cluster variances should be less than the parent's
        if var_a < variance and var_b < variance:
            return SplitCandidate(
                sub_clusters=[
                    {"centroid": centroid_a, "member_count": len(cluster_a)},
                    {"centroid": centroid_b, "member_count": len(cluster_b)},
                ],
                should_split=True,
            )

        return None
