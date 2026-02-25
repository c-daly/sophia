"""Cross-cluster relationship discovery via k-NN.

Finds nodes in other type clusters that are close to a given embedding,
suggesting potential relationships between nodes of different types.
"""

import logging

logger = logging.getLogger(__name__)

# Node type collections to search (excludes TypeCentroid and Edge).
_SEARCHABLE_TYPES = ("Entity", "Concept", "State", "Process")


class RelationshipDiscoverer:
    """Discovers relationship candidates across type clusters."""

    def __init__(self, milvus):
        self._milvus = milvus

    def find_candidates(
        self,
        embedding: list[float],
        own_type: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Find nodes in other type clusters that are close to the embedding.

        For each hit in a foreign cluster, checks whether the hit is closer
        to the query than to its own type centroid. If so, it's a relationship
        candidate; if not, it's a well-placed member of its own cluster.

        Args:
            embedding: The query node's embedding vector.
            own_type: The query node's type collection (e.g. "Entity").
            top_k: Max candidates per collection.

        Returns:
            Filtered candidates sorted by score (closest first).
        """
        candidates = []

        for coll_type in _SEARCHABLE_TYPES:
            if coll_type == own_type:
                continue

            try:
                hits = self._milvus.search_similar(
                    node_type=coll_type,
                    query_embedding=embedding,
                    top_k=top_k,
                )
            except Exception as e:
                logger.debug("Search in %s failed: %s", coll_type, e)
                continue

            for hit in hits:
                # Check if this node is closer to our query than to its
                # own type centroid (meaning it's a boundary node).
                try:
                    nearest_types = self._milvus.find_nearest_types(
                        query_embedding=embedding,
                        top_k=1,
                    )
                    if nearest_types:
                        centroid_distance = nearest_types[0]["score"]
                        query_distance = hit["score"]
                        # Keep only if closer to query than to own centroid
                        if query_distance >= centroid_distance:
                            continue
                except Exception as e:
                    logger.debug(
                        "Centroid check failed for %s: %s", hit["uuid"], e
                    )

                candidates.append(
                    {
                        "uuid": hit["uuid"],
                        "score": hit["score"],
                        "source_type": coll_type,
                    }
                )

        candidates.sort(key=lambda x: x["score"])
        return candidates
