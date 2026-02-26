"""Cross-cluster relationship discovery via k-NN.

Finds nodes in other type clusters that are close to a given embedding,
suggesting potential relationships between nodes of different types.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Node type collections to search (excludes TypeCentroid and Edge).
_SEARCHABLE_TYPES = ("Entity", "Concept", "State", "Process")


class RelationshipDiscoverer:
    """Discovers relationship candidates across type clusters."""

    def __init__(self, milvus: Any) -> None:
        self._milvus = milvus

    def find_candidates(
        self,
        embedding: list[float],
        own_type: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Find nodes in other type clusters that are close to the embedding.

        Searches all type collections except *own_type* and returns the
        closest hits as relationship candidates, sorted by distance.

        Note: A future refinement could filter hits by comparing each hit's
        distance to its own centroid vs. the query, but that requires
        fetching hit embeddings from Milvus (not returned by search).

        Args:
            embedding: The query node's embedding vector.
            own_type: The query node's type collection (e.g. "Entity").
            top_k: Max candidates per collection.

        Returns:
            Candidates sorted by score (closest first).
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
                candidates.append(
                    {
                        "uuid": hit["uuid"],
                        "score": hit["score"],
                        "source_type": coll_type,
                    }
                )

        candidates.sort(key=lambda x: x["score"])
        return candidates
