"""Embedding-based type classifier for Sophia.

Uses centroid comparison in Milvus to assign HCG node types.
Sophia owns the type vocabulary — Hermes type hints are ignored.
"""

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# If the gap between the top two centroid distances is less than this
# fraction of the top distance, the assignment is ambiguous.
AMBIGUITY_RATIO = 0.2

# Distances above this are considered too far from any type.
MAX_DISTANCE = 2.0

FALLBACK_TYPE_UUID = "type_entity"
FALLBACK_TYPE_NAME = "entity"


@dataclass
class TypeAssignment:
    """Result of type classification."""

    type_uuid: str
    type_name: str
    confidence: float
    # Ternary work-queue flag for #504 type correction:
    #   True  = needs reclassifying (the routine queue). First-pass placement is
    #           provisional, so every freshly ingested node starts True.
    #   False = nowhere better to put it right now -- #504 considered it and had
    #           no better home. NOT "settled"; revisitable as the vocab grows.
    #   None  = has a problem that needs fixing -- #504 hit a cycle / something
    #           unusual it couldn't resolve, and flagged it for resolution (the
    #           node-level cousin of an AMBIGUOUS_SUBSUMPTION cycle).
    # First-pass classification only ever yields True; False and None are written
    # only by #504.
    needs_reclassification: Optional[bool]


class TypeClassifier:
    """Assigns HCG node types using embedding-space centroid proximity."""

    def __init__(self, milvus: Any, hcg: Any) -> None:
        self._milvus = milvus
        self._hcg = hcg

    def classify(self, embedding: list[float], top_k: int = 3) -> TypeAssignment:
        """Classify an embedding by nearest type centroid.

        Args:
            embedding: The node's embedding vector.
            top_k: Number of candidate centroids to consider.

        Returns:
            TypeAssignment with type, confidence, and reclassification flag.
        """
        results = self._milvus.find_nearest_types(
            query_embedding=embedding,
            top_k=top_k + 5,  # fetch extra to compensate for filtering
        )

        # Filter out reserved types — only Sophia subsystems assign those.
        results = [r for r in results if not r["uuid"].startswith("type_reserved_")]

        if not results:
            return TypeAssignment(
                type_uuid=FALLBACK_TYPE_UUID,
                type_name=FALLBACK_TYPE_NAME,
                confidence=0.0,
                needs_reclassification=True,
            )

        best = results[0]
        best_distance = best["score"]
        best_uuid = best["uuid"]
        # Use the type-definition's clean human label (e.g. "organism"), NOT
        # the uuid stripped of "type_": minted uuids carry a random hex suffix,
        # so deriving the name from the uuid produced "organism_<hex>" labels
        # that then overwrote the clean emergence name on the type-def. The
        # lookup is best-effort: a graph error must fall back to the uuid-derived
        # name rather than abort classification (the caller may be mid-ingest
        # with Neo4j flaky).
        best_name = best_uuid.removeprefix("type_")
        if self._hcg is not None:
            try:
                best_node = self._hcg.get_node(best_uuid)
                if best_node and best_node.get("name"):
                    best_name = best_node["name"]
            except Exception:
                logger.debug(
                    "type name lookup failed for %s; using uuid-derived name",
                    best_uuid,
                )

        # Confidence: inverse of distance, clamped to [0, 1]
        if best_distance <= 0:
            confidence = 1.0
        elif best_distance >= MAX_DISTANCE:
            confidence = 0.0
        else:
            confidence = 1.0 - (best_distance / MAX_DISTANCE)

        # Ambiguity check: if the runner-up is close, lower confidence.
        if len(results) >= 2:
            runner_up_distance = results[1]["score"]
            gap = max(runner_up_distance - best_distance, 0.0)
            if gap < AMBIGUITY_RATIO * best_distance:
                confidence *= 0.5

        # needs_reclassification (see TypeAssignment): first-pass placement is
        # provisional, so it always flags True ("needs reclassifying") and joins
        # the #504 correction queue. False ("nowhere better right now") and None
        # ("a cycle / something unusual was hit -- flag for resolution") are only
        # ever written by #504, never on first pass.
        needs_reclass: Optional[bool] = True

        return TypeAssignment(
            type_uuid=best_uuid,
            type_name=best_name,
            confidence=round(confidence, 4),
            needs_reclassification=needs_reclass,
        )

    def update_centroid_for_assignment(
        self,
        type_uuid: str,
        new_embedding: list[float],
        current_centroid: list[float],
        member_count: int,
        model: str,
    ) -> list[float]:
        """Incrementally update a type centroid after assigning a new node.

        new_centroid = (old_centroid * count + new_embedding) / (count + 1)

        Returns:
            The updated centroid vector.
        """
        new_count = member_count + 1
        updated = [
            (old * member_count + new) / new_count
            for old, new in zip(current_centroid, new_embedding)
        ]

        self._milvus.update_centroid(
            type_uuid=type_uuid,
            centroid=updated,
            model=model,
        )

        return updated
