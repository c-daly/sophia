"""Structural type correction (#504): evict a member that is a PART/PRODUCT of a
same-type peer.

Deterministic and embedding-free. A cluster centroid groups a thing with its
parts and products (embeddings encode *association*, not *taxonomy*), so a type
can end up containing both a whole and its part -- e.g. ``tusk`` inside the
marine-mammal type, ``acorn`` inside the oak's type. An intra-type meronymic /
productive edge (``tusk PART_OF narwhal``, ``oak PRODUCES acorn``) is conclusive
structural evidence that one member is a part/product of another, NOT a
co-member. We evict the part/product back to the ``type_entity`` junk-drawer
(its ``PART_OF``/``PRODUCES`` edge stays, so its role is still recorded) by
writing the inverse of ``type_minting``'s retype.

This redirects #504 off the falsified centroid-cohesion approach (embedding
distance cannot separate "same kind" from "merely related") onto the proven
structural signal. No Milvus / Hermes required.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_BIG = 100000

# Node ``type`` values that are NOT emergent types (base ontology + plan
# scaffold). Correction only fires when two members share a *non-base* type.
_BASE_TYPES = frozenset({"entity", "concept", "process"})

# Relations meaning "different KIND of thing": a member linked to a same-type
# peer by one of these is a part/product, not a co-member.
_ANTILINK = frozenset(
    {
        "PART_OF",
        "HAS_PART",
        "ATTACHED_TO",
        "CONTAINS",
        "COVERS",
        "PRODUCES",
        "DROPS",
        "FORMS",
        "EMITTED_BY",
        "DISPENSES",
        "CARRIES",
        "GATHERS",
        "DELIVERED_TO",
        "DISPENSED_INTO",
        "HARVESTS",
        "HAULS",
        "DRAGS",
        "ABSORBS",
        "ACCUMULATES",
        "USES",
    }
)

# Which endpoint is the part/product to evict. For PART_OF / HAS_PART /
# ATTACHED_TO the *source* is the part (depends on the target); for the rest
# the *target* is the product/object.
_EVICT_SOURCE = frozenset({"PART_OF", "ATTACHED_TO"})


class TypeCorrectionHandler:
    """Evict part/product members from emergent types via the existing retype."""

    def __init__(self, *, config: Any, hcg: Any, event_bus: Any = None) -> None:
        self._config = config
        self._hcg = hcg
        self._event_bus = event_bus

    def run(self) -> None:
        try:
            edges = self._hcg.list_all_edges(limit=_BIG) or []
        except Exception:
            logger.exception("type_correction: list_all_edges failed")
            return

        anti = [
            e
            for e in edges
            if e.get("relation") in _ANTILINK
            and e.get("source")
            and e.get("target")
            and e.get("source") != e.get("target")
        ]
        if not anti:
            return

        uuids = {e["source"] for e in anti} | {e["target"] for e in anti}
        try:
            rows = self._hcg.get_nodes_batch(list(uuids)) or []
        except Exception:
            logger.exception("type_correction: get_nodes_batch failed")
            return
        # Membership is the node's emergent type. `type` (the slug) is set
        # alongside `type_uuid` on every retype (type_minting), so two members
        # of one emergent type share a non-base `type`.
        type_of: dict[str, str] = {
            r.get("uuid"): r.get("type") for r in rows if r.get("uuid")
        }

        evicted = 0
        for e in anti:
            s, t = e["source"], e["target"]
            st = type_of.get(s)
            # Same emergent type on both ends -> one is a part/product, not a peer.
            if not st or st != type_of.get(t) or st in _BASE_TYPES:
                continue
            victim = s if e["relation"] in _EVICT_SOURCE else t
            if self._evict(victim, st, e["relation"]):
                evicted += 1
                # A second anti-link edge to the same victim is now a no-op.
                type_of[victim] = "entity"

        if evicted:
            logger.info("type_correction: evicted %d part/product member(s)", evicted)

    def _evict(self, uuid: str, from_type: str, rel: str) -> bool:
        """Retype the member back to the junk-drawer (inverse of mint_type)."""
        try:
            self._hcg.update_node(
                uuid,
                {
                    "type": "entity",
                    "type_uuid": "type_entity",
                    "needs_reclassification": True,
                },
            )
            logger.info(
                "type_correction: evicted %s from %s (intra-type %s edge)",
                uuid,
                from_type,
                rel,
            )
            return True
        except Exception:
            logger.exception("type_correction: evict failed for %s", uuid)
            return False


def build_type_correction_handler(
    *, config: Any, hcg: Any, event_bus: Any = None
) -> Any:
    """Return the callable registered as ``handlers['type_correction']`` (#504).

    Deterministic + embedding-free: scans for intra-type meronymic edges and
    evicts the part/product member to ``type_entity`` via the existing
    ``update_node`` retype. No Milvus / Hermes needed.
    """
    handler = TypeCorrectionHandler(config=config, hcg=hcg, event_bus=event_bus)
    return lambda **_kw: handler.run()
