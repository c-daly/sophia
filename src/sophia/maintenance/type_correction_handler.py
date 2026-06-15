"""Structural type correction (#504): evict a member that is a PART/PRODUCT of a
same-type peer.

Deterministic and embedding-free. A cluster centroid groups a thing with its
parts and products (embeddings encode *association*, not *taxonomy*), so a type
can end up containing both a whole and its part -- e.g. ``tusk`` inside the
marine-mammal type, ``acorn`` inside the oak's type. An intra-type meronymic /
productive edge (``tusk PART_OF narwhal``, ``oak PRODUCES acorn``) is conclusive
structural evidence that one member is a part/product of another, NOT a
co-member. We evict the part/product back to the ``entity`` junk-drawer (its
``PART_OF``/``PRODUCES`` edge stays, so its role is still recorded) by re-pointing
its single upward ``IS_A`` membership edge onto the ``entity`` realm root via
``placement.reparent`` -- the inverse of ``type_minting``'s placement.

This redirects #504 off the falsified centroid-cohesion approach (embedding
distance cannot separate "same kind" from "merely related") onto the proven
structural signal. No Milvus / Hermes required.
"""

from __future__ import annotations

import logging
from typing import Any

from sophia.maintenance import placement

logger = logging.getLogger(__name__)

_BIG = 100000

# Node ``type`` values that are NOT emergent types (base ontology + plan
# scaffold). Correction only fires when two members share a *non-base* type.
# `_cognition` is the underscore-namespaced cognition realm root (seeder fold
# 54b49d2); keep the bare `cognition` too for the pre-reseed transition window so
# the 4th realm root stays protected from correction either way.
_BASE_TYPES = frozenset({"entity", "concept", "process", "cognition", "_cognition"})

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
        self._entity_uuid: str | None = None

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
        # Co-membership is detected from the `type` slug two members share on a
        # retype. Membership itself is the instance->type IS_A edge now, not a
        # `type_uuid` stamp; this detection scan is a read path left unconverted.
        # TODO #35: read still infers membership from property; convert to IS_A walk
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

    def _resolve_entity_uuid(self) -> str | None:
        """Resolve the real ``entity`` realm-root uuid (cached).

        Skeleton type-defs carry real uuids (uuid5), not the legacy
        ``type_<name>`` slug, so the eviction target is looked up by name
        rather than hard-coded -- a literal ``"type_entity"`` would dangle
        against a reseeded graph.
        """
        if self._entity_uuid is None:
            try:
                for n in self._hcg.get_all_type_definitions():
                    if n.get("name") == "entity":
                        self._entity_uuid = n.get("uuid")
                        break
            except Exception:
                logger.exception("type_correction: failed to resolve entity uuid")
        return self._entity_uuid

    def _evict(self, uuid: str, from_type: str, rel: str) -> bool:
        """Re-point the member's membership edge back to the junk-drawer.

        Membership is the instance->type IS_A edge now (B2/B3, DESIGN sec 3), not
        a `type_uuid` stamp -- eviction re-points that single upward edge onto the
        `entity` realm root via placement.reparent (the inverse of mint_type's
        placement). An empty `children_of` is safe: an evicted instance is a leaf
        that never roots a type IS_A subtree, so no cycle is possible.
        """
        entity_uuid = self._resolve_entity_uuid()
        if not entity_uuid:
            logger.warning(
                "type_correction: cannot evict %s -- entity realm root not found", uuid
            )
            return False
        try:
            placement.reparent(
                uuid,
                entity_uuid,
                hcg=self._hcg,
                children_of={},
                placed_by="root_fallback",
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
    evicts the part/product member back under the ``entity`` realm root by
    re-pointing its ``IS_A`` membership edge (placement.reparent). No Milvus /
    Hermes needed.
    """
    handler = TypeCorrectionHandler(config=config, hcg=hcg, event_bus=event_bus)
    return lambda **_kw: handler.run()
