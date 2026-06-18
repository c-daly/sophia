"""Publish the positional type snapshot to Redis (`logos:ontology:types`).

Single source for the type-catalog snapshot Hermes' TypeRegistry boots from.
Types are gathered positionally via ``get_all_type_definitions()`` (incoming-IS_A
over the IS_A subgraph), so the snapshot reflects the *real* type layer -- every
node something IS_A's, including content-node types (``engine``) and anything that
became a type after a graft -- not just label-stamped/minted nodes.

Driven from three places so it stays in sync:
  - ingest, after each proposal batch (proposal_processor);
  - the maintenance scheduler's periodic loop (reconcile / safety net);
  - the ``ontology.type_created`` event (incremental, on emergence mint).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sophia.maintenance.realms import GRAFTABLE_REALMS, STRUCTURAL_ROOTS

logger = logging.getLogger(__name__)

REDIS_KEY = "logos:ontology:types"


def publish_type_snapshot(hcg: Any, redis: Any) -> int:
    """Gather the positional type layer and write it to Redis; return the count.

    Fail-soft (sophia#195): a missing/failed snapshot just makes Hermes'
    TypeRegistry boot empty rather than crashing the caller.
    """
    if redis is None or hcg is None:
        return 0
    try:
        records = hcg.get_all_type_definitions()
        snapshot: dict[str, dict[str, Any]] = {}
        for record in records:
            name = record.get("name", "")
            if not name:
                continue
            # Reserved/internal scaffolding (`_reserved_*`, `_cognition`) is not a
            # graftable type: never publish it to the catalog, or the naming LLM
            # picks it as a parent and re-roots real subtrees under it (#152).
            # Guard isinstance first: a malformed non-string name would raise
            # AttributeError on .startswith and abort the whole snapshot.
            if not isinstance(name, str) or name.startswith("_"):
                continue
            # Normalize to a case-insensitive catalog key: a stored name with
            # non-standard casing/whitespace (e.g. "Entity") must not publish a
            # variant Hermes won't match (it canonicalizes on lookup).
            key = name.strip().lower()
            # Structural scaffolding (node/root) is never a graftable parent --
            # Hermes bars them as structural roots and they only pollute the catalog.
            if key in STRUCTURAL_ROOTS:
                continue
            props = record.get("properties")
            member_count = (
                props.get("member_count", 0) if isinstance(props, dict) else 0
            )
            if key in snapshot:
                logger.warning(
                    "publish_type_snapshot: name collision %r (uuid %s clobbers %s); "
                    "only the last will be visible to TypeRegistry",
                    key,
                    record.get("uuid", ""),
                    snapshot[key].get("uuid", ""),
                )
            snapshot[key] = {
                "uuid": record.get("uuid", ""),
                "member_count": member_count,
            }
        # Always publish the graftable realms (entity/concept/process), even with
        # no members: the positional pass omits a childless realm root, but the
        # naming LLM must still see them as valid closed-world parents -- else
        # every /type-cluster mint is rejected ("no valid parent") and emergence
        # can never bootstrap on a cold graph. Resolve the missing ones by name
        # (mirrors the ingest realm-park, #211); a by-name realm has no counted
        # members yet, so it is published with member_count 0.
        missing = GRAFTABLE_REALMS - {n.strip().lower() for n in snapshot}
        if missing:
            resolved = hcg.find_nodes_by_names(sorted(missing))
            if isinstance(resolved, dict):
                for nm, node in resolved.items():
                    key = nm.strip().lower()
                    if (
                        key in GRAFTABLE_REALMS
                        and isinstance(node, dict)
                        and node.get("uuid")
                    ):
                        snapshot.setdefault(
                            key, {"uuid": node["uuid"], "member_count": 0}
                        )
        redis.set(REDIS_KEY, json.dumps(snapshot))
        return len(snapshot)
    except Exception:
        logger.exception("Failed to publish type snapshot to Redis")
        return 0
