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
            if name.startswith("_"):
                continue
            props = record.get("properties")
            member_count = (
                props.get("member_count", 0) if isinstance(props, dict) else 0
            )
            if name in snapshot:
                logger.warning(
                    "publish_type_snapshot: name collision %r (uuid %s clobbers %s); "
                    "only the last will be visible to TypeRegistry",
                    name,
                    record.get("uuid", ""),
                    snapshot[name].get("uuid", ""),
                )
            snapshot[name] = {
                "uuid": record.get("uuid", ""),
                "member_count": member_count,
            }
        redis.set(REDIS_KEY, json.dumps(snapshot))
        return len(snapshot)
    except Exception:
        logger.exception("Failed to publish type snapshot to Redis")
        return 0
