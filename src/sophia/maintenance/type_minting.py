"""Mint an emergent type from a named cluster: type node + centroid + IS_A edge.

`mint_type` creates one NEW type-definition from a cluster:
- create a `:Node` type-definition under its parent with name_history lineage,
- seed its Milvus centroid (= mean of member embeddings),
- wire its own single upward IS_A edge to the parent type-definition.

Membership is the instance->type IS_A edge now, NOT a `type_uuid` property
(B2/B3, DESIGN §3): minting does NOT touch the members. The draining caller
(`EmergenceHandler._place_cluster`) owns member placement -- it re-points each
fitting member's single upward IS_A edge onto this type through
`placement.reparent`. The `retype_members` flag is therefore a no-op, kept only
so the gated-off rollup tier's `retype_members=False` call site still resolves.

HCGClient encodes nested properties (name_history) transparently. No `ancestors`
property is stored: structure (the IS_A edges) is the membership/typing fact,
walked on demand (DESIGN §3). Deciding whether to mint here vs. reuse an
existing same-name type is the caller's job -- see
`EmergenceHandler._place_cluster` (the flat parent-driven cascade).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sophia.maintenance.emergence_types import EmergentCluster, NameResult

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


def _slugify(label: str) -> str:
    """Normalize a free-text Hermes label into a slug-safe identifier component.

    ``name.label`` comes verbatim from Hermes' JSON response and flows into the
    ``type_uuid``, the node ``type`` property, and event payloads. A multi-word
    or punctuated label (e.g. ``"living thing"``, ``"sub-class"``) would inject
    spaces/punctuation into the graph's type namespace and corrupt subsequent
    lookups (greptile review #149). Lowercase and collapse any run of
    non-alphanumeric characters to a single underscore; fall back to ``unnamed``
    so the identifier is never empty (uniqueness still comes from the uuid suffix).
    """
    slug = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
    return slug or "unnamed"


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    n = len(vectors)
    dim = len(vectors[0])
    return [sum(v[d] for v in vectors) / n for d in range(dim)]


def mint_type(
    cluster: EmergentCluster,
    name: NameResult,
    *,
    hcg: Any,
    milvus: Any,
    source_cluster_id: str,
    parent_type_uuid: str = "type_entity",
    # DEPRECATED, unused since the `ancestors` property was removed (B1 T3): kept
    # only so the gated-off rollup tier's call site still resolves. Removed when
    # rollup is converted onto placement.py. The `parent_type_uuid="type_entity"`
    # default is likewise a legacy slug slated for that cleanup -- drainage callers
    # always pass a real uuid.
    parent_ancestors: list[str] | None = None,
    parent_name: str | None = None,
    # NO-OP since B2/B3: membership is the instance->type IS_A edge, re-pointed by
    # the draining caller via placement.reparent -- mint_type no longer retypes
    # members. Kept only so the gated-off rollup tier's `retype_members=False`
    # call site still resolves; removed when rollup is converted onto placement.py.
    retype_members: bool = True,
    placed_by: str | None = None,
) -> str:
    """Create the type-definition node, seed its centroid, and wire its IS_A edge.

    The type uuid carries a random suffix so that two clusters that Hermes
    happens to name identically mint *distinct* type-definition nodes (and
    distinct centroids) instead of overwriting each other. Members are NOT
    touched here -- membership is the instance->type IS_A edge, re-pointed onto
    this type by the draining caller through ``placement.reparent`` (B2/B3,
    DESIGN §3).
    """
    slug = _slugify(name.label)
    type_uuid = f"type_{slug}_{uuid4().hex[:8]}"
    now = datetime.now(timezone.utc).isoformat()
    name_history = [
        {
            "name": name.label,
            "named_at": now,
            "reason": "emergence",
            "source_cluster_id": source_cluster_id,
            "hermes_confidence": name.confidence,
        }
    ]
    hcg.add_node(
        name=name.label,
        node_type="type_definition",
        uuid=type_uuid,
        properties={
            # No `is_type_definition` flag (#171): the type layer is detected
            # structurally via node_type="type_definition" + the type_ uuid.
            # No `ancestors` snapshot either: the IS_A edge below IS the
            # membership/typing structure, walked on demand (DESIGN §3).
            "name_history": name_history,
        },
        source="emergence",
    )
    # Wire the minted type into the IS_A hierarchy under its parent. This edge
    # IS the membership/typing structure (walked on demand), so there is no
    # redundant `ancestors` property to keep in sync (new_type IS_A parent).
    hcg.add_edge(
        type_uuid,
        parent_type_uuid,
        "IS_A",
        properties={"placed_by": placed_by} if placed_by else None,
    )

    model = next((m.model for m in cluster.members if m.model), _DEFAULT_MODEL)
    milvus.update_centroid(
        type_uuid=type_uuid,
        centroid=_mean_vector(cluster.embeddings),
        model=model,
    )

    # Members are NOT retyped here. Membership is the instance->type IS_A edge
    # (B2/B3, DESIGN §3): the draining caller re-points each fitting member's
    # single upward IS_A edge onto this type via placement.reparent. mint_type
    # only owns the type-definition node, its centroid, and its own IS_A edge to
    # the parent (created above). No `type_uuid`/`type` property is stamped.
    logger.info(
        "Minted type %s (%s) from %d members", name.label, type_uuid, cluster.size
    )
    return type_uuid
