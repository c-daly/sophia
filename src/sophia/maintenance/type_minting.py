"""Mint an emergent type from a named cluster: type node + centroid + IS_A edge.

`mint_type` creates one NEW type-definition from a cluster:
- create a `:Node` type-definition under its parent with name_history lineage,
- seed its Milvus centroid (= mean of member embeddings),
- wire its own single upward IS_A edge to the parent type-definition.

Membership is the instance->type IS_A edge now, NOT a `type_uuid` property
(B2/B3, DESIGN §3): minting does NOT touch the members. The draining caller
(`EmergenceHandler._place_cluster`) owns member placement -- it re-points each
fitting member's single upward IS_A edge onto this type through
`placement.reparent`.

HCGClient encodes nested properties (name_history) transparently. No `ancestors`
property is stored: structure (the IS_A edges) is the membership/typing fact,
walked on demand (DESIGN §3). Deciding whether to mint here vs. reuse an
existing same-name type is the caller's job -- see
`EmergenceHandler._place_cluster` (the flat parent-driven cascade).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sophia.maintenance.emergence_types import EmergentCluster, NameResult

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


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
    # Required: drainage callers always resolve and pass a real parent uuid
    # (a resolved parent or the source realm root). No `type_<name>` slug
    # default -- type identity is an opaque uuid, never name-encoded.
    parent_type_uuid: str,
    # The realm this type roots into (entity/concept/process), stamped as the
    # node `.type`. Type-ness itself is NOT stamped -- it is positional (the
    # incoming IS_A edge); `.type` carries the realm, uniform with content nodes.
    realm: str,
    placed_by: str | None = None,
) -> str:
    """Create the type node, seed its centroid, and wire its IS_A edge.

    The type uuid is an opaque ``uuid4`` -- type identity is positional (the
    incoming IS_A edges), NOT encoded in the uuid. The node ``.type`` holds the
    REALM (entity/concept/process), the same field content nodes use; "is this a
    type?" is answered positionally (does anything IS_A it?), never by a stored
    ``type_definition`` label. Two clusters Hermes happens to name identically
    still mint *distinct* type nodes (distinct uuids, distinct centroids).
    Members are NOT touched here -- membership is the instance->type IS_A edge,
    re-pointed onto this type by the draining caller through
    ``placement.reparent`` (B2/B3, DESIGN §3).
    """
    type_uuid = str(uuid4())
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
        node_type=realm,
        uuid=type_uuid,
        properties={
            # No `is_type_definition` flag and no `type_`-encoded uuid: the type
            # layer is detected POSITIONALLY (a node is a type iff it has an
            # incoming IS_A edge), not from the uuid or a stored flag. No
            # `ancestors` snapshot either: the IS_A edge below IS the
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
