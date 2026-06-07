"""Mint an emergent type from a named cluster: type node + centroid + retype (#505).

`mint_type` creates one NEW type-definition from a cluster:
- create a `:Node` type-definition under its parent with name_history lineage,
- seed its Milvus centroid (= mean of member embeddings),
- retype each member (authoritative `type_uuid` property via update_node),
  unless `retype_members=False` (an internal super-type whose members are
  retyped at the leaf subtypes below it).

Membership is the `type_uuid` property -- emergence does NOT create an
instance->type IS_A edge (it was redundant with `type_uuid`). The taxonomy
IS_A (new type-definition -> parent type-definition) is created below.

HCGClient encodes nested properties (name_history) transparently. No `ancestors`
property is stored: structure (the IS_A edges) is the membership/typing fact,
walked on demand (DESIGN §3). Deciding whether to mint here vs. reconcile a
cluster into an *existing* type (#504 match-before-mint) is the caller's job --
see `EmergenceHandler._match_existing_type`.
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
    parent_ancestors: list[str] | None = None,
    parent_name: str | None = None,
    retype_members: bool = True,
) -> str:
    """Create the type-definition node, seed its centroid, and retype members.

    The type uuid carries a random suffix so that two clusters that Hermes
    happens to name identically mint *distinct* type-definition nodes (and
    distinct centroids) instead of overwriting each other -- members are tied
    to a specific minted type via their ``type_uuid`` property pointing at this
    uuid, not via the shared label string.
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
    hcg.add_edge(type_uuid, parent_type_uuid, "IS_A")

    model = next((m.model for m in cluster.members if m.model), _DEFAULT_MODEL)
    milvus.update_centroid(
        type_uuid=type_uuid,
        centroid=_mean_vector(cluster.embeddings),
        model=model,
    )

    # Retype members onto this type, unless this is an internal super-type whose
    # members belong to its child subtypes (the caller retypes them at the
    # leaves). Membership is a pure property: stamp the authoritative current-type
    # pointer (`type_uuid`, overwritten on each retype) and the human-facing
    # `type` slug. We deliberately do NOT create an instance->type IS_A edge --
    # that edge was redundant bookkeeping over `type_uuid` and only polluted the
    # edge graph. _member_rows loads members directly by this `type_uuid`
    # property, so a member retyped away from a parent is excluded automatically
    # (no stale-edge cleanup needed) (#505).
    if retype_members:
        for member in cluster.members:
            hcg.update_node(member.uuid, {"type": slug, "type_uuid": type_uuid})

    logger.info(
        "Minted type %s (%s) from %d members", name.label, type_uuid, cluster.size
    )
    return type_uuid
