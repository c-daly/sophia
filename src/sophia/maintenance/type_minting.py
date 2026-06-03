"""Mint an emergent type from a named cluster: type node + centroid + retype (#505).

Emergence always mints a NEW type from a cluster of the unmatched residue:
- create a `:Node` type-definition under `root` with name_history lineage,
- seed its Milvus centroid (= mean of member embeddings),
- retype each member (`type` property via update_node) and add an `IS_A` edge.

HCGClient encodes nested properties (name_history) transparently; ancestors is a
native string list. Reconciling members into an *existing* type is #504's job.
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
) -> str:
    """Create the type-definition node, seed its centroid, and retype members.

    The type uuid carries a random suffix so that two clusters that Hermes
    happens to name identically mint *distinct* type-definition nodes (and
    distinct centroids) instead of overwriting each other -- members are tied
    to a specific minted type via their ``IS_A`` edge to this uuid, not via the
    shared label string.
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
    # Descend from the parent type (default `type_entity`) rather than `root`:
    # the seeder represents the type hierarchy via IS_A edges, and spec 21.3
    # expects a minted type's `ancestors` to match that IS_A chain. For the
    # default entity parent this yields ["root", "node", "entity"].
    _anc = parent_ancestors or ["root", "node"]
    parent_name = parent_type_uuid.removeprefix("type_")
    hcg.add_node(
        name=name.label,
        node_type="type_definition",
        uuid=type_uuid,
        properties={
            "is_type_definition": True,
            "ancestors": _anc + [parent_name],
            "name_history": name_history,
        },
        source="emergence",
    )
    # Wire the minted type into the IS_A hierarchy under its parent so the
    # graph chain matches the stored `ancestors` (new_type IS_A parent).
    hcg.add_edge(type_uuid, parent_type_uuid, "IS_A")

    model = next((m.model for m in cluster.members if m.model), _DEFAULT_MODEL)
    milvus.update_centroid(
        type_uuid=type_uuid,
        centroid=_mean_vector(cluster.embeddings),
        model=model,
    )

    for member in cluster.members:
        # Remove the member's prior type-membership IS_A edge(s) before adding the
        # new one. Edges are reified :Node records, so delete_edge(edge_uuid)
        # removes the edge -- without this, a member split out of a parent type
        # keeps a stale IS_A->parent edge and re-emergence on the parent would
        # re-include and re-mint it (#149 review).
        for e in hcg.query_edges_from(member.uuid):
            if e.get("relation") == "IS_A":
                edge_uuid = e.get("id") or e.get("uuid")
                if edge_uuid:
                    hcg.delete_edge(edge_uuid)
        # Also stamp the authoritative current-membership pointer (type_uuid,
        # overwritten on each retype); _member_rows filters by it as
        # defense-in-depth should an edge delete above ever be missed.
        hcg.update_node(member.uuid, {"type": slug, "type_uuid": type_uuid})
        hcg.add_edge(member.uuid, type_uuid, "IS_A")

    logger.info(
        "Minted type %s (%s) from %d members", name.label, type_uuid, cluster.size
    )
    return type_uuid
