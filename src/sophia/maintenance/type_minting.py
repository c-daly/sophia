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
from datetime import datetime, timezone
from typing import Any

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
) -> str:
    """Create the type-definition node, seed its centroid, and retype members."""
    type_uuid = f"type_{name.label}"
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
            "is_type_definition": True,
            "ancestors": ["root"],
            "name_history": name_history,
        },
        source="emergence",
    )

    model = next((m.model for m in cluster.members if m.model), _DEFAULT_MODEL)
    milvus.update_centroid(
        type_uuid=type_uuid,
        centroid=_mean_vector(cluster.embeddings),
        model=model,
    )

    for member in cluster.members:
        hcg.update_node(member.uuid, {"type": name.label})
        hcg.add_edge(member.uuid, type_uuid, "IS_A")

    logger.info(
        "Minted type %s (%s) from %d members", name.label, type_uuid, cluster.size
    )
    return type_uuid
