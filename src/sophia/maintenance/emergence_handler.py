"""The 'type_emergence' maintenance handler (#505).

Dispatched by MaintenanceScheduler as handlers['type_emergence'](type_uuid=...).
Dependencies (load_members / name_fn / mint_fn / candidates_fn) are injected so
the orchestration is unit-testable without Neo4j / Milvus / Hermes. Emergence
always mints NEW types from the residue; it never attaches to existing ones.
"""

from __future__ import annotations

import logging
import uuid as uuid_lib
from collections.abc import Callable
from typing import Any

from sophia.maintenance.config import MaintenanceConfig
from sophia.maintenance.emergence_clustering import find_emergent_hierarchy
from sophia.maintenance.emergence_types import EmergentCluster, Member
from sophia.maintenance.structural_signature import build_signature

logger = logging.getLogger(__name__)

ONTOLOGY_CHANGED_CHANNEL = "ontology.type_created"

# The base "junk-drawer" type that holds un-specialised nodes. Its membership is
# resolved by node-type scan; minted types resolve membership via IS_A edges.
_BASE_TYPE = "entity"

# Lineage of the `entity` junk-drawer (root -> node -> entity). Emergence mints
# directly under `type_entity`, so a minted type's parent ancestors are these
# two prefixes and its own full ancestors append "entity" (#505).
_ENTITY_ANCESTORS = ["root", "node"]


def _type_name(type_uuid: str) -> str:
    """Best-effort label from a type-definition uuid.

    Minted uuids are ``type_<label>_<hex8>`` and base/legacy ones ``type_<name>``.
    Used only for the base junk-drawer scan and for embedding-collection routing.
    """
    return type_uuid[len("type_") :] if type_uuid.startswith("type_") else type_uuid


def current_categories(hcg: Any) -> list[str]:
    """Existing type-definition labels, excluding `entity` and reserved_* types."""
    out: list[str] = []
    for node in hcg.list_all_nodes(node_type="type_definition"):
        name = node.get("name")
        if not name or name == "entity" or name.startswith("reserved_"):
            continue
        out.append(name)
    return out


def _member_rows(hcg: Any, type_uuid: str, type_name: str) -> list[dict[str, Any]]:
    """Resolve the node rows that belong to ``type_uuid``.

    The base junk-drawer (``entity``) is resolved by node-type scan. Any minted
    type is resolved by the nodes that have an ``IS_A`` edge into *this specific*
    type-definition uuid -- so two minted types that share a label do not bleed
    members into one another.

    IS_A edges are additive (mint_type cannot delete the old edge), so a member
    that was split out of this type into a *child* type still carries its stale
    IS_A edge here. We therefore also require the node's authoritative
    ``type_uuid`` pointer (overwritten on each retype) to still equal this type --
    otherwise a re-emergence run would re-include and re-mint already-retyped
    members (#149 review).
    """
    if type_name == _BASE_TYPE:
        return list(hcg.list_all_nodes(node_type=type_name))

    edges = hcg.list_all_edges(relation_type="IS_A", target_uuid=type_uuid)
    member_uuids = [e["source"] for e in (edges or []) if e and e.get("source")]
    if not member_uuids:
        return []
    return [
        n
        for n in (hcg.get_nodes_batch(member_uuids) or [])
        if n and "uuid" in n and n.get("type_uuid") == type_uuid
    ]


def _build_member(
    hcg: Any, milvus: Any, row: dict[str, Any], type_name: str
) -> Member | None:
    """Build a Member from a node row, or None if it has no usable embedding."""
    from sophia.ingestion.proposal_processor import _collection_for

    uuid = row["uuid"]
    # Emergence members all descend from the `entity` junk drawer, so their
    # embeddings physically live in that base collection. Retyping a node to a
    # minted slug does NOT move its stored vector, so we must read from the base
    # collection -- NOT _collection_for(current type). A minted slug that maps to
    # a different collection (e.g. "concept" -> "Concept") would otherwise miss
    # and silently drop the member on re-emergence (greptile #149).
    emb = milvus.get_embedding(node_type=_collection_for(_BASE_TYPE), uuid=uuid)
    if not emb or not emb.get("embedding"):
        return None
    edges = hcg.query_edges_from(uuid)
    target_uuids = [e["target"] for e in edges if e.get("target")]
    target_nodes = {
        n["uuid"]: n
        for n in ((hcg.get_nodes_batch(target_uuids) or []) if target_uuids else [])
        if n and "uuid" in n
    }
    neighbors = [
        {
            "relation": e.get("relation"),
            "neighbor_name": target_nodes.get(e.get("target"), {}).get("name"),
            "neighbor_type": target_nodes.get(e.get("target"), {}).get("type"),
        }
        for e in edges
    ]
    props = row.get("properties", {}) or {}
    return Member(
        uuid=uuid,
        name=row.get("name", uuid),
        embedding=emb["embedding"],
        signature=build_signature(neighbors),
        current_type=row.get("type", type_name),
        hermes_type_hint=props.get("hermes_type_hint"),
        neighbors=neighbors,
        model=emb.get("model"),
    )


def load_type_members(hcg: Any, milvus: Any, type_uuid: str) -> list[Member]:
    """Load all members of a type as Member objects (embedding + structural signature).

    Membership is resolved by :func:`_member_rows` (node-type scan for the base
    junk-drawer, IS_A-edge lookup for minted types). Embeddings come from Milvus;
    the structural signature is built from the node's outgoing reified edges
    (relation + resolved neighbor type). Nodes without an embedding are skipped
    (they can't be clustered).
    """
    type_name = _type_name(type_uuid)
    members: list[Member] = []
    for row in _member_rows(hcg, type_uuid, type_name):
        if not row or "uuid" not in row:
            continue
        member = _build_member(hcg, milvus, row, type_name)
        if member is not None:
            members.append(member)
    return members


class EmergenceHandler:
    def __init__(
        self,
        *,
        config: MaintenanceConfig,
        hcg: Any,
        milvus: Any,
        event_bus: Any,
        hermes_url: str,
        token: str,
        load_members: Any,
        name_fn: Any,
        mint_fn: Any,
        candidates_fn: Any,
    ) -> None:
        self._config = config
        self._hcg = hcg
        self._milvus = milvus
        self._event_bus = event_bus
        self._hermes_url = hermes_url
        self._token = token
        self._load_members = load_members
        self._name_fn = name_fn
        self._mint_fn = mint_fn
        self._candidates_fn = candidates_fn

    def run(self, type_uuid: str) -> None:
        members = self._load_members(type_uuid)
        # Roll the fine clusters up into a hierarchy and mint from the top-level
        # nodes, so related fine groups consolidate into super-types (#505).
        hierarchy = find_emergent_hierarchy(
            members,
            min_cluster_size=self._config.min_cluster_size,
            variance_threshold=self._config.variance_threshold,
        )
        if not hierarchy:
            logger.info("emergence: no qualifying clusters in %s", type_uuid)
            return
        clusters = [EmergentCluster(members=node.members) for node in hierarchy]

        # Mutable copy: each successful mint adds its label so later clusters in
        # this same run see it as a candidate and don't silently re-mint a
        # same-label sibling.
        candidates = list(self._candidates_fn())
        for cluster in clusters:
            # Per-cluster isolation: a transient HCG/Milvus/Redis error while
            # minting one cluster must not abort the whole run -- log it and move
            # on so the remaining clusters still get minted (greptile #149).
            try:
                name = self._name_fn(cluster, candidates, self._hermes_url, self._token)
                if (
                    name is None
                    or name.confidence < self._config.hermes_confidence_floor
                ):
                    logger.info("emergence: skip cluster (no/low-confidence name)")
                    continue
                cluster_id = uuid_lib.uuid4().hex[:8]
                new_type_uuid = self._mint_fn(
                    cluster,
                    name,
                    hcg=self._hcg,
                    milvus=self._milvus,
                    source_cluster_id=cluster_id,
                    parent_type_uuid=f"type_{_BASE_TYPE}",
                    parent_ancestors=_ENTITY_ANCESTORS,
                )
                if name.label not in candidates:
                    candidates.append(name.label)
                if self._event_bus is not None:
                    self._event_bus.publish(
                        ONTOLOGY_CHANGED_CHANNEL,
                        {
                            "type_uuid": new_type_uuid,
                            "name": name.label,
                            "ancestors": _ENTITY_ANCESTORS + [_BASE_TYPE],
                        },
                    )
                logger.info(
                    "emergence: minted %s from %d members", name.label, cluster.size
                )
            except Exception:
                logger.exception("emergence: cluster failed, skipping")


def build_emergence_handler(
    *,
    config: MaintenanceConfig,
    hcg: Any,
    milvus: Any,
    event_bus: Any,
    hermes_url: str,
    token: str,
) -> Callable[[str], None]:
    """Return the callable registered as handlers['type_emergence']."""
    from sophia.maintenance.hermes_naming import name_cluster
    from sophia.maintenance.type_minting import mint_type

    handler = EmergenceHandler(
        config=config,
        hcg=hcg,
        milvus=milvus,
        event_bus=event_bus,
        hermes_url=hermes_url,
        token=token,
        load_members=lambda u: load_type_members(hcg, milvus, u),
        name_fn=lambda c, cand, url, tok: name_cluster(
            c,
            candidates=cand,
            hermes_url=url,
            token=tok,
            max_members=config.max_cluster_size,
        ),
        mint_fn=mint_type,
        candidates_fn=lambda: current_categories(hcg),
    )

    def _run(type_uuid: str) -> None:
        handler.run(type_uuid=type_uuid)

    return _run
