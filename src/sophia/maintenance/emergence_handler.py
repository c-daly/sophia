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
from sophia.maintenance.type_minting import _slugify

logger = logging.getLogger(__name__)

ONTOLOGY_CHANGED_CHANNEL = "ontology.type_created"

# The base "junk-drawer" type that holds un-specialised nodes. Its membership is
# resolved by node-type scan; minted types resolve membership via the
# authoritative `type_uuid` property (no instance->type IS_A edges -- #505).
_BASE_TYPE = "entity"

# Lineage of the `entity` junk-drawer (root -> node -> entity). Emergence mints
# directly under `type_entity`, so a minted type's parent ancestors are these
# two prefixes and its own full ancestors append "entity" (#505).
_ENTITY_ANCESTORS = ["root", "node"]


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 if either is zero)."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


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
    type is resolved by the nodes whose authoritative ``type_uuid`` property
    points at *this specific* type-definition uuid -- so two minted types that
    share a label do not bleed members into one another, and a member retyped
    out of this type (its ``type_uuid`` now points elsewhere) is excluded
    automatically.

    Membership is the ``type_uuid`` property; emergence no longer creates an
    instance->type IS_A edge, so there is nothing to traverse or guard against
    here (#505).
    """
    if type_name == _BASE_TYPE:
        return list(hcg.list_all_nodes(node_type=type_name))

    return [
        n for n in (hcg.get_nodes_by_type_uuid(type_uuid) or []) if n and "uuid" in n
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
    junk-drawer, `type_uuid`-property lookup for minted types). Embeddings come
    from Milvus;
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
        # Roll the fine clusters up into a multi-level hierarchy: leaves are the
        # fine clusters, internal nodes are super-types grouping related leaves
        # (e.g. "linear algebra" + "calculus" -> "mathematics") (#505).
        hierarchy = find_emergent_hierarchy(
            members,
            min_cluster_size=self._config.min_cluster_size,
            variance_threshold=self._config.variance_threshold,
        )
        if not hierarchy:
            logger.info("emergence: no qualifying clusters in %s", type_uuid)
            return

        # Mint the whole tree *under the type being subdivided*. For the base
        # `entity` junk-drawer that parent is `type_entity`; when re-emergence
        # runs on an already-minted type we nest the new subtypes under it, so
        # the hierarchy actually deepens instead of everything landing flat under
        # `entity` (#505).
        if _type_name(type_uuid) == _BASE_TYPE:
            parent_type_uuid = f"type_{_BASE_TYPE}"
            parent_ancestors = _ENTITY_ANCESTORS
        else:
            parent_type_uuid = type_uuid
            parent_node = self._hcg.get_node(type_uuid) or {}
            parent_props = parent_node.get("properties") or {}
            parent_ancestors = list(parent_props.get("ancestors") or _ENTITY_ANCESTORS)

        # Mutable copy: each successful mint adds its label so later clusters in
        # this same run see it as a candidate and don't silently re-mint a
        # same-label sibling.
        candidates = list(self._candidates_fn())
        for node in hierarchy:
            self._mint_subtree(node, parent_type_uuid, parent_ancestors, candidates)

    def _mint_subtree(
        self,
        node: Any,
        parent_type_uuid: str,
        parent_ancestors: list[str],
        candidates: list[str],
    ) -> None:
        """Mint (or reconcile) one hierarchy node, then recurse into its children.

        Leaf nodes carry real instance members and get them retyped onto the
        minted/reconciled type. Internal nodes are pure super-types: their type
        node is created but members are retyped at the leaves below. Every level
        is named by Hermes, and children are minted under their freshly-created
        parent so the IS_A chain and `ancestors` nest correctly.
        """
        # Per-node isolation: a transient HCG/Milvus/Redis error on one node must
        # not abort the whole run -- log it and let siblings proceed (#149).
        try:
            cluster = EmergentCluster(members=node.members)
            name = self._name_fn(cluster, candidates, self._hermes_url, self._token)
            if name is None or name.confidence < self._config.hermes_confidence_floor:
                logger.info("emergence: skip cluster (no/low-confidence name)")
                return

            is_leaf = not node.children
            existing = self._match_existing_type(node.centroid, parent_type_uuid)
            if existing is not None:
                # Reconcile into the existing type rather than mint a duplicate
                # (#504): retype the leaf's members onto it; children nest under
                # it using its stored ancestors.
                type_uuid = existing
                existing_node = self._hcg.get_node(type_uuid) or {}
                existing_props = existing_node.get("properties") or {}
                child_ancestors = list(
                    existing_props.get("ancestors") or parent_ancestors
                )
                if is_leaf:
                    self._attach_members(node.members, type_uuid, existing_node)
                logger.info(
                    "emergence: reconciled %d members into existing type %s",
                    len(node.members),
                    type_uuid,
                )
            else:
                cluster_id = uuid_lib.uuid4().hex[:8]
                type_uuid = self._mint_fn(
                    cluster,
                    name,
                    hcg=self._hcg,
                    milvus=self._milvus,
                    source_cluster_id=cluster_id,
                    parent_type_uuid=parent_type_uuid,
                    parent_ancestors=parent_ancestors,
                    retype_members=is_leaf,
                )
                if name.label not in candidates:
                    candidates.append(name.label)
                # The minted type's own ancestors are parent_ancestors + its
                # parent's name -- that's the chain its children descend from.
                child_ancestors = list(parent_ancestors) + [
                    parent_type_uuid.removeprefix("type_")
                ]
                if self._event_bus is not None:
                    self._event_bus.publish(
                        ONTOLOGY_CHANGED_CHANNEL,
                        {
                            "type_uuid": type_uuid,
                            "name": name.label,
                            "ancestors": child_ancestors,
                        },
                    )
                logger.info(
                    "emergence: minted %s from %d members", name.label, cluster.size
                )

            for child in node.children:
                self._mint_subtree(child, type_uuid, child_ancestors, candidates)
        except Exception:
            logger.exception("emergence: node failed, skipping")

    def _match_existing_type(
        self, centroid: list[float], parent_type_uuid: str
    ) -> str | None:
        """Nearest existing type whose centroid is within the match threshold of
        ``centroid``, or None to mint fresh (#504 match-before-mint).

        The parent being subdivided is excluded -- a cluster matching its own
        parent isn't a distinct type, so we mint/keep rather than self-attach.
        """
        try:
            nearest = self._milvus.find_nearest_types(centroid, top_k=1)
        except Exception:
            logger.exception("emergence: find_nearest_types failed")
            return None
        if not nearest:
            return None
        cand_uuid = nearest[0].get("uuid")
        if not cand_uuid or cand_uuid == parent_type_uuid:
            return None
        try:
            row = self._milvus.get_embedding(node_type="TypeCentroid", uuid=cand_uuid)
        except Exception:
            logger.exception("emergence: get_embedding failed for %s", cand_uuid)
            return None
        if not row or not row.get("embedding"):
            return None
        if _cosine(centroid, row["embedding"]) < self._config.type_match_threshold:
            return None
        return str(cand_uuid)

    def _attach_members(
        self, members: list[Member], type_uuid: str, type_node: dict[str, Any]
    ) -> None:
        """Retype members onto an existing type when reconciling (#504).

        Membership is the authoritative `type_uuid` property plus the `type`
        slug -- the same convention mint_type uses; no IS_A edge.
        """
        slug = _slugify(type_node.get("name") or _type_name(type_uuid))
        for member in members:
            self._hcg.update_node(member.uuid, {"type": slug, "type_uuid": type_uuid})


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
