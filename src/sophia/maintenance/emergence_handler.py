"""The 'type_emergence' maintenance handler (B1 flat parent-driven drainage).

Dispatched by MaintenanceScheduler as handlers['type_emergence'](type_uuid=...).
Dependencies (load_members / name_fn / mint_fn) are injected so the
orchestration is unit-testable without Neo4j / Milvus / Hermes.

Drainage is FLAT and parent-driven (DESIGN sec 5 / sec 6): embeddings only
PROPOSE clusters; the graph ASSERTS placement via the parent the LLM names,
validated closed-world through sophia.maintenance.placement. Centroids never
decide placement, depth accumulates across passes (never within one), and
structure (the single upward IS_A edge) is the only membership -- no ancestors
snapshot and no type_<name> slug (names resolve via the in-pass catalog map).
"""

from __future__ import annotations

import logging
import uuid as uuid_lib
from collections.abc import Callable
from typing import Any

from sophia.maintenance import placement
from sophia.maintenance.config import MaintenanceConfig
from sophia.maintenance.emergence_clustering import find_emergent_clusters
from sophia.maintenance.emergence_types import EmergentCluster, Member, NameResult
from sophia.maintenance.structural_signature import build_signature
from sophia.maintenance.type_minting import _slugify

logger = logging.getLogger(__name__)

ONTOLOGY_CHANGED_CHANNEL = "ontology.type_created"

# The base "junk-drawer" type that holds un-specialised nodes. Its membership is
# resolved by node-type scan; minted types resolve membership via the
# authoritative `type_uuid` property (no instance->type IS_A edges -- #505).
_BASE_TYPE = "entity"

# The three content realm roots. A realm-root pool (e.g. the base `entity`
# junk-drawer) IS its own realm; any other pool walks IS_A up to one of these.
# Names are matched case-folded against the in-pass catalog (DESIGN sec 3 / 5).
_GRAFTABLE_REALMS = frozenset({"entity", "concept", "process"})


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 if either is zero).

    Retained here (emergence no longer matches on centroids -- B1) because the
    gated-off rollup tier still imports it for its own centroid comparisons.
    """
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

    def run(self, type_uuid: str) -> None:
        """Drain one realm pool: cluster -> name -> flat parent-driven placement.

        Embeddings only PROPOSE the clusters; placement is asserted via the
        parent the LLM names (or the realm root), validated closed-world. The
        pass is FLAT -- one placement per cluster, no per-pass sub-tree.
        """
        members = self._load_members(type_uuid)
        # Embeddings propose cohesive sub-groups; they never decide placement.
        clusters = find_emergent_clusters(
            members,
            min_cluster_size=self._config.min_cluster_size,
            variance_threshold=self._config.variance_threshold,
        )
        if not clusters:
            logger.info("emergence: no qualifying clusters in %s", type_uuid)
            return

        # In-pass catalog (the de-slug): every type name -> its real uuid, lower
        # cased. Names resolve ONLY through this map -- never a fabricated
        # `type_<name>` slug (DESIGN sec 3).
        uuid_by_name = {
            n["name"].strip().lower(): n["uuid"]
            for n in self._hcg.list_all_nodes(node_type="type_definition")
            if n.get("name") and n.get("uuid")
        }

        # Resolve the pool's realm. A realm-root pool (entity/concept/process)
        # IS its own realm; any other pool walks IS_A up to its realm root. We
        # can only place inside a realm, so bail if there is none.
        node = self._hcg.get_node(type_uuid) or {}
        node_name = (node.get("name") or "").strip().lower()
        if node_name in _GRAFTABLE_REALMS:
            realm = node_name
        else:
            realm = placement.realm_of(type_uuid, hcg=self._hcg)
        if realm is None:
            logger.info("emergence: %s has no realm; skipping", type_uuid)
            return
        realm_root_uuid = uuid_by_name.get(realm)
        if realm_root_uuid is None:
            logger.info("emergence: realm %r has no catalog uuid; skipping", realm)
            return

        # Per-cluster isolation: a transient error on one cluster must not abort
        # the whole pass -- log it and let the others place.
        for cluster in clusters:
            try:
                self._place_cluster(
                    cluster,
                    realm=realm,
                    realm_root_uuid=realm_root_uuid,
                    uuid_by_name=uuid_by_name,
                    source_type_uuid=type_uuid,
                )
            except Exception:
                logger.exception("emergence: cluster failed, skipping")

    def _place_cluster(
        self,
        cluster: EmergentCluster,
        *,
        realm: str,
        realm_root_uuid: str,
        uuid_by_name: dict[str, str],
        source_type_uuid: str,
    ) -> None:
        """Place ONE cluster flat: the LLM names it, the cascade decides where.

        The parent the LLM suggests drives placement, validated closed-world via
        :mod:`placement` (centroids never decide). Outliers are left untouched in
        the realm pool -- no graveyard, reconsidered next pass (DESIGN sec 5).
        """
        tc = self._name_fn(cluster)
        if tc is None or not (tc.name or "").strip():
            logger.info("emergence: cluster left in pool (no name)")
            return

        # Outliers stay in the pool (DESIGN sec 5 -- no graveyard): drop them from
        # the cohort we place, but do NOT retype or park them. Untouched, they
        # keep pointing at the realm root and re-enter the next pass.
        residual = set(tc.residual_ids or [])
        fitting = [m for m in cluster.members if m.uuid not in residual]
        if not fitting:
            logger.info("emergence: cluster all outliers, leaving in pool")
            return

        # Cascade (DESIGN sec 6): resolvable parent -> mint under it; else reuse
        # an in-realm type of the same name; else mint under the realm root.
        parent_uuid: str | None = None
        placed_by: str | None = None
        reuse_target: str | None = None
        if tc.parent:
            pu = placement.resolve_parent(
                tc.parent, uuid_by_name=uuid_by_name, hcg=self._hcg, realm=realm
            )
            if pu:
                parent_uuid, placed_by = pu, "parent_resolution"
        if parent_uuid is None:
            existing = uuid_by_name.get(tc.name.strip().lower())
            if (
                existing
                and existing != source_type_uuid
                and placement.realm_of(existing, hcg=self._hcg) == realm
            ):
                reuse_target, placed_by = existing, "name_reuse"
            else:
                parent_uuid, placed_by = realm_root_uuid, "root_fallback"

        if reuse_target is not None:
            # Reuse: retype the cohort onto the existing same-name type (membership
            # is the type_uuid property -- no mint, no new edge).
            existing_node = self._hcg.get_node(reuse_target) or {}
            self._attach_members(fitting, reuse_target, existing_node)
            logger.info(
                "emergence: reused %s for %d members (%s)",
                reuse_target,
                len(fitting),
                placed_by,
            )
            return

        # Mint a new type under the chosen parent. mint_type stamps the
        # type->parent IS_A edge with placed_by (the parent-driven traceability).
        name_obj = NameResult(
            label=tc.name, description="", confidence=1.0, removed=[], parent=tc.parent
        )
        new_uuid = self._mint_fn(
            EmergentCluster(members=fitting),
            name_obj,
            hcg=self._hcg,
            milvus=self._milvus,
            source_cluster_id=uuid_lib.uuid4().hex[:8],
            parent_type_uuid=parent_uuid,
            placed_by=placed_by,
            retype_members=True,
        )
        # In-pass dedup: a later same-named cluster reuses this freshly-minted
        # uuid instead of re-minting a sibling.
        uuid_by_name[tc.name.strip().lower()] = new_uuid
        if self._event_bus is not None:
            self._event_bus.publish(
                ONTOLOGY_CHANGED_CHANNEL,
                {
                    "type_uuid": new_uuid,
                    "name": tc.name,
                    "parent_uuid": parent_uuid,
                    "placed_by": placed_by,
                },
            )
        logger.info(
            "emergence: minted %s (%s) from %d members",
            tc.name,
            placed_by,
            len(fitting),
        )

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
    """Return the callable registered as handlers['type_emergence'].

    name_fn wraps the Hermes v2 /type-cluster client: the catalog lives server
    side, so no candidates are sent. Dedup across clusters in a pass happens via
    the in-pass `uuid_by_name` map, not a candidates seam.
    """
    from sophia.maintenance.hermes_naming import type_cluster
    from sophia.maintenance.type_minting import mint_type

    handler = EmergenceHandler(
        config=config,
        hcg=hcg,
        milvus=milvus,
        event_bus=event_bus,
        hermes_url=hermes_url,
        token=token,
        load_members=lambda u: load_type_members(hcg, milvus, u),
        name_fn=lambda c: type_cluster(
            c,
            hermes_url=hermes_url,
            token=token,
            max_members=config.max_cluster_size,
        ),
        mint_fn=mint_type,
    )

    def _run(type_uuid: str) -> None:
        handler.run(type_uuid=type_uuid)

    return _run
