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

logger = logging.getLogger(__name__)

ONTOLOGY_CHANGED_CHANNEL = "ontology.type_created"

# The base "junk-drawer" realm whose Milvus collection physically holds the
# entity-derived embeddings. It is NOT a membership assertion any more (B2/B3):
# membership is the instance->type IS_A edge, read via get_members_of_type. This
# constant only routes embedding reads to the base collection (_build_member).
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
    Used only as the ``current_type`` display fallback in :func:`_build_member`;
    membership itself is the IS_A edge, never this slug (B2/B3, DESIGN §3).
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


def _member_rows(hcg: Any, type_uuid: str) -> list[dict[str, Any]]:
    """Resolve the node rows that are members of ``type_uuid``.

    Membership is the instance->type ``IS_A`` edge (B2/B3, DESIGN §3): a node is
    a member of ``type_uuid`` iff its single upward IS_A edge points there. The
    edge query is anchored on the type uuid, so it serves both cases uniformly --
    a REALM-ROOT uuid yields that realm's drainage pool (entities parked directly
    under the root) and a minted-type uuid yields that type's members. There is
    no node-type scan and no ``type_uuid``-property/slug read any more (de-slug);
    a member re-pointed away from this type is excluded automatically because its
    edge no longer targets it.
    """
    return [n for n in (hcg.get_members_of_type(type_uuid) or []) if n and "uuid" in n]


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

    Membership is resolved by :func:`_member_rows` (the instance->type IS_A edge
    query, uniform across the realm-root pool and minted types). Embeddings come
    from Milvus; the structural signature is built from the node's outgoing
    reified edges (relation + resolved neighbor type). Nodes without an embedding
    are skipped (they can't be clustered).
    """
    type_name = _type_name(type_uuid)
    members: list[Member] = []
    for row in _member_rows(hcg, type_uuid):
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

        # The in-pass IS_A adjacency placement.reparent reads (cycle guard) and
        # keeps consistent as members are re-pointed onto their types. Members are
        # leaves, so the cycle check is trivially false and the type-layer
        # hierarchy is never polluted; the map only tracks the instance->type
        # edges drawn this pass.
        children_of: dict[str, list[str]] = {}

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
                    children_of=children_of,
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
        children_of: dict[str, list[str]],
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

        # Cascade (DESIGN sec 6) as a single MINT-OR-ATTACH invariant (#200):
        # Hermes points at the placement (centroids never decide). We MINT a new
        # type ONLY when it hands us BOTH a new name AND a parent to hang it under.
        # Every other placeable answer points at an EXISTING type -- it named one
        # that already exists -- so we ATTACH the cohort to that node and mint
        # nothing. This makes same-name dedup implicit (#38: a name an earlier
        # cluster/prior pass minted resolves through uuid_by_name and is reused,
        # never re-minted) and, crucially, makes reuse REALM-AGNOSTIC: a cohort
        # Hermes names after another realm's root (e.g. processes pooled under
        # `entity`) is corrected onto that real root rather than spawning a
        # duplicate root-named type. (A no-name answer is already held above; a
        # new name with no parent has nothing to point at, so it is held too.)
        existing = uuid_by_name.get(tc.name.strip().lower())
        if existing is not None:
            if existing == source_type_uuid:
                # The name resolves to the cohort's own pool root: a no-op. Leave
                # the members where they are rather than re-pointing onto
                # themselves (and never mint a duplicate of the root).
                logger.info(
                    "emergence: cluster maps to its own pool root; left in pool"
                )
                return
            # Attach: re-point the cohort's instance->type IS_A edges onto the
            # existing type (membership is the edge -- no mint, no type_uuid
            # stamp), regardless of realm.
            self._place_members(fitting, existing, "name_reuse", children_of)
            logger.info(
                "emergence: attached %d members to %s (name_reuse)",
                len(fitting),
                existing,
            )
            return

        # MINT a new type (the name is not an existing type). Every emergent type
        # MUST root under a domain: resolve the proposed parent closed-world, and
        # if Hermes gave no usable parent fall back to the SOURCE pool's realm root
        # (entity/concept/process) -- never hold it unparented. The attach path
        # above already handled existing names, so this never duplicates a root.
        # Minting under a realm root is always legal.
        resolved_parent = (
            placement.resolve_parent(
                tc.parent, uuid_by_name=uuid_by_name, hcg=self._hcg, realm=realm
            )
            if tc.parent
            else None
        )
        parent_uuid = resolved_parent or realm_root_uuid
        # parent_resolution: Hermes named a parent that RESOLVED (even if it is a
        # realm root, e.g. "entity"). root_fallback: no usable parent was given /
        # it did not resolve, so we defaulted to the source realm root. Keeping
        # these distinct preserves event-bus traceability -- an explicit-root
        # graft and an unresolvable-parent fallback are different stories
        # (greptile #201).
        placed_by = (
            "parent_resolution" if resolved_parent is not None else "root_fallback"
        )

        # Mint a new type under the chosen parent. mint_type writes the type's
        # own type->parent IS_A edge (carrying placed_by) but does NOT touch the
        # members -- drainage owns member placement.
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
        )
        # Re-point each fitting member's instance->type IS_A edge onto the
        # freshly-minted type, carrying the SAME placed_by the type was placed by
        # (parent_resolution / root_fallback). Membership is the edge (DESIGN §3).
        self._place_members(fitting, new_uuid, placed_by, children_of)
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

    def _place_members(
        self,
        members: list[Member],
        type_uuid: str,
        placed_by: str,
        children_of: dict[str, list[str]],
    ) -> None:
        """Re-point each member's single upward instance->type IS_A edge to a type.

        Membership is the edge now (B2/B3, DESIGN §3): for every member,
        :func:`placement.reparent` drops its stale realm/parent IS_A edge and
        writes the new one to ``type_uuid`` carrying ``placed_by`` (the SAME
        reason the type itself was placed by). No ``type_uuid``/``type`` property
        is stamped. Members are leaves, so the cycle guard is trivially false and
        the type-layer hierarchy in ``children_of`` is unaffected.
        """
        for member in members:
            placement.reparent(
                member.uuid,
                type_uuid,
                hcg=self._hcg,
                children_of=children_of,
                placed_by=placed_by,
            )


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
