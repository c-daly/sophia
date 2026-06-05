"""Type-level rollup: build hierarchy over the flat shelf of emergent types (#160).

A slow-cadence maintenance pass. Emergence mints a wide flat layer of leaf
type-definitions under `entity` and never revisits it, so the ontology stays
flat. This pass groups those *types* into super-types. It RE-PARENTS existing
type-definitions (sets `ancestors` + a single `type->parent IS_A` edge); it never
retypes instance members.

Two tiers, run in order:
  1. **Explicit subsumption lift** -- member relations already cross type
     boundaries with meaningful labels (`HAS_PART`/`INCLUDES`/`COMPOSED_OF` =>
     parent->child, `PART_OF` => child->parent, `ALSO_KNOWN_AS` => synonym).
     Aggregate them into a type-type graph and lift the directed ones into
     type-level IS_A; alias synonyms under a canonical.
  2. **Residual clustering** -- types with no explicit subsumption edge are
     clustered recursively into super-types via the same `find_emergent_hierarchy`
     used for entity emergence, but with type centroids as the points.

Idempotent: `_reparent_one` change-detects and is a no-op when the type already
has the right parent/ancestors; super-types reconcile (match-before-mint) so a
re-run does not duplicate them. Safe to fire on a plain periodic trigger.
"""

from __future__ import annotations

import logging
import uuid as uuid_lib
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any

from sophia.maintenance.config import MaintenanceConfig
from sophia.maintenance.emergence_clustering import find_emergent_hierarchy
from sophia.maintenance.emergence_handler import _cosine, _type_name
from sophia.maintenance.emergence_types import EmergentCluster, Member

logger = logging.getLogger(__name__)

_BASE_TYPE = "entity"
_ENTITY_ANCESTORS = ["root", "node"]
_DEFAULT_MODEL = "all-MiniLM-L6-v2"
# Domain realm roots a TOP-LEVEL super-type may be grafted under (besides the
# `entity` junk-drawer). Sent as naming candidates so Hermes can name one as a
# super-type's parent -- this is how the flat `entity` shelf reaches the other
# roots. `cognition` is intentionally excluded: it is populated by metacognitive
# schema induction, not by domain type rollup.
_REALM_ROOTS = ("concept", "process")
# Seeded structural roots that must NEVER be minted as a super, matched/reused as
# a super, or reparented -- they are the fixed top of the ontology. Guards the
# rollup against corrupting them (review blocker + duplicate-root path).
_PROTECTED_ROOT_NAMES = frozenset(
    {"root", "node", "entity", "concept", "cognition", "process"}
)
_PROTECTED_ROOT_UUIDS = frozenset(f"type_{n}" for n in _PROTECTED_ROOT_NAMES)

# Member-relation labels that imply a type-level relationship.
_PARENT_REL = {"HAS_PART", "INCLUDES", "COMPOSED_OF"}  # source-type is the PARENT
_CHILD_REL = {"PART_OF"}  # source-type is the CHILD (invert)
_SYNONYM_REL = {"ALSO_KNOWN_AS"}  # undirected synonymy -> alias
_ALL_RELS = _PARENT_REL | _CHILD_REL | _SYNONYM_REL
_BIG = 100_000


def _is_rollup_candidate(td: dict) -> bool:
    """A minted, non-reserved entity-side type-def eligible for rollup."""
    name = (td.get("name") or "").strip()
    uuid = td.get("uuid") or ""
    if not name or not uuid.startswith("type_"):
        return False
    if name in _PROTECTED_ROOT_NAMES or name.startswith("reserved_"):
        return False
    props = td.get("properties") or {}
    anc = props.get("ancestors")
    if (
        isinstance(anc, list) and "edge_type" in anc
    ):  # edge-type-defs are not entity types
        return False
    return bool(props.get("is_type_definition"))


class TypeRollupHandler:
    """Groups the flat layer of emergent type-defs into super-types (#160)."""

    def __init__(
        self,
        *,
        config: MaintenanceConfig,
        hcg: Any,
        milvus: Any,
        event_bus: Any,
        hermes_url: str,
        token: str,
        name_fn: Any,
        mint_fn: Any,
    ) -> None:
        self._config = config
        self._hcg = hcg
        self._milvus = milvus
        self._event_bus = event_bus
        self._hermes_url = hermes_url
        self._token = token
        self._name_fn = name_fn
        self._mint_fn = mint_fn
        # Per-pass scratch (rebuilt each run()).
        self._children_of: dict[str, list[str]] = {}
        self._name_of: dict[str, str] = {}
        self._uuid_by_name: dict[str, str] = {}

    # ------------------------------------------------------------------ pass
    def run(self) -> None:
        rows = self._load_type_layer()
        if len(rows) < 2:
            logger.info(
                "type_rollup: %d candidate types, nothing to roll up", len(rows)
            )
            return
        self._build_is_a_adjacency()
        self._name_of = {r["uuid"]: r["name"] for r in rows}
        # name -> uuid, for name-based reconcile (no duplicate-named type-defs).
        self._uuid_by_name = {r["name"]: r["uuid"] for r in rows}

        # Include the realm roots so Hermes can name one as a super-type's
        # parent: the rollup is the single authority that roots types under
        # concept/process (emergence stays flat-under-entity).
        candidates = list({r["name"] for r in rows} | set(_REALM_ROOTS))
        # Tier 1: explicit subsumption + synonyms.
        explicit_children = self._tier1_explicit(rows)
        # Tier 2: cluster only the *flat leaves* -- types still directly under
        # `entity` (ancestor depth <= 3) that are not already a super-type (no
        # IS_A children). Types already lifted into a hierarchy, and the
        # super-types built by a prior pass, are left in place, so a re-run never
        # re-clusters established structure. That is the idempotency anchor: no
        # new layers and no duplicate super-types on repeated runs.
        residual = [
            r
            for r in rows
            if r["uuid"] not in explicit_children
            and r.get("centroid")
            and len(r["ancestors"]) <= 3
            and not self._children_of.get(r["uuid"])
        ]
        self._tier2_residual(residual, candidates)

    # ------------------------------------------------------------------ load
    def _load_type_layer(self) -> list[dict]:
        try:
            type_defs = self._hcg.get_all_type_definitions()
        except Exception:
            logger.exception("type_rollup: failed to list type definitions")
            return []
        rows: list[dict] = []
        for td in type_defs or []:
            if not _is_rollup_candidate(td):
                continue
            uuid = td["uuid"]
            props = td.get("properties") or {}
            centroid = None
            model = _DEFAULT_MODEL
            try:
                emb = self._milvus.get_embedding(node_type="TypeCentroid", uuid=uuid)
                if emb and emb.get("embedding"):
                    centroid = emb["embedding"]
                    model = emb.get("embedding_model") or _DEFAULT_MODEL
            except Exception:
                logger.debug("type_rollup: no centroid for %s", uuid)
            rows.append(
                {
                    "uuid": uuid,
                    "name": td.get("name") or _type_name(uuid),
                    "ancestors": list(props.get("ancestors") or _ENTITY_ANCESTORS),
                    "centroid": centroid,
                    "model": model,
                }
            )
        return rows

    # ------------------------------------------------------------ tier 1
    def _tier1_explicit(self, rows: list[dict]) -> set[str]:
        """Lift explicit member subsumption into type-level IS_A. Returns the set
        of child type uuids that received an explicit parent (excluded from tier 2)."""
        type_uuids = {r["uuid"] for r in rows}
        # weight[(src_type, rel, tgt_type)] = count of member edges
        weights: Counter[tuple[str, str, str]] = Counter()
        for rel in _ALL_RELS:
            try:
                edges = self._hcg.list_all_edges(relation_type=rel, limit=_BIG)
            except Exception:
                logger.exception("type_rollup: list_all_edges(%s) failed", rel)
                continue
            endpoints = {e["source"] for e in edges or []} | {
                e["target"] for e in edges or []
            }
            type_of = self._type_uuid_map(list(endpoints))
            for e in edges or []:
                st, tt = type_of.get(e["source"]), type_of.get(e["target"])
                if not st or not tt or st == tt:
                    continue
                if st not in type_uuids or tt not in type_uuids:
                    continue
                weights[(st, rel, tt)] += 1

        # Resolve a single best parent per child type.
        # parent candidates: (child, parent, weight)
        votes: dict[str, Counter[str]] = defaultdict(Counter)
        for (st, rel, tt), w in weights.items():
            if rel in _PARENT_REL:
                votes[tt][st] += w  # tt (child) <- st (parent)
            elif rel in _CHILD_REL:
                votes[st][tt] += w  # st (child) <- tt (parent)
            elif rel in _SYNONYM_REL:
                # Alias the lexicographically-later uuid under the earlier (stable canonical).
                child, parent = (st, tt) if st > tt else (tt, st)
                votes[child][parent] += w

        explicit_children: set[str] = set()
        for child, parent_counter in votes.items():
            parent, _ = parent_counter.most_common(1)[0]
            if parent == child:
                continue
            p_row = next((r for r in rows if r["uuid"] == parent), None)
            if p_row is None:
                continue
            # Guard against a 2-cycle (A parent-of B and B parent-of A): keep the
            # heavier direction. On an exact tie the strict `>` deliberately lets
            # both through -- the second reparent then hits the structural cycle
            # guard in _reparent_one and is recorded as AMBIGUOUS_SUBSUMPTION
            # rather than silently dropped. Do NOT change to `>=`, which would lose
            # that ambiguity signal (gemini #161).
            if votes.get(parent, Counter()).get(child, 0) > parent_counter.get(
                parent, 0
            ):
                continue
            self._reparent_one(child, parent, p_row["ancestors"], p_row["name"])
            explicit_children.add(child)
        if explicit_children:
            logger.info(
                "type_rollup: tier-1 lifted %d explicit subsumptions",
                len(explicit_children),
            )
        return explicit_children

    def _type_uuid_map(self, node_uuids: list[str]) -> dict[str, str]:
        # Resolve in chunks: a single get_nodes_batch over every member uuid can
        # exceed query-size/timeout limits on a large graph (gemini #161).
        out: dict[str, str] = {}
        chunk = 500
        for i in range(0, len(node_uuids), chunk):
            try:
                nodes = self._hcg.get_nodes_batch(node_uuids[i : i + chunk]) or []
            except Exception:
                logger.exception("type_rollup: get_nodes_batch failed")
                continue
            for n in nodes:
                if not n or "uuid" not in n:
                    continue
                tu = n.get("type_uuid") or (n.get("properties") or {}).get("type_uuid")
                if tu:
                    out[n["uuid"]] = tu
        return out

    # ------------------------------------------------------------ tier 2
    def _tier2_residual(self, residual: list[dict], candidates: list[str]) -> None:
        if len(residual) < self._config.rollup_min_supercluster_size:
            logger.info(
                "type_rollup: %d residual types, below supercluster floor",
                len(residual),
            )
            return
        members = [
            Member(
                uuid=r["uuid"],
                name=r["name"],
                embedding=r["centroid"],
                signature=Counter(),
                current_type="type_definition",
                hermes_type_hint=None,
                neighbors=[],
                model=r["model"],
            )
            for r in residual
        ]
        hierarchy = find_emergent_hierarchy(
            members,
            min_cluster_size=self._config.rollup_min_cluster_size,
            variance_threshold=0.0,  # no junk-drawer cohesion gate at the type layer
            min_supercluster_size=self._config.rollup_min_supercluster_size,
        )
        if not hierarchy:
            logger.info(
                "type_rollup: no super-structure in %d residual types", len(residual)
            )
            return
        for node in hierarchy:
            self._reparent_subtree(
                node, "type_entity", _ENTITY_ANCESTORS, _BASE_TYPE, candidates
            )

    def _reparent_subtree(
        self,
        node: Any,
        parent_type_uuid: str,
        parent_ancestors: list[str],
        parent_name: str,
        candidates: list[str],
    ) -> None:
        """Internal nodes => mint/reuse a super-type; leaves => re-parent the
        existing type-defs (a leaf Member's uuid IS a type_uuid)."""
        try:
            if not node.children:
                # Leaf: each member is an existing type-def; re-parent it directly.
                for m in node.members:
                    self._reparent_one(
                        m.uuid, parent_type_uuid, parent_ancestors, parent_name
                    )
                return
            # Internal node => a super-type over its children.
            name = self._name_fn(
                EmergentCluster(members=node.members),
                candidates,
                self._hermes_url,
                self._token,
            )
            if name is None or name.confidence < self._config.hermes_confidence_floor:
                logger.info("type_rollup: skip super-type (no/low-confidence name)")
                return
            # A cluster member must never be selected as the super-type of its
            # own peers (greptile #161): that yields peer-as-parent IS_A edges
            # and cascaded ancestor chains. Exclude the members from BOTH the
            # centroid match and the name reconcile below.
            member_uuids = {m.uuid for m in node.members}
            # Never use a seeded structural root as a super-type *name*: minting
            # would duplicate the root, and reusing it would reparent it. If
            # Hermes labelled the cluster after a root, skip this super level and
            # attach the children directly under the current parent.
            if f"type_{name.label}" in _PROTECTED_ROOT_UUIDS:
                logger.info(
                    "type_rollup: cluster named after root %r -- skipping super level",
                    name.label,
                )
                for child in node.children:
                    self._reparent_subtree(
                        child,
                        parent_type_uuid,
                        parent_ancestors,
                        parent_name,
                        candidates,
                    )
                return
            # Root a TOP-LEVEL super-type under the realm Hermes named (concept /
            # process) instead of flat under `entity` -- this is how the entity
            # shelf reaches the other roots. Top level only; deeper super-types
            # keep nesting under their tree parent. Guard the rebind: never graft
            # under one of the cluster's own members or under the coined label
            # itself (mint_fn has no cycle guard). Degrades to the default parent
            # when the suggestion does not resolve.
            if (
                parent_type_uuid == f"type_{_BASE_TYPE}"
                and name.parent
                and name.parent.strip().lower() != name.label.strip().lower()
            ):
                rooted = self._resolve_parent(name.parent)
                if (
                    rooted is not None
                    and rooted[0] not in member_uuids
                    and rooted[0] != self._uuid_by_name.get(name.label)
                ):
                    parent_type_uuid, parent_ancestors, parent_name = rooted
                    logger.info(
                        "type_rollup: rooting super-type %r under realm %r",
                        name.label,
                        parent_name,
                    )
            super_uuid = self._match_existing_type(
                node.centroid, parent_type_uuid, exclude=member_uuids
            )
            # Never reuse a seeded structural root as a domain super-type: a root
            # is not a cluster member, so exclude=members does not catch it, and
            # the reuse branch's _reparent_one would then move the root into the
            # domain tree (no cycle, so the cycle guard cannot stop it). Treat a
            # realm-root hit as no-match -> mint a fresh super instead (blocker).
            if super_uuid in _PROTECTED_ROOT_UUIDS:
                super_uuid = None
            if super_uuid is None:
                # Name-based reconcile before minting: never create a second
                # type-def with a name that already exists (the duplicate
                # `name`_hex problem). The centroid match can miss two clusters
                # Hermes named identically; an exact name is a strong reconcile
                # signal. Skip if it is the parent or a member of this very
                # cluster (reusing those would just re-introduce a cycle).
                by_name = self._uuid_by_name.get(name.label)
                if (
                    by_name
                    and by_name != parent_type_uuid
                    and by_name not in member_uuids
                ):
                    super_uuid = by_name
                    logger.info(
                        "type_rollup: reusing existing type '%s' (%s) by name "
                        "instead of minting a duplicate",
                        name.label,
                        by_name,
                    )
            if super_uuid is None:
                cid = uuid_lib.uuid4().hex[:8]
                super_uuid = self._mint_fn(
                    EmergentCluster(members=node.members),
                    name,
                    hcg=self._hcg,
                    milvus=self._milvus,
                    source_cluster_id=cid,
                    parent_type_uuid=parent_type_uuid,
                    parent_ancestors=parent_ancestors,
                    parent_name=parent_name,
                    retype_members=False,  # super-type owns no instances
                )
                if name.label not in candidates:
                    candidates.append(name.label)
                super_name = name.label
                self._name_of[super_uuid] = super_name
                self._uuid_by_name[super_name] = super_uuid
                if self._event_bus is not None:
                    try:
                        self._event_bus.publish(
                            "ontology.type_created",
                            {
                                "type_uuid": super_uuid,
                                "name": super_name,
                                "ancestors": list(parent_ancestors) + [parent_name],
                            },
                        )
                    except Exception:
                        logger.exception("type_rollup: event publish failed")
                logger.info(
                    "type_rollup: minted super-type %s over %d children",
                    super_name,
                    len(node.children),
                )
                # mint_fn wrote the super's IS_A edge to the graph; mirror it in
                # the in-memory adjacency so the cycle guard sees it this pass.
                self._children_of.setdefault(parent_type_uuid, [])
                if super_uuid not in self._children_of[parent_type_uuid]:
                    self._children_of[parent_type_uuid].append(super_uuid)
            else:
                super_name = (self._hcg.get_node(super_uuid) or {}).get(
                    "name"
                ) or _type_name(super_uuid)
                # The super was REUSED (centroid- or name-match), not minted, so
                # nothing has placed it under this parent. mint_fn does that for
                # fresh supers; the reuse paths must do it explicitly or the
                # super keeps its old position while its new children get
                # ancestors for the intended one (divergence). Idempotent no-op
                # when already correct; cycle-guarded if it would loop.
                self._reparent_one(
                    super_uuid, parent_type_uuid, parent_ancestors, parent_name
                )
            super_ancestors = list(parent_ancestors) + [parent_name]
            for child in node.children:
                self._reparent_subtree(
                    child, super_uuid, super_ancestors, super_name, candidates
                )
        except Exception:
            logger.exception("type_rollup: subtree failed, skipping")

    # ----------------------------------------------------- reparent + cascade
    def _reparent_one(
        self,
        child_uuid: str,
        new_parent_uuid: str,
        new_parent_ancestors: list[str],
        new_parent_name: str,
    ) -> None:
        """Idempotent re-parent of one type-def + ancestor cascade. No-op when the
        type already has the right parent and ancestors (the convergence anchor)."""
        if child_uuid == new_parent_uuid:
            return
        if self._creates_cycle(child_uuid, new_parent_uuid):
            # A cycle means we've claimed subsumption in both directions: the two
            # types are too alike to order into parent/child. Don't force a
            # (false) IS_A -- record the pair as a misunderstood, unresolved
            # relationship so Sophia keeps the signal instead of dropping it.
            self._record_ambiguous(child_uuid, new_parent_uuid)
            return
        target_ancestors = list(new_parent_ancestors) + [new_parent_name]
        cur = self._hcg.get_node(child_uuid) or {}
        cur_anc = list((cur.get("properties") or {}).get("ancestors") or [])
        cur_parent, cur_edge = self._current_is_a(child_uuid)
        if cur_parent == new_parent_uuid and cur_anc == target_ancestors:
            return  # already correct -> true no-op
        # 1. swing the IS_A edge
        if cur_parent is not None and cur_parent != new_parent_uuid:
            try:
                # Drop the stale parent edge by its own id when we have one;
                # fall back to a (source, target, relation) match when the edge
                # was persisted without an id/uuid. delete_edge(None) would
                # silently no-op and leave the old IS_A in place, so the child
                # would end up with two IS_A parents (greptile #161).
                if cur_edge:
                    self._hcg.delete_edge(cur_edge)
                else:
                    self._hcg.delete_edges_between(child_uuid, cur_parent, "IS_A")
                siblings = self._children_of.get(cur_parent)
                if siblings and child_uuid in siblings:
                    siblings.remove(child_uuid)
            except Exception:
                logger.exception(
                    "type_rollup: delete stale IS_A failed for %s", child_uuid
                )
        if cur_parent != new_parent_uuid:
            try:
                self._hcg.add_edge(
                    child_uuid, new_parent_uuid, "IS_A"
                )  # MERGE -> idempotent
                self._children_of.setdefault(new_parent_uuid, [])
                if child_uuid not in self._children_of[new_parent_uuid]:
                    self._children_of[new_parent_uuid].append(child_uuid)
            except Exception:
                logger.exception("type_rollup: add IS_A failed for %s", child_uuid)
        # 2. set ancestors
        if cur_anc != target_ancestors:
            try:
                self._hcg.update_node(child_uuid, {"ancestors": target_ancestors})
            except Exception:
                logger.exception(
                    "type_rollup: update ancestors failed for %s", child_uuid
                )
        # 3. cascade to descendants
        child_name = (
            cur.get("name") or self._name_of.get(child_uuid) or _type_name(child_uuid)
        )
        self._cascade_descendants(child_uuid, target_ancestors + [child_name], set())

    def _cascade_descendants(
        self, node_uuid: str, childrens_ancestors: list[str], seen: set[str]
    ) -> None:
        """Top-down recompute of each descendant's ancestors. `childrens_ancestors`
        is what a direct child of `node_uuid` must have."""
        if node_uuid in seen:
            return
        seen.add(node_uuid)
        for child in list(self._children_of.get(node_uuid, [])):
            cn = self._hcg.get_node(child) or {}
            cur = list((cn.get("properties") or {}).get("ancestors") or [])
            if cur != childrens_ancestors:
                try:
                    self._hcg.update_node(child, {"ancestors": childrens_ancestors})
                except Exception:
                    logger.exception("type_rollup: cascade update failed for %s", child)
            cname = cn.get("name") or self._name_of.get(child) or _type_name(child)
            self._cascade_descendants(child, childrens_ancestors + [cname], seen)

    # ----------------------------------------------------------- helpers
    def _creates_cycle(self, child_uuid: str, new_parent_uuid: str) -> bool:
        """True if ``new_parent_uuid`` already sits in ``child_uuid``'s descendant
        subtree -- making it the parent would close an IS_A loop. Walks the live
        ``_children_of`` adjacency with a visited guard so it terminates even if
        the adjacency is already corrupt."""
        stack = [child_uuid]
        seen: set[str] = set()
        while stack:
            node = stack.pop()
            if node == new_parent_uuid:
                return True
            if node in seen:
                continue
            seen.add(node)
            stack.extend(self._children_of.get(node, []))
        return False

    def _record_ambiguous(self, a_uuid: str, b_uuid: str) -> None:
        """Record a would-be-cyclic subsumption as an unresolved relationship.

        A 2-cycle (A IS_A B *and* B IS_A A) is the rollup telling us the two
        types are too similar to order -- a *misunderstood* relationship, not a
        hierarchy. We persist a single bidirectional ``AMBIGUOUS_SUBSUMPTION``
        edge (MERGEd on source/target/relation, with a canonical endpoint order
        -> idempotent) so the signal survives for later resolution, rather than
        silently dropping it. tier 1 ignores this relation, so it never feeds
        back into IS_A."""
        lo, hi = sorted((a_uuid, b_uuid))
        try:
            self._hcg.add_edge(
                lo,
                hi,
                "AMBIGUOUS_SUBSUMPTION",
                bidirectional=True,
                properties={"reason": "is_a_cycle", "detected_by": "type_rollup"},
            )
            logger.info(
                "type_rollup: ambiguous subsumption recorded %s <-> %s "
                "(types too alike to order)",
                lo,
                hi,
            )
        except Exception:
            logger.exception(
                "type_rollup: failed to record ambiguity %s <-> %s", a_uuid, b_uuid
            )

    def _build_is_a_adjacency(self) -> None:
        """parent_uuid -> [child_uuid], from all IS_A edges (child IS_A parent)."""
        self._children_of = defaultdict(list)
        try:
            for e in self._hcg.list_all_edges(relation_type="IS_A", limit=_BIG) or []:
                src, tgt = e.get("source"), e.get("target")
                # Type-level hierarchy only. IS_A also carries instance taxonomy
                # (e.g. `fish IS_A natural_resource`); walking those would let the
                # cascade and cycle-check wander out of the type layer and report
                # spurious cycles between unrelated types.
                if src and tgt and src.startswith("type_") and tgt.startswith("type_"):
                    self._children_of[tgt].append(src)
        except Exception:
            logger.exception("type_rollup: build IS_A adjacency failed")

    def _current_is_a(self, child_uuid: str) -> tuple[str | None, str | None]:
        try:
            for e in self._hcg.query_edges_from(child_uuid) or []:
                if e.get("relation") == "IS_A":
                    return e.get("target"), (e.get("id") or e.get("uuid"))
        except Exception:
            logger.exception("type_rollup: query_edges_from failed for %s", child_uuid)
        return None, None

    def _resolve_parent(self, parent_name: str) -> tuple[str, list[str], str] | None:
        """Resolve a Hermes-suggested realm parent *name* (concept / process) to
        ``(type_uuid, ancestors, clean_name)`` for rooting a top-level
        super-type. Minted types live in the per-run name->uuid map; the seeded
        realm roots do not, so fall back to their canonical ``type_<name>`` uuid.
        Returns None when nothing resolves, so an invalid suggestion degrades to
        the default `entity` parent rather than a dangling root."""
        target = (parent_name or "").strip().lower()
        if not target:
            return None
        uuid = self._uuid_by_name.get(target) or f"type_{target}"
        try:
            node = self._hcg.get_node(uuid)
        except Exception:
            logger.exception("type_rollup: _resolve_parent(%s) failed", parent_name)
            return None
        if not node:
            return None
        ancestors = list((node.get("properties") or {}).get("ancestors") or [])
        if not ancestors:
            # A seeded realm root's canonical ancestors are [root, node]; for
            # anything else with no chain, fail closed (degrade to the default
            # parent) rather than fabricate a truncated chain (review).
            if uuid in _PROTECTED_ROOT_UUIDS:
                ancestors = list(_ENTITY_ANCESTORS)
            else:
                return None
        return uuid, ancestors, node.get("name") or target

    def _match_existing_type(
        self,
        centroid: list[float],
        parent_type_uuid: str,
        exclude: frozenset[str] | set[str] = frozenset(),
    ) -> str | None:
        """Nearest existing type within the match threshold (idempotency anchor),
        excluding the parent AND the cluster's own members. A member must never be
        selected as the super-type of its peers: a cluster centroid is the mean of
        its members, so its nearest type is often a member -- reconciling to it
        would persist a peer-as-parent IS_A edge + a wrong ancestor cascade
        (greptile #161). We therefore scan beyond the top hit so a real super-type
        past the excluded members can still match."""
        if not centroid:
            return None
        try:
            nearest = self._milvus.find_nearest_types(centroid, top_k=10)
        except Exception:
            logger.exception("type_rollup: find_nearest_types failed")
            return None
        for hit in nearest or []:
            cand = hit.get("uuid")
            if not cand or cand == parent_type_uuid or cand in exclude:
                continue
            try:
                row = self._milvus.get_embedding(node_type="TypeCentroid", uuid=cand)
            except Exception:
                logger.exception("type_rollup: get_embedding failed for %s", cand)
                continue
            emb = (row or {}).get("embedding")
            if not emb:
                continue
            try:
                if _cosine(centroid, emb) >= self._config.type_match_threshold:
                    return str(cand)
            except ValueError:
                continue
        return None


def build_type_rollup_handler(
    *,
    config: MaintenanceConfig,
    hcg: Any,
    milvus: Any,
    event_bus: Any,
    hermes_url: str,
    token: str,
) -> Callable[[], None]:
    """Return the callable registered as handlers['type_rollup']."""
    from sophia.maintenance.hermes_naming import name_cluster
    from sophia.maintenance.type_minting import mint_type

    handler = TypeRollupHandler(
        config=config,
        hcg=hcg,
        milvus=milvus,
        event_bus=event_bus,
        hermes_url=hermes_url,
        token=token,
        name_fn=lambda c, cand, url, tok: name_cluster(
            c,
            candidates=cand,
            hermes_url=url,
            token=tok,
            max_members=config.max_cluster_size,
        ),
        mint_fn=mint_type,
    )
    return lambda: handler.run()
