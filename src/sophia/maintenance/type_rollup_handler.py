"""Type-level rollup: build hierarchy over the flat shelf of emergent types (#160).

A slow-cadence maintenance pass. Emergence mints a wide flat layer of leaf
type-definitions under `entity` and never revisits it, so the ontology stays
flat. This pass groups those *types* into super-types. It RE-PARENTS existing
type-definitions (re-points the single `type->parent IS_A` edge via
`placement.reparent`, the only membership-write path); it never retypes instance
members.

Two tiers, run in order:
  1. **Explicit subsumption lift** -- member relations already cross type
     boundaries with meaningful labels (`HAS_PART`/`INCLUDES`/`COMPOSED_OF` =>
     parent->child, `PART_OF` => child->parent, `ALSO_KNOWN_AS` => synonym).
     Aggregate them into a type-type graph and lift the directed ones into
     type-level IS_A; alias synonyms under a canonical.
  2. **Residual clustering** -- types with no explicit subsumption edge are
     clustered recursively into super-types via the same `find_emergent_hierarchy`
     used for entity emergence, but with type centroids as the points.

Idempotent: `placement.reparent` is a no-op when the type already has the right
parent; super-types reconcile (match-before-mint) so a re-run does not duplicate
them. Safe to fire on a plain periodic trigger.
"""

from __future__ import annotations

import logging
import uuid as uuid_lib
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any

from sophia.maintenance import placement
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
# Seeded roots a super-type MAY be grafted under as the *fallback* when no closer
# domain type covers the cluster (siblings of `entity` under `node`). `cognition`
# is excluded -- it is populated by metacognitive schema induction, not domain
# rollup -- and the pure structural roots `root`/`node` are never domain parents.
# A super normally roots under the closest covering domain type Hermes names; a
# realm root is only the top-of-chain fallback.
_GRAFTABLE_REALMS = frozenset({"entity", "concept", "process"})
# Seeded structural roots that must NEVER be minted as a super, matched/reused as
# a super, or reparented -- they are the fixed top of the ontology. Guards the
# rollup against corrupting them (review blocker + duplicate-root path).
_PROTECTED_ROOT_NAMES = frozenset(
    {"root", "node", "entity", "concept", "cognition", "process"}
)
# Their real uuids are resolved POSITIONALLY at run() time (name -> uuid from the
# type layer), never assumed to be a `type_<name>` slug -- the seeder mints roots
# with uuid5 ids and emergence mints opaque uuid4s, so a `type_` prefix is gone.

# Member-relation labels that imply a type-level relationship.
_PARENT_REL = {"HAS_PART", "INCLUDES", "COMPOSED_OF"}  # source-type is the PARENT
_CHILD_REL = {"PART_OF"}  # source-type is the CHILD (invert)
_SYNONYM_REL = {"ALSO_KNOWN_AS"}  # undirected synonymy -> alias
_ALL_RELS = _PARENT_REL | _CHILD_REL | _SYNONYM_REL
_BIG = 100_000


def _is_rollup_candidate(td: dict) -> bool:
    """A non-reserved entity-side type-def (seeded or minted) eligible for rollup."""
    name = (td.get("name") or "").strip()
    uuid = td.get("uuid") or ""
    if not name or not uuid:
        return False
    if name in _PROTECTED_ROOT_NAMES or name.startswith("reserved_"):
        return False
    # Every record reaching here came from get_all_type_definitions(), i.e. it is
    # already a type POSITIONALLY (a node with an incoming IS_A edge). Neither the
    # `type_definition` label nor a `type_`-prefixed uuid is consulted -- the IS_A
    # subgraph is the sole definition of "is a type". Non-reserved, non-protected
    # types (whatever their uuid scheme: uuid5 seed roots, opaque uuid4 mints, or
    # accreted content nodes like `engine`) are all eligible to roll up.
    return True


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
        # Positional root resolution (name -> real uuid for the seeded roots),
        # built from the full type layer so we never assume a `type_<name>` slug.
        self._root_uuid_by_name: dict[str, str] = {}
        self._entity_root_uuid: str | None = None
        self._protected_root_uuids: frozenset[str] = frozenset()
        # All positional type uuids (nodes with an incoming IS_A), for telling
        # type-level IS_A edges (type IS_A type) from instance-level ones.
        self._type_uuids: set[str] = set()

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
        # Keyed case-insensitively (lowercased): Hermes labels are normalized to
        # lower case, so a stored type-def with legacy/mixed casing must still
        # match a coined label and never bypass the reuse / protected-root guards
        # (review #165: case-insensitive lookup on the name->uuid map).
        self._uuid_by_name = {r["name"].strip().lower(): r["uuid"] for r in rows}
        # Resolve the seeded roots to their REAL uuids positionally (by name from
        # the full type layer), not the dangling `type_<name>` slug. Roots are not
        # rollup candidates, so they live in `_root_uuid_by_name`, not `_uuid_by_name`.
        self._entity_root_uuid = self._root_uuid_by_name.get(_BASE_TYPE)
        self._protected_root_uuids = frozenset(
            self._root_uuid_by_name[n]
            for n in _PROTECTED_ROOT_NAMES
            if n in self._root_uuid_by_name
        )
        if not self._entity_root_uuid:
            logger.warning(
                "type_rollup: entity root not found positionally; skipping pass"
            )
            return
        entity_root_uuid = self._entity_root_uuid

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
        self._tier2_residual(residual, candidates, entity_root_uuid)

    # ------------------------------------------------------------------ load
    def _load_type_layer(self) -> list[dict]:
        try:
            type_defs = self._hcg.get_all_type_definitions()
        except Exception:
            logger.exception("type_rollup: failed to list type definitions")
            return []
        # Full positional name -> uuid (incl. the seeded realm/protected roots,
        # which are NOT rollup candidates). This is how root parents resolve to
        # their REAL uuids by name, replacing the old `type_<name>` slug.
        #
        # Protected-root precedence: an accreted content node that happens to
        # share a protected realm-root name (e.g. a minted "entity" or "concept"
        # type-def) must NEVER overwrite the seeded realm-root uuid in this map.
        # If it did, _entity_root_uuid / _protected_root_uuids would point at the
        # accreted node and every subsequent graft would land under the wrong
        # parent, silently corrupting the ontology hierarchy.
        #
        # The seeded structural root sits structurally higher than any accreted
        # node of the same name: it has a shorter ancestors chain (["root","node"]
        # vs. ["root","node","entity",...]).  We therefore resolve protected names
        # by picking the candidate with the FEWEST ancestors -- ties go to the
        # first entry (stable under repeated graph scans).  Non-protected names
        # keep last-wins behaviour as before.
        _all: dict[str, str] = {}
        _protected_best: dict[str, tuple[int, str]] = {}  # name -> (depth, uuid)
        for td in type_defs or []:
            if not td.get("name") or not td.get("uuid"):
                continue
            key = td["name"].strip().lower()
            _all[key] = td["uuid"]
            if key in _PROTECTED_ROOT_NAMES:
                depth = len((td.get("properties") or {}).get("ancestors") or [])
                prev = _protected_best.get(key)
                if prev is None or depth < prev[0]:
                    _protected_best[key] = (depth, td["uuid"])
        self._root_uuid_by_name = {
            **_all,
            **{k: v for k, (_, v) in _protected_best.items()},
        }
        # Every node returned here is a type positionally; keep the uuid set so
        # `_build_is_a_adjacency` can keep to type-level IS_A edges.
        self._type_uuids = {td["uuid"] for td in (type_defs or []) if td.get("uuid")}
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
            self._reparent_one(child, parent)
            explicit_children.add(child)
        if explicit_children:
            logger.info(
                "type_rollup: tier-1 lifted %d explicit subsumptions",
                len(explicit_children),
            )
        return explicit_children

    def _type_uuid_map(self, node_uuids: list[str]) -> dict[str, str]:
        # TODO #35: read still infers membership from property; convert to IS_A walk
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
    def _tier2_residual(
        self, residual: list[dict], candidates: list[str], entity_root_uuid: str
    ) -> None:
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
                # These synthetic members ARE types (positionally); the naming
                # context just marks them as such, matching emergence_clustering's
                # generic "type" marker (no `type_definition` label anywhere).
                current_type="type",
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
            self._reparent_subtree(node, entity_root_uuid, candidates)

    def _reparent_subtree(
        self,
        node: Any,
        parent_type_uuid: str,
        candidates: list[str],
    ) -> None:
        """Internal nodes => mint/reuse a super-type; leaves => re-parent the
        existing type-defs (a leaf Member's uuid IS a type_uuid)."""
        try:
            if not node.children:
                # Leaf: each member is an existing type-def; re-parent it directly.
                for m in node.members:
                    self._reparent_one(m.uuid, parent_type_uuid)
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
            # Hermes labels come from an LLM; normalize before any name/uuid
            # comparison so a capitalized label (e.g. "Concept") cannot bypass the
            # protected-root / self-graft guards below (review).
            clean_label = name.label.strip().lower()
            # Never use a seeded structural root as a super-type *name*: minting
            # would duplicate the root, and reusing it would reparent it. If
            # Hermes labelled the cluster after a root, skip this super level and
            # attach the children directly under the current parent.
            if clean_label in _PROTECTED_ROOT_NAMES:
                logger.info(
                    "type_rollup: cluster named after root %r -- skipping super level",
                    name.label,
                )
                for child in node.children:
                    self._reparent_subtree(child, parent_type_uuid, candidates)
                return
            # Root a TOP-LEVEL super-type under the CLOSEST covering type Hermes
            # named -- a realm root (concept / process) OR a deeper existing
            # domain type -- instead of flat under `entity`. "Create the group as
            # close as you can to the cluster name": realm roots are the fallback,
            # not the only option. Top level only; deeper super-types keep nesting
            # under their tree parent. Guard the rebind: never graft under one of
            # the cluster's own members or the coined label itself. Degrades to
            # the default `entity` parent when nothing valid resolves.
            if (
                parent_type_uuid == self._entity_root_uuid
                and name.parent
                and name.parent.strip().lower() != name.label.strip().lower()
            ):
                rooted = self._resolve_parent(name.parent)
                if (
                    rooted is not None
                    and rooted not in member_uuids
                    and rooted != self._uuid_by_name.get(clean_label)
                ):
                    parent_type_uuid = rooted
                    logger.info(
                        "type_rollup: rooting super-type %r under %s (%s)",
                        name.label,
                        self._name_of.get(parent_type_uuid, parent_type_uuid),
                        parent_type_uuid,
                    )
            super_uuid = self._match_existing_type(
                node.centroid, parent_type_uuid, exclude=member_uuids
            )
            # Never reuse a seeded structural root as a domain super-type: a root
            # is not a cluster member, so exclude=members does not catch it, and
            # the reuse branch's _reparent_one would then move the root into the
            # domain tree (no cycle, so the cycle guard cannot stop it). Treat a
            # realm-root hit as no-match -> mint a fresh super instead (blocker).
            if super_uuid in self._protected_root_uuids:
                super_uuid = None
            if super_uuid is None:
                # Name-based reconcile before minting: never create a second
                # type-def with a name that already exists (the duplicate
                # `name`_hex problem). The centroid match can miss two clusters
                # Hermes named identically; an exact name is a strong reconcile
                # signal. Skip if it is the parent or a member of this very
                # cluster (reusing those would just re-introduce a cycle).
                by_name = self._uuid_by_name.get(clean_label)
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
                    # The super roots into the realm of its graft parent.
                    realm=placement.realm_of(parent_type_uuid, hcg=self._hcg)
                    or _BASE_TYPE,
                )
                if name.label not in candidates:
                    candidates.append(name.label)
                super_name = name.label
                self._name_of[super_uuid] = super_name
                # Preserve the case-insensitive invariant of the name->uuid map.
                self._uuid_by_name[super_name.strip().lower()] = super_uuid
                if self._event_bus is not None:
                    try:
                        self._event_bus.publish(
                            "ontology.type_created",
                            {
                                "type_uuid": super_uuid,
                                "name": super_name,
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
                # The super was REUSED (centroid- or name-match), not minted, so
                # nothing has placed it under this parent. mint_fn does that for
                # fresh supers; the reuse paths must do it explicitly or the super
                # keeps its old position (divergence). Idempotent no-op when
                # already correct; cycle-guarded if it would loop.
                self._reparent_one(super_uuid, parent_type_uuid)
            for child in node.children:
                self._reparent_subtree(child, super_uuid, candidates)
        except Exception:
            logger.exception("type_rollup: subtree failed, skipping")

    # --------------------------------------------------------------- reparent
    def _reparent_one(self, child_uuid: str, new_parent_uuid: str) -> None:
        """Re-point one type-def's single upward IS_A edge onto a new parent via
        the shared placement module (cycle-safe, idempotent). Membership/ancestry
        is the edge, walked on demand -- nothing below the moved node changes and
        no `ancestors` property is written (DESIGN sec 3)."""
        placement.reparent(
            child_uuid,
            new_parent_uuid,
            hcg=self._hcg,
            children_of=self._children_of,
            placed_by="parent_resolution",
        )

    # ----------------------------------------------------------- helpers
    def _build_is_a_adjacency(self) -> None:
        """parent_uuid -> [child_uuid], from all IS_A edges (child IS_A parent)."""
        self._children_of = defaultdict(list)
        try:
            for e in self._hcg.list_all_edges(relation_type="IS_A", limit=_BIG) or []:
                src, tgt = e.get("source"), e.get("target")
                # Type-level hierarchy only. IS_A also carries instance taxonomy
                # (e.g. `fish IS_A natural_resource`); walking those would let the
                # cascade and cycle-check wander out of the type layer and report
                # spurious cycles between unrelated types. A type-level edge is one
                # whose BOTH endpoints are positional types (in `_type_uuids`) --
                # the child must itself be a type, not a leaf instance.
                if src in self._type_uuids and tgt in self._type_uuids:
                    self._children_of[tgt].append(src)
        except Exception:
            logger.exception("type_rollup: build IS_A adjacency failed")

    def _resolve_parent(self, parent_name: str) -> str | None:
        """Resolve a Hermes-named graft parent to a type uuid, or None.

        The parent is the CLOSEST covering type Hermes named -- a realm root OR a
        deeper existing domain type-def -- so a super-type roots as near the
        cluster as possible instead of flat under `entity`. Realm roots are the
        top-of-chain fallback, not the only allowed parents.

        Returns None for an unresolvable name or a forbidden target, so the caller
        degrades to the default `entity` parent. Forbidden: `cognition` (reserved
        for metacognitive schema induction) and the pure structural roots
        `root`/`node`. The caller adds the cycle guards (member / self / IS_A
        descendant).
        """
        target = (parent_name or "").strip().lower()
        if not target:
            return None
        if target in _PROTECTED_ROOT_NAMES:
            # A seeded root: only the graftable realms (entity / concept /
            # process) are valid; `cognition` and the structural roots are not.
            # Resolve to the root's REAL uuid positionally by name (the run map
            # is lowercased, so a case-variant domain type-def cannot shadow it).
            if target not in _GRAFTABLE_REALMS:
                return None
            uuid = self._root_uuid_by_name.get(target)
            if not uuid:
                return None
        else:
            # A deeper domain type Hermes named as the closest cover. Resolve it
            # via the run's name->uuid map (reserved/edge/protected types are
            # already excluded from it). Its domain-rootedness is verified below --
            # the map alone does not guarantee a domain ancestry.
            resolved = self._uuid_by_name.get(target)
            if not resolved:
                return None
            uuid = resolved
        try:
            node = self._hcg.get_node(uuid)
        except Exception:
            logger.exception("type_rollup: _resolve_parent(%s) failed", parent_name)
            return None
        if not node:
            return None
        # Domain-rootedness guard: the resolved type must root in a graftable realm
        # (entity / concept / process), computed by WALKING IS_A edges upward via
        # placement.realm_of -- never by reading a forbidden `ancestors` property.
        # This stops a domain super from grafting into the `cognition` subtree or
        # onto bare structure.
        if placement.realm_of(uuid, hcg=self._hcg) is None:
            return None
        return uuid

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
