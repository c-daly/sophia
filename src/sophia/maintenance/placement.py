"""Shared placement module -- the single path for every type-layer IS_A write.

Naming-driven-typing keystone (DESIGN sec 6). Consolidation invariant: *every*
``IS_A`` write the engine performs goes through this module, so the placement
mechanics and their guards live in exactly one place. Extracted and decoupled
from ``type_rollup_handler`` -- the functions here take their collaborators
(``hcg``, ``uuid_by_name``, ``children_of``) as arguments rather than off
``self``, so any caller can route through them.

Load-bearing invariants (DESIGN sec 3):

* **Structure is the ONLY representation of membership.** A type of a node is
  its single upward ``IS_A`` edge; there is no ``ancestors`` property and no
  ``type_uuid``-as-truth. Nothing here ever writes an ``ancestors`` property and
  nothing cascades to descendants -- membership and ancestry are *read* by
  walking the edges (``realm_of`` / ``_walk_to_realm``), never stored. A move
  re-points one edge; nothing below it changes.
* **Names resolve only via ``uuid_by_name``.** The seeder now mints real
  uuid5/uuid4, so the legacy ``type_<name>`` slug is dead. This module never
  fabricates a slug; an unresolvable name is closed-world ``None``.
* **Guards deflect-and-record, never force.** A graft that would close an IS_A
  cycle is recorded as ``AMBIGUOUS_SUBSUMPTION`` competing evidence rather than
  written; a protected or unresolvable parent coerces fail-closed (``None``) so
  the caller mints under the realm root.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# The three content realm roots. A content type IS_A chain always walks up to one
# of these; they are themselves valid graft parents (DESIGN sec 3 / sec 7).
_GRAFTABLE_REALMS = frozenset({"entity", "concept", "process"})

# Bare names the content engine may never graft under (DESIGN sec 6 guard set,
# identical to the sec 7 catalog exclusion). ``entity`` / ``concept`` /
# ``process`` are deliberately NOT here -- they are graftable realms, not
# protected.
_PROTECTED_NAMES = frozenset({"root", "node", "cognition"})

# The traceability tags every IS_A edge this module writes must carry, proving
# placement was parent-driven, not centroid-driven (DESIGN sec 6). Callers
# validate their ``placed_by`` against this set.
PLACED_BY_REASONS = frozenset({"parent_resolution", "name_reuse", "root_fallback"})


def resolve_parent(
    parent_name: str,
    *,
    uuid_by_name: dict[str, str],
    hcg: Any,
    realm: str | None = None,
) -> str | None:
    """Resolve a Hermes-suggested graft-parent NAME to a real type uuid, or None.

    Closed-world and fail-closed: an empty, protected/reserved, unknown, or
    non-domain-rooted name returns ``None`` -- the caller then mints under the
    realm root. The parent suggestion is validated, never trusted blindly
    (DESIGN sec 6).

    When ``realm`` is given (the draining cluster realm), the reused type must
    root in that SAME realm; a cross-realm name collision in the flat
    ``uuid_by_name`` catalog is rejected so entity content never grafts under a
    same-named concept/process type (DESIGN sec 3). ``realm=None`` (default)
    skips the check, keeping callers that do not know the realm unchanged.
    """
    target = (parent_name or "").strip().lower()
    if not target:
        return None
    # Protected / reserved targets: deflect-and-record (log + None). Both the
    # bare and the `_`-prefixed reserved namespaces are rejected so the guard
    # holds across the bare->underscore reseed transition (live graph has bare
    # `cognition` / `reserved_*` now, `_cognition` / `_reserved_*` after).
    if (
        target in _PROTECTED_NAMES
        or target.startswith("_")
        or target.startswith("reserved_")
    ):
        logger.info("placement: rejected protected/reserved parent %r", target)
        return None
    # Names resolve only via the supplied map -- never a fabricated slug. An
    # absent name -> closed-world None.
    uuid = uuid_by_name.get(target)
    if not uuid:
        return None
    try:
        node = hcg.get_node(uuid)
    except Exception:
        logger.exception("placement: resolve_parent get_node failed for %s", uuid)
        return None
    if not node:
        return None
    # Domain-rootedness guard: the resolved type must be a graftable realm root
    # OR have one in its ancestry, computed by WALKING IS_A edges upward (never
    # by reading a forbidden `ancestors` property). This stops a domain super
    # from grafting into the `cognition` subtree or onto bare structure.
    resolved_realm = _walk_to_realm(uuid, hcg=hcg)
    if resolved_realm is None:
        return None
    # Cross-realm guard: the flat name->uuid catalog can collide a same-named
    # type across realms. When the caller knows the cluster realm, the reused
    # type must root in that SAME realm -- otherwise grafting crosses realms and
    # breaks IS_A uniformity (DESIGN sec 3). Reuse the realm already walked above;
    # never walk twice.
    if realm is not None and resolved_realm != realm:
        logger.info(
            "placement: rejected cross-realm parent %r (realm %s != cluster %s)",
            target,
            resolved_realm,
            realm,
        )
        return None
    return uuid


def realm_of(uuid: str, *, hcg: Any) -> str | None:
    """Return the graftable-realm root name (entity/concept/process) that ``uuid``
    sits under, by walking IS_A edges upward; ``None`` if the chain reaches no
    realm.
    """
    return _walk_to_realm(uuid, hcg=hcg)


def reparent(
    child_uuid: str,
    new_parent_uuid: str,
    *,
    hcg: Any,
    children_of: dict[str, list[str]],
    placed_by: str,
) -> None:
    """Re-point the top element single upward IS_A edge: +1 edge, -1 edge.

    The cost is O(1) in the moved top regardless of subtree size -- nothing below
    changes (membership and ancestry are read by walking, not stored). On a graft
    that would close a cycle, record it as competing evidence and return rather
    than force a false IS_A (deflect-and-record, DESIGN sec 6).
    """
    if placed_by not in PLACED_BY_REASONS:
        raise ValueError(f"invalid placed_by: {placed_by!r}")
    if child_uuid == new_parent_uuid:
        return
    if creates_cycle(child_uuid, new_parent_uuid, children_of):
        # Both directions claimed: the two types are too alike to order. Keep the
        # signal as AMBIGUOUS_SUBSUMPTION instead of forcing (or dropping) it.
        record_ambiguous(child_uuid, new_parent_uuid, hcg=hcg)
        return
    cur_parent, cur_edge = _current_is_a(child_uuid, hcg=hcg)
    if cur_parent == new_parent_uuid:
        return  # already correct -> true no-op
    if cur_parent is not None:
        # Drop the stale parent edge by its own id when we have one; fall back to
        # a (source, target, relation) match otherwise. delete_edge(None) would
        # silently no-op and leave a 2nd IS_A parent (greptile #161).
        try:
            if cur_edge:
                hcg.delete_edge(cur_edge)
            else:
                hcg.delete_edges_between(child_uuid, cur_parent, "IS_A")
            siblings = children_of.get(cur_parent)
            if siblings and child_uuid in siblings:
                siblings.remove(child_uuid)
        except Exception:
            # Do NOT add the new edge if the stale one couldn't be removed -- that
            # would leave the node with two upward IS_A edges, violating the
            # single-upward-pointer invariant (DESIGN sec 3). Fail safe: keep the
            # old parent rather than create a fork.
            logger.exception("placement: delete stale IS_A failed for %s", child_uuid)
            return
    try:
        hcg.add_edge(
            child_uuid,
            new_parent_uuid,
            "IS_A",
            properties={"placed_by": placed_by},
        )  # MERGE -> idempotent
        children_of.setdefault(new_parent_uuid, [])
        if child_uuid not in children_of[new_parent_uuid]:
            children_of[new_parent_uuid].append(child_uuid)
    except Exception:
        logger.exception("placement: add IS_A failed for %s", child_uuid)


def attach(
    child_uuid: str,
    parent_uuid: str,
    *,
    hcg: Any,
    children_of: dict[str, list[str]],
    placed_by: str,
) -> None:
    """Create the first upward IS_A edge for a loose node -- the degenerate
    re-parent with no stale edge to remove. General signature (child may be any
    node): used for entity->type membership (B2) and type->type placement. The
    IS_A EDGE is the membership/typing fact -- there is no ``type_uuid`` property.
    """
    reparent(
        child_uuid,
        parent_uuid,
        hcg=hcg,
        children_of=children_of,
        placed_by=placed_by,
    )


def repark(
    child_uuid: str,
    realm_root_uuid: str,
    *,
    hcg: Any,
    children_of: dict[str, list[str]],
) -> None:
    """Re-point a residual/outlier back to its realm root; it re-enters the pool
    by location (DESIGN sec 5 / sec 6).
    """
    reparent(
        child_uuid,
        realm_root_uuid,
        hcg=hcg,
        children_of=children_of,
        placed_by="root_fallback",
    )


def creates_cycle(
    child_uuid: str,
    new_parent_uuid: str,
    children_of: dict[str, list[str]],
) -> bool:
    """True if ``new_parent_uuid`` already sits in the descendant subtree of
    ``child_uuid`` -- making it the parent would close an IS_A loop. Walks the
    live ``children_of`` adjacency with a visited guard so it terminates even if
    the adjacency is already corrupt.
    """
    stack = [child_uuid]
    seen: set[str] = set()
    while stack:
        node = stack.pop()
        if node == new_parent_uuid:
            return True
        if node in seen:
            continue
        seen.add(node)
        stack.extend(children_of.get(node, []))
    return False


def record_ambiguous(a_uuid: str, b_uuid: str, *, hcg: Any) -> None:
    """Record a would-be-cyclic subsumption as an unresolved relationship.

    A 2-cycle (A IS_A B *and* B IS_A A) means the two types are too similar to
    order into parent/child -- a misunderstood relationship, not a hierarchy. We
    persist one bidirectional ``AMBIGUOUS_SUBSUMPTION`` edge (MERGEd on a
    canonical endpoint order -> idempotent) so the signal survives for later
    resolution rather than being silently dropped (the seed of the future
    split-confidence model, DESIGN sec 3).
    """
    lo, hi = sorted((a_uuid, b_uuid))
    try:
        hcg.add_edge(
            lo,
            hi,
            "AMBIGUOUS_SUBSUMPTION",
            bidirectional=True,
            properties={"reason": "is_a_cycle", "detected_by": "placement"},
        )
        logger.info(
            "placement: ambiguous subsumption recorded %s <-> %s "
            "(types too alike to order)",
            lo,
            hi,
        )
    except Exception:
        logger.exception(
            "placement: failed to record ambiguity %s <-> %s", a_uuid, b_uuid
        )


def _current_is_a(child_uuid: str, *, hcg: Any) -> tuple[str | None, str | None]:
    """Return ``(parent_uuid, edge_id)`` for the single upward IS_A edge of
    ``child_uuid``, or ``(None, None)``. ``edge_id`` may be ``None`` when the
    edge was persisted without an id.
    """
    try:
        is_a = [
            e
            for e in (hcg.query_edges_from(child_uuid) or [])
            if e.get("relation") == "IS_A"
        ]
        if len(is_a) > 1:
            # The single-upward-pointer invariant says there is exactly one;
            # surface a corruption rather than silently picking the first.
            logger.warning(
                "placement: node %s has %d upward IS_A edges (expected 1)",
                child_uuid,
                len(is_a),
            )
        if is_a:
            e = is_a[0]
            return e.get("target"), (e.get("id") or e.get("uuid"))
    except Exception:
        logger.exception("placement: query_edges_from failed for %s", child_uuid)
    return None, None


def _walk_to_realm(uuid: str, *, hcg: Any) -> str | None:
    """Walk IS_A edges upward from ``uuid`` and return the graftable-realm name
    (entity/concept/process) at the top of the chain, or ``None``.

    The node itself counts: a realm root resolves to its own name. Ancestry is
    *read* here by walking edges -- never by reading a stored ``ancestors``
    property (which is forbidden, DESIGN sec 3). A visited guard bounds the walk
    against a corrupt cycle.
    """
    seen: set[str] = set()
    cur: str | None = uuid
    while cur and cur not in seen:
        seen.add(cur)
        try:
            node = hcg.get_node(cur)
        except Exception:
            logger.exception("placement: _walk_to_realm get_node failed for %s", cur)
            return None
        if not node:
            return None
        name = str(node.get("name") or "").strip().lower()
        if name in _GRAFTABLE_REALMS:
            return name
        cur, _edge = _current_is_a(cur, hcg=hcg)
    return None
