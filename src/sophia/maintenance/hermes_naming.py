"""Sophia -> Hermes name_cluster client (#505).

Sophia knows *that* a cluster's members belong together; this asks Hermes *what*
they are. Existing category labels travel along for naming consistency.
"""

from __future__ import annotations

import logging
import random

import httpx

from sophia.maintenance.emergence_types import (
    EmergentCluster,
    Member,
    NameResult,
    TypeClusterResult,
)

logger = logging.getLogger(__name__)


def _sample_members(members: list[Member], max_members: int | None) -> list[Member]:
    """Down-sample a cluster's membership for the naming request.

    Naming only needs a representative handful; sending thousands of members is
    wasteful and can exceed Hermes' context. A deterministic seed keeps the
    sample stable across retries of the same cluster.
    """
    if not max_members or len(members) <= max_members:
        return members
    rng = random.Random(" ".join(sorted(m.uuid for m in members)))
    return rng.sample(members, max_members)


def name_cluster(
    cluster: EmergentCluster,
    *,
    candidates: list[str],
    hermes_url: str,
    token: str,
    timeout: float = 30.0,
    max_members: int | None = None,
) -> NameResult | None:
    """Ask Hermes to name what binds the cluster. Returns None on failure.

    When ``max_members`` is set, clusters larger than that are down-sampled to a
    representative subset before the request.
    """
    members = _sample_members(cluster.members, max_members)
    payload = {
        "members": [
            {
                "id": m.uuid,
                "name": m.name,
                "type": m.current_type,
                "hermes_type_hint": m.hermes_type_hint,
                "neighbors": m.neighbors,
            }
            for m in members
        ],
        "candidates": candidates,
    }
    url = f"{hermes_url.rstrip('/')}/name-cluster"
    try:
        resp = httpx.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(timeout),
        )
        resp.raise_for_status()
        data = resp.json()
        removed = data.get("removed") or []
        parent = data.get("parent")
        return NameResult(
            label=data["label"],
            description=data.get("description", ""),
            confidence=float(data.get("confidence", 0.0)),
            removed=[str(r) for r in removed if r],
            parent=(
                str(parent).strip()
                if isinstance(parent, str) and parent.strip()
                else None
            ),
        )
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        logger.warning("name_cluster failed: %s", exc)
        return None


def type_cluster(
    cluster: EmergentCluster,
    *,
    hermes_url: str,
    token: str,
    timeout: float = 30.0,
    max_members: int | None = None,
) -> TypeClusterResult | None:
    """Ask Hermes v2 /type-cluster what type the cluster members are.

    Parallel to :func:`name_cluster`, but for the v2 typing tier: the catalog
    lives server-side, so no ``candidates`` are sent and members carry only
    id/name/hint/neighbors (no ``type``). Returns None on failure. When
    ``max_members`` is set, larger clusters are down-sampled first.
    """
    members = _sample_members(cluster.members, max_members)
    payload = {
        "members": [
            {
                "id": m.uuid,
                "name": m.name,
                "hermes_type_hint": m.hermes_type_hint,
                "neighbors": m.neighbors,
            }
            for m in members
        ],
    }
    base = hermes_url.rstrip("/")
    url = f"{base}/type-cluster"
    try:
        resp = httpx.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(timeout),
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            logger.warning("type_cluster returned non-dict JSON: %r", data)
            return None
        # Hermes v2 returns exactly ONE group (the most-specific type it can form
        # for the cluster; members that blur it are excluded as residuals) under
        # `groups`, NOT a top-level `name`. The group's IS_A `chain` runs
        # specific->general with chain[0]==name, so the proposed parent is
        # chain[1] -- and it is guaranteed to already exist. `parent` is None only
        # when the group reuses an existing type (assign_to != "NEW"); the handler
        # then re-points members onto the existing same-name type rather than
        # minting.
        groups = data.get("groups")
        if not isinstance(groups, list) or not groups:
            logger.warning("type_cluster returned no groups; treating as failure")
            return None
        group = groups[0]
        if len(groups) > 1:
            # Hermes v2 guarantees exactly one group; surface a contract
            # violation (e.g. a partial batch) rather than silently dropping the
            # rest.
            logger.warning(
                "type_cluster returned %d groups; using only the first",
                len(groups),
            )
        if not isinstance(group, dict):
            logger.warning("type_cluster group is not an object; treating as failure")
            return None
        name = str(group.get("name") or "").strip()
        if not name:
            logger.warning("type_cluster group has no name; treating as failure")
            return None
        # Coerce to str + strip BEFORE defaulting, so a whitespace-only or
        # non-string assign_to falls back to "NEW" instead of slipping through
        # as a falsy value on the existing-type-reuse path.
        assign_to = str(group.get("assign_to") or "").strip() or "NEW"
        chain = group.get("chain")
        parent: str | None = None
        if assign_to == "NEW" and isinstance(chain, list) and len(chain) > 1:
            # chain[1] may be JSON null; guard so it never becomes the literal
            # string "None" used as a parent name.
            parent = None if chain[1] is None else (str(chain[1]).strip() or None)
        residual = data.get("residual_ids")
        if not isinstance(residual, list):
            # A non-list residual_ids (string/int from a serialisation glitch)
            # would iterate char-by-char or raise; treat anything non-list as none.
            residual = []
        return TypeClusterResult(
            name=name,
            parent=parent,
            residual_ids=[str(r) for r in residual if r],
        )
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        logger.warning("type_cluster failed: %s", exc)
        return None
