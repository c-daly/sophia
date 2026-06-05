"""Sophia -> Hermes name_cluster client (#505).

Sophia knows *that* a cluster's members belong together; this asks Hermes *what*
they are. Existing category labels travel along for naming consistency.
"""

from __future__ import annotations

import logging
import random

import httpx

from sophia.maintenance.emergence_types import EmergentCluster, Member, NameResult

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
