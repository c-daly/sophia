"""Sophia -> Hermes name_cluster client (#505).

Sophia knows *that* a cluster's members belong together; this asks Hermes *what*
they are. Existing category labels travel along for naming consistency.
"""

from __future__ import annotations

import logging

import httpx

from sophia.maintenance.emergence_types import EmergentCluster, NameResult

logger = logging.getLogger(__name__)


def name_cluster(
    cluster: EmergentCluster,
    *,
    candidates: list[str],
    hermes_url: str,
    token: str,
    timeout: float = 30.0,
) -> NameResult | None:
    """Ask Hermes to name what binds the cluster. Returns None on failure."""
    payload = {
        "members": [
            {
                "name": m.name,
                "type": m.current_type,
                "hermes_type_hint": m.hermes_type_hint,
                "neighbors": m.neighbors,
            }
            for m in cluster.members
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
        return NameResult(
            label=data["label"],
            description=data.get("description", ""),
            confidence=float(data.get("confidence", 0.0)),
        )
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        logger.warning("name_cluster failed: %s", exc)
        return None
