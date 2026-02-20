"""Experiment tracking query endpoints.

Provides read-only access to experiment_run nodes created during proposal
ingestion, enabling comparison of pipeline configurations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

# Lazy reference to the HCG client getter, set by app.py during router
# inclusion to avoid circular imports.
_hcg_client_getter = None


def set_hcg_client_getter(getter):
    """Set the callable that returns the HCG client instance."""
    global _hcg_client_getter
    _hcg_client_getter = getter


router = APIRouter(prefix="/experiments", tags=["experiments"])


def _get_hcg():
    """Get the HCG client from app state.

    Uses a module-level reference set during router inclusion to avoid
    importing from app (which would create a circular dependency risk).
    """
    if _hcg_client_getter is None:
        raise HTTPException(status_code=503, detail="HCG client not initialized")
    client = _hcg_client_getter()
    if client is None:
        raise HTTPException(status_code=503, detail="HCG client not initialized")
    return client


@router.get("", response_model=List[Dict[str, Any]])
async def list_experiments(
    ner_provider: Optional[str] = Query(None, description="Filter by NER provider"),
    embedding_provider: Optional[str] = Query(
        None, description="Filter by embedding provider"
    ),
    tag: Optional[str] = Query(None, description="Filter by experiment tag"),
    limit: int = Query(50, ge=1, le=500, description="Max results"),
) -> List[Dict[str, Any]]:
    """List experiment runs with optional filtering."""
    hcg = _get_hcg()

    # Use list_all_nodes to get experiment_run nodes
    try:
        runs = hcg.list_all_nodes(node_type="experiment_run", limit=limit)
    except Exception as e:
        logger.error("Failed to query experiment runs: %s", e)
        return []

    # Apply filters
    filtered = []
    for run in runs:
        props = run.get("properties", {})
        if ner_provider and props.get("ner_provider") != ner_provider:
            continue
        if embedding_provider and props.get("embedding_provider") != embedding_provider:
            continue
        if tag:
            run_tags = props.get("experiment_tags", [])
            if isinstance(run_tags, str):
                run_tags = [run_tags]
            if tag not in (run_tags or []):
                continue
        filtered.append(run)

    return filtered


@router.get("/compare", response_model=List[Dict[str, Any]])
async def compare_experiments(
    group_by: str = Query(
        "ner_provider",
        description="Property to group by (ner_provider, embedding_provider)",
    ),
    limit: int = Query(100, ge=1, le=500),
) -> List[Dict[str, Any]]:
    """Compare experiment runs grouped by provider configuration."""
    hcg = _get_hcg()

    if group_by not in ("ner_provider", "embedding_provider"):
        group_by = "ner_provider"

    try:
        runs = hcg.list_all_nodes(node_type="experiment_run", limit=limit)
    except Exception as e:
        logger.error("Failed to compare experiments: %s", e)
        return []

    # Group in Python (avoids needing raw Cypher access)
    groups: Dict[str, list] = {}
    for run in runs:
        props = run.get("properties", {})
        key = props.get(group_by, "unknown")
        groups.setdefault(key, []).append(props)

    result = []
    for provider, group_runs in groups.items():
        n = len(group_runs)
        result.append(
            {
                "provider": provider,
                "run_count": n,
                "avg_duration_ms": round(
                    sum(r.get("total_duration_ms", 0) for r in group_runs) / n, 1
                ),
                "avg_entities": round(
                    sum(r.get("entity_count", 0) for r in group_runs) / n, 1
                ),
                "avg_edges": round(
                    sum(r.get("edge_count", 0) for r in group_runs) / n, 1
                ),
                "avg_ner_ms": round(
                    sum(r.get("ner_duration_ms", 0) for r in group_runs) / n, 1
                ),
                "avg_emb_ms": round(
                    sum(r.get("embedding_duration_ms", 0) for r in group_runs) / n, 1
                ),
            }
        )

    result.sort(key=lambda x: x["run_count"], reverse=True)
    return result


@router.get("/{run_id}/entities", response_model=List[Dict[str, Any]])
async def get_experiment_entities(
    run_id: str,
) -> List[Dict[str, Any]]:
    """Get entities produced by an experiment run."""
    hcg = _get_hcg()

    # Get outgoing edges from the run node, filter for PRODUCED relation
    try:
        edges = hcg.query_edges_from(run_id)
    except Exception as e:
        logger.error("Failed to get experiment entities for %s: %s", run_id, e)
        return []

    entities = []
    for edge in edges:
        if edge.get("relation") != "PRODUCED":
            continue
        target_uuid = edge.get("target_uuid")
        if target_uuid:
            node = hcg.get_node(target_uuid)
            if node:
                entities.append(node)

    return entities
