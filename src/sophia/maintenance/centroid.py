"""Type-centroid maintenance: incremental online-mean on graft + full reconcile.

A type's centroid is the mean of its DIRECT members' embeddings -- the same thing
``mint_type`` writes once from the founding cluster. It must stay current as
members are grafted, because the mint/reuse decision
(``type_rollup_handler._match_existing_type``) matches a candidate cluster's
centroid against EXISTING type centroids: stale centroids there mean minting a
duplicate or reusing the wrong type.

Full recompute per graft is O(members) and prohibitively expensive at scale, so:
  - on graft we fold the new member(s) into the running mean in O(dim)
    (:func:`online_mean_add` / :func:`bump_centroid`);
  - a periodic job recomputes everything from scratch (:func:`reconcile_centroids`)
    to repair float drift, removals, and anything the incremental path missed.

Nothing here invokes an embedding model: the member vectors already exist; a
centroid is their mean, and the ``embedding_model`` label is read off the
existing vectors (the vector's dimension is the real invariant -- sophia rejects
a dim mismatch at write time -- so the label is metadata, never fabricated).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Member content embeddings physically live in the base "Entity" collection
# (_CONTENT_COLLECTION); type centroids in "TypeCentroid". Both are realm-agnostic.
_CONTENT_COLLECTION = "Entity"
_TYPE_CENTROID = "TypeCentroid"
# Seeded structural/realm roots: never carry a content centroid (not rollup
# candidates; some have no member embeddings to average).
_PROTECTED_ROOT_NAMES = frozenset(
    {"root", "node", "entity", "concept", "cognition", "process"}
)


def online_mean_add(
    centroid: list[float] | None, n: int, embeddings: list[list[float]]
) -> list[float] | None:
    """Mean of ``n`` vectors (whose mean is ``centroid``) after adding ``embeddings``.

    Pure, O(dim * k). When ``centroid`` is falsy or ``n`` <= 0 this initialises:
    it returns the plain mean of ``embeddings`` (the founding case). Returns the
    unchanged ``centroid`` if there is nothing to add.
    """
    vecs = [e for e in embeddings if e]
    if not vecs:
        return centroid
    k = len(vecs)
    dim = len(vecs[0])
    sums = [sum(e[i] for e in vecs) for i in range(dim)]
    if not centroid or n <= 0:
        return [s / k for s in sums]
    assert len(centroid) == dim, (
        f"online_mean_add: centroid dim {len(centroid)} != new-vec dim {dim}; "
        "dimension mismatch should be caught at Milvus write time"
    )
    return [(centroid[i] * n + sums[i]) / (n + k) for i in range(dim)]


def bump_centroid(
    milvus: Any,
    type_uuid: str,
    embeddings: list[list[float]],
    n: int,
    *,
    model: str | None = None,
) -> None:
    """Fold ``embeddings`` into ``type_uuid``'s centroid incrementally (fail-soft).

    ``n`` is the type's direct-member count BEFORE this add (the cheap
    ``member_count`` already tracked in the type snapshot / a direct in-degree
    count). The model label is read off the existing centroid when present,
    otherwise taken from ``model`` (the members' recorded embedding_model) --
    never invented. A failure just leaves the centroid for the periodic reconcile
    to repair rather than breaking the graft.
    """
    if milvus is None or not type_uuid:
        return
    vecs = [e for e in embeddings if e]
    if not vecs:
        return
    try:
        cur = milvus.get_embedding(node_type=_TYPE_CENTROID, uuid=type_uuid) or {}
        old = cur.get("embedding")
        # A centroid exists but we don't have its prior count (n<=0): folding
        # would re-init it to just the new members' mean and lose the existing
        # population. Don't corrupt -- leave it for the periodic reconcile.
        if old and n <= 0:
            return
        mdl = cur.get("embedding_model") or model
        new = online_mean_add(old, n, vecs)
        if new is None:
            return
        milvus.update_centroid(type_uuid=type_uuid, centroid=new, model=mdl)
    except Exception:
        logger.exception("centroid bump failed for %s", type_uuid)


def reconcile_centroids(hcg: Any, milvus: Any) -> dict[str, int]:
    """Recompute every type's centroid from its direct members' current embeddings.

    The periodic backstop + initialiser: corrects drift/removals the incremental
    path can't, and seeds a centroid for any type that never had one (e.g. an
    accreted content type that became a type by graft, not by mint). Skips the
    protected/reserved roots and any type whose members lack embeddings or carry
    more than one model (no fabrication). Returns a small stats dict; fail-soft.
    """
    stats = {
        "written": 0,
        "skipped_protected": 0,
        "skipped_no_embeddings": 0,
        "skipped_mixed_model": 0,
        "errors": 0,
    }
    if hcg is None or milvus is None:
        return stats
    try:
        types = hcg.get_all_type_definitions()
    except Exception:
        logger.exception("centroid reconcile: failed to list types")
        stats["errors"] += 1
        return stats
    for t in types or []:
        tu = t.get("uuid")
        name = (t.get("name") or "").strip().lower()
        if not tu:
            continue
        if (
            name in _PROTECTED_ROOT_NAMES
            or name.startswith("_")
            or name.startswith("reserved_")
        ):
            stats["skipped_protected"] += 1
            continue
        try:
            member_uuids = _member_uuids(hcg, tu)
            embs: list[list[float]] = []
            models: set[str] = set()
            for m in member_uuids:
                e = milvus.get_embedding(node_type=_CONTENT_COLLECTION, uuid=m)
                if e and e.get("embedding"):
                    embs.append(e["embedding"])
                    if e.get("embedding_model"):
                        models.add(e["embedding_model"])
            if not embs:
                stats["skipped_no_embeddings"] += 1
                continue
            if len(models) > 1:
                stats["skipped_mixed_model"] += 1
                continue
            centroid = online_mean_add(None, 0, embs)
            milvus.update_centroid(
                type_uuid=tu, centroid=centroid, model=next(iter(models), None)
            )
            stats["written"] += 1
        except Exception:
            logger.exception("centroid reconcile failed for %s", tu)
            stats["errors"] += 1
    logger.info("centroid reconcile: %s", stats)
    return stats


def _member_uuids(hcg: Any, type_uuid: str) -> list[str]:
    """Direct members of a type: nodes whose reified IS_A edge points :TO it."""
    query = """
    MATCH (m:Node)<-[:FROM]-(:Node {relation: "IS_A"})-[:TO]->(t:Node {uuid: $u})
    RETURN m.uuid AS uuid
    """
    return [r["uuid"] for r in hcg._execute_read(query, {"u": type_uuid})]
