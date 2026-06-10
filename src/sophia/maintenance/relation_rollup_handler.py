"""Relation-vocabulary rollup (sophia#192).

The Sophia-side batch consolidation of the descriptive-relation vocabulary --
the edge-axis analog of the node-side type rollup, and the load-bearing piece
the relation-vocabulary fix needs (H1/H2 collapse only morphological variants;
the sprawl is semantic). Off the extraction hot path, scheduled like the other
maintenance handlers.

Pipeline (embeddings POINT, the graph ASSERTS):
  1. read the descriptive-relation vocabulary (reserved typing relations
     already excluded);
  2. EMBED the relation labels and cluster by cosine, so true synonyms
     (HAULS/DRAGS/CARRIES -- no shared tokens) co-locate;
  3. one Hermes /relation-synonyms call per cluster NAMES the synonym groups
     (the codec; fail-closed, IS_A rejected);
  4. consolidate above a confidence gate via rename_relation (dedup-safe,
     idempotent, reserved-guarded).

Dependencies are injected so the orchestration is unit-testable; the default
wiring uses Hermes /embed_text and /relation-synonyms.
"""

from __future__ import annotations

import logging
from typing import Callable, Sequence

logger = logging.getLogger(__name__)

EmbedFn = Callable[[list[str]], Sequence[Sequence[float]]]
SynonymFn = Callable[..., list]  # (predicates, context=None) -> list[RelationSynonymGroup]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return num / (na * nb)


def _greedy_cluster(
    labels: list[str], vectors: Sequence[Sequence[float]], threshold: float
) -> list[list[str]]:
    """Greedy single-pass cosine clustering: each label joins the first
    cluster whose seed it is within ``threshold`` of, else seeds a new one.
    Deterministic in input order."""
    clusters: list[list[str]] = []
    seeds: list[Sequence[float]] = []
    for label, vec in zip(labels, vectors):
        placed = False
        for i, seed in enumerate(seeds):
            if _cosine(vec, seed) >= threshold:
                clusters[i].append(label)
                placed = True
                break
        if not placed:
            clusters.append([label])
            seeds.append(vec)
    return clusters


class RelationRollupHandler:
    """Cluster -> name -> consolidate the descriptive-relation vocabulary."""

    def __init__(
        self,
        hcg,
        *,
        embed_fn: EmbedFn,
        synonym_fn: SynonymFn,
        min_confidence: float = 0.6,
        cluster_threshold: float = 0.7,
        max_cluster_size: int = 200,
        on_consolidated: Callable[[], None] | None = None,
    ) -> None:
        self._hcg = hcg
        self._embed_fn = embed_fn
        self._synonym_fn = synonym_fn
        self._min_confidence = min_confidence
        self._cluster_threshold = cluster_threshold
        self._max_cluster_size = max_cluster_size
        # called after edges are consolidated (e.g. refresh the Redis snapshot)
        self._on_consolidated = on_consolidated

    def run(self) -> dict:
        vocab = self._hcg.get_relation_vocabulary() or []
        labels = [v["relation"] for v in vocab if v.get("relation")]
        if not labels:
            return {"groups_applied": 0, "edges_renamed": 0, "clusters": 0}

        try:
            vectors = self._embed_fn(labels)
        except Exception:
            logger.exception("relation rollup: embedding failed; skipping run")
            return {"groups_applied": 0, "edges_renamed": 0, "clusters": 0}

        clusters = _greedy_cluster(labels, vectors, self._cluster_threshold)
        # only multi-member clusters can contain synonyms worth a Hermes call
        batches = [c[: self._max_cluster_size] for c in clusters if len(c) >= 2]

        groups_applied = 0
        edges_renamed = 0
        for batch in batches:
            try:
                groups = self._synonym_fn(batch, context=None)
            except Exception:
                logger.exception("relation rollup: synonym call failed for a cluster")
                continue
            for g in groups:
                if g.confidence < self._min_confidence:
                    continue
                applied = False
                for member in g.members:
                    if member == g.canonical:
                        continue
                    try:
                        edges_renamed += self._hcg.rename_relation(member, g.canonical)
                        applied = True
                    except ValueError:
                        # reserved relation slipped through -- never consolidate it
                        logger.warning(
                            "relation rollup: refused reserved rename %s -> %s",
                            member,
                            g.canonical,
                        )
                if applied:
                    groups_applied += 1
                    logger.info(
                        "relation rollup: consolidated %s -> %s (conf %.2f)",
                        g.members,
                        g.canonical,
                        g.confidence,
                    )

        if edges_renamed and self._on_consolidated is not None:
            try:
                self._on_consolidated()
            except Exception:
                logger.exception("relation rollup: snapshot refresh failed")

        return {
            "groups_applied": groups_applied,
            "edges_renamed": edges_renamed,
            "clusters": len(batches),
        }


def build_relation_rollup_handler(
    *,
    hcg,
    hermes_url: str,
    token: str,
    redis_client=None,
    min_confidence: float = 0.6,
    cluster_threshold: float = 0.7,
) -> RelationRollupHandler:
    """Default wiring: Hermes /embed_text + /relation-synonyms, snapshot refresh.

    embed_fn loops /embed_text per distinct label (the pass is off the hot path
    and bounded per run; batch embedding is a follow-up). on_consolidated
    re-publishes logos:ontology:relations so Hermes (hermes#137) re-seeds the
    collapsed vocabulary.
    """
    import json

    import httpx

    from sophia.maintenance.hermes_naming import relation_synonyms

    base = hermes_url.rstrip("/")
    auth = {"Authorization": f"Bearer {token}"}

    def embed_fn(labels: list[str]):
        vectors = []
        with httpx.Client(timeout=httpx.Timeout(30.0)) as client:
            for label in labels:
                resp = client.post(
                    f"{base}/embed_text", json={"text": label}, headers=auth
                )
                resp.raise_for_status()
                vectors.append(resp.json()["embedding"])
        return vectors

    def synonym_fn(predicates, context=None):
        return relation_synonyms(
            predicates, hermes_url=hermes_url, token=token, context=context
        )

    def refresh_snapshot():
        if redis_client is None:
            return
        records = hcg.get_relation_vocabulary()
        snapshot = {
            r["relation"]: {"edge_count": r.get("edge_count", 0)}
            for r in records
            if r.get("relation")
        }
        redis_client.set("logos:ontology:relations", json.dumps(snapshot))

    return RelationRollupHandler(
        hcg,
        embed_fn=embed_fn,
        synonym_fn=synonym_fn,
        min_confidence=min_confidence,
        cluster_threshold=cluster_threshold,
        on_consolidated=refresh_snapshot,
    )
