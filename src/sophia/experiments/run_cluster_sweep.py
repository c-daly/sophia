"""Compare clustering algorithms for type emergence (#505) ON the experiment harness.

Runs each candidate clustering algorithm through
``logos_experiment.ExperimentRunner`` (arrange/act/assert) on a labeled fixture
of entity embeddings, then scores each result against domain ground truth
(science / engineering / arts_humanities).

Usage: poetry run python -m sophia.experiments.run_cluster_sweep
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np

from logos_experiment.config import ExperimentConfig, PipelineStep
from logos_experiment.runner import ExperimentRunner
from sophia.experiments.agents.clustering import (
    CosineKMeansAgent,
    ProductionAgglomerativeAgent,
    _normalize,
    silhouette_cosine,
)

FIXTURE = Path(__file__).parent / "data" / "cluster_fixture.json"


def _comb2(x: float) -> float:
    return x * (x - 1) / 2.0


def adjusted_rand_index(true: list, pred: list) -> float:
    classes = sorted(set(true))
    clusters = sorted(set(pred))
    ci = {c: i for i, c in enumerate(classes)}
    ki = {c: i for i, c in enumerate(clusters)}
    cont = np.zeros((len(classes), len(clusters)))
    for t, p in zip(true, pred):
        cont[ci[t], ki[p]] += 1
    sum_comb = sum(_comb2(v) for v in cont.flatten())
    a, b = cont.sum(1), cont.sum(0)
    sa = sum(_comb2(v) for v in a)
    sb = sum(_comb2(v) for v in b)
    total = _comb2(len(true))
    exp = sa * sb / total if total else 0.0
    mx = (sa + sb) / 2
    return 0.0 if mx == exp else (sum_comb - exp) / (mx - exp)


def purity(true: list, pred: list) -> float:
    groups: dict = {}
    for t, p in zip(true, pred):
        groups.setdefault(p, []).append(t)
    return sum(Counter(v).most_common(1)[0][1] for v in groups.values()) / len(true)


def run_variant(
    name: str, factory: Callable[..., Any], cfg: dict[str, Any], embeddings: list
) -> list[list[int]]:
    """Run one clustering algorithm THROUGH the experiment harness."""
    config = ExperimentConfig(
        name=name,
        seed=0,
        pipeline=[PipelineStep(name="cluster", factory="injected", config=cfg)],
    )
    runner = ExperimentRunner(config)
    runner.arrange(factories={"cluster": lambda c: factory(**c)})
    runner.act([embeddings])  # single-item corpus = the whole member set
    return cast("list[list[int]]", runner.assert_results()["results"][0])


def main() -> None:
    rows = json.loads(FIXTURE.read_text())
    names = [r["name"] for r in rows]
    domains = [r["domain"] for r in rows]
    embs = [r["embedding"] for r in rows]
    x = _normalize(np.asarray(embs, dtype=float))
    n = len(rows)
    print(f"fixture: {n} entities  domains={dict(Counter(domains))}\n")

    # NOTE: the prior recursive-binary-split algorithm scored 0 clusters on this
    # data (curse of dimensionality -- see #505); it was replaced by agglomerative.
    variants: list[tuple[str, Callable[..., Any], dict[str, Any]]] = [
        (
            "cosine_kmeans_silhouette",
            CosineKMeansAgent,
            {"min_cluster_size": 3, "seed": 0},
        ),
        (
            "agglomerative (SHIPPED)",
            ProductionAgglomerativeAgent,
            {"min_cluster_size": 3, "variance_threshold": 0.6},
        ),
    ]

    print(
        f"{'algorithm':36} {'#cl':>4} {'cover':>6} {'silhou':>7} {'purity':>7} {'ARI':>6}"
    )
    print("-" * 72)
    detail = []
    for vname, fac, cfg in variants:
        clusters = run_variant(vname, fac, cfg, embs)
        clustered = [(i, ci) for ci, cl in enumerate(clusters) for i in cl]
        cov = len(clustered) / n
        if clustered and len({ci for _, ci in clustered}) > 1:
            idx = [i for i, _ in clustered]
            pred = [ci for _, ci in clustered]
            true = [domains[i] for i in idx]
            sil = silhouette_cosine(x[idx], np.asarray(pred))
            pur = purity(true, pred)
            ari = adjusted_rand_index(true, pred)
        else:
            sil = pur = ari = float("nan")
        print(
            f"{vname:36} {len(clusters):>4} {cov:>5.0%} {sil:>7.3f} {pur:>7.2f} {ari:>6.2f}"
        )
        detail.append((vname, clusters))

    print("\n=== cluster contents ===")
    for vname, clusters in detail:
        print(f"\n[{vname}]")
        if not clusters:
            print("   (no clusters)")
        for ci, cl in enumerate(clusters):
            doms = dict(Counter(domains[i] for i in cl))
            print(
                f"   c{ci} (n={len(cl)}) {doms}: {', '.join(names[i] for i in cl[:6])}"
            )


if __name__ == "__main__":
    main()
