"""One-shot trigger for the relation-vocabulary rollup on the live graph.

Uses the disk-cached embeddings (no re-embedding -> no OpenAI rate-limit
risk), the live /relation-synonyms endpoint, and the live graph for
consolidation. Bounded to a deterministic 350-relation slice. Writes the
before/after df=1 result to /tmp/rollup_result.txt.
"""

import json
import os
import random
import time

import httpx

from sophia.hcg_client.client import HCGClient
from sophia.maintenance.hermes_naming import relation_synonyms
from sophia.maintenance.relation_rollup_handler import RelationRollupHandler

OUT = "/tmp/rollup_result.txt"
CACHE = "/tmp/rel_emb_cache.json"


def log(msg):
    with open(OUT, "a") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


def df1(vocab):
    d = len(vocab)
    o = sum(1 for v in vocab if v["edge_count"] == 1)
    return d, o, (o / d if d else 0.0)


def main():
    open(OUT, "w").close()
    hcg = HCGClient(
        neo4j_uri="bolt://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="logosdev",
    )
    full = hcg.get_relation_vocabulary()
    bd, bo, bf = df1(full)
    log(f"BEFORE (full graph): {bd} distinct, df=1={bo} ({bf:.3f})")

    by_count = sorted(full, key=lambda v: -v["edge_count"])
    heads = [v["relation"] for v in by_count[:100]]
    tail = [v["relation"] for v in by_count if v["edge_count"] == 1]
    random.Random(7).shuffle(tail)
    candidates = heads + tail[:250]
    counts = {v["relation"]: v["edge_count"] for v in full}

    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

    def embed_fn(labels):
        missing = [l for l in labels if l not in cache]
        if missing:
            raise RuntimeError(
                f"{len(missing)} labels not cached; refusing to embed live"
            )
        return [cache[l] for l in labels]

    def synonym_fn(preds, context=None):
        return relation_synonyms(
            preds, hermes_url="http://localhost:17000", token="", context=context
        )

    hcg.get_relation_vocabulary = lambda: [
        {"relation": r, "edge_count": counts[r]} for r in candidates
    ]
    handler = RelationRollupHandler(
        hcg,
        embed_fn=embed_fn,
        synonym_fn=synonym_fn,
        min_confidence=0.6,
        cluster_threshold=0.5,
        max_cluster_size=40,
    )
    t = time.time()
    summary = handler.run()
    log(f"rollup summary: {summary} in {time.time() - t:.0f}s")

    hcg2 = HCGClient(
        neo4j_uri="bolt://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="logosdev",
    )
    after = hcg2.get_relation_vocabulary()
    ad, ao, af = df1(after)
    log(f"AFTER  (full graph): {ad} distinct, df=1={ao} ({af:.3f})  gate<0.25")
    log(
        f"DELTA: distinct {bd}->{ad} ({bd - ad} consolidated); "
        f"df=1 {bf:.3f}->{af:.3f}; L_model relation bits saved {16 * (bd - ad):,}"
    )
    log("DONE")


if __name__ == "__main__":
    main()
