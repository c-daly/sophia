"""End-to-end test for embedding persistence at ingestion (#146).

Regression guard for the keystone defect: the batch-flush of pending
embeddings used to be wrapped in a warn-only ``try/except`` in
``ProposalProcessor.process``. A failed Milvus write was swallowed, so
ingestion reported success while the ``hcg_*_embeddings`` collections stayed
empty -- starving the type classifier and #505 emergent type discovery.

This test ingests a small fixture through the real ``ProposalProcessor`` against
a live Neo4j + Milvus stack and asserts that the relevant
``hcg_*_embeddings`` collection row count grows by exactly the number of
embedded nodes/edges (previously it stayed at 0).

Requires Neo4j + Milvus from the shared test stack. Connection comes from env:
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, MILVUS_HOST, MILVUS_PORT

Depends on the logos #528 fix (``HCGMilvusSync`` upsert-by-uuid) being
installed in the environment so the upsert actually lands.
"""

import logging
import os
import uuid as uuid_lib

import pytest

from logos_hcg.sync import COLLECTION_NAMES, HCGMilvusSync

from sophia.hcg_client import HCGClient
from sophia.ingestion.proposal_processor import ProposalProcessor

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.integration

DIM = int(os.getenv("LOGOS_EMBEDDING_DIM", "384"))


def _make_centroid(index: int) -> list[float]:
    """A vector with energy concentrated at ``index`` (mirrors the e2e helper)."""
    vec = [0.0] * DIM
    vec[index % DIM] = 1.0
    for i in range(DIM):
        vec[i] += 0.01
    return vec


def _make_near(centroid: list[float], noise: float = 0.05) -> list[float]:
    return [v + noise * (0.5 - (i % 3) * 0.25) for i, v in enumerate(centroid)]


# Seed centroids so the classifier has something to classify against and the
# nodes land in a predictable (Entity) collection.
ENTITY_CENTROID = _make_centroid(200)
SEED_TYPES = {"type_entity": ENTITY_CENTROID}


def _count_rows(sync: HCGMilvusSync, node_type: str) -> int:
    """Return the number of persisted rows in a collection.

    Uses ``num_entities`` after an explicit flush, which is reliable here
    because ``batch_upsert_embeddings`` flushes on write. Resolves the
    collection from the public name + connection alias rather than the private
    ``sync._get_collection`` accessor, so a rename there fails as a clean
    assertion rather than an AttributeError.
    """
    from pymilvus import Collection

    collection = Collection(name=COLLECTION_NAMES[node_type], using=sync.alias)
    collection.flush()
    return int(collection.num_entities)


@pytest.fixture(scope="module")
def milvus_sync():
    host = os.getenv("MILVUS_HOST", "localhost")
    port = os.getenv("MILVUS_PORT", "47530")

    sync = HCGMilvusSync(milvus_host=host, milvus_port=port)
    try:
        sync.connect()
    except Exception as e:  # pragma: no cover - infra gate
        pytest.skip(f"Milvus not available at {host}:{port}: {e}")

    # Isolation: start from a clean embedding space. Post-#146 embeddings actually
    # persist, so nodes ingested by earlier integration modules survive in the
    # shared test Milvus and would skew this module's row-count delta assertions
    # (and dedup an ingest down to zero growth). Drop first so the module sees
    # only its own data.
    from pymilvus import utility

    for _name in utility.list_collections(using=sync.alias):
        utility.drop_collection(_name, using=sync.alias)
    # The drops above bypass HCGMilvusSync's internal collection cache; clear it
    # so ensure_collection rebuilds fresh handles (v0.7.2 #542 fast-path would
    # otherwise reuse a stale Collection id -> "collection not found" on upsert).
    sync._collections.clear()

    for node_type in COLLECTION_NAMES:
        sync.ensure_collection(node_type, DIM)

    for type_uuid, centroid in SEED_TYPES.items():
        sync.update_centroid(
            type_uuid=type_uuid, centroid=centroid, model="synthetic-test"
        )

    yield sync

    sync.disconnect()


@pytest.fixture(scope="module")
def hcg_client():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "logosdev")

    try:
        client = HCGClient(neo4j_uri=uri, neo4j_username=user, neo4j_password=password)
    except Exception as e:  # pragma: no cover - infra gate
        pytest.skip(f"Neo4j not available at {uri}: {e}")
    yield client
    client.close()


@pytest.fixture(scope="module")
def processor(hcg_client, milvus_sync):
    return ProposalProcessor(hcg_client=hcg_client, milvus_sync=milvus_sync)


class TestEmbeddingPersistenceE2E:
    """Ingestion must actually land embeddings in Milvus, not silently drop them."""

    def test_ingested_node_embeddings_land_in_milvus(self, processor, milvus_sync):
        """Row count for the target collection grows by the embedded-node count.

        Before #146 this delta was 0 (warn-only swallow). After the fix, the
        write must succeed and the rows must be present.
        """
        before = _count_rows(milvus_sync, "Entity")

        # Two nodes, both near the entity centroid -> both land in the Entity
        # collection (hcg_entity_embeddings). Unique names avoid dedup collisions.
        run = uuid_lib.uuid4().hex[:8]
        node_names = [f"PersistNodeA_{run}", f"PersistNodeB_{run}"]
        emb = _make_near(ENTITY_CENTROID)
        proposed_nodes = [
            {
                "name": name,
                "type": "entity",
                "embedding": emb,
                "embedding_id": f"emb-{name}",
                "dimension": DIM,
                "model": "synthetic-test",
                "properties": {},
            }
            for name in node_names
        ]

        proposal = {
            "proposal_id": f"e2e-persist-{run}",
            "source_service": "hermes",
            "confidence": 0.8,
            "raw_text": "Persistence regression fixture",
            "proposed_nodes": proposed_nodes,
            "proposed_edges": [],
            "document_embedding": {
                "embedding": emb,
                "embedding_id": f"doc-{run}",
                "dimension": DIM,
                "model": "synthetic-test",
            },
        }

        result = processor.process(proposal)

        # All nodes were stored in the graph.
        stored = result["stored_node_ids"]
        assert len(stored) == len(
            node_names
        ), f"Expected {len(node_names)} stored nodes, got {len(stored)}"

        after = _count_rows(milvus_sync, "Entity")
        delta = after - before
        logger.info(
            "hcg_entity_embeddings count: before=%d after=%d delta=%d",
            before,
            after,
            delta,
        )

        # The keystone assertion: embeddings actually persisted. Previously 0.
        assert delta >= len(node_names), (
            "Embedding persistence regressed: expected the Entity collection to "
            f"grow by at least {len(node_names)} rows, but it grew by {delta}. "
            "A failed/swallowed Milvus write leaves hcg_*_embeddings empty."
        )
        assert after > 0, "hcg_entity_embeddings must be non-empty after ingestion"
