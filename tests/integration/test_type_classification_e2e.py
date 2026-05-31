"""End-to-end test for Sophia-driven type classification.

Requires: Neo4j (bolt://localhost:7687) and Milvus (localhost:19530) running.
Does NOT require Sophia or Hermes HTTP services — tests ProposalProcessor directly.

Verifies that Sophia classifies node types using embedding-space centroids,
ignoring the type hint provided by Hermes.
"""

import logging
import os

import pytest

from logos_hcg.sync import HCGMilvusSync

from sophia.hcg_client import HCGClient
from sophia.ingestion.proposal_processor import ProposalProcessor

logger = logging.getLogger(__name__)

# Mark all tests as integration
pytestmark = pytest.mark.integration

# Embedding dimension — must match logos_hcg default (384)
DIM = int(os.getenv("LOGOS_EMBEDDING_DIM", "384"))

# --- Synthetic embedding helpers ---
# We create centroids as unit-ish vectors pointing in different directions,
# then create test embeddings close to specific centroids.


def _make_centroid(index: int) -> list[float]:
    """Create a centroid vector with energy concentrated at `index`."""
    vec = [0.0] * DIM
    vec[index % DIM] = 1.0
    # Add small spread so it's not perfectly sparse
    for i in range(DIM):
        vec[i] += 0.01
    return vec


def _make_near(centroid: list[float], noise: float = 0.05) -> list[float]:
    """Create a vector close to `centroid` with slight perturbation."""
    return [v + noise * (0.5 - (i % 3) * 0.25) for i, v in enumerate(centroid)]


# Centroids for each type (placed at distinct directions in embedding space)
LOCATION_CENTROID = _make_centroid(0)
CONCEPT_CENTROID = _make_centroid(50)
ENTITY_CENTROID = _make_centroid(200)

# Reserved internal types — centroids placed far from general knowledge
RESERVED_STATE_CENTROID = _make_centroid(300)  # CWM states only
RESERVED_PROCESS_CENTROID = _make_centroid(350)  # Plan execution only

SEED_TYPES = {
    "type_location": LOCATION_CENTROID,
    "type_concept": CONCEPT_CENTROID,
    "type_entity": ENTITY_CENTROID,
    "type_reserved_state": RESERVED_STATE_CENTROID,
    "type_reserved_process": RESERVED_PROCESS_CENTROID,
}


@pytest.fixture(scope="module")
def milvus_sync():
    """Connect to real Milvus and seed type centroids."""
    host = os.getenv("MILVUS_HOST", "localhost")
    port = os.getenv("MILVUS_PORT", "19530")

    sync = HCGMilvusSync(milvus_host=host, milvus_port=port)
    sync.connect()

    # Isolation: start from a clean embedding space. Post-#146 a failed embedding
    # write is no longer swallowed, so embeddings actually persist — which means
    # nodes ingested by *earlier* integration modules survive in the shared test
    # Milvus and get returned by this module's similarity/dedup search (e.g. an
    # ENTITY_CENTROID node from test_embedding_persistence_e2e), causing this
    # module's near-duplicate ingests to be skipped. Drop first so the module's
    # assertions see only its own seeded centroids and ingested nodes.
    from pymilvus import utility

    for _name in utility.list_collections(using=sync.alias):
        utility.drop_collection(_name, using=sync.alias)
    # The drops above bypass HCGMilvusSync's internal collection cache; clear it
    # so ensure_collection rebuilds fresh handles. Without this, v0.7.2's #542
    # fast-path returns a stale Collection pointing at the just-dropped id and
    # upserts fail with "collection not found".
    sync._collections.clear()

    # Ensure all collections exist
    for node_type in ["Entity", "Concept", "State", "Process", "Edge", "TypeCentroid"]:
        sync.ensure_collection(node_type, DIM)

    # Seed type centroids
    for type_uuid, centroid in SEED_TYPES.items():
        sync.update_centroid(
            type_uuid=type_uuid,
            centroid=centroid,
            model="synthetic-test",
        )
    logger.info("Seeded %d type centroids in Milvus", len(SEED_TYPES))

    yield sync

    sync.disconnect()


@pytest.fixture(scope="module")
def hcg_client():
    """Connect to real Neo4j."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "logosdev")

    client = HCGClient(
        neo4j_uri=uri,
        neo4j_username=user,
        neo4j_password=password,
    )
    yield client
    client.close()


@pytest.fixture(scope="module")
def processor(hcg_client, milvus_sync):
    """Create a ProposalProcessor with real backends."""
    return ProposalProcessor(
        hcg_client=hcg_client,
        milvus_sync=milvus_sync,
    )


class TestTypeClassificationE2E:
    """Sophia classifies nodes by embedding proximity, not Hermes type hints."""

    def test_location_classified_despite_hermes_saying_state(
        self, processor, hcg_client, milvus_sync
    ):
        """Dublin's embedding is near the location centroid.

        Hermes says 'state' but Sophia should classify it as 'location'.
        """
        dublin_embedding = _make_near(LOCATION_CENTROID)

        proposal = {
            "proposal_id": "e2e-type-class-1",
            "source_service": "hermes",
            "confidence": 0.8,
            "raw_text": "Dublin is a city in Ireland",
            "proposed_nodes": [
                {
                    "name": "Dublin_e2e_test",
                    "type": "state",  # Wrong Hermes classification
                    "embedding": dublin_embedding,
                    "embedding_id": "emb-dublin-e2e",
                    "dimension": DIM,
                    "model": "synthetic-test",
                    "properties": {"start": 0, "end": 6},
                }
            ],
            "proposed_edges": [],
            "document_embedding": {
                "embedding": dublin_embedding,
                "embedding_id": "doc-e2e-1",
                "dimension": DIM,
                "model": "synthetic-test",
            },
        }

        result = processor.process(proposal)

        # Node should have been created
        assert len(result["stored_node_ids"]) == 1
        node_uuid = result["stored_node_ids"][0]

        # Verify in Neo4j: node should be classified as location, NOT state
        node = hcg_client.get_node(node_uuid)
        assert node is not None, f"Node {node_uuid} not found in Neo4j"

        node_type = node.get("type", "")
        logger.info("Dublin_e2e_test classified as: %s (Hermes said: state)", node_type)
        assert node_type == "location", (
            f"Expected 'location' but got '{node_type}'. "
            "Sophia should override Hermes type hint using centroid proximity."
        )

        # Verify confidence metadata was stored
        props = node.get("properties", {})
        assert "type_confidence" in props, "type_confidence should be in properties"
        assert props["type_confidence"] > 0, "Confidence should be positive"

    def test_concept_classified_correctly(self, processor, hcg_client, milvus_sync):
        """An embedding near the concept centroid gets classified as concept."""
        concept_embedding = _make_near(CONCEPT_CENTROID)

        proposal = {
            "proposal_id": "e2e-type-class-2",
            "source_service": "hermes",
            "confidence": 0.7,
            "raw_text": "Democracy is a system of governance",
            "proposed_nodes": [
                {
                    "name": "Democracy_e2e_test",
                    "type": "entity",  # Hermes says entity
                    "embedding": concept_embedding,
                    "embedding_id": "emb-democracy-e2e",
                    "dimension": DIM,
                    "model": "synthetic-test",
                    "properties": {},
                }
            ],
            "proposed_edges": [],
            "document_embedding": {
                "embedding": concept_embedding,
                "embedding_id": "doc-e2e-2",
                "dimension": DIM,
                "model": "synthetic-test",
            },
        }

        result = processor.process(proposal)

        assert len(result["stored_node_ids"]) == 1
        node_uuid = result["stored_node_ids"][0]

        node = hcg_client.get_node(node_uuid)
        assert node is not None
        assert (
            node.get("type") == "concept"
        ), f"Expected 'concept' but got '{node.get('type')}'"

    def test_general_entity_not_assigned_reserved_type(
        self, processor, hcg_client, milvus_sync
    ):
        """An entity embedding near 'entity' centroid should NOT land in reserved types.

        Photosynthesis is a real-world process but 'reserved_process' is for
        Sophia plan execution only. General knowledge goes to Entity/Concept.
        """
        entity_embedding = _make_near(ENTITY_CENTROID)

        proposal = {
            "proposal_id": "e2e-type-class-3",
            "source_service": "hermes",
            "confidence": 0.7,
            "raw_text": "Photosynthesis converts light to energy",
            "proposed_nodes": [
                {
                    "name": "Photosynthesis_e2e_test",
                    "type": "process",  # Hermes says process
                    "embedding": entity_embedding,
                    "embedding_id": "emb-photo-e2e",
                    "dimension": DIM,
                    "model": "synthetic-test",
                    "properties": {},
                }
            ],
            "proposed_edges": [],
            "document_embedding": {
                "embedding": entity_embedding,
                "embedding_id": "doc-e2e-3",
                "dimension": DIM,
                "model": "synthetic-test",
            },
        }

        result = processor.process(proposal)

        assert len(result["stored_node_ids"]) == 1
        node_uuid = result["stored_node_ids"][0]

        node = hcg_client.get_node(node_uuid)
        assert node is not None
        node_type = node.get("type", "")
        assert node_type not in (
            "reserved_state",
            "reserved_process",
        ), f"General knowledge should not be assigned reserved type '{node_type}'"
