"""Integration tests for Hermes proposal ingestion with real Neo4j.

These tests require Neo4j to be running.
Run with: pytest tests/integration/test_hermes_ingestion_integration.py -v -m integration

In CI, these run automatically with containerized Neo4j.
Locally, start services with: docker compose -f docker-compose.test.yml up -d
"""

import os
import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient

from sophia.api.app import create_app
from sophia.hcg_client import HCGClient


# Integration tests are run by CI with real services.
pytestmark = [
    pytest.mark.integration,
]


@pytest.fixture
def neo4j_uri():
    """Neo4j connection URI - uses offset port 37687."""
    return os.getenv("NEO4J_URI", "bolt://localhost:37687")


@pytest.fixture
def neo4j_username():
    """Neo4j username."""
    return os.getenv("NEO4J_USER", "neo4j")


@pytest.fixture
def neo4j_password():
    """Neo4j password."""
    return os.getenv("NEO4J_PASSWORD", "neo4jtest")


@pytest.fixture
def hcg_client(neo4j_uri, neo4j_username, neo4j_password):
    """Create HCG client for test verification."""
    client = HCGClient(
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password,
    )
    yield client
    client.close()


@pytest.fixture
def app():
    """Create test application."""
    # Use offset ports for sophia test stack
    os.environ.setdefault("NEO4J_URI", "bolt://localhost:37687")
    os.environ.setdefault("NEO4J_USER", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "neo4jtest")
    return create_app()


@pytest.fixture
def client(app):
    """Create test client."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_proposal():
    """Sample Hermes proposal payload."""
    return {
        "proposal_id": f"hermes_integration_test_{datetime.now(timezone.utc).timestamp()}",
        "source_service": "hermes",
        "llm_provider": "openai",
        "model": "gpt-4",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "confidence": 0.85,
        "raw_text": "Move the red block to the bin",
        "plan_steps": [
            {
                "action": "move_to_red_block",
                "target": "red_block",
                "parameters": {},
            },
            {
                "action": "grasp_red_block",
                "target": "red_block",
                "parameters": {"force": 0.5},
            },
            {
                "action": "move_to_bin",
                "target": "bin",
                "parameters": {},
            },
        ],
        "imagined_states": [
            {
                "state_id": "state_1",
                "entities": {"red_block": {"location": "table"}},
            },
            {
                "state_id": "state_2",
                "entities": {"red_block": {"location": "bin"}},
            },
        ],
        "diagnostics": {
            "reasoning": "Block needs to be moved from table to bin",
            "steps_count": 3,
        },
        "tool_calls": [
            {
                "tool": "get_object_location",
                "parameters": {"object_id": "red_block"},
            }
        ],
        "metadata": {
            "session_id": "integration_test_session",
            "user_id": "integration_test_user",
        },
    }


class TestHermesIngestionIntegration:
    """Integration tests for Hermes proposal ingestion with real Neo4j."""

    def test_health_check_includes_neo4j(self, client):
        """Test that health check reports Neo4j status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "components" in data
        assert "neo4j" in data["components"]
        assert data["components"]["neo4j"] is True

    def test_ingest_proposal_creates_neo4j_nodes(self, client, sample_proposal):
        """Test that ingesting a proposal creates nodes in Neo4j."""
        response = client.post("/ingest/hermes_proposal", json=sample_proposal)

        assert (
            response.status_code == 201
        ), f"Expected 201, got {response.status_code}: {response.text}"
        data = response.json()

        # Verify response structure
        assert data["proposal_id"] == sample_proposal["proposal_id"]
        assert data["status"] == "accepted"
        assert "stored_node_ids" in data
        assert "created_at" in data

        # Should have: 1 proposal + 3 plan steps + 2 states + 1 tool call = 7 nodes
        assert len(data["stored_node_ids"]) == 7
        assert data["stored_node_ids"][0] == sample_proposal["proposal_id"]

    def test_ingest_proposal_nodes_retrievable(
        self, client, sample_proposal, hcg_client
    ):
        """Test that ingested proposal nodes can be retrieved from Neo4j."""
        response = client.post("/ingest/hermes_proposal", json=sample_proposal)
        assert response.status_code == 201

        proposal_id = sample_proposal["proposal_id"]

        # Verify proposal node exists in Neo4j
        proposal_node = hcg_client.get_node(proposal_id)
        assert proposal_node is not None
        assert proposal_node["type"] == "hermes_proposal"

        # Verify properties were stored
        props = proposal_node.get("properties", proposal_node)
        assert props.get("llm_provider") == "openai"
        assert props.get("model") == "gpt-4"
        assert props.get("confidence") == 0.85

    def test_ingest_minimal_proposal(self, client):
        """Test ingesting a proposal with only required fields."""
        minimal_proposal = {
            "proposal_id": f"hermes_minimal_{datetime.now(timezone.utc).timestamp()}",
            "llm_provider": "anthropic",
            "model": "claude-3-opus",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "confidence": 0.90,
        }

        response = client.post("/ingest/hermes_proposal", json=minimal_proposal)

        assert response.status_code == 201
        data = response.json()
        assert data["proposal_id"] == minimal_proposal["proposal_id"]
        assert data["status"] == "accepted"
        # Only the proposal node itself
        assert len(data["stored_node_ids"]) == 1

    def test_ingest_proposal_with_raw_text_only(self, client):
        """Test ingesting a proposal with raw text but no structured steps."""
        proposal = {
            "proposal_id": f"hermes_rawtext_{datetime.now(timezone.utc).timestamp()}",
            "llm_provider": "openai",
            "model": "gpt-4",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "confidence": 0.75,
            "raw_text": "Pick up the blue cube and place it on the shelf",
        }

        response = client.post("/ingest/hermes_proposal", json=proposal)

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "accepted"

    def test_ingest_multiple_proposals(self, client):
        """Test ingesting multiple proposals in sequence."""
        proposals = []
        for i in range(3):
            proposal = {
                "proposal_id": f"hermes_batch_{i}_{datetime.now(timezone.utc).timestamp()}",
                "llm_provider": "openai",
                "model": "gpt-4",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "confidence": 0.80 + i * 0.05,
            }
            proposals.append(proposal)

        # Ingest all proposals
        for proposal in proposals:
            response = client.post("/ingest/hermes_proposal", json=proposal)
            assert response.status_code == 201

    def test_ingest_proposal_preserves_provenance(
        self, client, sample_proposal, hcg_client
    ):
        """Test that all provenance metadata is preserved in Neo4j."""
        response = client.post("/ingest/hermes_proposal", json=sample_proposal)
        assert response.status_code == 201

        # Retrieve and verify provenance
        proposal_node = hcg_client.get_node(sample_proposal["proposal_id"])
        assert proposal_node is not None

        props = proposal_node.get("properties", proposal_node)

        # Check all provenance fields
        assert "source_service" in props or props.get("source_service") == "hermes"
        assert "llm_provider" in props
        assert "model" in props
        assert "generated_at" in props
        assert "confidence" in props
        assert "ingested_at" in props


class TestHermesIngestionValidation:
    """Integration tests for validation of Hermes proposals."""

    def test_ingest_missing_required_field_returns_422(self, client):
        """Test that missing required fields return 422."""
        invalid_proposal = {
            "proposal_id": "hermes_invalid",
            # Missing llm_provider, model, generated_at, confidence
        }

        response = client.post("/ingest/hermes_proposal", json=invalid_proposal)
        assert response.status_code == 422

    def test_ingest_invalid_confidence_returns_422(self, client):
        """Test that confidence outside [0, 1] returns 422."""
        invalid_proposal = {
            "proposal_id": "hermes_invalid_confidence",
            "llm_provider": "openai",
            "model": "gpt-4",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "confidence": 1.5,  # Invalid: > 1.0
        }

        response = client.post("/ingest/hermes_proposal", json=invalid_proposal)
        assert response.status_code == 422

    def test_ingest_negative_confidence_returns_422(self, client):
        """Test that negative confidence returns 422."""
        invalid_proposal = {
            "proposal_id": "hermes_negative_confidence",
            "llm_provider": "openai",
            "model": "gpt-4",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "confidence": -0.1,  # Invalid: < 0.0
        }

        response = client.post("/ingest/hermes_proposal", json=invalid_proposal)
        assert response.status_code == 422
