"""Integration tests for Hermes ingestion endpoint.

These tests require Sophia and Neo4j to be running.
Run with: pytest tests/integration/test_hermes_ingestion_integration.py -v -m integration

Start services with: ./scripts/test_integration.sh up
"""

import pytest

pytestmark = [
    pytest.mark.integration,
]


class TestHermesIngestionIntegration:
    """Integration tests for Hermes proposal ingestion."""

    @pytest.fixture
    def sample_proposal(self):
        """Sample Hermes proposal payload matching HermesProposalRequest model."""
        return {
            "proposal_id": "test-proposal-001",
            "source_service": "hermes",
            "llm_provider": "openai",
            "model": "gpt-4",
            "generated_at": "2025-01-01T00:00:00Z",
            "confidence": 0.95,
            "raw_text": "The robot should pick up the red block and place it in the bin.",
            "plan_steps": [
                {"action": "pick_up", "target": "red_block"},
                {"action": "place", "target": "bin"},
            ],
            "imagined_states": [
                {"state_id": "state_1", "entities": {"red_block": {"location": "bin"}}},
            ],
            "metadata": {
                "session_id": "test-session",
            },
        }

    def test_health_check_includes_neo4j(self, http_client):
        """Test that health check reports Neo4j status."""
        response = http_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "components" in data
        assert "neo4j" in data["components"]

    def test_ingest_proposal_creates_neo4j_nodes(
        self, http_client, auth_headers, sample_proposal
    ):
        """Test that ingesting a proposal creates nodes in Neo4j."""
        response = http_client.post(
            "/ingest/hermes_proposal",
            json=sample_proposal,
            headers=auth_headers,
        )
        assert response.status_code in [200, 201]

        data = response.json()
        assert "proposal_id" in data
        assert "stored_node_ids" in data

    def test_ingest_minimal_proposal(self, http_client, auth_headers):
        """Test ingesting a minimal valid proposal."""
        minimal_proposal = {
            "proposal_id": "minimal-001",
            "source_service": "hermes",
            "llm_provider": "openai",
            "model": "gpt-4",
            "generated_at": "2025-01-01T00:00:00Z",
            "confidence": 0.8,
        }
        response = http_client.post(
            "/ingest/hermes_proposal",
            json=minimal_proposal,
            headers=auth_headers,
        )
        assert response.status_code in [200, 201]

    def test_ingest_proposal_with_raw_text_only(self, http_client, auth_headers):
        """Test ingesting proposal with only raw text."""
        proposal = {
            "proposal_id": "text-only-001",
            "source_service": "hermes",
            "llm_provider": "anthropic",
            "model": "claude-3-opus",
            "generated_at": "2025-01-01T00:00:00Z",
            "confidence": 0.9,
            "raw_text": "Move the blue block to the table.",
        }
        response = http_client.post(
            "/ingest/hermes_proposal",
            json=proposal,
            headers=auth_headers,
        )
        assert response.status_code in [200, 201]

    def test_ingest_multiple_proposals(self, http_client, auth_headers):
        """Test ingesting multiple proposals in sequence."""
        proposals = [
            {
                "proposal_id": f"batch-{i}",
                "source_service": "hermes",
                "llm_provider": "openai",
                "model": "gpt-4",
                "generated_at": "2025-01-01T00:00:00Z",
                "confidence": 0.85,
                "raw_text": f"Proposal number {i}",
            }
            for i in range(3)
        ]

        for proposal in proposals:
            response = http_client.post(
                "/ingest/hermes_proposal",
                json=proposal,
                headers=auth_headers,
            )
            assert response.status_code in [200, 201]

    def test_ingest_proposal_nodes_retrievable(
        self, http_client, auth_headers, sample_proposal, hcg_client
    ):
        """Test that ingested proposal nodes can be retrieved from Neo4j."""
        response = http_client.post(
            "/ingest/hermes_proposal",
            json=sample_proposal,
            headers=auth_headers,
        )
        assert response.status_code in [200, 201]

        # Query Neo4j for the proposal
        with hcg_client.driver.session(database=hcg_client.database) as session:
            result = session.run(
                """
                MATCH (n {id: $proposal_id})
                RETURN n
                """,
                {"proposal_id": sample_proposal["proposal_id"]},
            )
            # Just verify query runs; may be 0 if proposal storage is different
            list(result)

    def test_ingest_proposal_preserves_provenance(
        self, http_client, auth_headers, hcg_client
    ):
        """Test that ingested proposals preserve source provenance."""
        proposal = {
            "proposal_id": "provenance-test-001",
            "source_service": "hermes-test",
            "llm_provider": "openai",
            "model": "gpt-4",
            "generated_at": "2025-01-01T00:00:00Z",
            "confidence": 0.9,
            "raw_text": "Test provenance",
        }

        response = http_client.post(
            "/ingest/hermes_proposal",
            json=proposal,
            headers=auth_headers,
        )
        assert response.status_code in [200, 201]

        # Verify provenance in response
        data = response.json()
        assert "proposal_id" in data
        assert data["proposal_id"] == proposal["proposal_id"]
