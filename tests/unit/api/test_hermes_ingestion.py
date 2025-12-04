"""Unit tests for Hermes proposal ingestion endpoint.

These tests use mocks to verify endpoint behavior in isolation.
For tests with real Neo4j, see tests/integration/test_hermes_ingestion_integration.py
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, ANY
from datetime import datetime, timezone

from sophia.api.app import create_app


pytestmark = pytest.mark.unit


@pytest.fixture
def app():
    """Create test application."""
    return create_app()


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_proposal():
    """Sample Hermes proposal payload."""
    return {
        "proposal_id": "hermes_test_001",
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
            "session_id": "test_session_123",
            "user_id": "test_user",
        },
    }


class TestHermesIngestionEndpoint:
    """Tests for the /ingest/hermes_proposal endpoint."""

    def test_ingestion_no_auth_required(self, client, sample_proposal):
        """Test that ingestion endpoint doesn't require authentication (local dev)."""
        with patch("sophia.api.app._hcg_client") as mock_hcg:
            mock_hcg.add_node = Mock()
            mock_hcg.add_edge = Mock()

            response = client.post("/ingest/hermes_proposal", json=sample_proposal)

            # Should not return 403 Forbidden
            assert response.status_code != 403

    def test_ingestion_success(self, client, sample_proposal):
        """Test successful proposal ingestion."""
        with patch("sophia.api.app._hcg_client") as mock_hcg:
            mock_hcg.add_node = Mock()
            mock_hcg.add_edge = Mock()

            response = client.post("/ingest/hermes_proposal", json=sample_proposal)

            assert response.status_code == 201
            data = response.json()

            assert data["proposal_id"] == sample_proposal["proposal_id"]
            assert data["status"] == "accepted"
            assert "stored_node_ids" in data
            assert len(data["stored_node_ids"]) > 0
            assert "created_at" in data

    def test_ingestion_creates_proposal_node(self, client, sample_proposal):
        """Test that ingestion creates the main proposal node."""
        with patch("sophia.api.app._hcg_client") as mock_hcg:
            mock_hcg.add_node = Mock()
            mock_hcg.add_edge = Mock()

            response = client.post("/ingest/hermes_proposal", json=sample_proposal)

            assert response.status_code == 201

            # Verify proposal node was created
            mock_hcg.add_node.assert_any_call(
                node_id=sample_proposal["proposal_id"],
                node_type="hermes_proposal",
                properties={
                    "source_service": "hermes",
                    "llm_provider": "openai",
                    "model": "gpt-4",
                    "generated_at": sample_proposal["generated_at"],
                    "confidence": 0.85,
                    "raw_text": "Move the red block to the bin",
                    "diagnostics": sample_proposal["diagnostics"],
                    "session_id": "test_session_123",
                    "user_id": "test_user",
                    "ingested_at": ANY,
                },
            )

    def test_ingestion_creates_plan_step_nodes(self, client, sample_proposal):
        """Test that ingestion creates plan step nodes and edges."""
        with patch("sophia.api.app._hcg_client") as mock_hcg:
            mock_hcg.add_node = Mock()
            mock_hcg.add_edge = Mock()

            response = client.post("/ingest/hermes_proposal", json=sample_proposal)

            assert response.status_code == 201

            # Verify plan step nodes were created (3 steps)
            plan_step_calls = [
                call
                for call in mock_hcg.add_node.call_args_list
                if call[1]["node_type"] == "proposed_plan_step"
            ]
            assert len(plan_step_calls) == 3

            # Verify edges were created
            edge_calls = [
                call
                for call in mock_hcg.add_edge.call_args_list
                if call[1]["relation"] == "contains_plan_step"
            ]
            assert len(edge_calls) == 3

    def test_ingestion_creates_imagined_state_nodes(self, client, sample_proposal):
        """Test that ingestion creates imagined state nodes and edges."""
        with patch("sophia.api.app._hcg_client") as mock_hcg:
            mock_hcg.add_node = Mock()
            mock_hcg.add_edge = Mock()

            response = client.post("/ingest/hermes_proposal", json=sample_proposal)

            assert response.status_code == 201

            # Verify imagined state nodes were created (2 states)
            state_calls = [
                call
                for call in mock_hcg.add_node.call_args_list
                if call[1]["node_type"] == "proposed_imagined_state"
            ]
            assert len(state_calls) == 2

            # Verify edges were created
            edge_calls = [
                call
                for call in mock_hcg.add_edge.call_args_list
                if call[1]["relation"] == "contains_imagined_state"
            ]
            assert len(edge_calls) == 2

    def test_ingestion_creates_tool_call_nodes(self, client, sample_proposal):
        """Test that ingestion creates tool call nodes and edges."""
        with patch("sophia.api.app._hcg_client") as mock_hcg:
            mock_hcg.add_node = Mock()
            mock_hcg.add_edge = Mock()

            response = client.post("/ingest/hermes_proposal", json=sample_proposal)

            assert response.status_code == 201

            # Verify tool call nodes were created (1 tool call)
            tool_calls = [
                call
                for call in mock_hcg.add_node.call_args_list
                if call[1]["node_type"] == "proposed_tool_call"
            ]
            assert len(tool_calls) == 1

            # Verify edges were created
            edge_calls = [
                call
                for call in mock_hcg.add_edge.call_args_list
                if call[1]["relation"] == "contains_tool_call"
            ]
            assert len(edge_calls) == 1

    def test_ingestion_minimal_payload(self, client):
        """Test ingestion with minimal required fields."""
        minimal_proposal = {
            "proposal_id": "hermes_minimal_001",
            "llm_provider": "openai",
            "model": "gpt-4",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "confidence": 0.75,
        }

        with patch("sophia.api.app._hcg_client") as mock_hcg:
            mock_hcg.add_node = Mock()
            mock_hcg.add_edge = Mock()

            response = client.post("/ingest/hermes_proposal", json=minimal_proposal)

            assert response.status_code == 201
            data = response.json()
            assert data["proposal_id"] == "hermes_minimal_001"
            assert data["status"] == "accepted"

    def test_ingestion_missing_required_field(self, client):
        """Test that missing required fields return 422."""
        invalid_proposal = {
            "proposal_id": "hermes_invalid_001",
            # Missing llm_provider, model, generated_at, confidence
        }

        response = client.post("/ingest/hermes_proposal", json=invalid_proposal)

        assert response.status_code == 422

    def test_ingestion_invalid_confidence_range(self, client):
        """Test that confidence outside [0, 1] returns 422."""
        invalid_proposal = {
            "proposal_id": "hermes_invalid_002",
            "llm_provider": "openai",
            "model": "gpt-4",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "confidence": 1.5,  # Invalid: > 1.0
        }

        response = client.post("/ingest/hermes_proposal", json=invalid_proposal)

        assert response.status_code == 422

    def test_ingestion_shacl_validation_failure(self, client, sample_proposal):
        """Test that SHACL validation failure returns 422."""
        with patch("sophia.api.app._hcg_client") as mock_hcg:
            # Mock SHACL validation failure
            mock_hcg.add_node = Mock(side_effect=ValueError("SHACL validation failed"))

            response = client.post("/ingest/hermes_proposal", json=sample_proposal)

            assert response.status_code == 422
            data = response.json()
            assert "validation failed" in data["detail"].lower()

    def test_ingestion_hcg_unavailable(self, client, sample_proposal):
        """Test that HCG unavailability returns 503."""
        with patch("sophia.api.app._hcg_client", None):
            response = client.post("/ingest/hermes_proposal", json=sample_proposal)

            assert response.status_code == 503
            data = response.json()
            assert "not available" in data["detail"].lower()

    def test_ingestion_internal_error(self, client, sample_proposal):
        """Test that internal errors return 500."""
        with patch("sophia.api.app._hcg_client") as mock_hcg:
            # Mock unexpected error
            mock_hcg.add_node = Mock(side_effect=Exception("Database connection lost"))

            response = client.post("/ingest/hermes_proposal", json=sample_proposal)

            assert response.status_code == 500
            data = response.json()
            assert "failed to ingest" in data["detail"].lower()

    def test_ingestion_returns_all_node_ids(self, client, sample_proposal):
        """Test that response includes all created node IDs."""
        with patch("sophia.api.app._hcg_client") as mock_hcg:
            mock_hcg.add_node = Mock()
            mock_hcg.add_edge = Mock()

            response = client.post("/ingest/hermes_proposal", json=sample_proposal)

            assert response.status_code == 201
            data = response.json()

            # Should have: 1 proposal + 3 plan steps + 2 states + 1 tool call = 7 nodes
            assert len(data["stored_node_ids"]) == 7

            # First node should be the proposal itself
            assert data["stored_node_ids"][0] == sample_proposal["proposal_id"]

    def test_ingestion_without_optional_fields(self, client):
        """Test ingestion without optional arrays (plan_steps, imagined_states, tool_calls)."""
        proposal = {
            "proposal_id": "hermes_no_optionals_001",
            "llm_provider": "anthropic",
            "model": "claude-3-opus",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "confidence": 0.90,
            "raw_text": "Simple proposal without structured data",
        }

        with patch("sophia.api.app._hcg_client") as mock_hcg:
            mock_hcg.add_node = Mock()
            mock_hcg.add_edge = Mock()

            response = client.post("/ingest/hermes_proposal", json=proposal)

            assert response.status_code == 201
            data = response.json()

            # Should only have the proposal node itself
            assert len(data["stored_node_ids"]) == 1
            assert data["stored_node_ids"][0] == proposal["proposal_id"]

    def test_ingestion_preserves_provenance(self, client, sample_proposal):
        """Test that provenance metadata is preserved in Neo4j."""
        with patch("sophia.api.app._hcg_client") as mock_hcg:
            mock_hcg.add_node = Mock()
            mock_hcg.add_edge = Mock()

            response = client.post("/ingest/hermes_proposal", json=sample_proposal)

            assert response.status_code == 201

            # Check that proposal node has all provenance fields
            proposal_call = mock_hcg.add_node.call_args_list[0]
            properties = proposal_call[1]["properties"]

            assert "source_service" in properties
            assert "llm_provider" in properties
            assert "model" in properties
            assert "generated_at" in properties
            assert "confidence" in properties
            assert "ingested_at" in properties
