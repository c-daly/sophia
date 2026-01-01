"""Unit tests for Hermes proposal ingestion endpoint.

These tests verify that the endpoint correctly receives, validates, and logs
proposals from Hermes. Proposals are not stored as nodes - Sophia will process
them cognitively and create semantic nodes based on her evaluation.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
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
            {"action": "move_to_red_block", "target": "red_block", "parameters": {}},
            {
                "action": "grasp_red_block",
                "target": "red_block",
                "parameters": {"force": 0.5},
            },
            {"action": "move_to_bin", "target": "bin", "parameters": {}},
        ],
        "imagined_states": [
            {"state_id": "state_1", "entities": {"red_block": {"location": "table"}}},
            {"state_id": "state_2", "entities": {"red_block": {"location": "bin"}}},
        ],
        "diagnostics": {
            "reasoning": "Block needs to be moved from table to bin",
            "steps_count": 3,
        },
        "metadata": {
            "session_id": "test_session_123",
            "user_id": "test_user",
        },
    }


class TestHermesIngestionEndpoint:
    """Tests for the /ingest/hermes_proposal endpoint."""

    def test_ingestion_accepts_valid_proposal(self, client, sample_proposal):
        """Test that valid proposals are accepted."""
        response = client.post("/ingest/hermes_proposal", json=sample_proposal)

        assert response.status_code == 201
        data = response.json()
        assert data["proposal_id"] == sample_proposal["proposal_id"]
        assert data["status"] == "accepted"
        # No nodes created - proposals are logged, not stored
        assert data["stored_node_ids"] == []

    def test_ingestion_logs_proposal(self, client, sample_proposal):
        """Test that proposals are logged for observability."""
        with patch("sophia.api.app.logger") as mock_logger:
            response = client.post("/ingest/hermes_proposal", json=sample_proposal)

            assert response.status_code == 201
            # Verify logging was called
            mock_logger.info.assert_called()
            log_message = mock_logger.info.call_args[0][0]
            assert sample_proposal["proposal_id"] in log_message
            assert sample_proposal["llm_provider"] in log_message

    def test_ingestion_minimal_payload(self, client):
        """Test ingestion with minimal required fields."""
        minimal_proposal = {
            "proposal_id": "hermes_minimal_001",
            "llm_provider": "openai",
            "model": "gpt-4",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "confidence": 0.75,
        }

        response = client.post("/ingest/hermes_proposal", json=minimal_proposal)

        assert response.status_code == 201
        data = response.json()
        assert data["proposal_id"] == minimal_proposal["proposal_id"]
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

    def test_ingestion_no_auth_required(self, client, sample_proposal):
        """Test that ingestion endpoint doesn't require authentication (local dev)."""
        response = client.post("/ingest/hermes_proposal", json=sample_proposal)
        # Should not return 401 or 403
        assert response.status_code not in (401, 403)
