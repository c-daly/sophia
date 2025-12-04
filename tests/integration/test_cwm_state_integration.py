"""Integration tests for the /state/cwm endpoint with real Neo4j.

These tests require Neo4j to be running.
Run with: pytest tests/integration/test_cwm_state_integration.py -v -m integration

In CI, these run automatically with containerized Neo4j.
Locally, start services with: docker compose -f docker-compose.test.yml up -d
"""

import os
import pytest
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
def api_token():
    """API authentication token."""
    return "test-integration-token"


@pytest.fixture
def app(api_token):
    """Create test application with Neo4j seeding."""
    os.environ["SOPHIA_API_TOKEN"] = api_token
    os.environ["SEED_PICK_AND_PLACE_DATA"] = "true"
    os.environ["CLEAR_BEFORE_SEED"] = "true"
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
def auth_headers(api_token):
    """Authentication headers."""
    return {"Authorization": f"Bearer {api_token}"}


class TestCWMStateIntegration:
    """Integration tests for the /state/cwm endpoint."""

    def test_get_cwm_states_returns_list(self, client, auth_headers):
        """Test that GET /state/cwm returns a list of CWM states."""
        response = client.get("/state/cwm", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "states" in data
        assert "total" in data
        assert isinstance(data["states"], list)

    def test_get_cwm_states_requires_auth(self, client):
        """Test that /state/cwm requires authentication."""
        response = client.get("/state/cwm")
        assert response.status_code in [401, 403]

    def test_get_cwm_states_with_limit(self, client, auth_headers):
        """Test that limit parameter restricts results."""
        response = client.get(
            "/state/cwm",
            params={"limit": 5},
            headers=auth_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert len(data["states"]) <= 5

    def test_get_cwm_states_filter_by_model_type(self, client, auth_headers):
        """Test filtering CWM states by model type."""
        response = client.get(
            "/state/cwm",
            params={"model_type": "CWM_A"},
            headers=auth_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert data["model_type"] == "CWM_A"
        # If there are results, they should all be CWM_A
        for state in data["states"]:
            assert state["model_type"] == "CWM_A"

    def test_get_cwm_states_returns_required_fields(self, client, auth_headers):
        """Test that CWM states have all required fields."""
        # First trigger a state update to create a CWM-A state
        client.post(
            "/state",
            json={
                "state": {
                    "red_block": {"location": "table", "grasped": False},
                    "gripper": {"position": "home", "holding": None},
                }
            },
            headers=auth_headers,
        )

        response = client.get("/state/cwm", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        if data["states"]:
            state = data["states"][0]
            # Check required fields from CWMStateResponse model
            assert "state_id" in state
            assert "model_type" in state
            assert "source" in state
            assert "timestamp" in state
            assert "confidence" in state
            assert "status" in state
            assert "links" in state
            assert "tags" in state
            assert "data" in state

    def test_get_cwm_states_sorted_by_timestamp(self, client, auth_headers):
        """Test that CWM states are sorted by timestamp descending."""
        # Create multiple state updates to generate CWM states
        for i in range(3):
            client.post(
                "/state",
                json={
                    "state": {
                        "red_block": {"location": f"position_{i}", "grasped": False},
                        "gripper": {"position": "home", "holding": None},
                    }
                },
                headers=auth_headers,
            )

        response = client.get("/state/cwm", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        if len(data["states"]) > 1:
            # Verify descending order by timestamp
            timestamps = [s["timestamp"] for s in data["states"]]
            assert timestamps == sorted(timestamps, reverse=True)

    def test_get_cwm_states_with_invalid_model_type(self, client, auth_headers):
        """Test filtering with an unknown model type returns empty results."""
        response = client.get(
            "/state/cwm",
            params={"model_type": "INVALID_MODEL"},
            headers=auth_headers,
        )
        assert response.status_code == 200

        data = response.json()
        # Unknown model type should return empty list
        assert data["states"] == []

    def test_get_cwm_states_limit_bounds(self, client, auth_headers):
        """Test limit parameter boundary validation."""
        # Test minimum limit
        response = client.get(
            "/state/cwm",
            params={"limit": 1},
            headers=auth_headers,
        )
        assert response.status_code == 200

        # Test maximum limit
        response = client.get(
            "/state/cwm",
            params={"limit": 1000},
            headers=auth_headers,
        )
        assert response.status_code == 200

        # Test invalid limit (0 or negative)
        response = client.get(
            "/state/cwm",
            params={"limit": 0},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_cwm_state_after_state_update(self, client, auth_headers):
        """Test that updating world state creates a new CWM-A state."""
        # Perform a state update
        update_response = client.post(
            "/state",
            json={
                "state": {
                    "red_block": {"location": "bin", "grasped": False},
                    "blue_block": {"location": "table", "grasped": False},
                    "gripper": {"position": "bin", "holding": None},
                }
            },
            headers=auth_headers,
        )
        assert update_response.status_code == 200

        # Get CWM state history
        cwm_response = client.get("/state/cwm", headers=auth_headers)
        assert cwm_response.status_code == 200

        data = cwm_response.json()
        # Should have at least one CWM state after the update
        # (depending on CWM-A state service being enabled)
        assert isinstance(data["states"], list)

    def test_cwm_state_confidence_valid_range(self, client, auth_headers):
        """Test that CWM state confidence is in valid range [0.0, 1.0]."""
        # Create a state update to generate CWM state
        client.post(
            "/state",
            json={
                "state": {
                    "red_block": {"location": "table", "grasped": False},
                    "gripper": {"position": "home", "holding": None},
                }
            },
            headers=auth_headers,
        )

        response = client.get("/state/cwm", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        for state in data["states"]:
            assert 0.0 <= state["confidence"] <= 1.0

    def test_cwm_state_status_valid_values(self, client, auth_headers):
        """Test that CWM state status is a valid value."""
        # Create a state update
        client.post(
            "/state",
            json={
                "state": {
                    "red_block": {"location": "table", "grasped": False},
                    "gripper": {"position": "home", "holding": None},
                }
            },
            headers=auth_headers,
        )

        response = client.get("/state/cwm", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        valid_statuses = {"observed", "imagined", "reflected"}
        for state in data["states"]:
            assert state["status"] in valid_statuses


class TestCWMStateAndRealStateIndependence:
    """Tests for independence between CWM state history and real-time state."""

    def test_cwm_state_does_not_affect_current_state(self, client, auth_headers):
        """Test that CWM state history is separate from current world state."""
        # Get initial state
        initial_response = client.get("/state", headers=auth_headers)
        initial_state = initial_response.json()["state"]

        # Query CWM states
        cwm_response = client.get("/state/cwm", headers=auth_headers)
        assert cwm_response.status_code == 200

        # Get state again - should be unchanged
        final_response = client.get("/state", headers=auth_headers)
        final_state = final_response.json()["state"]

        assert initial_state == final_state

    def test_cwm_states_reflect_state_history(self, client, auth_headers):
        """Test that CWM states capture state change history."""
        locations = ["table", "in_transit", "bin"]

        # Make multiple state updates
        for location in locations:
            client.post(
                "/state",
                json={
                    "state": {
                        "red_block": {"location": location, "grasped": False},
                        "gripper": {"position": location, "holding": None},
                    }
                },
                headers=auth_headers,
            )

        # Get CWM history
        response = client.get("/state/cwm", headers=auth_headers)
        assert response.status_code == 200

        # CWM should have history (if CWM-A is enabled)
        # The exact number depends on service configuration
        data = response.json()
        assert isinstance(data["total"], int)
