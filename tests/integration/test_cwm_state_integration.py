"""Integration tests for CWM state endpoints.

These tests require Sophia and Neo4j to be running.
Run with: pytest tests/integration/test_cwm_state_integration.py -v -m integration

Start services with: ./scripts/test_integration.sh up
"""

import pytest

pytestmark = [
    pytest.mark.integration,
]


class TestCWMStateIntegration:
    """Integration tests for CWM state endpoints."""

    def test_get_cwm_state_returns_structure(self, http_client, auth_headers):
        """Test that /state/cwm returns proper structure."""
        response = http_client.get("/state/cwm", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "states" in data or "cwm_states" in data

    def test_cwm_state_after_state_update(self, http_client, auth_headers):
        """Test CWM state after updating world state."""
        # Update state
        update_response = http_client.post(
            "/state",
            json={
                "state": {
                    "red_block": {"location": "bin", "grasped": False},
                }
            },
            headers=auth_headers,
        )
        # May succeed or fail depending on whether node exists
        assert update_response.status_code in [200, 201, 404]

        # Get CWM state
        response = http_client.get("/state/cwm", headers=auth_headers)
        assert response.status_code == 200

    def test_cwm_state_list_pagination(self, http_client, auth_headers):
        """Test CWM state list supports pagination."""
        response = http_client.get(
            "/state/cwm",
            params={"limit": 10, "offset": 0},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_cwm_state_requires_auth(self, http_client):
        """Test that /state/cwm requires authentication."""
        response = http_client.get("/state/cwm")
        assert response.status_code in [401, 403]


class TestCWMStateAndRealStateIndependence:
    """Test that CWM state is independent from actual world state."""

    def test_cwm_state_does_not_affect_current_state(self, http_client, auth_headers):
        """Test that CWM state operations don't affect current state."""
        # Get initial current state
        initial_response = http_client.get("/state", headers=auth_headers)
        initial_state = initial_response.json()

        # Get CWM state (should not modify current state)
        cwm_response = http_client.get("/state/cwm", headers=auth_headers)
        assert cwm_response.status_code == 200

        # Verify current state unchanged
        final_response = http_client.get("/state", headers=auth_headers)
        final_state = final_response.json()

        # States should be equal (ignoring timestamp differences)
        assert initial_state.get("nodes") == final_state.get("nodes")
