"""Integration tests for HCG API endpoints.

These tests require Sophia and Neo4j to be running.
Run with: pytest tests/integration/test_hcg_api_integration.py -v -m integration

Start services with: ./scripts/test_integration.sh up
"""

import pytest

pytestmark = [
    pytest.mark.integration,
]


class TestHCGSnapshotEndpoint:
    """Integration tests for GET /hcg/snapshot endpoint."""

    def test_snapshot_returns_graph_structure(self, http_client, auth_headers):
        """Test that /hcg/snapshot returns entities and edges."""
        response = http_client.get("/hcg/snapshot", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "entities" in data
        assert "edges" in data
        assert isinstance(data["entities"], list)
        assert isinstance(data["edges"], list)

    def test_snapshot_supports_limit(self, http_client, auth_headers):
        """Test that /hcg/snapshot supports limit parameter."""
        response = http_client.get(
            "/hcg/snapshot",
            params={"limit": 5},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["entities"]) <= 5

    def test_snapshot_requires_auth(self, http_client):
        """Test that /hcg/snapshot requires authentication."""
        response = http_client.get("/hcg/snapshot")
        assert response.status_code in [401, 403]


class TestHCGEntitiesEndpoint:
    """Integration tests for GET /hcg/entities endpoint."""

    def test_entities_returns_list(self, http_client, auth_headers):
        """Test that /hcg/entities returns a list of entities."""
        response = http_client.get("/hcg/entities", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

    def test_entities_supports_type_filter(self, http_client, auth_headers):
        """Test that /hcg/entities supports type filter."""
        response = http_client.get(
            "/hcg/entities",
            params={"type": "state"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_entities_supports_pagination(self, http_client, auth_headers):
        """Test that /hcg/entities supports limit and offset."""
        response = http_client.get(
            "/hcg/entities",
            params={"limit": 10, "offset": 0},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_entities_requires_auth(self, http_client):
        """Test that /hcg/entities requires authentication."""
        response = http_client.get("/hcg/entities")
        assert response.status_code in [401, 403]


class TestHCGEntityByIdEndpoint:
    """Integration tests for GET /hcg/entities/{entity_id} endpoint."""

    def test_entity_by_id_returns_404_for_nonexistent(self, http_client, auth_headers):
        """Test that /hcg/entities/{id} returns 404 for nonexistent entity."""
        response = http_client.get(
            "/hcg/entities/nonexistent-entity-12345",
            headers=auth_headers,
        )
        assert response.status_code == 404

    def test_entity_by_id_requires_auth(self, http_client):
        """Test that /hcg/entities/{id} requires authentication."""
        response = http_client.get("/hcg/entities/some-id")
        assert response.status_code in [401, 403]


class TestHCGEdgesEndpoint:
    """Integration tests for GET /hcg/edges endpoint."""

    def test_edges_returns_list(self, http_client, auth_headers):
        """Test that /hcg/edges returns a list of edges."""
        response = http_client.get("/hcg/edges", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

    def test_edges_supports_type_filter(self, http_client, auth_headers):
        """Test that /hcg/edges supports edge_type filter."""
        response = http_client.get(
            "/hcg/edges",
            params={"edge_type": "CAUSES"},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_edges_requires_auth(self, http_client):
        """Test that /hcg/edges requires authentication."""
        response = http_client.get("/hcg/edges")
        assert response.status_code in [401, 403]


class TestHCGStatesEndpoint:
    """Integration tests for GET /hcg/states endpoint."""

    def test_states_returns_list(self, http_client, auth_headers):
        """Test that /hcg/states returns a list of state entities."""
        response = http_client.get("/hcg/states", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

    def test_states_supports_pagination(self, http_client, auth_headers):
        """Test that /hcg/states supports limit and offset."""
        response = http_client.get(
            "/hcg/states",
            params={"limit": 5, "offset": 0},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_states_requires_auth(self, http_client):
        """Test that /hcg/states requires authentication."""
        response = http_client.get("/hcg/states")
        assert response.status_code in [401, 403]


class TestHCGProcessesEndpoint:
    """Integration tests for GET /hcg/processes endpoint."""

    def test_processes_returns_list(self, http_client, auth_headers):
        """Test that /hcg/processes returns a list of process entities."""
        response = http_client.get("/hcg/processes", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

    def test_processes_supports_status_filter(self, http_client, auth_headers):
        """Test that /hcg/processes supports status filter."""
        response = http_client.get(
            "/hcg/processes",
            params={"status": "active"},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_processes_requires_auth(self, http_client):
        """Test that /hcg/processes requires authentication."""
        response = http_client.get("/hcg/processes")
        assert response.status_code in [401, 403]


class TestHCGPlansEndpoint:
    """Integration tests for GET /hcg/plans endpoint."""

    def test_plans_returns_list(self, http_client, auth_headers):
        """Test that /hcg/plans returns a list of plan entities."""
        response = http_client.get("/hcg/plans", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

    def test_plans_supports_goal_filter(self, http_client, auth_headers):
        """Test that /hcg/plans supports goal_id filter."""
        response = http_client.get(
            "/hcg/plans",
            params={"goal_id": "some-goal-id"},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_plans_requires_auth(self, http_client):
        """Test that /hcg/plans requires authentication."""
        response = http_client.get("/hcg/plans")
        assert response.status_code in [401, 403]


class TestHCGHistoryEndpoint:
    """Integration tests for GET /hcg/history endpoint."""

    def test_history_returns_list(self, http_client, auth_headers):
        """Test that /hcg/history returns a list of history entries."""
        response = http_client.get("/hcg/history", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

    def test_history_supports_state_filter(self, http_client, auth_headers):
        """Test that /hcg/history supports state_id filter."""
        response = http_client.get(
            "/hcg/history",
            params={"state_id": "some-state-id"},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_history_requires_auth(self, http_client):
        """Test that /hcg/history requires authentication."""
        response = http_client.get("/hcg/history")
        assert response.status_code in [401, 403]


class TestHCGHealthEndpoint:
    """Integration tests for GET /hcg/health endpoint."""

    def test_health_returns_status(self, http_client, auth_headers):
        """Test that /hcg/health returns connection status."""
        response = http_client.get("/hcg/health", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "neo4j_connected" in data
        assert isinstance(data["neo4j_connected"], bool)

    def test_health_requires_auth(self, http_client):
        """Test that /hcg/health requires authentication."""
        response = http_client.get("/hcg/health")
        assert response.status_code in [401, 403]
