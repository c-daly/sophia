"""Unit tests for HCG API endpoints.

These tests use mocks to verify endpoint behavior (auth, validation, error handling)
without requiring external services.
"""

import os
import pytest
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

from sophia.api.app import create_app

pytestmark = pytest.mark.unit


@pytest.fixture
def api_token():
    """Fixture for API token."""
    return "test-token-12345"


@pytest.fixture
def test_app(api_token):
    """Create test FastAPI application with mocked HCG client."""
    os.environ["SOPHIA_API_TOKEN"] = api_token
    with patch("sophia.api.app._hcg_client") as mock_hcg:
        mock_hcg.list_all_nodes = Mock(return_value=[])
        mock_hcg.get_node = Mock(return_value=None)
        mock_hcg.list_all_edges = Mock(return_value=[])
        yield create_app()


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


@pytest.fixture
def auth_headers(api_token):
    """Create authentication headers."""
    return {"Authorization": f"Bearer {api_token}"}


class TestHCGSnapshotEndpoint:
    """Tests for GET /hcg/snapshot endpoint."""

    @patch("sophia.api.app._hcg_client")
    def test_snapshot_requires_authentication(self, mock_hcg, client):
        """Test that /hcg/snapshot requires authentication."""
        response = client.get("/hcg/snapshot")
        assert response.status_code == 403

    @patch("sophia.api.app._hcg_client")
    def test_snapshot_rejects_invalid_token(self, mock_hcg, client):
        """Test that /hcg/snapshot rejects invalid tokens."""
        headers = {"Authorization": "Bearer invalid-token"}
        response = client.get("/hcg/snapshot", headers=headers)
        assert response.status_code == 403

    @patch("sophia.api.app._hcg_client")
    def test_snapshot_returns_nodes_and_edges(self, mock_hcg, client, auth_headers):
        """Test that /hcg/snapshot returns proper structure."""
        mock_hcg.list_all_nodes.return_value = [
            {"uuid": "node1", "type": "state", "name": "test_node", "properties": {}}
        ]
        mock_hcg.list_all_edges.return_value = [
            {"id": "edge1", "source": "node1", "target": "node2", "relation": "CAUSES"}
        ]

        response = client.get("/hcg/snapshot", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "entities" in data
        assert "edges" in data

    @patch("sophia.api.app._hcg_client")
    def test_snapshot_embeddings_batched_and_routed_by_collection(
        self, mock_hcg, client, auth_headers
    ):
        """include_embeddings looks each node up in the collection that holds its
        vector (Entity vs Concept), and batches the `uuid in [...]` filter so the
        expression stays under Milvus's length cap (greptile #158 + collection
        sharding)."""
        import re

        # 3 entity nodes + 1 concept node. With the batch size forced to 2, the
        # entity collection must be queried in 2 chunks; concept in 1.
        mock_hcg.list_all_nodes.return_value = [
            {"uuid": "e1", "type": "entity", "name": "e1", "properties": {}},
            {"uuid": "e2", "type": "entity", "name": "e2", "properties": {}},
            {"uuid": "e3", "type": "entity", "name": "e3", "properties": {}},
            {"uuid": "c1", "type": "concept", "name": "c1", "properties": {}},
        ]
        mock_hcg.list_all_edges.return_value = []

        store = {
            "hcg_entity_embeddings": {
                "e1": [0.1, 0.2],
                "e2": [0.3, 0.4],
                "e3": [0.5, 0.6],
            },
            "hcg_concept_embeddings": {"c1": [0.7, 0.8]},
        }

        class FakeCollection:
            instances: list = []

            def __init__(self, name):
                self.name = name
                self.queries = []
                FakeCollection.instances.append(self)

            def load(self):
                pass

            def query(self, expr, output_fields, limit):
                uuids = re.findall(r'"([^"]+)"', expr)
                assert len(uuids) <= 2  # batched: never the whole list at once
                assert limit == len(uuids)
                self.queries.append(uuids)
                col = store.get(self.name, {})
                return [{"uuid": u, "embedding": col[u]} for u in uuids if u in col]

        with (
            patch("pymilvus.Collection", FakeCollection),
            patch("sophia.api.app._SNAPSHOT_EMB_BATCH", 2),
        ):
            response = client.get(
                "/hcg/snapshot",
                params={"include_embeddings": "true"},
                headers=auth_headers,
            )

        assert response.status_code == 200
        ents = {e["id"]: e["embedding"] for e in response.json()["entities"]}
        # Every node got its vector -- including the concept node, whose vector
        # lives in a different collection than hcg_entity_embeddings.
        assert ents == {
            "e1": [0.1, 0.2],
            "e2": [0.3, 0.4],
            "e3": [0.5, 0.6],
            "c1": [0.7, 0.8],
        }
        by_name = {c.name: c for c in FakeCollection.instances}
        # Entity collection queried in 2 batches ([e1,e2], [e3]); concept once.
        assert by_name["hcg_entity_embeddings"].queries == [["e1", "e2"], ["e3"]]
        assert by_name["hcg_concept_embeddings"].queries == [["c1"]]


class TestHCGEntitiesEndpoint:
    """Tests for GET /hcg/entities endpoint."""

    @patch("sophia.api.app._hcg_client")
    def test_entities_requires_authentication(self, mock_hcg, client):
        """Test that /hcg/entities requires authentication."""
        response = client.get("/hcg/entities")
        assert response.status_code == 403

    @patch("sophia.api.app._hcg_client")
    def test_entities_returns_list(self, mock_hcg, client, auth_headers):
        """Test that /hcg/entities returns a list."""
        mock_hcg.list_all_nodes.return_value = [
            {"uuid": "node1", "type": "state", "properties": {}},
            {"uuid": "node2", "type": "process", "properties": {}},
        ]

        response = client.get("/hcg/entities", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2

    @patch("sophia.api.app._hcg_client")
    def test_entities_filters_by_type(self, mock_hcg, client, auth_headers):
        """Test that /hcg/entities filters by type."""
        mock_hcg.list_all_nodes.return_value = []

        response = client.get(
            "/hcg/entities",
            params={"type": "state"},
            headers=auth_headers,
        )
        assert response.status_code == 200

        # Verify the filter was passed
        mock_hcg.list_all_nodes.assert_called()


class TestHCGEntityByIdEndpoint:
    """Tests for GET /hcg/entities/{entity_id} endpoint."""

    @patch("sophia.api.app._hcg_client")
    def test_entity_by_id_requires_authentication(self, mock_hcg, client):
        """Test that /hcg/entities/{id} requires authentication."""
        response = client.get("/hcg/entities/some-id")
        assert response.status_code == 403

    @patch("sophia.api.app._hcg_client")
    def test_entity_by_id_returns_404_when_not_found(
        self, mock_hcg, client, auth_headers
    ):
        """Test that /hcg/entities/{id} returns 404 for nonexistent entity."""
        mock_hcg.get_node.return_value = None

        response = client.get("/hcg/entities/nonexistent", headers=auth_headers)
        assert response.status_code == 404

    @patch("sophia.api.app._hcg_client")
    def test_entity_by_id_returns_entity(self, mock_hcg, client, auth_headers):
        """Test that /hcg/entities/{id} returns the entity."""
        mock_hcg.get_node.return_value = {
            "uuid": "entity123",
            "type": "state",
            "name": "test_entity",
            "properties": {},
        }

        response = client.get("/hcg/entities/entity123", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == "entity123"


class TestHCGEdgesEndpoint:
    """Tests for GET /hcg/edges endpoint."""

    @patch("sophia.api.app._hcg_client")
    def test_edges_requires_authentication(self, mock_hcg, client):
        """Test that /hcg/edges requires authentication."""
        response = client.get("/hcg/edges")
        assert response.status_code == 403

    @patch("sophia.api.app._hcg_client")
    def test_edges_returns_list(self, mock_hcg, client, auth_headers):
        """Test that /hcg/edges returns a list."""
        mock_hcg.list_all_edges.return_value = [
            {"id": "edge1", "source": "node1", "target": "node2", "relation": "CAUSES"}
        ]

        response = client.get("/hcg/edges", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)


class TestHCGStatesEndpoint:
    """Tests for GET /hcg/states endpoint."""

    @patch("sophia.api.app._hcg_client")
    def test_states_requires_authentication(self, mock_hcg, client):
        """Test that /hcg/states requires authentication."""
        response = client.get("/hcg/states")
        assert response.status_code == 403

    @patch("sophia.api.app._hcg_client")
    def test_states_returns_list(self, mock_hcg, client, auth_headers):
        """Test that /hcg/states returns filtered list."""
        mock_hcg.list_all_nodes.return_value = [
            {"uuid": "state1", "type": "state", "properties": {}}
        ]

        response = client.get("/hcg/states", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)


class TestHCGProcessesEndpoint:
    """Tests for GET /hcg/processes endpoint."""

    @patch("sophia.api.app._hcg_client")
    def test_processes_requires_authentication(self, mock_hcg, client):
        """Test that /hcg/processes requires authentication."""
        response = client.get("/hcg/processes")
        assert response.status_code == 403

    @patch("sophia.api.app._hcg_client")
    def test_processes_returns_list(self, mock_hcg, client, auth_headers):
        """Test that /hcg/processes returns filtered list."""
        mock_hcg.list_all_nodes.return_value = []

        response = client.get("/hcg/processes", headers=auth_headers)
        assert response.status_code == 200


class TestHCGPlansEndpoint:
    """Tests for GET /hcg/plans endpoint."""

    @patch("sophia.api.app._hcg_client")
    def test_plans_requires_authentication(self, mock_hcg, client):
        """Test that /hcg/plans requires authentication."""
        response = client.get("/hcg/plans")
        assert response.status_code == 403

    @patch("sophia.api.app._hcg_client")
    def test_plans_returns_list(self, mock_hcg, client, auth_headers):
        """Test that /hcg/plans returns filtered list."""
        mock_hcg.list_all_nodes.return_value = []

        response = client.get("/hcg/plans", headers=auth_headers)
        assert response.status_code == 200


class TestHCGHistoryEndpoint:
    """Tests for GET /hcg/history endpoint."""

    @patch("sophia.api.app._hcg_client")
    def test_history_requires_authentication(self, mock_hcg, client):
        """Test that /hcg/history requires authentication."""
        response = client.get("/hcg/history")
        assert response.status_code == 403

    @patch("sophia.api.app._hcg_client")
    def test_history_returns_list(self, mock_hcg, client, auth_headers):
        """Test that /hcg/history returns list."""
        mock_hcg.list_all_nodes.return_value = []

        response = client.get("/hcg/history", headers=auth_headers)
        assert response.status_code == 200


class TestHCGHealthEndpoint:
    """Tests for GET /hcg/health endpoint."""

    @patch("sophia.api.app._hcg_client")
    def test_health_requires_authentication(self, mock_hcg, client):
        """Test that /hcg/health requires authentication."""
        response = client.get("/hcg/health")
        assert response.status_code == 403

    @patch("sophia.api.app._hcg_client")
    def test_health_returns_connected_status(self, mock_hcg, client, auth_headers):
        """Test that /hcg/health returns connection status."""
        mock_hcg.list_all_nodes.return_value = []  # Successful call = connected

        response = client.get("/hcg/health", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "neo4j_connected" in data
        assert data["neo4j_connected"] is True

    @patch("sophia.api.app._hcg_client")
    def test_health_returns_disconnected_status(self, mock_hcg, client, auth_headers):
        """Test that /hcg/health returns disconnected when Neo4j is down."""
        mock_hcg.list_all_nodes.side_effect = Exception("Connection failed")

        response = client.get("/hcg/health", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert data["neo4j_connected"] is False
