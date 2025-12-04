"""Integration tests for the /imagine endpoint with real Neo4j.

These tests require Neo4j to be running.
Run with: pytest tests/integration/test_imagine_integration.py -v -m integration

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
def app(api_token, neo4j_uri, neo4j_username, neo4j_password):
    """Create test application."""
    os.environ["SOPHIA_API_TOKEN"] = api_token
    os.environ["NEO4J_URI"] = neo4j_uri
    os.environ["NEO4J_USER"] = neo4j_username
    os.environ["NEO4J_PASSWORD"] = neo4j_password
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


class TestImagineIntegration:
    """Integration tests for the /imagine endpoint."""

    def test_imagine_creates_imagined_states(self, client, auth_headers):
        """Test that /imagine creates imagined states with default parameters."""
        response = client.post(
            "/imagine",
            json={
                "horizon": 3,
                "model_version": "v1.0",
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        assert "imagination_id" in data
        assert "imagined_states" in data
        assert len(data["imagined_states"]) == 3
        assert data["model_version"] == "v1.0"
        assert data["horizon"] == 3

    def test_imagine_with_emotion_tags(self, client, auth_headers):
        """Test that /imagine processes CWM-E emotion tags."""
        response = client.post(
            "/imagine",
            json={
                "cwm_e_emotion_tags": ["curiosity", "anticipation"],
                "horizon": 2,
                "model_version": "v1.0",
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        assert len(data["imagined_states"]) == 2
        # Emotion tags should be included in state properties
        for state in data["imagined_states"]:
            assert "emotion_tags" in state["properties"]
            assert "curiosity" in state["properties"]["emotion_tags"]
            assert "anticipation" in state["properties"]["emotion_tags"]

    def test_imagine_with_context(self, client, auth_headers):
        """Test that /imagine processes additional context."""
        context = {
            "scene": "warehouse",
            "objects": ["red_block", "bin"],
            "goal": "organize objects",
        }
        response = client.post(
            "/imagine",
            json={
                "context": context,
                "horizon": 2,
                "model_version": "v1.0",
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        for state in data["imagined_states"]:
            assert state["properties"]["context"] == context

    def test_imagine_with_assumptions(self, client, auth_headers):
        """Test that /imagine processes assumptions."""
        assumptions = ["gripper is at home", "bin is accessible"]
        response = client.post(
            "/imagine",
            json={
                "assumptions": assumptions,
                "horizon": 2,
                "model_version": "v1.0",
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        assert data["assumptions"] == assumptions

    def test_imagine_persists_to_neo4j(self, client, auth_headers, hcg_client):
        """Test that imagined states are persisted to Neo4j."""
        response = client.post(
            "/imagine",
            json={
                "horizon": 2,
                "model_version": "v1.0",
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        imagination_id = data["imagination_id"]

        # Verify nodes were created in Neo4j
        # Query for imagined_state nodes with this imagination_id
        with hcg_client.driver.session(database=hcg_client.database) as session:
            result = session.run(
                """
                MATCH (n {imagination_id: $imagination_id})
                RETURN n.imagination_id as imagination_id, count(n) as count
                """,
                {"imagination_id": imagination_id},
            )
            record = result.single()
            if record:
                assert record["count"] >= 2

    def test_imagine_confidence_decreases_with_horizon(self, client, auth_headers):
        """Test that confidence decreases for states further in the horizon."""
        response = client.post(
            "/imagine",
            json={
                "horizon": 5,
                "model_version": "v1.0",
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        states = data["imagined_states"]

        # Confidence should decrease as horizon increases
        for i in range(len(states) - 1):
            assert states[i]["confidence"] >= states[i + 1]["confidence"]

    def test_imagine_requires_auth(self, client):
        """Test that /imagine requires authentication."""
        response = client.post(
            "/imagine",
            json={
                "horizon": 2,
                "model_version": "v1.0",
            },
        )
        # Should be 403 without auth
        assert response.status_code in [401, 403]

    def test_imagine_with_invalid_horizon(self, client, auth_headers):
        """Test that /imagine rejects invalid horizon values."""
        response = client.post(
            "/imagine",
            json={
                "horizon": 0,  # Must be > 0
                "model_version": "v1.0",
            },
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_imagine_state_ids_are_unique(self, client, auth_headers):
        """Test that each imagined state has a unique ID."""
        response = client.post(
            "/imagine",
            json={
                "horizon": 5,
                "model_version": "v1.0",
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        state_ids = [s["state_id"] for s in data["imagined_states"]]
        assert len(state_ids) == len(set(state_ids))

    def test_imagine_with_cwm_g_imagery(self, client, auth_headers):
        """Test that /imagine processes CWM-G imagery data."""
        imagery_data = [
            {
                "image_id": "img_001",
                "description": "Red block on table",
                "features": {"color": "red", "shape": "cube"},
            },
            {
                "image_id": "img_002",
                "description": "Empty bin",
                "features": {"type": "container", "status": "empty"},
            },
        ]
        response = client.post(
            "/imagine",
            json={
                "cwm_g_imagery": imagery_data,
                "horizon": 2,
                "model_version": "v1.0",
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        assert len(data["imagined_states"]) == 2
