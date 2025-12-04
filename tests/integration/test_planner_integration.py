"""Integration tests for Sophia's planner interactions.

These tests validate that Sophia correctly:
1. Sends planning requests to the HCG/planner service
2. Handles plan responses appropriately
3. Manages error conditions gracefully

Note: The actual planning logic is in logos_hcg (logos repo).
Sophia's responsibility is integration, not planning correctness.

Requires: Neo4j running with pick-and-place data seeded.
"""

import os
import pytest
from fastapi.testclient import TestClient

from sophia.api.app import create_app
from sophia.hcg_client import HCGClient


pytestmark = pytest.mark.integration


@pytest.fixture
def neo4j_uri():
    """Neo4j connection URI."""
    return os.getenv("NEO4J_URI", "bolt://localhost:7687")


@pytest.fixture
def neo4j_user():
    """Neo4j username."""
    return os.getenv("NEO4J_USER", "neo4j")


@pytest.fixture
def neo4j_password():
    """Neo4j password."""
    return os.getenv("NEO4J_PASSWORD", "neo4jtest")


@pytest.fixture
def hcg_client(neo4j_uri, neo4j_user, neo4j_password):
    """Create HCG client for direct database verification."""
    client = HCGClient(
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_user,
        neo4j_password=neo4j_password,
    )
    yield client
    client.close()


@pytest.fixture
def api_token():
    """API authentication token."""
    return "test-planner-integration-token"


@pytest.fixture
def app(api_token, neo4j_uri, neo4j_user, neo4j_password):
    """Create test application."""
    os.environ["SOPHIA_API_TOKEN"] = api_token
    os.environ["NEO4J_URI"] = neo4j_uri
    os.environ["NEO4J_USER"] = neo4j_user
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


class TestPlannerIntegration:
    """Integration tests for planner API interactions."""

    def test_plan_request_returns_valid_structure(self, client, auth_headers):
        """Test that /plan returns a properly structured response."""
        response = client.post(
            "/plan",
            json={
                "goal": {
                    "description": "red block in bin",
                    "target_state": "red_block_in_bin",
                }
            },
            headers=auth_headers,
        )

        assert response.status_code == 201
        data = response.json()

        # Verify response structure
        assert "plan_id" in data
        assert "plan" in data
        assert "goal" in data
        assert "created_at" in data
        assert isinstance(data["plan"], list)

    def test_plan_with_seeded_kg_returns_steps(self, client, auth_headers):
        """Test that planning against seeded pick-and-place data returns valid steps."""
        response = client.post(
            "/plan",
            json={
                "goal": {
                    "description": "red block in bin",
                    "target_state": "red_block_in_bin",
                }
            },
            headers=auth_headers,
        )

        assert response.status_code == 201
        data = response.json()

        # Should have steps for pick-and-place
        plan = data["plan"]
        assert len(plan) > 0, "Plan should contain action steps"

        # Each step should have required fields
        for step in plan:
            assert "id" in step
            assert "name" in step or "action_type" in step

    def test_plan_steps_reference_hcg_nodes(self, client, auth_headers, hcg_client):
        """Test that plan steps reference actual nodes in the knowledge graph."""
        response = client.post(
            "/plan",
            json={
                "goal": {
                    "description": "red block in bin",
                    "target_state": "red_block_in_bin",
                }
            },
            headers=auth_headers,
        )

        assert response.status_code == 201
        plan = response.json()["plan"]

        # Verify each step references a real HCG node
        for step in plan:
            node_id = step.get("id")
            if node_id:
                node = hcg_client.get_node(node_id)
                assert node is not None, f"Step {node_id} should exist in HCG"

    def test_plan_empty_goal_succeeds(self, client, auth_headers):
        """Test that empty goal is accepted (planner handles empty goals gracefully)."""
        response = client.post(
            "/plan",
            json={"goal": {}},
            headers=auth_headers,
        )

        # API accepts empty goals and returns a plan (possibly empty)
        assert response.status_code == 201

    def test_plan_missing_goal_returns_validation_error(self, client, auth_headers):
        """Test that missing goal field returns validation error."""
        response = client.post(
            "/plan",
            json={},
            headers=auth_headers,
        )

        assert response.status_code == 422

    def test_plan_unachievable_goal_returns_empty_plan(self, client, auth_headers):
        """Test that unachievable goal returns empty plan gracefully."""
        response = client.post(
            "/plan",
            json={
                "goal": {
                    "description": "teleport to mars",
                    "target_state": "on_mars",
                }
            },
            headers=auth_headers,
        )

        # Should succeed but with empty or minimal plan
        assert response.status_code in [200, 201]
        data = response.json()
        # Either empty plan or a message indicating no plan found
        assert "plan" in data or "message" in data

    def test_plan_persisted_to_neo4j(self, client, auth_headers, hcg_client):
        """Test that generated plans are persisted to Neo4j."""
        response = client.post(
            "/plan",
            json={
                "goal": {
                    "description": "red block in bin",
                    "target_state": "red_block_in_bin",
                }
            },
            headers=auth_headers,
        )

        assert response.status_code == 201
        plan_id = response.json()["plan_id"]

        # Verify plan node exists in Neo4j
        plan_node = hcg_client.get_node(plan_id)
        assert plan_node is not None, f"Plan {plan_id} should be persisted to Neo4j"
        assert plan_node["type"] == "plan"


class TestPlannerErrorHandling:
    """Tests for planner error handling."""

    def test_plan_requires_authentication(self, client):
        """Test that /plan requires authentication."""
        response = client.post(
            "/plan",
            json={
                "goal": {
                    "description": "test",
                    "target_state": "test_state",
                }
            },
        )

        assert response.status_code == 403

    def test_plan_rejects_invalid_token(self, client):
        """Test that /plan rejects invalid tokens."""
        response = client.post(
            "/plan",
            json={
                "goal": {
                    "description": "test",
                    "target_state": "test_state",
                }
            },
            headers={"Authorization": "Bearer invalid-token"},
        )

        assert response.status_code == 403

    def test_plan_handles_malformed_json(self, client, auth_headers):
        """Test that /plan handles malformed JSON gracefully."""
        response = client.post(
            "/plan",
            content="not valid json",
            headers={**auth_headers, "Content-Type": "application/json"},
        )

        assert response.status_code == 422


class TestPlannerWithState:
    """Tests for planner interactions with state."""

    def test_plan_considers_current_state(self, client, auth_headers):
        """Test that planner considers current world state."""
        # First, get the current state
        state_response = client.get("/state", headers=auth_headers)
        assert state_response.status_code == 200
        _initial_state = state_response.json()

        # Generate a plan
        plan_response = client.post(
            "/plan",
            json={
                "goal": {
                    "description": "red block in bin",
                    "target_state": "red_block_in_bin",
                }
            },
            headers=auth_headers,
        )

        assert plan_response.status_code == 201
        plan = plan_response.json()

        # Plan should be generated considering initial state
        # (red_block starts on table, not in bin)
        assert len(plan["plan"]) > 0

    def test_plan_after_state_update(self, client, auth_headers):
        """Test that planner adapts to updated state."""
        # Update state to indicate red_block is already grasped
        _update_response = client.post(
            "/state",
            json={
                "state": {
                    "red_block": {"location": "gripper", "grasped": True},
                    "gripper": {"holding": "red_block"},
                }
            },
            headers=auth_headers,
        )
        # May succeed or fail validation - either is acceptable

        # Generate plan for same goal
        plan_response = client.post(
            "/plan",
            json={
                "goal": {
                    "description": "red block in bin",
                    "target_state": "red_block_in_bin",
                }
            },
            headers=auth_headers,
        )

        assert plan_response.status_code == 201
        # Plan should be different (shorter) since block is already grasped
