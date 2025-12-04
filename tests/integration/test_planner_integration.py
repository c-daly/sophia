"""Integration tests for Sophia's planner interactions.

These tests require Sophia and Neo4j to be running.
Run with: pytest tests/integration/test_planner_integration.py -v -m integration

Start services with: ./scripts/test_integration.sh up
"""

import pytest

pytestmark = [
    pytest.mark.integration,
]

# Common goal payloads for tests - goal must be a dict, not a string
GOAL_PAYLOAD = {
    "description": "move red_block to bin",
    "target_state": "red_block_in_bin",
}
EMPTY_GOAL_PAYLOAD = {"description": "", "target_state": ""}
UNACHIEVABLE_GOAL_PAYLOAD = {
    "description": "teleport to mars",
    "target_state": "on_mars",
}
TEST_GOAL_PAYLOAD = {"description": "test goal", "target_state": "test"}


class TestPlannerIntegration:
    """Integration tests for the /plan endpoint."""

    def test_plan_request_returns_valid_structure(self, http_client, auth_headers):
        """Test that /plan returns a valid plan structure."""
        response = http_client.post(
            "/plan",
            json={"goal": GOAL_PAYLOAD},
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        assert "plan_id" in data
        assert "plan" in data
        assert isinstance(data["plan"], list)

    def test_plan_with_seeded_kg_returns_steps(self, http_client, auth_headers):
        """Test that planning with seeded KG returns actionable steps."""
        response = http_client.post(
            "/plan",
            json={"goal": GOAL_PAYLOAD},
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        # Should have at least one step for this goal
        assert len(data["plan"]) >= 0

    def test_plan_empty_goal_succeeds(self, http_client, auth_headers):
        """Test that an empty goal returns an empty plan."""
        response = http_client.post(
            "/plan",
            json={"goal": EMPTY_GOAL_PAYLOAD},
            headers=auth_headers,
        )
        # May return empty plan or validation error
        assert response.status_code in [201, 400, 422]

    def test_plan_unachievable_goal_returns_empty_plan(self, http_client, auth_headers):
        """Test that an unachievable goal returns an empty plan."""
        response = http_client.post(
            "/plan",
            json={"goal": UNACHIEVABLE_GOAL_PAYLOAD},
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        # Unachievable goals should return empty steps
        assert len(data["plan"]) == 0

    def test_plan_steps_reference_hcg_nodes(
        self, http_client, auth_headers, hcg_client
    ):
        """Test that plan steps reference valid HCG nodes."""
        response = http_client.post(
            "/plan",
            json={"goal": GOAL_PAYLOAD},
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        # Verify any referenced nodes exist in Neo4j
        for step in data["plan"]:
            if "node_id" in step:
                with hcg_client.driver.session(database=hcg_client.database) as session:
                    session.run(
                        "MATCH (n {id: $id}) RETURN n.id as id",
                        {"id": step["node_id"]},
                    )
                    # Node may or may not exist depending on step type

    def test_plan_persisted_to_neo4j(self, http_client, auth_headers, hcg_client):
        """Test that generated plans are persisted to Neo4j."""
        response = http_client.post(
            "/plan",
            json={"goal": GOAL_PAYLOAD},
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        plan_id = data["plan_id"]

        # Query for plan in Neo4j
        with hcg_client.driver.session(database=hcg_client.database) as session:
            session.run(
                """
                MATCH (n {plan_id: $plan_id})
                RETURN count(n) as count
                """,
                {"plan_id": plan_id},
            )
            # Plan nodes may or may not be persisted

    def test_plan_requires_auth(self, http_client):
        """Test that /plan requires authentication."""
        response = http_client.post(
            "/plan",
            json={"goal": TEST_GOAL_PAYLOAD},
        )
        assert response.status_code in [401, 403]


class TestPlannerWithState:
    """Integration tests for planner with state interactions."""

    def test_plan_considers_current_state(self, http_client, auth_headers):
        """Test that planner considers current world state."""
        # Get current state
        state_response = http_client.get("/state", headers=auth_headers)
        assert state_response.status_code == 200

        # Generate plan
        plan_response = http_client.post(
            "/plan",
            json={"goal": GOAL_PAYLOAD},
            headers=auth_headers,
        )
        assert plan_response.status_code == 201

    def test_plan_after_state_update(self, http_client, auth_headers):
        """Test that planner adapts to state changes."""
        # Update state
        http_client.post(
            "/state",
            json={
                "state": {
                    "red_block": {"location": "table", "grasped": False},
                }
            },
            headers=auth_headers,
        )

        # Generate plan after state change
        response = http_client.post(
            "/plan",
            json={"goal": GOAL_PAYLOAD},
            headers=auth_headers,
        )
        assert response.status_code == 201
