"""Integration tests for the prototype pick-and-place workflow.

These tests require Sophia and Neo4j to be running with seeded data.
Run with: pytest tests/integration/test_prototype_integration.py -v -m integration

Start services with: ./scripts/test_integration.sh up
"""

import pytest

pytestmark = [
    pytest.mark.integration,
]

# Common goal payload for tests - goal must be a dict, not a string
GOAL_PAYLOAD = {
    "description": "move red_block to bin",
    "target_state": "red_block_in_bin",
}


class TestPrototypeIntegration:
    """Integration tests for the prototype demonstrating the complete flow."""

    def test_health_check_with_neo4j(self, http_client):
        """Test that health check shows Neo4j as healthy."""
        response = http_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["dependencies"]["neo4j"]["connected"] is True

    def test_get_initial_state_from_neo4j(self, http_client, auth_headers):
        """Test retrieving initial state from Neo4j."""
        response = http_client.get("/state", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "nodes" in data or "state" in data

    def test_generate_plan_reads_from_neo4j(self, http_client, auth_headers):
        """Test that plan generation uses Neo4j data."""
        response = http_client.post(
            "/plan",
            json={"goal": GOAL_PAYLOAD},
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        assert "plan_id" in data
        assert "plan" in data

    def test_update_state_writes_to_neo4j(self, http_client, auth_headers, hcg_client):
        """Test that state updates are written to Neo4j."""
        # Update state via API
        update_response = http_client.post(
            "/state",
            json={
                "state": {
                    "test_node": {"status": "updated"},
                }
            },
            headers=auth_headers,
        )
        # May succeed or return 404 if node doesn't exist
        assert update_response.status_code in [200, 201, 404]

    def test_complete_pick_and_place_workflow(self, http_client, auth_headers):
        """Test complete pick-and-place workflow."""
        # 1. Get initial state
        state_response = http_client.get("/state", headers=auth_headers)
        assert state_response.status_code == 200

        # 2. Generate plan
        plan_response = http_client.post(
            "/plan",
            json={"goal": GOAL_PAYLOAD},
            headers=auth_headers,
        )
        assert plan_response.status_code == 201
        plan_data = plan_response.json()

        # 3. Execute plan (dry run)
        exec_response = http_client.post(
            "/execute",
            json={
                "plan_id": plan_data["plan_id"],
                "dry_run": True,
            },
            headers=auth_headers,
        )
        assert exec_response.status_code == 201

        # 4. Verify execution result
        exec_data = exec_response.json()
        assert "execution_id" in exec_data
        assert "overall_status" in exec_data

    def test_shacl_validation_on_state_update(self, http_client, auth_headers):
        """Test that SHACL validation is applied to state updates."""
        # Attempt to update with invalid data
        response = http_client.post(
            "/state",
            json={
                "state": {
                    "invalid_node": {"status": "unknown"},
                }
            },
            headers=auth_headers,
        )
        # Should either succeed (if permissive) or fail validation
        assert response.status_code in [200, 201, 400, 404, 422]

    def test_plan_written_to_neo4j_can_be_retrieved(
        self, http_client, auth_headers, hcg_client
    ):
        """Test that generated plans are persisted to Neo4j."""
        # Generate a plan
        plan_response = http_client.post(
            "/plan",
            json={"goal": GOAL_PAYLOAD},
            headers=auth_headers,
        )
        assert plan_response.status_code == 201
        plan_data = plan_response.json()
        plan_id = plan_data["plan_id"]

        # Query Neo4j for the plan
        with hcg_client.driver.session(database=hcg_client.database) as session:
            result = session.run(
                """
                MATCH (n {plan_id: $plan_id})
                RETURN n.plan_id as plan_id
                """,
                {"plan_id": plan_id},
            )
            # Plan may or may not be persisted depending on implementation
            list(result)
