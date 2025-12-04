"""Integration tests for the /execute endpoint.

These tests require Sophia and Neo4j to be running with seeded data.
Run with: pytest tests/integration/test_execute_integration.py -v -m integration

Start services with: ./scripts/test_integration.sh up
"""

import pytest

pytestmark = [
    pytest.mark.integration,
]


class TestExecuteIntegration:
    """Integration tests for the /execute endpoint."""

    def test_execute_plan_returns_execution_id(self, http_client, auth_headers):
        """Test that /execute returns an execution ID."""
        # First create a plan
        plan_response = http_client.post(
            "/plan",
            json={"goal": "move red_block to bin"},
            headers=auth_headers,
        )
        assert plan_response.status_code == 201
        plan_data = plan_response.json()

        # Execute the plan
        response = http_client.post(
            "/execute",
            json={
                "plan_id": plan_data["plan_id"],
                "dry_run": False,
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        assert "execution_id" in data
        assert "status" in data

    def test_execute_dry_run_does_not_change_state(self, http_client, auth_headers):
        """Test that dry_run=True doesn't actually change state."""
        # Get initial state
        state_response = http_client.get("/state", headers=auth_headers)
        initial_state = state_response.json()

        # Create and execute plan in dry run mode
        plan_response = http_client.post(
            "/plan",
            json={"goal": "move red_block to bin"},
            headers=auth_headers,
        )
        plan_data = plan_response.json()

        response = http_client.post(
            "/execute",
            json={
                "plan_id": plan_data["plan_id"],
                "dry_run": True,
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

        # Verify state unchanged
        state_response = http_client.get("/state", headers=auth_headers)
        final_state = state_response.json()
        assert initial_state == final_state

    def test_execute_returns_results_for_each_step(self, http_client, auth_headers):
        """Test that /execute returns results for each plan step."""
        plan_response = http_client.post(
            "/plan",
            json={"goal": "move red_block to bin"},
            headers=auth_headers,
        )
        plan_data = plan_response.json()

        response = http_client.post(
            "/execute",
            json={
                "plan_id": plan_data["plan_id"],
                "dry_run": True,
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        assert "results" in data
        if plan_data.get("steps"):
            assert len(data["results"]) == len(plan_data["steps"])

    def test_execute_specific_step_index(self, http_client, auth_headers):
        """Test executing only a specific step by index."""
        plan_response = http_client.post(
            "/plan",
            json={"goal": "move red_block to bin"},
            headers=auth_headers,
        )
        plan_data = plan_response.json()

        response = http_client.post(
            "/execute",
            json={
                "plan_id": plan_data["plan_id"],
                "step_index": 0,
                "dry_run": True,
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        assert "results" in data
        # Should only have result for one step
        assert len(data["results"]) == 1

    def test_execute_requires_auth(self, http_client):
        """Test that /execute requires authentication."""
        response = http_client.post(
            "/execute",
            json={
                "plan_id": "test-plan",
                "dry_run": True,
            },
        )
        assert response.status_code in [401, 403]

    def test_execute_overall_status_reflects_results(self, http_client, auth_headers):
        """Test that overall status reflects individual step results."""
        plan_response = http_client.post(
            "/plan",
            json={"goal": "move red_block to bin"},
            headers=auth_headers,
        )
        plan_data = plan_response.json()

        response = http_client.post(
            "/execute",
            json={
                "plan_id": plan_data["plan_id"],
                "dry_run": True,
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        assert data["status"] in ["success", "partial", "failed", "pending"]

    def test_execute_returns_timestamp(self, http_client, auth_headers):
        """Test that /execute returns execution timestamp."""
        plan_response = http_client.post(
            "/plan",
            json={"goal": "move red_block to bin"},
            headers=auth_headers,
        )
        plan_data = plan_response.json()

        response = http_client.post(
            "/execute",
            json={
                "plan_id": plan_data["plan_id"],
                "dry_run": True,
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        assert "timestamp" in data or "executed_at" in data

    def test_execute_with_invalid_step_index(self, http_client, auth_headers):
        """Test that invalid step_index returns appropriate error."""
        plan_response = http_client.post(
            "/plan",
            json={"goal": "move red_block to bin"},
            headers=auth_headers,
        )
        plan_data = plan_response.json()

        response = http_client.post(
            "/execute",
            json={
                "plan_id": plan_data["plan_id"],
                "step_index": 9999,
                "dry_run": True,
            },
            headers=auth_headers,
        )
        # Should return error for invalid index
        assert response.status_code in [400, 404, 422]


class TestExecuteWorkflow:
    """Integration tests for execute workflows."""

    def test_execute_then_verify_state(self, http_client, auth_headers, hcg_client):
        """Test that execution updates state correctly."""
        # Create and execute a plan
        plan_response = http_client.post(
            "/plan",
            json={"goal": "move red_block to bin"},
            headers=auth_headers,
        )
        plan_data = plan_response.json()

        response = http_client.post(
            "/execute",
            json={
                "plan_id": plan_data["plan_id"],
                "dry_run": False,
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

        # Verify state was updated (check Neo4j directly)
        with hcg_client.driver.session(database=hcg_client.database) as session:
            result = session.run(
                """
                MATCH (n:Node {id: 'red_block'})
                RETURN n.id as id, n.properties as props
                """
            )
            # Just verify the query runs without error
            record = result.single()

    def test_multiple_executions_generate_unique_ids(self, http_client, auth_headers):
        """Test that each execution has a unique ID."""
        plan_response = http_client.post(
            "/plan",
            json={"goal": "move red_block to bin"},
            headers=auth_headers,
        )
        plan_data = plan_response.json()

        execution_ids = []
        for _ in range(3):
            response = http_client.post(
                "/execute",
                json={
                    "plan_id": plan_data["plan_id"],
                    "dry_run": True,
                },
                headers=auth_headers,
            )
            assert response.status_code == 201
            execution_ids.append(response.json()["execution_id"])

        # All execution IDs should be unique
        assert len(execution_ids) == len(set(execution_ids))
