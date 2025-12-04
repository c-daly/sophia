"""Integration tests for the /execute endpoint with real Neo4j.

These tests require Neo4j to be running.
Run with: pytest tests/integration/test_execute_integration.py -v -m integration

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
    """Create test application with Neo4j seeding."""
    os.environ["SOPHIA_API_TOKEN"] = api_token
    os.environ["NEO4J_URI"] = neo4j_uri
    os.environ["NEO4J_USER"] = neo4j_username
    os.environ["NEO4J_PASSWORD"] = neo4j_password
    os.environ["SEED_PICK_AND_PLACE_DATA"] = "true"
    os.environ["CLEAR_BEFORE_SEED"] = "true"
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


@pytest.fixture
def created_plan(client, auth_headers):
    """Create a plan for testing execution."""
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
    return response.json()


class TestExecuteIntegration:
    """Integration tests for the /execute endpoint."""

    def test_execute_plan_returns_execution_id(
        self, client, auth_headers, created_plan
    ):
        """Test that executing a plan returns an execution ID."""
        plan_id = created_plan["plan_id"]

        response = client.post(
            "/execute",
            json={
                "plan_id": plan_id,
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        assert "execution_id" in data
        assert data["plan_id"] == plan_id
        assert "results" in data
        assert "overall_status" in data

    def test_execute_dry_run_does_not_change_state(
        self, client, auth_headers, created_plan
    ):
        """Test that dry run execution doesn't modify state."""
        plan_id = created_plan["plan_id"]

        # Get initial state
        initial_state_response = client.get("/state", headers=auth_headers)
        initial_state = initial_state_response.json()["state"]

        # Execute in dry run mode
        response = client.post(
            "/execute",
            json={
                "plan_id": plan_id,
                "dry_run": True,
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        # In dry run, results should show 'simulated' status
        for result in data["results"]:
            assert result["status"] == "simulated"
            # State changes should be empty in dry run
            assert result["state_changes"] == {}

        # Verify state hasn't changed
        final_state_response = client.get("/state", headers=auth_headers)
        final_state = final_state_response.json()["state"]
        assert initial_state == final_state

    def test_execute_returns_results_for_each_step(
        self, client, auth_headers, created_plan
    ):
        """Test that execution returns results for executed steps."""
        plan_id = created_plan["plan_id"]

        response = client.post(
            "/execute",
            json={
                "plan_id": plan_id,
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        assert len(data["results"]) >= 1

        for result in data["results"]:
            assert "step" in result
            assert "status" in result
            assert result["status"] in ["success", "failed", "skipped", "simulated"]

    def test_execute_specific_step_index(self, client, auth_headers, created_plan):
        """Test executing a specific step by index."""
        plan_id = created_plan["plan_id"]

        response = client.post(
            "/execute",
            json={
                "plan_id": plan_id,
                "step_index": 0,  # Execute only first step
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        assert "results" in data

    def test_execute_requires_auth(self, client, created_plan):
        """Test that /execute requires authentication."""
        plan_id = created_plan["plan_id"]

        response = client.post(
            "/execute",
            json={
                "plan_id": plan_id,
            },
        )
        assert response.status_code in [401, 403]

    def test_execute_requires_plan_id(self, client, auth_headers):
        """Test that /execute requires a plan_id."""
        response = client.post(
            "/execute",
            json={},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_execute_with_nonexistent_plan_id(self, client, auth_headers):
        """Test execution with a non-existent plan ID."""
        response = client.post(
            "/execute",
            json={
                "plan_id": "nonexistent_plan_id_12345",
            },
            headers=auth_headers,
        )
        # Should still return 201 (current implementation is mock)
        # In a real implementation, this might return 404
        assert response.status_code in [201, 404]

    def test_execute_overall_status_reflects_results(
        self, client, auth_headers, created_plan
    ):
        """Test that overall_status reflects the execution results."""
        plan_id = created_plan["plan_id"]

        response = client.post(
            "/execute",
            json={
                "plan_id": plan_id,
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        overall_status = data["overall_status"]

        # Verify overall_status is one of the expected values
        assert overall_status in ["success", "partial", "failed"]

        # If all results are success, overall should be success
        if all(r["status"] == "success" for r in data["results"]):
            assert overall_status == "success"

    def test_execute_returns_timestamp(self, client, auth_headers, created_plan):
        """Test that execution response includes a timestamp."""
        plan_id = created_plan["plan_id"]

        response = client.post(
            "/execute",
            json={
                "plan_id": plan_id,
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        assert "created_at" in data

    def test_execute_with_invalid_step_index(self, client, auth_headers, created_plan):
        """Test execution with an out-of-range step index."""
        plan_id = created_plan["plan_id"]

        response = client.post(
            "/execute",
            json={
                "plan_id": plan_id,
                "step_index": 9999,  # Large invalid index
            },
            headers=auth_headers,
        )
        # Current implementation returns 201 (mock)
        # A real implementation might return 400 or 404
        assert response.status_code in [201, 400, 404]


class TestExecuteWorkflow:
    """Integration tests for execute workflow with state updates."""

    def test_execute_then_verify_state(self, client, auth_headers, created_plan):
        """Test executing a plan and verifying state can still be retrieved."""
        plan_id = created_plan["plan_id"]

        # Execute the plan
        exec_response = client.post(
            "/execute",
            json={
                "plan_id": plan_id,
            },
            headers=auth_headers,
        )
        assert exec_response.status_code == 201

        # Verify state endpoint still works after execution
        state_response = client.get("/state", headers=auth_headers)
        assert state_response.status_code == 200
        assert "state" in state_response.json()

    def test_multiple_executions_generate_unique_ids(
        self, client, auth_headers, created_plan
    ):
        """Test that multiple executions generate unique execution IDs."""
        plan_id = created_plan["plan_id"]
        execution_ids = []

        for _ in range(3):
            response = client.post(
                "/execute",
                json={
                    "plan_id": plan_id,
                    "dry_run": True,  # Use dry run to avoid state changes
                },
                headers=auth_headers,
            )
            assert response.status_code == 201
            execution_ids.append(response.json()["execution_id"])

        # All execution IDs should be unique
        assert len(execution_ids) == len(set(execution_ids))
