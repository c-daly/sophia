"""Integration tests for the Sophia prototype (plan/state API over HCG).

These tests require Neo4j and Milvus to be running.
Run with: pytest tests/api/test_prototype_integration.py -v -m integration
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
def neo4j_username():
    """Neo4j username."""
    return os.getenv("NEO4J_USER", "neo4j")


@pytest.fixture
def neo4j_password():
    """Neo4j password."""
    return os.getenv("NEO4J_PASSWORD", "sophiadev")


@pytest.fixture
def milvus_host():
    """Milvus host."""
    return os.getenv("MILVUS_HOST", "localhost")


@pytest.fixture
def milvus_port():
    """Milvus port."""
    return int(os.getenv("MILVUS_PORT", "19530"))


@pytest.fixture
def hcg_client(neo4j_uri, neo4j_username, neo4j_password, milvus_host, milvus_port):
    """Create HCG client for test setup."""
    try:
        client = HCGClient(
            neo4j_uri=neo4j_uri,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password,
            milvus_host=milvus_host,
            milvus_port=milvus_port,
        )
        yield client
        client.close()
    except Exception as e:
        pytest.skip(f"HCG services not available: {e}")


@pytest.fixture
def api_token():
    """API authentication token."""
    return "test-integration-token"


@pytest.fixture
def app(api_token):
    """Create test application with Neo4j seeding enabled."""
    os.environ["SOPHIA_API_TOKEN"] = api_token
    os.environ["SEED_PICK_AND_PLACE_DATA"] = "true"
    os.environ["CLEAR_BEFORE_SEED"] = "true"
    return create_app()


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def auth_headers(api_token):
    """Authentication headers."""
    return {"Authorization": f"Bearer {api_token}"}


class TestPrototypeIntegration:
    """Integration tests for the prototype demonstrating the complete flow."""

    def test_health_check_with_neo4j(self, client):
        """Test that health check shows Neo4j as healthy."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["components"]["neo4j"] is True

    def test_get_initial_state_from_neo4j(self, client, auth_headers):
        """Test reading initial state from Neo4j after seeding."""
        response = client.get("/state", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "state" in data
        assert "state_id" in data
        assert "timestamp" in data

        # Verify initial state structure
        state = data["state"]
        assert "red_block" in state
        assert "blue_block" in state
        assert "gripper" in state

        # Verify initial values
        assert state["red_block"]["location"] == "table"
        assert state["red_block"]["grasped"] is False
        assert state["gripper"]["position"] == "home"
        assert state["gripper"]["holding"] is None

    def test_generate_plan_reads_from_neo4j(self, client, auth_headers):
        """Test that plan generation reads state from Neo4j."""
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
        assert "plan" in data
        assert "plan_id" in data
        assert len(data["plan"]) == 4  # MOVE, GRASP, MOVE, RELEASE

        # Verify plan sequence
        plan_sequence = [step["action_type"] for step in data["plan"]]
        assert plan_sequence == ["MOVE", "GRASP", "MOVE", "RELEASE"]

        # Verify plan step details
        assert data["plan"][0]["id"] == "move_to_red_block"
        assert data["plan"][1]["id"] == "grasp_red_block"
        assert data["plan"][2]["id"] == "move_to_bin"
        assert data["plan"][3]["id"] == "release_red_block"

    def test_update_state_writes_to_neo4j(self, client, auth_headers):
        """Test that state updates are written to Neo4j."""
        # Update state to reflect execution
        new_state = {
            "red_block": {"location": "bin", "grasped": False},
            "blue_block": {"location": "table", "grasped": False},
            "gripper": {"position": "bin", "holding": None},
        }

        response = client.post(
            "/state",
            json={"state": new_state},
            headers=auth_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert data["validation_passed"] is True

        # Verify state was persisted in Neo4j
        response = client.get("/state", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        state = data["state"]
        assert state["red_block"]["location"] == "bin"
        assert state["gripper"]["position"] == "bin"

    def test_complete_pick_and_place_workflow(self, client, auth_headers):
        """Test the complete workflow: read state -> plan -> execute -> update state."""
        # Step 1: Read initial state
        response = client.get("/state", headers=auth_headers)
        assert response.status_code == 200
        initial_state = response.json()["state"]
        assert initial_state["red_block"]["location"] == "table"

        # Step 2: Generate plan
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
        plan_data = response.json()
        assert len(plan_data["plan"]) == 4
        # Plan ID is stored but not used in this test
        # Verify plan is written to Neo4j (would need direct Neo4j query in real test)
        # For now, we trust that the plan endpoint wrote it

        # Step 3: Simulate execution by updating state
        final_state = {
            "red_block": {"location": "bin", "grasped": False},
            "blue_block": {"location": "table", "grasped": False},
            "gripper": {"position": "bin", "holding": None},
        }

        response = client.post(
            "/state",
            json={"state": final_state},
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert response.json()["validation_passed"] is True

        # Step 4: Verify final state persisted
        response = client.get("/state", headers=auth_headers)
        assert response.status_code == 200
        persisted_state = response.json()["state"]
        assert persisted_state["red_block"]["location"] == "bin"
        assert persisted_state["gripper"]["position"] == "bin"

    def test_plan_written_to_neo4j_can_be_retrieved(
        self, client, auth_headers, hcg_client
    ):
        """Test that plans written to Neo4j can be retrieved."""
        # Generate a plan
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
        plan_data = response.json()
        plan_id = plan_data["plan_id"]

        # Verify plan node exists in Neo4j
        plan_node = hcg_client.get_node(plan_id)
        assert plan_node is not None
        assert plan_node["type"] == "plan"
        assert "goal" in plan_node["properties"]
        assert "steps" in plan_node["properties"]

        # Verify plan has correct steps
        steps = plan_node["properties"]["steps"]
        assert len(steps) == 4
        assert steps[0]["action_type"] == "MOVE"
        assert steps[1]["action_type"] == "GRASP"
        assert steps[2]["action_type"] == "MOVE"
        assert steps[3]["action_type"] == "RELEASE"

    def test_shacl_validation_on_state_update(self, client, auth_headers):
        """Test that SHACL validation is enforced on state updates."""
        # This test depends on the SHACL rules in the validator
        # For now, we just verify that validation occurs (any state is accepted)
        valid_state = {
            "red_block": {"location": "bin", "grasped": False},
        }

        response = client.post(
            "/state",
            json={"state": valid_state},
            headers=auth_headers,
        )
        # Should succeed with validation
        assert response.status_code == 200
        assert response.json()["validation_passed"] is True
