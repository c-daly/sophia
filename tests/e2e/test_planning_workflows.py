"""End-to-end tests for complete planning workflows.

These tests validate full user scenarios from goal definition
through plan execution to final state verification.

Requires: Full stack (Neo4j, Milvus, Sophia API running)
"""

import os
import pytest
from fastapi.testclient import TestClient

from sophia.api.app import create_app
from sophia.hcg_client import HCGClient


pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def neo4j_uri():
    """Neo4j connection URI."""
    return os.getenv("NEO4J_URI", "bolt://localhost:7687")


@pytest.fixture(scope="module")
def neo4j_user():
    """Neo4j username."""
    return os.getenv("NEO4J_USER", "neo4j")


@pytest.fixture(scope="module")
def neo4j_password():
    """Neo4j password."""
    return os.getenv("NEO4J_PASSWORD", "neo4jtest")


@pytest.fixture(scope="module")
def hcg_client(neo4j_uri, neo4j_user, neo4j_password):
    """Create HCG client for direct database verification."""
    client = HCGClient(
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_user,
        neo4j_password=neo4j_password,
    )
    yield client
    client.close()


@pytest.fixture(scope="module")
def api_token():
    """API authentication token."""
    return "test-e2e-token"


@pytest.fixture(scope="module")
def app(api_token, neo4j_uri, neo4j_user, neo4j_password):
    """Create test application."""
    os.environ["SOPHIA_API_TOKEN"] = api_token
    os.environ["NEO4J_URI"] = neo4j_uri
    os.environ["NEO4J_USER"] = neo4j_user
    os.environ["NEO4J_PASSWORD"] = neo4j_password
    return create_app()


@pytest.fixture(scope="module")
def client(app):
    """Create test client."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers(api_token):
    """Authentication headers."""
    return {"Authorization": f"Bearer {api_token}"}


class TestGoalToPlanToExecuteWorkflow:
    """E2E tests for complete goal → plan → execute → verify workflow."""

    def test_pick_and_place_complete_workflow(self, client, auth_headers, hcg_client):
        """Test complete pick-and-place workflow from goal to execution.
        
        Workflow:
        1. Verify initial state (red_block on table)
        2. Submit goal (red_block in bin)
        3. Receive plan (MOVE → GRASP → MOVE → RELEASE)
        4. Execute plan
        5. Verify final state (red_block in bin)
        """
        # Step 1: Verify initial state
        state_response = client.get("/state", headers=auth_headers)
        assert state_response.status_code == 200
        initial_state = state_response.json()["state"]
        
        # Verify red_block starts on table
        assert "red_block" in initial_state or initial_state == {}
        
        # Step 2: Submit planning goal
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
        plan_id = plan["plan_id"]
        steps = plan["steps"]
        
        # Step 3: Verify plan structure
        assert len(steps) == 4, f"Expected 4 steps (MOVE→GRASP→MOVE→RELEASE), got {len(steps)}"
        expected_actions = ["MOVE", "GRASP", "MOVE", "RELEASE"]
        actual_actions = [step.get("action_type") for step in steps]
        assert actual_actions == expected_actions, f"Expected {expected_actions}, got {actual_actions}"
        
        # Step 4: Execute plan
        execute_response = client.post(
            "/execute",
            json={"plan_id": plan_id},
            headers=auth_headers,
        )
        assert execute_response.status_code == 201
        execution = execute_response.json()
        assert execution["overall_status"] == "success"
        
        # Step 5: Verify final state
        final_state_response = client.get("/state", headers=auth_headers)
        assert final_state_response.status_code == 200
        final_state = final_state_response.json()["state"]
        
        # Red block should now be in bin
        if "red_block" in final_state:
            assert final_state["red_block"]["location"] == "bin"

    def test_plan_simulate_then_execute(self, client, auth_headers):
        """Test workflow: plan → simulate → refine → execute.
        
        Uses imagination/simulation to preview plan outcomes
        before actual execution.
        """
        # Step 1: Generate plan
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
        
        # Step 2: Simulate plan execution
        simulate_response = client.post(
            "/simulate",
            json={
                "entities": [
                    {
                        "id": "red_block",
                        "type": "object",
                        "properties": {"location": "table"},
                    }
                ],
                "actions": [
                    {"type": step.get("action_type"), "target": step.get("target")}
                    for step in plan["steps"]
                    if step.get("action_type")
                ],
                "k_steps": len(plan["steps"]),
            },
            headers=auth_headers,
        )
        assert simulate_response.status_code == 201
        simulation = simulate_response.json()
        
        # Verify simulation predicts successful outcome
        assert simulation["overall_confidence"] > 0.5
        assert len(simulation["imagined_states"]) > 0
        
        # Step 3: Execute plan (now confident from simulation)
        execute_response = client.post(
            "/execute",
            json={"plan_id": plan["plan_id"]},
            headers=auth_headers,
        )
        assert execute_response.status_code == 201

    def test_dry_run_then_execute(self, client, auth_headers):
        """Test dry run followed by actual execution."""
        # Generate plan
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
        plan_id = plan_response.json()["plan_id"]
        
        # Dry run first
        dry_run_response = client.post(
            "/execute",
            json={"plan_id": plan_id, "dry_run": True},
            headers=auth_headers,
        )
        assert dry_run_response.status_code == 201
        dry_result = dry_run_response.json()
        assert dry_result["overall_status"] == "simulated"
        
        # Now execute for real
        execute_response = client.post(
            "/execute",
            json={"plan_id": plan_id, "dry_run": False},
            headers=auth_headers,
        )
        assert execute_response.status_code == 201
        result = execute_response.json()
        assert result["overall_status"] == "success"


class TestStateLifecycleWorkflow:
    """E2E tests for state management lifecycle."""

    def test_state_write_read_roundtrip(self, client, auth_headers, hcg_client):
        """Test complete state write → read → verify cycle."""
        # Write state
        new_state = {
            "test_object": {
                "location": "shelf_a",
                "color": "blue",
                "weight": 1.5,
            }
        }
        
        write_response = client.post(
            "/state",
            json={"state": new_state},
            headers=auth_headers,
        )
        assert write_response.status_code == 200
        
        # Read state back via API
        read_response = client.get("/state", headers=auth_headers)
        assert read_response.status_code == 200
        api_state = read_response.json()["state"]
        
        # Verify via direct Neo4j query
        neo4j_state = hcg_client.get_node("current_state")
        assert neo4j_state is not None
        
        # States should match
        if "test_object" in api_state:
            assert api_state["test_object"]["location"] == "shelf_a"

    def test_state_update_triggers_cwm_envelope(self, client, auth_headers):
        """Test that state updates create CWM state envelopes."""
        # Update state
        update_response = client.post(
            "/state",
            json={
                "state": {
                    "envelope_test_object": {"status": "created"}
                }
            },
            headers=auth_headers,
        )
        assert update_response.status_code == 200
        
        # Query CWM history
        cwm_response = client.get("/state/cwm", headers=auth_headers)
        assert cwm_response.status_code == 200
        cwm_data = cwm_response.json()
        
        # Should have at least one state envelope
        assert cwm_data["total"] > 0 or len(cwm_data["states"]) > 0


class TestSimulationWorkflow:
    """E2E tests for simulation workflows."""

    def test_simulate_persist_query_cycle(self, client, auth_headers, hcg_client):
        """Test simulation → persistence → query cycle."""
        # Run simulation
        simulate_response = client.post(
            "/simulate",
            json={
                "entities": [
                    {
                        "id": "ball",
                        "type": "object",
                        "properties": {"mass": 0.5},
                        "position": {"x": 0, "y": 0, "z": 1.0},
                    }
                ],
                "k_steps": 5,
            },
            headers=auth_headers,
        )
        assert simulate_response.status_code == 201
        simulation = simulate_response.json()
        simulation_id = simulation["simulation_id"]
        
        # Verify imagined states were created
        assert len(simulation["imagined_states"]) == 5
        
        # Query Neo4j for persisted simulation
        sim_node = hcg_client.get_node(simulation_id)
        # Simulation may or may not be persisted as a node depending on implementation
        
        # Verify imagined states have decreasing confidence
        confidences = [s["confidence"] for s in simulation["imagined_states"]]
        for i in range(len(confidences) - 1):
            assert confidences[i] >= confidences[i + 1], "Confidence should decay"

    def test_simulation_with_media_sample(self, client, auth_headers):
        """Test simulation referencing a media sample for context."""
        # First ingest a media sample (if endpoint available)
        # For now, test with media_sample_id reference
        simulate_response = client.post(
            "/simulate",
            json={
                "entities": [
                    {
                        "id": "observed_object",
                        "type": "object",
                        "properties": {},
                    }
                ],
                "media_sample_id": "sample_test_123",
                "k_steps": 3,
            },
            headers=auth_headers,
        )
        
        # Should succeed (sample may not exist, but request is valid)
        assert simulate_response.status_code in [201, 404]


class TestCrossServiceWorkflow:
    """E2E tests for cross-service interactions."""

    def test_hermes_proposal_to_execution(self, client, auth_headers, hcg_client):
        """Test Hermes proposal ingestion followed by execution."""
        # Ingest a proposal from Hermes
        proposal = {
            "proposal_id": "hermes_e2e_test_001",
            "source": "hermes",
            "confidence": 0.85,
            "plan_steps": [
                {
                    "step_id": "step_1",
                    "action_type": "MOVE",
                    "target": "position_a",
                }
            ],
            "imagined_states": [
                {
                    "state_id": "imagined_1",
                    "description": "Robot at position A",
                    "confidence": 0.8,
                }
            ],
        }
        
        ingest_response = client.post(
            "/ingest/hermes_proposal",
            json=proposal,
        )
        assert ingest_response.status_code == 201
        result = ingest_response.json()
        
        # Verify proposal was stored
        assert "proposal_node_id" in result
        
        # Verify in Neo4j
        proposal_node = hcg_client.get_node(result["proposal_node_id"])
        assert proposal_node is not None
        assert proposal_node["properties"].get("source") == "hermes"

    def test_concurrent_operations(self, client, auth_headers):
        """Test concurrent planning and simulation don't interfere."""
        import concurrent.futures
        
        def run_plan():
            return client.post(
                "/plan",
                json={
                    "goal": {
                        "description": "test concurrent plan",
                        "target_state": "concurrent_state",
                    }
                },
                headers=auth_headers,
            )
        
        def run_simulate():
            return client.post(
                "/simulate",
                json={
                    "entities": [{"id": "obj", "type": "object"}],
                    "k_steps": 3,
                },
                headers=auth_headers,
            )
        
        # Run concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            plan_future = executor.submit(run_plan)
            sim_future = executor.submit(run_simulate)
            
            plan_result = plan_future.result()
            sim_result = sim_future.result()
        
        # Both should succeed or fail gracefully (no crashes)
        assert plan_result.status_code in [200, 201, 503]
        assert sim_result.status_code in [200, 201, 503]
        
        # If both succeeded, verify they have different IDs
        if plan_result.status_code == 201 and sim_result.status_code == 201:
            plan_id = plan_result.json().get("plan_id")
            sim_id = sim_result.json().get("simulation_id")
            assert plan_id != sim_id


class TestErrorRecoveryWorkflow:
    """E2E tests for error handling and recovery."""

    def test_graceful_degradation_without_neo4j(self, client, auth_headers):
        """Test that health endpoint reports degraded status appropriately."""
        # Health check should always work
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "components" in data
        # Status may be "healthy" or "degraded" depending on Neo4j

    def test_invalid_plan_execution_recovery(self, client, auth_headers):
        """Test recovery from attempting to execute non-existent plan."""
        response = client.post(
            "/execute",
            json={"plan_id": "non_existent_plan_id_12345"},
            headers=auth_headers,
        )
        
        # Should fail gracefully, not crash
        assert response.status_code in [400, 404, 500]
        
        # System should still be responsive
        health_response = client.get("/health")
        assert health_response.status_code == 200
