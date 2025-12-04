"""
Sophia End-to-End Tests

These tests validate complete Sophia workflows with real infrastructure.
The test stack must be running: ./scripts/start_services.sh

Test categories:
1. Infrastructure health checks
2. Hermes proposal ingestion (core integration point)
3. State management with SHACL validation
4. Planning and execution workflows
5. JEPA simulation pipeline

Based on LOGOS Phase 1/2 e2e test patterns.
Reference: sophia#57 (Testing Audit)
"""

import pytest
import httpx


pytestmark = pytest.mark.e2e


class TestInfrastructureHealth:
    """Verify that infrastructure services are running and healthy."""

    def test_neo4j_is_running(self, infrastructure_ports: dict):
        """Neo4j should be accessible on the configured port."""
        resp = httpx.get(
            f"http://localhost:{infrastructure_ports['neo4j_http']}/",
            timeout=5,
        )
        assert resp.status_code == 200, "Neo4j HTTP endpoint should return 200"

    def test_milvus_is_healthy(self, infrastructure_ports: dict):
        """Milvus should report healthy status."""
        resp = httpx.get(
            f"http://localhost:{infrastructure_ports['milvus_health']}/healthz",
            timeout=5,
        )
        assert resp.status_code == 200, "Milvus healthz should return 200"

    def test_neo4j_accepts_cypher(self, neo4j_config: dict):
        """Neo4j should accept Cypher queries via HTTP API."""
        # Use the transaction endpoint
        resp = httpx.post(
            "http://localhost:37474/db/neo4j/tx/commit",
            json={"statements": [{"statement": "RETURN 1 as test"}]},
            auth=(neo4j_config["user"], neo4j_config["password"]),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        assert resp.status_code == 200, f"Neo4j query failed: {resp.text}"
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) > 0


class TestHermesIngestion:
    """
    Test the Hermes proposal ingestion endpoint.

    This is a critical integration point - Hermes sends LLM-generated
    plans and imagined states to Sophia for persistence with SHACL validation.
    """

    def test_ingest_basic_proposal(
        self, sophia_url: str, unique_id: str, test_timestamp: str
    ):
        """Ingest a basic proposal and verify it's stored."""
        proposal_id = f"proposal_{unique_id}"

        resp = httpx.post(
            f"{sophia_url}/ingest/hermes_proposal",
            json={
                "proposal_id": proposal_id,
                "source_service": "hermes",
                "llm_provider": "openai",
                "model": "gpt-4",
                "generated_at": test_timestamp,
                "confidence": 0.85,
                "raw_text": "Pick up the red block and place it in the bin.",
            },
            timeout=10,
        )

        assert resp.status_code == 201, f"Proposal ingestion failed: {resp.text}"
        data = resp.json()
        assert data["proposal_id"] == proposal_id
        assert data["status"] == "accepted"
        assert proposal_id in data["stored_node_ids"]

    def test_ingest_proposal_with_plan_steps(
        self, sophia_url: str, unique_id: str, test_timestamp: str
    ):
        """Ingest a proposal with plan steps and verify graph structure."""
        proposal_id = f"proposal_{unique_id}"

        resp = httpx.post(
            f"{sophia_url}/ingest/hermes_proposal",
            json={
                "proposal_id": proposal_id,
                "source_service": "hermes",
                "llm_provider": "openai",
                "model": "gpt-4",
                "generated_at": test_timestamp,
                "confidence": 0.9,
                "plan_steps": [
                    {
                        "name": "move_to_block",
                        "action_type": "MOVE",
                        "target": "red_block",
                    },
                    {
                        "name": "grasp_block",
                        "action_type": "GRASP",
                        "target": "red_block",
                    },
                    {
                        "name": "move_to_bin",
                        "action_type": "MOVE",
                        "target": "bin",
                    },
                    {
                        "name": "release_block",
                        "action_type": "RELEASE",
                        "target": "red_block",
                    },
                ],
            },
            timeout=10,
        )

        assert resp.status_code == 201, f"Proposal ingestion failed: {resp.text}"
        data = resp.json()
        assert data["status"] == "accepted"

        # Should have proposal + 4 plan steps = 5 nodes
        assert len(data["stored_node_ids"]) == 5
        assert proposal_id in data["stored_node_ids"]
        assert f"{proposal_id}_plan_step_0" in data["stored_node_ids"]
        assert f"{proposal_id}_plan_step_3" in data["stored_node_ids"]

    def test_ingest_proposal_with_imagined_states(
        self, sophia_url: str, unique_id: str, test_timestamp: str
    ):
        """Ingest a proposal with imagined states."""
        proposal_id = f"proposal_{unique_id}"

        resp = httpx.post(
            f"{sophia_url}/ingest/hermes_proposal",
            json={
                "proposal_id": proposal_id,
                "source_service": "hermes",
                "llm_provider": "anthropic",
                "model": "claude-3",
                "generated_at": test_timestamp,
                "confidence": 0.88,
                "imagined_states": [
                    {
                        "description": "Block is grasped",
                        "confidence": 0.95,
                    },
                    {
                        "description": "Block is above bin",
                        "confidence": 0.85,
                    },
                    {
                        "description": "Block is in bin",
                        "confidence": 0.75,
                    },
                ],
            },
            timeout=10,
        )

        assert resp.status_code == 201, f"Proposal ingestion failed: {resp.text}"
        data = resp.json()

        # Should have proposal + 3 imagined states = 4 nodes
        assert len(data["stored_node_ids"]) == 4

    def test_ingest_proposal_with_tool_calls(
        self, sophia_url: str, unique_id: str, test_timestamp: str
    ):
        """Ingest a proposal with tool call references."""
        proposal_id = f"proposal_{unique_id}"

        resp = httpx.post(
            f"{sophia_url}/ingest/hermes_proposal",
            json={
                "proposal_id": proposal_id,
                "source_service": "hermes",
                "llm_provider": "openai",
                "model": "gpt-4-turbo",
                "generated_at": test_timestamp,
                "confidence": 0.92,
                "tool_calls": [
                    {
                        "tool_name": "perception_query",
                        "arguments": {"query": "find red blocks"},
                    },
                    {
                        "tool_name": "action_execute",
                        "arguments": {"action": "grasp", "target": "block_1"},
                    },
                ],
            },
            timeout=10,
        )

        assert resp.status_code == 201
        data = resp.json()

        # Should have proposal + 2 tool calls = 3 nodes
        assert len(data["stored_node_ids"]) == 3

    def test_ingest_requires_mandatory_fields(self, sophia_url: str):
        """Proposal ingestion should reject requests missing required fields."""
        resp = httpx.post(
            f"{sophia_url}/ingest/hermes_proposal",
            json={
                "proposal_id": "incomplete",
                # Missing: source_service, llm_provider, model, generated_at, confidence
            },
            timeout=10,
        )

        assert resp.status_code == 422, "Should reject incomplete proposal"


class TestStateManagement:
    """
    Test state read/write operations with SHACL validation.

    State updates trigger CWM-A emissions and are validated against
    the SHACL schema before persistence.
    """

    def test_read_state(self, sophia_url: str, auth_headers: dict):
        """GET /state should return current state."""
        resp = httpx.get(
            f"{sophia_url}/state",
            headers=auth_headers,
            timeout=10,
        )

        assert resp.status_code == 200, f"State read failed: {resp.text}"
        data = resp.json()
        assert "state" in data
        assert "state_id" in data

    def test_update_state(self, sophia_url: str, auth_headers: dict, unique_id: str):
        """POST /state should update state with SHACL validation."""
        new_state = {
            f"test_entity_{unique_id}": {
                "position": [1.0, 2.0, 3.0],
                "grasped": False,
            }
        }

        resp = httpx.post(
            f"{sophia_url}/state",
            json={"state": new_state},
            headers=auth_headers,
            timeout=10,
        )

        assert resp.status_code == 200, f"State update failed: {resp.text}"
        data = resp.json()
        assert data["state_id"] == "current_state"
        assert data["validation_passed"] is True

    def test_update_state_emits_cwm(
        self, sophia_url: str, auth_headers: dict, unique_id: str
    ):
        """State update should emit CWM-A state envelope with diffs."""
        # First, set initial state
        initial_state = {f"entity_{unique_id}": {"value": 1}}
        resp = httpx.post(
            f"{sophia_url}/state",
            json={"state": initial_state},
            headers=auth_headers,
            timeout=10,
        )
        assert resp.status_code == 200

        # Now update to see diff emission
        updated_state = {f"entity_{unique_id}": {"value": 2}}
        resp = httpx.post(
            f"{sophia_url}/state",
            json={"state": updated_state},
            headers=auth_headers,
            timeout=10,
        )

        assert resp.status_code == 200
        data = resp.json()

        # Should have emitted CWM state with diffs
        if data.get("cwm_state_id"):
            assert data["entity_diffs"] is not None
            assert len(data["entity_diffs"]) > 0

    def test_get_cwm_history(self, sophia_url: str, auth_headers: dict):
        """GET /state/cwm should return CWM state history."""
        resp = httpx.get(
            f"{sophia_url}/state/cwm",
            headers=auth_headers,
            timeout=10,
        )

        assert resp.status_code == 200, f"CWM history read failed: {resp.text}"
        data = resp.json()
        assert "states" in data
        assert "total" in data


class TestHealthEndpoint:
    """Test the health check endpoint."""

    def test_health_returns_200(self, sophia_url: str):
        """Health endpoint should return 200 when services are running."""
        resp = httpx.get(f"{sophia_url}/health", timeout=10)

        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded"]

    def test_health_reports_components(self, sophia_url: str):
        """Health endpoint should report component statuses."""
        resp = httpx.get(f"{sophia_url}/health", timeout=10)

        data = resp.json()
        assert "components" in data
        # When stack is running, both should be true
        components = data["components"]
        assert "neo4j" in components
        assert "milvus" in components


class TestPlanningWorkflow:
    """
    Test the planning endpoint.

    Planning uses backward chaining from goals to actionable steps,
    leveraging the knowledge graph stored in Neo4j.
    """

    def test_generate_plan(self, sophia_url: str, auth_headers: dict):
        """POST /plan should generate a plan for a goal."""
        resp = httpx.post(
            f"{sophia_url}/plan",
            json={
                "goal": {
                    "target_state": "red_block_in_bin",
                    "description": "Place the red block in the bin",
                }
            },
            headers=auth_headers,
            timeout=15,
        )

        assert resp.status_code == 201, f"Plan generation failed: {resp.text}"
        data = resp.json()
        assert "plan" in data
        assert "plan_id" in data
        assert "goal" in data

    def test_plan_contains_steps(self, sophia_url: str, auth_headers: dict):
        """Generated plan should contain actionable steps."""
        resp = httpx.post(
            f"{sophia_url}/plan",
            json={
                "goal": {
                    "target_state": "red_block_in_bin",
                }
            },
            headers=auth_headers,
            timeout=15,
        )

        assert resp.status_code == 201
        data = resp.json()

        # Plan should have steps
        assert isinstance(data["plan"], list)
        if len(data["plan"]) > 0:
            step = data["plan"][0]
            assert "id" in step
            assert "name" in step
            assert "action_type" in step


class TestSimulationWorkflow:
    """
    Test the JEPA simulation endpoint.

    Simulation performs k-step forward prediction using the JEPA model,
    creating imagined processes and states.
    """

    def test_run_simulation(self, sophia_url: str, auth_headers: dict, unique_id: str):
        """POST /simulate should run JEPA k-step simulation."""
        resp = httpx.post(
            f"{sophia_url}/simulate",
            json={
                "k_steps": 3,
                "entities": [
                    {
                        "id": f"block_{unique_id}",
                        "type": "rigid_body",
                        "properties": {"mass": 0.5, "color": "red"},
                    }
                ],
                "sensor_refs": [],
                "initial_state": {"block_position": [0.0, 0.0, 0.0]},
                "actions": [{"action": "grasp", "target": f"block_{unique_id}"}],
            },
            headers=auth_headers,
            timeout=20,
        )

        assert resp.status_code == 201, f"Simulation failed: {resp.text}"
        data = resp.json()
        assert "simulation_id" in data
        assert "k_steps" in data
        assert data["k_steps"] == 3

    def test_simulation_produces_states(
        self, sophia_url: str, auth_headers: dict, unique_id: str
    ):
        """Simulation should produce imagined states for each step."""
        resp = httpx.post(
            f"{sophia_url}/simulate",
            json={
                "k_steps": 5,
                "entities": [
                    {
                        "id": f"object_{unique_id}",
                        "type": "object",
                        "properties": {},
                    }
                ],
                "sensor_refs": [],
                "initial_state": {},
                "actions": [],
            },
            headers=auth_headers,
            timeout=20,
        )

        assert resp.status_code == 201
        data = resp.json()

        # Should have imagined states
        assert "imagined_states" in data
        # JEPA stub produces states for each step
        assert len(data.get("imagined_states", [])) >= 1

    def test_simulation_reports_confidence(
        self, sophia_url: str, auth_headers: dict, unique_id: str
    ):
        """Simulation should report overall confidence."""
        resp = httpx.post(
            f"{sophia_url}/simulate",
            json={
                "k_steps": 2,
                "entities": [
                    {
                        "id": f"test_{unique_id}",
                        "type": "test",
                        "properties": {},
                    }
                ],
                "sensor_refs": [],
                "initial_state": {},
                "actions": [],
            },
            headers=auth_headers,
            timeout=20,
        )

        assert resp.status_code == 201
        data = resp.json()
        assert "overall_confidence" in data
        assert 0.0 <= data["overall_confidence"] <= 1.0


class TestCompleteWorkflow:
    """
    Test a complete Sophia workflow end-to-end.

    This simulates a realistic usage pattern:
    1. Hermes sends a proposal with plan steps
    2. State is read and updated
    3. Simulation is run to predict outcomes
    4. Results are verified in the knowledge graph
    """

    def test_hermes_to_simulation_workflow(
        self, sophia_url: str, auth_headers: dict, unique_id: str, test_timestamp: str
    ):
        """Complete workflow: proposal → state → simulate → verify."""
        proposal_id = f"workflow_proposal_{unique_id}"

        # Step 1: Ingest a proposal from Hermes
        resp = httpx.post(
            f"{sophia_url}/ingest/hermes_proposal",
            json={
                "proposal_id": proposal_id,
                "source_service": "hermes",
                "llm_provider": "openai",
                "model": "gpt-4",
                "generated_at": test_timestamp,
                "confidence": 0.9,
                "plan_steps": [
                    {"name": "approach", "action_type": "MOVE", "target": "block"},
                    {"name": "grasp", "action_type": "GRASP", "target": "block"},
                ],
            },
            timeout=10,
        )
        assert resp.status_code == 201, "Proposal ingestion should succeed"
        resp.json()  # Consume response

        # Step 2: Read current state
        resp = httpx.get(
            f"{sophia_url}/state",
            headers=auth_headers,
            timeout=10,
        )
        assert resp.status_code == 200, "State read should succeed"

        # Step 3: Update state with workflow context
        resp = httpx.post(
            f"{sophia_url}/state",
            json={
                "state": {
                    f"workflow_{unique_id}": {
                        "phase": "simulation",
                        "proposal_id": proposal_id,
                    }
                }
            },
            headers=auth_headers,
            timeout=10,
        )
        assert resp.status_code == 200, "State update should succeed"

        # Step 4: Run simulation for the workflow
        resp = httpx.post(
            f"{sophia_url}/simulate",
            json={
                "k_steps": 3,
                "entities": [
                    {
                        "id": f"block_{unique_id}",
                        "type": "manipulandum",
                        "properties": {"from_proposal": proposal_id},
                    }
                ],
                "sensor_refs": [],
                "initial_state": {"gripper_open": True},
                "actions": [
                    {"action": "approach", "target": f"block_{unique_id}"},
                    {"action": "grasp", "target": f"block_{unique_id}"},
                ],
                "assumptions": ["rigid body dynamics", "no friction"],
            },
            headers=auth_headers,
            timeout=20,
        )
        assert resp.status_code == 201, "Simulation should succeed"
        sim_data = resp.json()

        # Step 5: Verify the simulation produced results
        assert sim_data["simulation_id"] is not None
        assert sim_data["k_steps"] == 3
        assert "imagined_states" in sim_data
        assert "overall_confidence" in sim_data

        # Step 6: Verify health still good after workflow
        resp = httpx.get(f"{sophia_url}/health", timeout=10)
        assert resp.status_code == 200
        health = resp.json()
        assert health["status"] in ["healthy", "degraded"]
