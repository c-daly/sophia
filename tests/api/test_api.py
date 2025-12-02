"""Tests for Sophia API endpoints."""

import os
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from logos_sophia_sdk.models.plan_request import PlanRequest as SDKPlanRequest
from logos_sophia_sdk.models.plan_request_goal import PlanRequestGoal

from sophia.api.app import create_app


# Mock the authentication token
@pytest.fixture
def api_token():
    """Fixture for API token."""
    return "test-token-12345"


@pytest.fixture
def app(api_token):
    """Create test application."""
    os.environ["SOPHIA_API_TOKEN"] = api_token
    return create_app()


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def auth_headers(api_token):
    """Create authentication headers."""
    return {"Authorization": f"Bearer {api_token}"}


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check_no_auth_required(self, client):
        """Test that health endpoint doesn't require authentication."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data
        assert "version" in data

    def test_health_check_returns_component_status(self, client):
        """Test that health check returns component statuses."""
        response = client.get("/health")
        data = response.json()
        assert "neo4j" in data["components"]
        assert "milvus" in data["components"]


class TestPlanEndpoint:
    """Tests for the /plan endpoint."""

    def test_plan_requires_authentication(self, client):
        """Test that /plan requires authentication."""
        response = client.post(
            "/plan",
            json={"goal": {"description": "test", "target_state": "test_state"}},
        )
        assert response.status_code == 403

    def test_plan_rejects_invalid_token(self, client):
        """Test that /plan rejects invalid tokens."""
        response = client.post(
            "/plan",
            json={"goal": {"description": "test", "target_state": "test_state"}},
            headers={"Authorization": "Bearer invalid-token"},
        )
        assert response.status_code == 403

    def test_plan_accepts_valid_token(self, client, auth_headers):
        """Test that /plan accepts valid token."""
        response = client.post(
            "/plan",
            json={"goal": {"description": "test", "target_state": "test_state"}},
            headers=auth_headers,
        )
        # May be 201 or 500 depending on planner state, but not 403
        assert response.status_code != 403

    def test_plan_validates_request_body(self, client, auth_headers):
        """Test that /plan validates request body."""
        # Missing required field
        response = client.post(
            "/plan",
            json={},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_plan_returns_plan_response(self, client, auth_headers):
        """Test that /plan returns proper response structure."""
        response = client.post(
            "/plan",
            json={"goal": {"description": "test", "target_state": "test_state"}},
            headers=auth_headers,
        )

        # If successful, check response structure
        if response.status_code == 201:
            data = response.json()
            assert "plan" in data
            assert "goal" in data
            assert "plan_id" in data
            assert "created_at" in data
            assert isinstance(data["plan"], list)

    def test_plan_sdk_request_schema(self, client, auth_headers):
        """Ensure shared SDK payload uses PlanRequestGoal for goal."""
        sdk_request = SDKPlanRequest(
            goal=PlanRequestGoal(
                description="Place red block in bin",
                target_state="block_in_bin",
            )
        )
        response = client.post(
            "/plan",
            json=sdk_request.to_dict(),
            headers=auth_headers,
        )
        # The SDK payload should now be accepted and move further into planning
        assert response.status_code != 422


class TestImagineEndpoint:
    """Tests for the /imagine endpoint."""

    def test_imagine_requires_authentication(self, client):
        """Test that /imagine requires authentication."""
        response = client.post(
            "/imagine",
            json={},
        )
        assert response.status_code == 403

    def test_imagine_rejects_invalid_token(self, client):
        """Test that /imagine rejects invalid tokens."""
        response = client.post(
            "/imagine",
            json={},
            headers={"Authorization": "Bearer invalid-token"},
        )
        assert response.status_code == 403

    def test_imagine_accepts_valid_token(self, client, auth_headers):
        """Test that /imagine accepts valid token."""
        response = client.post(
            "/imagine",
            json={},
            headers=auth_headers,
        )
        # May be 201 or 503 depending on HCG state, but not 403
        assert response.status_code != 403

    def test_imagine_validates_horizon(self, client, auth_headers):
        """Test that /imagine validates horizon parameter."""
        response = client.post(
            "/imagine",
            json={"horizon": 0},  # Invalid: must be > 0
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_imagine_accepts_optional_fields(self, client, auth_headers):
        """Test that /imagine accepts optional fields."""
        response = client.post(
            "/imagine",
            json={
                "cwm_g_imagery": [{"type": "visual"}],
                "cwm_e_emotion_tags": ["curious"],
                "context": {"key": "value"},
                "model_version": "v2.0",
                "horizon": 3,
                "assumptions": ["test assumption"],
            },
            headers=auth_headers,
        )
        # May be 201 or 503, but not 403 or 422
        assert response.status_code not in [403, 422]

    def test_imagine_returns_imagined_states(self, client, auth_headers):
        """Test that /imagine returns proper response structure."""
        # Mock HCG client to avoid dependency on live database
        with patch("sophia.api.app._hcg_client") as mock_hcg:
            mock_hcg.add_node = Mock()

            response = client.post(
                "/imagine",
                json={"horizon": 2},
                headers=auth_headers,
            )

            if response.status_code == 201:
                data = response.json()
                assert "imagined_states" in data
                assert "imagination_id" in data
                assert "model_version" in data
                assert "horizon" in data
                assert "assumptions" in data
                assert "created_at" in data
                assert isinstance(data["imagined_states"], list)
                assert len(data["imagined_states"]) == 2


class TestExecuteEndpoint:
    """Tests for the /execute endpoint."""

    def test_execute_requires_authentication(self, client):
        """Test that /execute requires authentication."""
        response = client.post(
            "/execute",
            json={"plan_id": "test-plan-id"},
        )
        assert response.status_code == 403

    def test_execute_rejects_invalid_token(self, client):
        """Test that /execute rejects invalid tokens."""
        response = client.post(
            "/execute",
            json={"plan_id": "test-plan-id"},
            headers={"Authorization": "Bearer invalid-token"},
        )
        assert response.status_code == 403

    def test_execute_accepts_valid_token(self, client, auth_headers):
        """Test that /execute accepts valid token."""
        response = client.post(
            "/execute",
            json={"plan_id": "test-plan-id"},
            headers=auth_headers,
        )
        # May be 201 or 503, but not 403
        assert response.status_code != 403

    def test_execute_validates_request_body(self, client, auth_headers):
        """Test that /execute validates request body."""
        # Missing required field
        response = client.post(
            "/execute",
            json={},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_execute_accepts_optional_fields(self, client, auth_headers):
        """Test that /execute accepts optional fields."""
        response = client.post(
            "/execute",
            json={
                "plan_id": "test-plan-id",
                "step_index": 0,
                "dry_run": True,
            },
            headers=auth_headers,
        )
        # May be 201 or 503, but not 403 or 422
        assert response.status_code not in [403, 422]

    def test_execute_returns_execution_response(self, client, auth_headers):
        """Test that /execute returns proper response structure."""
        response = client.post(
            "/execute",
            json={"plan_id": "test-plan-id", "dry_run": True},
            headers=auth_headers,
        )

        if response.status_code == 201:
            data = response.json()
            assert "plan_id" in data
            assert "results" in data
            assert "overall_status" in data
            assert "execution_id" in data
            assert "created_at" in data
            assert isinstance(data["results"], list)


class TestStateEndpoint:
    """Tests for the /state endpoint."""

    def test_get_state_requires_authentication(self, client):
        """Test that GET /state requires authentication."""
        response = client.get("/state")
        assert response.status_code == 403

    def test_get_state_rejects_invalid_token(self, client):
        """Test that GET /state rejects invalid tokens."""
        response = client.get(
            "/state",
            headers={"Authorization": "Bearer invalid-token"},
        )
        assert response.status_code == 403

    def test_get_state_accepts_valid_token(self, client, auth_headers):
        """Test that GET /state accepts valid token."""
        response = client.get("/state", headers=auth_headers)
        # May be 200 or 503 (if HCG unavailable), but not 403
        assert response.status_code != 403

    def test_get_state_returns_state_response(self, client, auth_headers):
        """Test that GET /state returns proper response structure."""
        response = client.get("/state", headers=auth_headers)
        if response.status_code == 200:
            data = response.json()
            assert "state" in data
            assert "state_id" in data
            assert "timestamp" in data
            assert isinstance(data["state"], dict)

    def test_post_state_requires_authentication(self, client):
        """Test that POST /state requires authentication."""
        response = client.post(
            "/state",
            json={"state": {"test": "value"}},
        )
        assert response.status_code == 403

    def test_post_state_rejects_invalid_token(self, client):
        """Test that POST /state rejects invalid tokens."""
        response = client.post(
            "/state",
            json={"state": {"test": "value"}},
            headers={"Authorization": "Bearer invalid-token"},
        )
        assert response.status_code == 403

    def test_post_state_validates_request_body(self, client, auth_headers):
        """Test that POST /state validates request body."""
        # Missing required field
        response = client.post(
            "/state",
            json={},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_post_state_accepts_valid_state(self, client, auth_headers):
        """Test that POST /state accepts valid state updates."""
        response = client.post(
            "/state",
            json={
                "state": {
                    "red_block": {"location": "table", "grasped": False},
                    "gripper": {"position": "home", "holding": None},
                }
            },
            headers=auth_headers,
        )
        # May be 200, 422 (validation failure), or 503 (HCG unavailable), but not 403
        assert response.status_code != 403

    def test_post_state_returns_update_response(self, client, auth_headers):
        """Test that POST /state returns proper response structure."""
        response = client.post(
            "/state",
            json={
                "state": {
                    "red_block": {"location": "bin", "grasped": False},
                }
            },
            headers=auth_headers,
        )
        if response.status_code == 200:
            data = response.json()
            assert "state_id" in data
            assert "updated_at" in data
            assert "validation_passed" in data


class TestAPIDocumentation:
    """Tests for API documentation endpoints."""

    def test_openapi_schema_available(self, client):
        """Test that OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema

    def test_swagger_ui_available(self, client):
        """Test that Swagger UI is available."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "swagger" in response.text.lower()

    def test_redoc_available(self, client):
        """Test that ReDoc is available."""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "redoc" in response.text.lower()


class TestSimulateEndpoint:
    """Tests for the /simulate endpoint."""

    def test_simulate_requires_authentication(self, client):
        """Test that /simulate requires authentication."""
        response = client.post("/simulate", json={})
        assert response.status_code == 403

    def test_simulate_rejects_invalid_token(self, client):
        """Test that /simulate rejects invalid tokens."""
        headers = {"Authorization": "Bearer invalid-token"}
        response = client.post("/simulate", headers=headers, json={})
        assert response.status_code == 403

    def test_simulate_validates_request_body(self, client, auth_headers):
        """Test that /simulate validates request body."""
        # Missing required 'entities' field
        response = client.post("/simulate", headers=auth_headers, json={})
        assert response.status_code == 422

    def test_simulate_accepts_valid_request(self, client, auth_headers):
        """Test that /simulate accepts valid request."""
        request_data = {
            "entities": [
                {
                    "id": "test_obj",
                    "type": "object",
                    "properties": {"mass": 1.0},
                    "position": {"x": 0.0, "y": 0.0, "z": 0.1},
                }
            ],
            "k_steps": 3,
        }
        response = client.post("/simulate", headers=auth_headers, json=request_data)
        assert response.status_code in [201, 503]  # 503 if HCG not available

    def test_simulate_returns_simulation_response(self, client, auth_headers):
        """Test that /simulate returns proper response structure."""
        request_data = {
            "entities": [
                {
                    "id": "test_obj",
                    "type": "object",
                    "properties": {},
                }
            ],
            "k_steps": 2,
        }
        response = client.post("/simulate", headers=auth_headers, json=request_data)

        if response.status_code == 201:
            data = response.json()
            assert "simulation_id" in data
            assert "imagined_processes" in data
            assert "imagined_states" in data
            assert "k_steps" in data
            assert data["k_steps"] == 2
            assert "model_version" in data
            assert "overall_confidence" in data
            assert "created_at" in data

    def test_simulate_with_actions(self, client, auth_headers):
        """Test /simulate with action sequence."""
        request_data = {
            "entities": [
                {"id": "robot", "type": "agent", "properties": {"status": "idle"}}
            ],
            "actions": [
                {"type": "MOVE", "target": "robot"},
                {"type": "GRASP", "target": "robot"},
            ],
            "k_steps": 2,
        }
        response = client.post("/simulate", headers=auth_headers, json=request_data)
        assert response.status_code in [201, 503]

    def test_simulate_with_sensors(self, client, auth_headers):
        """Test /simulate with sensor references."""
        request_data = {
            "entities": [{"id": "obj", "type": "object"}],
            "sensor_refs": [
                {
                    "sensor_id": "camera_1",
                    "sensor_type": "camera",
                    "frame_id": "base_link",
                }
            ],
            "k_steps": 2,
        }
        response = client.post("/simulate", headers=auth_headers, json=request_data)
        assert response.status_code in [201, 503]

    def test_simulate_with_talos_metadata(self, client, auth_headers):
        """Test /simulate with Talos metadata."""
        request_data = {
            "entities": [{"id": "obj", "type": "object"}],
            "talos_metadata": {
                "simulator_version": "talos-v2.0",
                "physics_engine": "ODE",
                "use_hardware": True,
            },
            "k_steps": 2,
        }
        response = client.post("/simulate", headers=auth_headers, json=request_data)
        assert response.status_code in [201, 503]

    def test_simulate_validates_k_steps_range(self, client, auth_headers):
        """Test that /simulate validates k_steps parameter."""
        # k_steps must be > 0
        request_data = {"entities": [{"id": "obj", "type": "object"}], "k_steps": 0}
        response = client.post("/simulate", headers=auth_headers, json=request_data)
        assert response.status_code == 422

        # k_steps must be <= 100
        request_data["k_steps"] = 101
        response = client.post("/simulate", headers=auth_headers, json=request_data)
        assert response.status_code == 422
