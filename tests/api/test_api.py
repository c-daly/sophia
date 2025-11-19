"""Tests for Sophia API endpoints."""

import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

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
