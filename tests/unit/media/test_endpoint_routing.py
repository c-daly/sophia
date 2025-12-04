"""Unit tests for media API endpoint routing and validation.

These tests use mocks to verify endpoint behavior (auth, validation, error handling)
without requiring external services. They test the API layer in isolation.

For tests with real Neo4j/Milvus, see tests/integration/test_media_ingestion_integration.py
"""

import io
import os
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock
from PIL import Image
from fastapi.testclient import TestClient

from sophia.storage.media_storage import MediaStorageService
from sophia.models.media_models import (
    MediaType,
    MediaIngestResponse,
    MediaMetadata,
    MediaSampleResponse,
    MediaSamplesListResponse,
)
from sophia.api.app import create_app


pytestmark = pytest.mark.unit


# Fixtures


@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create temporary storage directory."""
    storage_dir = tmp_path / "media_storage"
    storage_dir.mkdir()
    return storage_dir


@pytest.fixture
def api_token():
    """Fixture for API token."""
    return "test-token-12345"


@pytest.fixture
def test_app(api_token, temp_storage_dir):
    """Create test FastAPI application."""
    os.environ["SOPHIA_API_TOKEN"] = api_token
    os.environ["MEDIA_STORAGE_ROOT"] = str(temp_storage_dir)
    return create_app()


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


@pytest.fixture
def auth_headers(api_token):
    """Create authentication headers."""
    return {"Authorization": f"Bearer {api_token}"}


@pytest.fixture
def sample_image_file():
    """Create sample image file in memory."""
    img = Image.new("RGB", (800, 600), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer


class TestMediaIngestEndpoint:
    """Tests for POST /ingest/media endpoint."""

    @patch("sophia.api.app._media_ingestion")
    @patch("sophia.api.app._hcg_client")
    def test_ingest_media_requires_authentication(
        self, mock_hcg, mock_ingestion, client
    ):
        """Test that /ingest/media requires authentication."""
        files = {"file": ("test.jpg", b"fake image data", "image/jpeg")}
        data = {"media_type": "image"}

        response = client.post("/ingest/media", files=files, data=data)
        assert response.status_code == 403

    @patch("sophia.api.app._media_ingestion")
    @patch("sophia.api.app._hcg_client")
    def test_ingest_media_rejects_invalid_token(self, mock_hcg, mock_ingestion, client):
        """Test that /ingest/media rejects invalid tokens."""
        files = {"file": ("test.jpg", b"fake image data", "image/jpeg")}
        data = {"media_type": "image"}
        headers = {"Authorization": "Bearer invalid-token"}

        response = client.post("/ingest/media", files=files, data=data, headers=headers)
        assert response.status_code == 403

    @patch("sophia.api.app._media_ingestion")
    @patch("sophia.api.app._hcg_client")
    def test_ingest_media_validates_media_type(
        self, mock_hcg, mock_ingestion, client, auth_headers
    ):
        """Test that /ingest/media validates media_type field."""
        files = {"file": ("test.jpg", b"fake image data", "image/jpeg")}
        data = {"media_type": "invalid_type"}

        response = client.post(
            "/ingest/media", files=files, data=data, headers=auth_headers
        )
        assert response.status_code == 422

    @patch("sophia.api.app._media_ingestion")
    @patch("sophia.api.app._hcg_client")
    def test_ingest_media_success(
        self, mock_hcg, mock_ingestion, client, auth_headers, sample_image_file
    ):
        """Test successful media ingestion."""
        # Mock the ingestion service response
        mock_response = MediaIngestResponse(
            sample_id="test_sample_123",
            media_type=MediaType.IMAGE,
            file_path="/app/media_storage/image/test_sample_123.jpg",
            file_size=1024,
            timestamp=datetime.now(timezone.utc),
            metadata={"width": 800, "height": 600, "format": "JPEG"},
            neo4j_node_id="node_456",
        )
        mock_ingestion.ingest_media = AsyncMock(return_value=mock_response)

        sample_image_file.seek(0)
        files = {"file": ("test.jpg", sample_image_file.read(), "image/jpeg")}
        data = {"media_type": "image", "question": "What is in this image?"}

        response = client.post(
            "/ingest/media", files=files, data=data, headers=auth_headers
        )

        assert response.status_code == 201
        result = response.json()
        assert result["sample_id"] == "test_sample_123"
        assert result["media_type"] == "image"
        assert result["file_size"] == 1024
        assert "metadata" in result

    @patch("sophia.api.app._media_ingestion")
    @patch("sophia.api.app._hcg_client")
    def test_ingest_media_handles_validation_error(
        self, mock_hcg, mock_ingestion, client, auth_headers
    ):
        """Test that /ingest/media handles validation errors."""
        mock_ingestion.ingest_media = AsyncMock(
            side_effect=ValueError("Invalid file extension")
        )

        files = {"file": ("test.txt", b"not an image", "text/plain")}
        data = {"media_type": "image"}

        response = client.post(
            "/ingest/media", files=files, data=data, headers=auth_headers
        )

        assert response.status_code == 400
        assert "Invalid file extension" in response.json()["detail"]

    @patch("sophia.api.app._media_ingestion")
    def test_ingest_media_returns_503_when_service_unavailable(
        self, mock_ingestion, client, auth_headers
    ):
        """Test that /ingest/media returns 503 when service is unavailable."""
        with patch("sophia.api.app._media_ingestion", None):
            files = {"file": ("test.jpg", b"fake image", "image/jpeg")}
            data = {"media_type": "image"}

            response = client.post(
                "/ingest/media", files=files, data=data, headers=auth_headers
            )

            assert response.status_code == 503
            assert "not available" in response.json()["detail"]


class TestMediaSamplesListEndpoint:
    """Tests for GET /media/samples endpoint."""

    @patch("sophia.api.app._media_ingestion")
    @patch("sophia.api.app._hcg_client")
    def test_list_samples_requires_authentication(
        self, mock_hcg, mock_ingestion, client
    ):
        """Test that /media/samples requires authentication."""
        response = client.get("/media/samples")
        assert response.status_code == 403

    @patch("sophia.api.app._media_ingestion")
    @patch("sophia.api.app._hcg_client")
    def test_list_samples_returns_paginated_results(
        self, mock_hcg, mock_ingestion, client, auth_headers
    ):
        """Test that /media/samples returns paginated results."""
        # Mock the response
        mock_samples = MediaSamplesListResponse(
            samples=[
                MediaSampleResponse(
                    sample_id="sample1",
                    media_type=MediaType.IMAGE,
                    file_path="/path/to/sample1.jpg",
                    file_size=1024,
                    file_hash="hash1",
                    timestamp=datetime.now(timezone.utc),
                    neo4j_node_id="node1",
                    simulation_count=2,
                    metadata=MediaMetadata(width=800, height=600, format="JPEG"),
                )
            ],
            total=1,
            limit=50,
            offset=0,
        )
        mock_ingestion.list_media_samples.return_value = mock_samples

        response = client.get("/media/samples", headers=auth_headers)

        assert response.status_code == 200
        result = response.json()
        assert "samples" in result
        assert "total" in result
        assert len(result["samples"]) == 1

    @patch("sophia.api.app._media_ingestion")
    @patch("sophia.api.app._hcg_client")
    def test_list_samples_supports_filtering(
        self, mock_hcg, mock_ingestion, client, auth_headers
    ):
        """Test that /media/samples supports filtering by media_type."""
        mock_ingestion.list_media_samples.return_value = MediaSamplesListResponse(
            samples=[], total=0, limit=50, offset=0
        )

        response = client.get(
            "/media/samples",
            params={"media_type": "video", "limit": 10, "offset": 5},
            headers=auth_headers,
        )

        assert response.status_code == 200


class TestMediaSampleDetailEndpoint:
    """Tests for GET /media/samples/{sample_id} endpoint."""

    @patch("sophia.api.app._media_ingestion")
    @patch("sophia.api.app._hcg_client")
    def test_get_sample_requires_authentication(self, mock_hcg, mock_ingestion, client):
        """Test that /media/samples/{id} requires authentication."""
        response = client.get("/media/samples/sample123")
        assert response.status_code == 403

    @patch("sophia.api.app._media_ingestion")
    @patch("sophia.api.app._hcg_client")
    def test_get_sample_returns_sample_details(
        self, mock_hcg, mock_ingestion, client, auth_headers
    ):
        """Test that /media/samples/{id} returns sample details."""
        mock_sample = MediaSampleResponse(
            sample_id="sample123",
            media_type=MediaType.IMAGE,
            file_path="/path/to/sample123.jpg",
            file_size=2048,
            file_hash="hash123",
            timestamp=datetime.now(timezone.utc),
            neo4j_node_id="node123",
            simulation_count=3,
            metadata=MediaMetadata(width=1920, height=1080, format="PNG"),
        )
        mock_ingestion.get_media_sample.return_value = mock_sample

        response = client.get("/media/samples/sample123", headers=auth_headers)

        assert response.status_code == 200
        result = response.json()
        assert result["sample_id"] == "sample123"
        assert result["simulation_count"] == 3

    @patch("sophia.api.app._media_ingestion")
    @patch("sophia.api.app._hcg_client")
    def test_get_sample_returns_404_when_not_found(
        self, mock_hcg, mock_ingestion, client, auth_headers
    ):
        """Test that /media/samples/{id} returns 404 for non-existent sample."""
        mock_ingestion.get_media_sample.return_value = None

        response = client.get("/media/samples/nonexistent", headers=auth_headers)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
