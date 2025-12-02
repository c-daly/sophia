"""Integration tests for media ingestion service with real Neo4j.

These tests require Neo4j to be running.
Run with: pytest tests/integration/test_media_ingestion_integration.py -v -m integration
"""

import io
import os
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from sophia.api.app import create_app


RUN_MEDIA_INTEGRATION = os.getenv("RUN_MEDIA_INTEGRATION") == "1"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not RUN_MEDIA_INTEGRATION,
        reason="Media integration tests require external Neo4j; set RUN_MEDIA_INTEGRATION=1 to enable.",
    ),
]


@pytest.fixture
def test_token():
    """Get test token for API auth."""
    return os.getenv("API_TOKEN", "dev-token")


@pytest.fixture
def client():
    """Create FastAPI test client with lifespan context."""
    # Set up environment for integration testing
    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_USER", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "neo4jtest")
    os.environ.setdefault("MEDIA_STORAGE_ROOT", "./test_media_storage")

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_image():
    """Create sample image for upload."""
    img = Image.new("RGB", (800, 600), color="blue")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer


class TestMediaIngestionIntegration:
    """Integration tests for media ingestion with real Neo4j."""

    def test_health_check_includes_neo4j(self, client):
        """Test that health check reports Neo4j status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "components" in data
        assert "neo4j" in data["components"]

    def test_ingest_media_creates_neo4j_node(self, client, sample_image, test_token):
        """Test that ingesting media creates a node in Neo4j."""
        response = client.post(
            "/ingest/media",
            files={"file": ("test_image.jpg", sample_image, "image/jpeg")},
            data={"media_type": "image"},
            headers={"Authorization": f"Bearer {test_token}"},
        )

        assert response.status_code == 201, (
            f"Expected 201, got {response.status_code}: {response.text}"
        )
        data = response.json()

        # Verify response structure
        assert "sample_id" in data
        assert data["media_type"] == "image"
        assert "file_path" in data
        assert data["file_size"] > 0
        assert "timestamp" in data
        assert "neo4j_node_id" in data

        # Store sample_id for later tests
        sample_id = data["sample_id"]

        # Verify we can retrieve it
        get_response = client.get(
            f"/media/samples/{sample_id}",
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert get_response.status_code == 200
        get_data = get_response.json()
        assert get_data["sample_id"] == sample_id
        assert get_data["media_type"] == "image"

    def test_ingest_media_with_question(self, client, sample_image, test_token):
        """Test ingesting media with a perception question."""
        response = client.post(
            "/ingest/media",
            files={"file": ("jump_test.jpg", sample_image, "image/jpeg")},
            data={
                "media_type": "image",
                "question": "Will this jump clear the obstacle?",
            },
            headers={"Authorization": f"Bearer {test_token}"},
        )

        assert response.status_code == 201
        data = response.json()
        assert "sample_id" in data
        # Question should be stored in metadata/node properties

    def test_list_media_samples(self, client, sample_image, test_token):
        """Test listing media samples with pagination."""
        # First, ingest a sample
        ingest_response = client.post(
            "/ingest/media",
            files={"file": ("list_test.jpg", sample_image, "image/jpeg")},
            data={"media_type": "image"},
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert ingest_response.status_code == 201

        # List samples
        list_response = client.get(
            "/media/samples",
            params={"limit": 10, "offset": 0},
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert list_response.status_code == 200
        data = list_response.json()

        assert "samples" in data
        assert "total" in data
        assert data["total"] > 0
        assert len(data["samples"]) > 0

    def test_list_media_samples_filter_by_type(self, client, sample_image, test_token):
        """Test filtering media samples by type."""
        # List only image samples
        response = client.get(
            "/media/samples",
            params={"media_type": "image", "limit": 10},
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert response.status_code == 200
        data = response.json()

        # All returned samples should be images
        for sample in data["samples"]:
            assert sample["media_type"] == "image"

    def test_media_metadata_extraction(self, client, sample_image, test_token):
        """Test that image metadata is correctly extracted."""
        response = client.post(
            "/ingest/media",
            files={"file": ("metadata_test.jpg", sample_image, "image/jpeg")},
            data={"media_type": "image"},
            headers={"Authorization": f"Bearer {test_token}"},
        )

        assert response.status_code == 201
        data = response.json()

        # Verify metadata extraction
        assert "metadata" in data
        metadata = data["metadata"]
        # For an 800x600 image
        assert metadata.get("width") == 800 or metadata.get("metadata_width") == 800
        assert metadata.get("height") == 600 or metadata.get("metadata_height") == 600


class TestMediaIngestionAuth:
    """Test authentication for media ingestion endpoints."""

    def test_ingest_requires_auth(self, client, sample_image):
        """Test that /ingest/media requires authentication."""
        response = client.post(
            "/ingest/media",
            files={"file": ("auth_test.jpg", sample_image, "image/jpeg")},
            data={"media_type": "image"},
            # No auth header
        )
        assert response.status_code in [401, 403]

    def test_list_samples_requires_auth(self, client):
        """Test that /media/samples requires authentication."""
        response = client.get("/media/samples")
        assert response.status_code in [401, 403]

    def test_get_sample_requires_auth(self, client):
        """Test that /media/samples/{id} requires authentication."""
        response = client.get("/media/samples/test-sample-id")
        assert response.status_code in [401, 403]


class TestMediaIngestionErrors:
    """Test error handling for media ingestion."""

    def test_invalid_media_type(self, client, sample_image, test_token):
        """Test that invalid media type is rejected."""
        response = client.post(
            "/ingest/media",
            files={"file": ("test.jpg", sample_image, "image/jpeg")},
            data={"media_type": "invalid_type"},
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert response.status_code == 422  # Validation error

    def test_wrong_extension_for_type(self, client, sample_image, test_token):
        """Test that wrong file extension for media type is rejected."""
        response = client.post(
            "/ingest/media",
            files={
                "file": ("test.mp4", sample_image, "video/mp4")
            },  # JPG content with MP4 name
            data={"media_type": "image"},  # Claiming it's an image
            headers={"Authorization": f"Bearer {test_token}"},
        )
        # Should reject because .mp4 is not valid for image type
        # Depending on validation order, could be 400 (validation) or 500 (processing error)
        assert response.status_code in [400, 500]

    def test_get_nonexistent_sample(self, client, test_token):
        """Test that getting non-existent sample returns 404."""
        response = client.get(
            "/media/samples/nonexistent-sample-id-12345",
            headers={"Authorization": f"Bearer {test_token}"},
        )
        assert response.status_code == 404
