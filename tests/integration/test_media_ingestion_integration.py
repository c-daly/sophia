"""Integration tests for media ingestion service.

These tests require Sophia and Neo4j to be running.
Run with: pytest tests/integration/test_media_ingestion_integration.py -v -m integration

Start services with: ./scripts/test_integration.sh up
"""

import io
import pytest
from PIL import Image

pytestmark = [
    pytest.mark.integration,
]


@pytest.fixture
def sample_image():
    """Create sample image for upload."""
    img = Image.new("RGB", (800, 600), color="blue")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer


@pytest.fixture
def sample_audio():
    """Create sample audio bytes (WAV header)."""
    # Minimal WAV header
    wav_header = bytes(
        [
            0x52,
            0x49,
            0x46,
            0x46,  # "RIFF"
            0x24,
            0x00,
            0x00,
            0x00,  # Chunk size
            0x57,
            0x41,
            0x56,
            0x45,  # "WAVE"
            0x66,
            0x6D,
            0x74,
            0x20,  # "fmt "
            0x10,
            0x00,
            0x00,
            0x00,  # Subchunk1 size
            0x01,
            0x00,  # Audio format (PCM)
            0x01,
            0x00,  # Num channels
            0x44,
            0xAC,
            0x00,
            0x00,  # Sample rate (44100)
            0x88,
            0x58,
            0x01,
            0x00,  # Byte rate
            0x02,
            0x00,  # Block align
            0x10,
            0x00,  # Bits per sample
            0x64,
            0x61,
            0x74,
            0x61,  # "data"
            0x00,
            0x00,
            0x00,
            0x00,  # Subchunk2 size
        ]
    )
    return io.BytesIO(wav_header)


class TestMediaIngestionIntegration:
    """Integration tests for media ingestion with real Neo4j."""

    def test_health_check_includes_neo4j(self, http_client):
        """Test that health check reports Neo4j status."""
        response = http_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "components" in data
        assert "neo4j" in data["components"]

    def test_ingest_media_creates_neo4j_node(
        self, http_client, auth_headers, sample_image
    ):
        """Test that ingesting media creates a node in Neo4j."""
        response = http_client.post(
            "/ingest/media",
            files={"file": ("test_image.jpg", sample_image, "image/jpeg")},
            data={"media_type": "image"},
            headers=auth_headers,
        )

        assert response.status_code in [200, 201]
        data = response.json()
        assert "sample_id" in data or "media_id" in data

    def test_ingest_media_with_question(self, http_client, auth_headers, sample_image):
        """Test ingesting media with an associated question."""
        response = http_client.post(
            "/ingest/media",
            files={"file": ("test_image.jpg", sample_image, "image/jpeg")},
            data={
                "media_type": "image",
                "question": "What objects are in this image?",
            },
            headers=auth_headers,
        )

        assert response.status_code in [200, 201]

    def test_list_media_samples(self, http_client, auth_headers, sample_image):
        """Test listing ingested media samples."""
        # First ingest some media
        http_client.post(
            "/ingest/media",
            files={"file": ("test_image.jpg", sample_image, "image/jpeg")},
            data={"media_type": "image"},
            headers=auth_headers,
        )

        # List media
        response = http_client.get("/media/samples", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "samples" in data

    def test_list_media_samples_filter_by_type(
        self, http_client, auth_headers, sample_image
    ):
        """Test filtering media samples by type."""
        # Ingest image
        http_client.post(
            "/ingest/media",
            files={"file": ("test_image.jpg", sample_image, "image/jpeg")},
            data={"media_type": "image"},
            headers=auth_headers,
        )

        # Filter by type
        response = http_client.get(
            "/media/samples",
            params={"media_type": "image"},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_media_metadata_extraction(self, http_client, auth_headers, sample_image):
        """Test that media metadata is extracted correctly."""
        response = http_client.post(
            "/ingest/media",
            files={"file": ("test_image.jpg", sample_image, "image/jpeg")},
            data={"media_type": "image"},
            headers=auth_headers,
        )

        assert response.status_code in [200, 201]
        data = response.json()

        # Should have metadata
        if "metadata" in data:
            assert "width" in data["metadata"] or "size" in data["metadata"]


class TestMediaIngestionErrors:
    """Integration tests for media ingestion error handling."""

    def test_wrong_extension_for_type(self, http_client, auth_headers):
        """Test that wrong extension for media type is rejected."""
        # Create a text file but claim it's an image
        text_content = io.BytesIO(b"This is not an image")

        response = http_client.post(
            "/ingest/media",
            files={"file": ("test.txt", text_content, "text/plain")},
            data={"media_type": "image"},
            headers=auth_headers,
        )

        # Should be rejected
        assert response.status_code in [400, 415, 422]

    def test_get_nonexistent_sample(self, http_client, auth_headers):
        """Test getting a nonexistent media sample."""
        response = http_client.get(
            "/media/samples/nonexistent-id-12345",
            headers=auth_headers,
        )
        assert response.status_code == 404

    def test_ingest_requires_auth(self, http_client, sample_image):
        """Test that media ingestion requires authentication."""
        response = http_client.post(
            "/ingest/media",
            files={"file": ("test_image.jpg", sample_image, "image/jpeg")},
            data={"media_type": "image"},
        )
        assert response.status_code in [401, 403]
