"""Tests for media ingestion functionality."""

import io
import os
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from PIL import Image
from fastapi.testclient import TestClient
from fastapi import HTTPException

from sophia.storage.media_storage import MediaStorageService
from sophia.ingestion.media_service import MediaIngestionService
from sophia.api.media_models import (
    MediaType,
    MediaIngestResponse,
    MediaMetadata,
    MediaSampleQuery,
    MediaSampleResponse,
    MediaSamplesListResponse,
)
from sophia.api.app import create_app


# Fixtures


@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create temporary storage directory."""
    storage_dir = tmp_path / "media_storage"
    storage_dir.mkdir()
    return storage_dir


@pytest.fixture
def media_storage_service(temp_storage_dir):
    """Create MediaStorageService instance."""
    return MediaStorageService(storage_root=str(temp_storage_dir))


@pytest.fixture
def mock_hcg_client():
    """Create mock HCG client."""
    client = Mock()
    client.add_node = Mock(return_value="node123")
    client.get_node = Mock(
        return_value={
            "sample_id": "sample123",
            "type": "media_sample",
            "media_type": "image",
            "file_path": "/path/to/file.jpg",
            "file_size": 1024,
            "file_hash": "abc123",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "metadata_width": 800,
            "metadata_height": 600,
            "metadata_format": "JPEG",
        }
    )
    client.add_edge = Mock()
    client.driver = Mock()

    # Mock Neo4j session for Cypher queries
    mock_session = Mock()
    mock_result = Mock()
    mock_result.single.return_value = {"count": 2}
    mock_session.run.return_value = mock_result
    client.driver.session.return_value.__enter__.return_value = mock_session

    return client


@pytest.fixture
def media_ingestion_service(mock_hcg_client, media_storage_service):
    """Create MediaIngestionService instance."""
    return MediaIngestionService(
        hcg_client=mock_hcg_client,
        storage_service=media_storage_service,
    )


@pytest.fixture
def sample_image_file():
    """Create sample image file in memory."""
    img = Image.new("RGB", (800, 600), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer


@pytest.fixture
def mock_upload_file(sample_image_file):
    """Create mock UploadFile."""
    upload_file = Mock()
    upload_file.filename = "test_image.jpg"
    upload_file.read = AsyncMock(return_value=sample_image_file.read())
    return upload_file


@pytest.fixture
def api_token():
    """Fixture for API token."""
    return "test-token-12345"


@pytest.fixture
def test_app(api_token, temp_storage_dir):
    """Create test FastAPI application."""
    os.environ["SOPHIA_API_TOKEN"] = api_token
    os.environ["MEDIA_STORAGE_ROOT"] = str(temp_storage_dir)
    # Disable Neo4j/Milvus for unit tests
    os.environ["NEO4J_URI"] = "bolt://mock:7687"
    return create_app()


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


@pytest.fixture
def auth_headers(api_token):
    """Create authentication headers."""
    return {"Authorization": f"Bearer {api_token}"}


# MediaStorageService Tests


def test_validate_file_valid_image(media_storage_service):
    """Test validation of valid image file."""
    mock_file = Mock()
    mock_file.filename = "test.jpg"
    # Should not raise
    media_storage_service.validate_file(mock_file, MediaType.IMAGE)

    mock_file.filename = "test.png"
    media_storage_service.validate_file(mock_file, MediaType.IMAGE)


def test_validate_file_valid_video(media_storage_service):
    """Test validation of valid video file."""
    mock_file = Mock()
    mock_file.filename = "test.mp4"
    # Should not raise
    media_storage_service.validate_file(mock_file, MediaType.VIDEO)

    mock_file.filename = "test.avi"
    media_storage_service.validate_file(mock_file, MediaType.VIDEO)


def test_validate_file_valid_audio(media_storage_service):
    """Test validation of valid audio file."""
    mock_file = Mock()
    mock_file.filename = "test.mp3"
    # Should not raise
    media_storage_service.validate_file(mock_file, MediaType.AUDIO)

    mock_file.filename = "test.wav"
    media_storage_service.validate_file(mock_file, MediaType.AUDIO)


def test_validate_file_invalid_extension(media_storage_service):
    """Test validation rejects invalid file extension."""
    mock_file = Mock()
    mock_file.filename = "test.txt"
    with pytest.raises(HTTPException) as exc_info:
        media_storage_service.validate_file(mock_file, MediaType.IMAGE)
    assert exc_info.value.status_code == 400
    assert "Invalid file extension" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_store_file(media_storage_service, mock_upload_file):
    """Test file storage to disk."""
    sample_id, file_path, file_size = await media_storage_service.store_file(
        mock_upload_file, MediaType.IMAGE
    )

    assert sample_id is not None
    assert "image" in file_path
    assert file_path.endswith(".jpg")
    assert file_size > 0
    assert Path(file_path).exists()


def test_extract_image_metadata(
    media_storage_service, sample_image_file, temp_storage_dir
):
    """Test metadata extraction from image."""
    # Save test image
    image_path = temp_storage_dir / "test_image.jpg"
    with open(image_path, "wb") as f:
        f.write(sample_image_file.read())

    metadata = media_storage_service.extract_metadata(str(image_path), MediaType.IMAGE)

    assert metadata.width == 800
    assert metadata.height == 600
    assert metadata.format == "JPEG"


def test_extract_video_metadata_stub(media_storage_service, temp_storage_dir):
    """Test video metadata extraction returns stub."""
    video_path = temp_storage_dir / "test.mp4"
    video_path.touch()

    metadata = media_storage_service.extract_metadata(str(video_path), MediaType.VIDEO)

    # Stub should return empty metadata
    assert metadata.width is None
    assert metadata.duration_seconds is None


def test_compute_file_hash(media_storage_service, temp_storage_dir):
    """Test SHA256 hash computation."""
    test_file = temp_storage_dir / "test.txt"
    test_file.write_text("hello world")

    hash1 = media_storage_service.compute_file_hash(str(test_file))
    hash2 = media_storage_service.compute_file_hash(str(test_file))

    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 hex digest length


# MediaIngestionService Tests


@pytest.mark.asyncio
async def test_ingest_media(media_ingestion_service, mock_upload_file, mock_hcg_client):
    """Test full media ingestion flow."""
    result = await media_ingestion_service.ingest_media(
        file=mock_upload_file,
        media_type=MediaType.IMAGE,
        question="What objects are in this image?",
    )

    assert isinstance(result, MediaIngestResponse)
    assert result.sample_id is not None
    assert result.media_type == MediaType.IMAGE
    assert result.file_path is not None
    assert result.file_size > 0
    assert result.neo4j_node_id == "node123"

    # Verify HCG client was called
    mock_hcg_client.add_node.assert_called_once()
    call_args = mock_hcg_client.add_node.call_args
    assert call_args[1]["node_type"] == "media_sample"
    assert call_args[1]["properties"]["question"] == "What objects are in this image?"


@pytest.mark.asyncio
async def test_get_media_sample(media_ingestion_service, mock_hcg_client):
    """Test retrieving media sample by ID."""
    result = media_ingestion_service.get_media_sample("sample123")

    assert isinstance(result, MediaSampleResponse)
    assert result.sample_id == "sample123"
    assert result.media_type == MediaType.IMAGE
    assert result.simulation_count == 2
    assert result.metadata.width == 800
    assert result.metadata.height == 600


@pytest.mark.asyncio
async def test_get_media_sample_not_found(media_ingestion_service, mock_hcg_client):
    """Test retrieving non-existent media sample."""
    mock_hcg_client.get_node.return_value = None

    result = media_ingestion_service.get_media_sample("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_list_media_samples(media_ingestion_service, mock_hcg_client):
    """Test listing media samples with filters."""
    # Mock Cypher query results
    mock_records = [
        {
            "n": {
                "sample_id": f"sample{i}",
                "type": "media_sample",
                "media_type": "image",
                "file_path": f"/path/to/sample{i}.jpg",
                "file_size": 1024 * i,
                "file_hash": f"hash{i}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        }
        for i in range(1, 4)
    ]

    mock_session = Mock()
    mock_result = Mock()
    mock_result.data.return_value = mock_records
    mock_session.run.return_value = mock_result
    mock_hcg_client.driver.session.return_value.__enter__.return_value = mock_session

    query = MediaSampleQuery(
        media_type=MediaType.IMAGE,
        limit=10,
        offset=0,
    )

    result = media_ingestion_service.list_media_samples(query)

    assert isinstance(result, MediaSamplesListResponse)
    assert len(result.samples) == 3
    assert result.total == 3
    assert all(s.media_type == MediaType.IMAGE for s in result.samples)


def test_link_sample_to_simulation(media_ingestion_service, mock_hcg_client):
    """Test linking media sample to simulation."""
    success = media_ingestion_service.link_sample_to_simulation(
        sample_id="sample123",
        simulation_id="sim456",
    )

    assert success is True
    mock_hcg_client.add_edge.assert_called_once()
    call_args = mock_hcg_client.add_edge.call_args
    assert call_args[1]["source_id"] == "sample123"
    assert call_args[1]["target_id"] == "sim456"
    assert call_args[1]["relation"] == "used_in"


# Integration Tests


@pytest.mark.asyncio
async def test_end_to_end_image_ingestion(media_ingestion_service, sample_image_file):
    """Test complete image ingestion workflow."""
    # Create mock upload file
    upload_file = Mock()
    upload_file.filename = "test_photo.jpg"

    async def mock_read():
        sample_image_file.seek(0)
        return sample_image_file.read()

    upload_file.read = mock_read

    # Ingest
    result = await media_ingestion_service.ingest_media(
        file=upload_file,
        media_type=MediaType.IMAGE,
        question="Describe this scene",
    )

    # Verify ingestion response
    assert result.media_type == MediaType.IMAGE
    assert result.metadata["width"] == 800
    assert result.metadata["height"] == 600
    assert result.metadata["format"] == "JPEG"

    # Verify file was stored
    assert Path(result.file_path).exists()

    # Verify can retrieve
    retrieved = media_ingestion_service.get_media_sample(result.sample_id)
    assert retrieved.sample_id == result.sample_id
    assert retrieved.metadata.width == 800


@pytest.mark.asyncio
async def test_jepa_notification_hook(media_ingestion_service, mock_upload_file):
    """Test JEPA runner notification on ingestion."""
    mock_jepa = Mock()
    media_ingestion_service.jepa_runner = mock_jepa

    await media_ingestion_service.ingest_media(
        file=mock_upload_file,
        media_type=MediaType.IMAGE,
    )

    # For now, JEPA hook is a no-op that logs
    # In future: assert mock_jepa.process_media_sample.called


# FastAPI Endpoint Tests


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
                    simulation_count=2,
                    metadata=MediaMetadata(width=800, height=600, format="JPEG"),
                )
            ],
            total=1,
            limit=50,
            offset=0,
        )
        mock_ingestion.list_media_samples = AsyncMock(return_value=mock_samples)

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
        mock_ingestion.list_media_samples = AsyncMock(
            return_value=MediaSamplesListResponse(
                samples=[], total=0, limit=50, offset=0
            )
        )

        response = client.get(
            "/media/samples",
            params={"media_type": "video", "limit": 10, "offset": 5},
            headers=auth_headers,
        )

        assert response.status_code == 200
        # Verify query parameters were passed
        call_args = mock_ingestion.list_media_samples.call_args
        query = call_args[0][0]
        assert query.media_type == MediaType.VIDEO
        assert query.limit == 10
        assert query.offset == 5


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
            simulation_count=3,
            metadata=MediaMetadata(width=1920, height=1080, format="PNG"),
        )
        mock_ingestion.get_media_sample = AsyncMock(return_value=mock_sample)

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
        mock_ingestion.get_media_sample = AsyncMock(return_value=None)

        response = client.get("/media/samples/nonexistent", headers=auth_headers)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
