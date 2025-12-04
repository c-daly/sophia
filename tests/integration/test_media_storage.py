"""Integration tests for media storage file operations.

These tests verify real file system operations (storage, metadata extraction, hashing).
They don't require Neo4j/Milvus but do require file system access.

For API endpoint tests with real services, see test_media_ingestion_integration.py
For API endpoint tests with mocks (unit tests), see tests/unit/media/test_endpoint_routing.py
"""

import io
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock
from PIL import Image

from sophia.storage.media_storage import MediaStorageService
from sophia.models.media_models import MediaType


pytestmark = pytest.mark.integration


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
def sample_image_file():
    """Create sample image file in memory."""
    img = Image.new("RGB", (800, 600), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer


@pytest.fixture
def mock_upload_file(sample_image_file):
    """Create mock UploadFile for storage tests."""
    upload_file = Mock()
    upload_file.filename = "test_image.jpg"
    upload_file.read = AsyncMock(return_value=sample_image_file.read())
    return upload_file


# File Storage Tests


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
