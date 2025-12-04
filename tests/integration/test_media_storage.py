"""Integration tests for media storage file operations.

These tests verify real file system operations (storage, metadata extraction, hashing).
They don't require Neo4j/Milvus but do require file system access.

For API endpoint tests with real services, see test_media_ingestion_integration.py
For API endpoint tests with mocks (unit tests), see tests/unit/media/test_endpoint_routing.py
"""

import io
import pytest
from pathlib import Path
from starlette.datastructures import UploadFile

from sophia.storage.media_storage import MediaStorageService
from sophia.models.media_models import MediaType


pytestmark = pytest.mark.integration

# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


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
def test_image_path():
    """Path to real test image fixture."""
    path = FIXTURES_DIR / "test_image.jpg"
    assert path.exists(), f"Test fixture not found: {path}"
    return path


@pytest.fixture
def real_upload_file(test_image_path):
    """Create a real UploadFile from test fixture.

    This uses the actual file from tests/fixtures/ for realistic testing.
    The file is opened, used, and cleaned up after the test.
    """
    with open(test_image_path, "rb") as f:
        content = f.read()

    # Create UploadFile with real file content
    file_obj = io.BytesIO(content)
    upload = UploadFile(filename="test_image.jpg", file=file_obj)
    yield upload
    # Cleanup: close the file object
    file_obj.close()


# File Storage Tests


@pytest.mark.asyncio
async def test_store_file(media_storage_service, real_upload_file, temp_storage_dir):
    """Test file storage to disk with real file upload.

    Uses a real test image from fixtures, stores it, and verifies:
    - File is stored with correct extension
    - File size matches original
    - File exists on disk
    - File is cleaned up after test (via tmp_path)
    """
    sample_id, file_path, file_size = await media_storage_service.store_file(
        real_upload_file, MediaType.IMAGE
    )

    assert sample_id is not None
    assert sample_id.startswith("ms_")  # media storage prefix
    assert "image" in file_path
    assert file_path.endswith(".jpg")
    assert file_size > 0
    assert file_size == 8230  # exact size of our test fixture
    assert Path(file_path).exists()

    # Verify the stored file is a valid image
    from PIL import Image

    with Image.open(file_path) as img:
        assert img.size == (800, 600)
        assert img.format == "JPEG"


def test_extract_image_metadata(media_storage_service, test_image_path):
    """Test metadata extraction from real image fixture."""
    metadata = media_storage_service.extract_metadata(
        str(test_image_path), MediaType.IMAGE
    )

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
