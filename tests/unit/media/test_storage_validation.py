"""Unit tests for MediaStorageService validation logic.

These tests validate file extension and type checking without external services.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock
from fastapi import HTTPException

from sophia.storage.media_storage import MediaStorageService
from sophia.models.media_models import MediaType


pytestmark = pytest.mark.unit


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


class TestFileValidation:
    """Tests for file validation logic."""

    def test_validate_file_valid_image(self, media_storage_service):
        """Test validation of valid image file."""
        mock_file = Mock()
        mock_file.filename = "test.jpg"
        # Should not raise
        media_storage_service.validate_file(mock_file, MediaType.IMAGE)

        mock_file.filename = "test.png"
        media_storage_service.validate_file(mock_file, MediaType.IMAGE)

    def test_validate_file_valid_video(self, media_storage_service):
        """Test validation of valid video file."""
        mock_file = Mock()
        mock_file.filename = "test.mp4"
        # Should not raise
        media_storage_service.validate_file(mock_file, MediaType.VIDEO)

        mock_file.filename = "test.avi"
        media_storage_service.validate_file(mock_file, MediaType.VIDEO)

    def test_validate_file_valid_audio(self, media_storage_service):
        """Test validation of valid audio file."""
        mock_file = Mock()
        mock_file.filename = "test.mp3"
        # Should not raise
        media_storage_service.validate_file(mock_file, MediaType.AUDIO)

        mock_file.filename = "test.wav"
        media_storage_service.validate_file(mock_file, MediaType.AUDIO)

    def test_validate_file_invalid_extension(self, media_storage_service):
        """Test validation rejects invalid file extension."""
        mock_file = Mock()
        mock_file.filename = "test.txt"
        with pytest.raises(HTTPException) as exc_info:
            media_storage_service.validate_file(mock_file, MediaType.IMAGE)
        assert exc_info.value.status_code == 400
        assert "Invalid file extension" in str(exc_info.value.detail)
