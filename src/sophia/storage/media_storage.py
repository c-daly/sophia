"""Media storage service for handling file uploads and persistence."""

import logging
import hashlib
from pathlib import Path
from typing import Optional, Tuple
from uuid import uuid4

from PIL import Image  # type: ignore[import-not-found]
from fastapi import UploadFile, HTTPException, status

from sophia.models.media_models import MediaType, MediaMetadata


logger = logging.getLogger(__name__)


class MediaStorageService:
    """Handles media file storage and metadata extraction."""

    def __init__(self, storage_root: str = "./media_storage"):
        """Initialize media storage service.

        Args:
            storage_root: Root directory for media storage
        """
        self.storage_root = Path(storage_root)
        self._ensure_storage_directories()

    def _ensure_storage_directories(self) -> None:
        """Create storage directory structure if it doesn't exist."""
        for media_type in MediaType:
            type_dir = self.storage_root / media_type.value
            type_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Media storage initialized at {self.storage_root}")

    def validate_file(self, file: UploadFile, media_type: MediaType) -> None:
        """Validate uploaded file against media type constraints.

        Args:
            file: Uploaded file
            media_type: Expected media type

        Raises:
            HTTPException: If validation fails
        """
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Filename is required"
            )

        # Get file extension
        file_ext = Path(file.filename).suffix.lower()

        # Define allowed extensions by media type
        allowed_extensions = {
            MediaType.IMAGE: {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"},
            MediaType.VIDEO: {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"},
            MediaType.AUDIO: {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"},
        }

        if file_ext not in allowed_extensions.get(media_type, set()):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file extension '{file_ext}' for media type '{media_type.value}'. "
                f"Allowed: {', '.join(sorted(allowed_extensions[media_type]))}",
            )

    async def store_file(
        self, file: UploadFile, media_type: MediaType, sample_id: Optional[str] = None
    ) -> Tuple[str, str, int]:
        """Store uploaded file to disk.

        Args:
            file: Uploaded file
            media_type: Type of media
            sample_id: Optional custom sample ID, generates UUID if not provided

        Returns:
            Tuple of (sample_id, file_path, file_size)

        Raises:
            HTTPException: If storage fails
        """
        if sample_id is None:
            sample_id = f"ms_{uuid4().hex[:16]}"

        # Validate file before processing
        self.validate_file(file, media_type)

        # Generate file path
        file_ext = Path(file.filename or "unknown").suffix
        filename = f"{sample_id}{file_ext}"
        file_path = self.storage_root / media_type.value / filename
        max_size = 100 * 1024 * 1024  # 100MB
        chunk_size = 1024 * 1024  # 1MB

        try:
            file_size = 0
            with open(file_path, "wb") as f:
                while True:
                    chunk = await file.read(chunk_size)
                    if not chunk:
                        break
                    next_size = file_size + len(chunk)
                    if next_size > max_size:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail=(
                                f"File size exceeds maximum allowed size of {max_size} bytes"
                            ),
                        )
                    f.write(chunk)
                    file_size = next_size

            logger.info(
                f"Stored {media_type.value} file {filename} "
                f"({file_size} bytes) at {file_path}"
            )

            return sample_id, str(file_path), file_size

        except HTTPException:
            Path(file_path).unlink(missing_ok=True)
            raise
        except Exception as e:
            logger.error(f"Failed to store file: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to store file: {str(e)}",
            ) from e

    def extract_metadata(self, file_path: str, media_type: MediaType) -> MediaMetadata:
        """Extract metadata from stored media file.

        Args:
            file_path: Path to stored file
            media_type: Type of media

        Returns:
            Extracted metadata
        """
        # Create metadata with all fields as None initially
        metadata = MediaMetadata.model_construct()
        path = Path(file_path)

        try:
            # Extract format from file extension
            metadata.format = path.suffix.lstrip(".").upper()

            if media_type == MediaType.IMAGE:
                self._extract_image_metadata(file_path, metadata)
            elif media_type == MediaType.VIDEO:
                self._extract_video_metadata(file_path, metadata)
            elif media_type == MediaType.AUDIO:
                self._extract_audio_metadata(file_path, metadata)

        except Exception as e:
            logger.warning(f"Failed to extract metadata from {file_path}: {e}")
            # Return partial metadata rather than failing

        return metadata

    def _extract_image_metadata(self, file_path: str, metadata: MediaMetadata) -> None:
        """Extract image-specific metadata using Pillow.

        Args:
            file_path: Path to image file
            metadata: Metadata object to populate
        """
        try:
            with Image.open(file_path) as img:
                metadata.width = img.width
                metadata.height = img.height
                metadata.format = img.format or metadata.format
        except Exception as e:
            logger.debug(f"Could not extract image metadata: {e}")

    def _extract_video_metadata(self, file_path: str, metadata: MediaMetadata) -> None:
        """Extract video-specific metadata.

        Args:
            file_path: Path to video file
            metadata: Metadata object to populate

        Note:
            This is a stub implementation. Full implementation would use
            ffmpeg-python or similar to extract video metadata.
        """
        # TODO: Implement with ffmpeg-python when available
        # For now, leave video metadata as None values
        logger.debug(f"Video metadata extraction not yet implemented for {file_path}")

    def _extract_audio_metadata(self, file_path: str, metadata: MediaMetadata) -> None:
        """Extract audio-specific metadata.

        Args:
            file_path: Path to audio file
            metadata: Metadata object to populate

        Note:
            This is a stub implementation. Full implementation would use
            ffmpeg-python or mutagen to extract audio metadata.
        """
        # TODO: Implement with mutagen or ffmpeg-python when available
        logger.debug(f"Audio metadata extraction not yet implemented for {file_path}")

    def compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file for deduplication.

        Args:
            file_path: Path to file

        Returns:
            Hex digest of SHA256 hash
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def delete_file(self, file_path: str) -> bool:
        """Delete a stored media file.

        Args:
            file_path: Path to file to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            Path(file_path).unlink(missing_ok=True)
            logger.info(f"Deleted file {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return False
