"""Pydantic models for media ingestion API."""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class MediaType(str, Enum):
    """Supported media types for ingestion."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class MediaIngestResponse(BaseModel):
    """Response from media ingestion endpoint."""

    sample_id: str = Field(..., description="Unique identifier for the media sample")
    media_type: MediaType = Field(..., description="Type of media uploaded")
    file_path: str = Field(..., description="Storage path for the uploaded file")
    file_size: int = Field(..., description="File size in bytes")
    timestamp: datetime = Field(..., description="Upload timestamp (UTC)")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted metadata (dimensions, duration, etc.)",
    )
    neo4j_node_id: str = Field(..., description="Neo4j MediaSample node ID")
    message: str = Field(default="Media uploaded successfully")


class MediaMetadata(BaseModel):
    """Metadata extracted from uploaded media."""

    # Image metadata
    width: Optional[int] = Field(None, description="Image/video width in pixels")
    height: Optional[int] = Field(None, description="Image/video height in pixels")
    format: Optional[str] = Field(
        None, description="File format (e.g., 'PNG', 'JPEG', 'MP4')"
    )

    # Video metadata
    duration_seconds: Optional[float] = Field(
        None, description="Video/audio duration in seconds"
    )
    frame_rate: Optional[float] = Field(None, description="Video frame rate (fps)")
    frame_count: Optional[int] = Field(
        None, description="Total number of frames in video"
    )

    # Audio metadata
    sample_rate: Optional[int] = Field(None, description="Audio sample rate (Hz)")
    channels: Optional[int] = Field(None, description="Number of audio channels")

    # Common metadata
    codec: Optional[str] = Field(None, description="Codec used for encoding")
    bitrate: Optional[int] = Field(None, description="Bitrate (bits per second)")


class MediaSampleQuery(BaseModel):
    """Query parameters for retrieving media samples."""

    media_type: Optional[MediaType] = Field(None, description="Filter by media type")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    offset: int = Field(0, ge=0, description="Pagination offset")
    after_timestamp: Optional[datetime] = Field(
        None, description="Filter samples after this timestamp"
    )


class MediaSampleResponse(BaseModel):
    """Response containing media sample metadata."""

    sample_id: str
    media_type: MediaType
    file_path: str
    file_size: int
    file_hash: str
    timestamp: datetime
    metadata: MediaMetadata
    neo4j_node_id: str
    simulation_count: int = Field(
        0, description="Number of simulations using this sample"
    )


class MediaSamplesListResponse(BaseModel):
    """Response for listing multiple media samples."""

    samples: list[MediaSampleResponse]
    total: int = Field(..., description="Total number of matching samples")
    limit: int
    offset: int
