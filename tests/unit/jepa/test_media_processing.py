"""Unit tests for JEPA media processing and model construction.

These tests verify JEPARunner behavior and Pydantic model construction
without external services.
"""

import pytest

from sophia.jepa.runner import JEPARunner
from sophia.api.models import SimulateRequest, SimulateResponse


pytestmark = pytest.mark.unit


@pytest.fixture
def jepa_runner():
    """Fixture for JEPA runner."""
    return JEPARunner(model_version="jepa-stub-v1.0")


class TestJEPAMediaProcessing:
    """Tests for JEPA media processing output format."""

    @pytest.mark.asyncio
    async def test_process_media_sample_generates_embeddings(self, jepa_runner):
        """Test that JEPA runner generates embeddings for media samples."""
        result = await jepa_runner.process_media_sample(
            sample_id="test_sample_123",
            file_path="/path/to/image.jpg",
            media_type="image",
            metadata={"width": 800, "height": 600},
            question="What happens next?",
        )

        assert result["sample_id"] == "test_sample_123"
        assert result["media_type"] == "image"
        assert "embeddings" in result
        assert "visual" in result["embeddings"]
        assert "physics" in result["embeddings"]
        assert result["embedding_dim"] == 768
        assert len(result["embeddings"]["visual"]) == 768
        assert len(result["embeddings"]["physics"]) == 768
        assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_process_media_sample_includes_metadata(self, jepa_runner):
        """Test that processing result includes metadata."""
        result = await jepa_runner.process_media_sample(
            sample_id="test_sample_456",
            file_path="/path/to/video.mp4",
            media_type="video",
            metadata={"duration": 5.0, "fps": 30},
            question="Will the ball clear the obstacle?",
        )

        assert result["metadata"]["file_path"] == "/path/to/video.mp4"
        assert result["metadata"]["question"] == "Will the ball clear the obstacle?"
        assert result["metadata"]["media_metadata"]["duration"] == 5.0


class TestSimulationRequestModels:
    """Tests for simulation request/response Pydantic models."""

    def test_simulate_request_with_media_sample_id(self):
        """Test that /simulate request accepts and uses media_sample_id."""
        request = SimulateRequest(
            entities=[
                {
                    "id": "ball_1",
                    "type": "object",
                    "properties": {"mass": 0.5},
                    "position": {"x": 0.0, "y": 0.0, "z": 1.0},
                }
            ],
            media_sample_id="sample_abc123",
            k_steps=5,
        )

        assert request.media_sample_id == "sample_abc123"
        assert request.k_steps == 5

    def test_simulate_request_without_media_is_optional(self):
        """Test that media_sample_id is optional for simulations."""
        request = SimulateRequest(
            entities=[
                {
                    "id": "block_1",
                    "type": "object",
                    "properties": {"mass": 1.0},
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                }
            ],
            k_steps=3,
        )

        assert request.media_sample_id is None
        assert request.k_steps == 3

    def test_simulate_response_includes_media_reference(self):
        """Test that simulation results include media sample reference."""
        response = SimulateResponse(
            simulation_id="sim_123",
            imagined_processes=[],
            imagined_states=[],
            k_steps=5,
            model_version="jepa-stub-v1.0",
            overall_confidence=0.85,
            media_sample_id="sample_abc123",
            media_embeddings=["sample_abc123_visual", "sample_abc123_physics"],
        )

        assert response.media_sample_id == "sample_abc123"
        assert len(response.media_embeddings) == 2
        assert "sample_abc123_visual" in response.media_embeddings
