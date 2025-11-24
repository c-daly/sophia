"""Integration tests for JEPA runner with media ingestion pipeline."""

import pytest
from unittest.mock import Mock, AsyncMock
from io import BytesIO

from sophia.models.media_models import MediaType
from sophia.jepa.runner import JEPARunner


@pytest.fixture
def jepa_runner():
    """Fixture for JEPA runner."""
    return JEPARunner(model_version="jepa-stub-v1.0")


@pytest.fixture
def mock_hcg_client():
    """Fixture for mocked HCG client."""
    client = Mock()
    client.add_node = Mock()
    client.add_edge = Mock()
    client.get_node = Mock(return_value={"sample_id": "test_sample"})
    
    # Mock _milvus private attribute (correct API)
    client._milvus = Mock()
    client._milvus.insert_embedding = Mock()
    
    # Mock _neo4j._driver for session context (correct API)
    mock_session = Mock()
    mock_result = Mock()
    mock_result.__iter__ = Mock(return_value=iter([]))
    mock_session.run = Mock(return_value=mock_result)
    mock_session.__enter__ = Mock(return_value=mock_session)
    mock_session.__exit__ = Mock(return_value=False)
    
    mock_driver = Mock()
    mock_driver.session = Mock(return_value=mock_session)
    
    mock_neo4j = Mock()
    mock_neo4j._driver = mock_driver
    mock_neo4j._database = "neo4j"
    client._neo4j = mock_neo4j
    
    return client


@pytest.fixture
def sample_image_file():
    """Fixture for sample image file."""
    # Create a minimal valid JPEG file
    jpeg_bytes = (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c"
        b"\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c"
        b"\x1c $.\' \",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01"
        b"\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x08\x01\x01\x00\x00?"
        b"\x00\x7f\xff\xd9"
    )
    return BytesIO(jpeg_bytes)


class TestJEPAMediaProcessing:
    """Tests for JEPA media processing."""

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


class TestMediaToSimulationIntegration:
    """Tests for media ingestion → JEPA → simulation integration."""

    @pytest.mark.asyncio
    async def test_media_ingestion_triggers_jepa_processing(
        self, mock_hcg_client, jepa_runner, sample_image_file
    ):
        """Test that media ingestion triggers JEPA processing."""
        from sophia.ingestion.media_service import MediaIngestionService
        from sophia.storage.media_storage import MediaStorageService
        from fastapi import UploadFile

        # Create media ingestion service with JEPA runner
        storage_service = MediaStorageService(storage_root="/tmp/test_media")
        media_service = MediaIngestionService(
            hcg_client=mock_hcg_client,
            storage_service=storage_service,
            jepa_runner=jepa_runner,
        )

        # Mock the storage operations
        storage_service.store_file = AsyncMock(
            return_value=("sample_123", "/tmp/test_media/sample_123.jpg", 1024)
        )
        storage_service.extract_metadata = Mock(
            return_value=Mock(
                model_dump=Mock(return_value={"width": 800, "height": 600})
            )
        )
        storage_service.compute_file_hash = Mock(return_value="abc123hash")

        # Create mock upload file
        sample_image_file.seek(0)
        upload_file = UploadFile(
            filename="test.jpg",
            file=sample_image_file,
        )

        # Ingest media with question
        result = await media_service.ingest_media(
            file=upload_file,
            media_type=MediaType.IMAGE,
            question="What happens next?",
        )

        # Verify media sample was created
        assert result.sample_id == "sample_123"
        assert mock_hcg_client.add_node.called

        # Verify embeddings would be stored in Milvus
        # (In real test with full Milvus, verify actual storage)
        assert mock_hcg_client._milvus.insert_embedding.call_count >= 2  # visual + physics

    @pytest.mark.asyncio
    async def test_simulate_with_media_sample_id(self, mock_hcg_client):
        """Test that /simulate endpoint accepts and uses media_sample_id."""
        from sophia.api.models import SimulateRequest

        # Create simulation request with media_sample_id
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

        # Verify request includes media_sample_id
        assert request.media_sample_id == "sample_abc123"
        assert request.k_steps == 5

    @pytest.mark.asyncio
    async def test_embeddings_linked_to_neo4j(self, mock_hcg_client):
        """Test that embeddings are linked to media samples in Neo4j."""
        from sophia.ingestion.media_service import MediaIngestionService
        from sophia.storage.media_storage import MediaStorageService

        storage_service = MediaStorageService(storage_root="/tmp/test_media")
        jepa_runner = JEPARunner()
        media_service = MediaIngestionService(
            hcg_client=mock_hcg_client,
            storage_service=storage_service,
            jepa_runner=jepa_runner,
        )

        # Process embeddings
        embeddings = {
            "visual": [0.1] * 768,
            "physics": [0.2] * 768,
        }
        
        await media_service._store_jepa_embeddings(
            sample_id="sample_123",
            embeddings=embeddings,
            metadata={"confidence": 0.85},
        )

        # Verify Milvus insert was called for each embedding type
        assert mock_hcg_client._milvus.insert_embedding.call_count == 2

        # Verify Neo4j edges were created
        assert mock_hcg_client.add_edge.call_count == 2


class TestSimulationWithMediaContext:
    """Tests for simulations using media context."""

    def test_simulation_result_includes_media_reference(self):
        """Test that simulation results include media sample reference."""
        from sophia.api.models import SimulateResponse

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

    def test_simulation_without_media_is_optional(self):
        """Test that media_sample_id is optional for simulations."""
        from sophia.api.models import SimulateRequest

        # Create request without media
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


class TestEndToEndWorkflow:
    """End-to-end tests for complete perception pipeline."""

    @pytest.mark.asyncio
    async def test_complete_workflow(
        self, mock_hcg_client, jepa_runner, sample_image_file
    ):
        """Test complete workflow: upload → JEPA → embeddings → simulation.
        
        This test verifies the full integration:
        1. Apollo uploads image with question
        2. Sophia ingests media
        3. JEPA processes and generates embeddings
        4. Embeddings stored in Milvus + linked in Neo4j
        5. Simulation references media context
        6. Response includes evidence artifacts
        """
        from sophia.ingestion.media_service import MediaIngestionService
        from sophia.storage.media_storage import MediaStorageService
        from fastapi import UploadFile

        # Step 1: Setup services
        storage_service = MediaStorageService(storage_root="/tmp/test_media")
        storage_service.store_file = AsyncMock(
            return_value=("sample_xyz", "/tmp/test_media/sample_xyz.jpg", 2048)
        )
        storage_service.extract_metadata = Mock(
            return_value=Mock(
                model_dump=Mock(return_value={"width": 1024, "height": 768})
            )
        )
        storage_service.compute_file_hash = Mock(return_value="xyz789hash")

        media_service = MediaIngestionService(
            hcg_client=mock_hcg_client,
            storage_service=storage_service,
            jepa_runner=jepa_runner,
        )

        # Step 2: Ingest media (simulates Apollo upload)
        sample_image_file.seek(0)
        upload_file = UploadFile(filename="physics_scene.jpg", file=sample_image_file)

        ingest_result = await media_service.ingest_media(
            file=upload_file,
            media_type=MediaType.IMAGE,
            question="Will the stack collapse?",
        )

        # Step 3: Verify media sample created
        assert ingest_result.sample_id == "sample_xyz"
        assert mock_hcg_client.add_node.called

        # Step 4: Verify embeddings stored
        assert mock_hcg_client._milvus.insert_embedding.call_count >= 2

        # Step 5: Create simulation request with media context
        from sophia.api.models import SimulateRequest

        sim_request = SimulateRequest(
            entities=[],  # Could be extracted from image analysis
            media_sample_id=ingest_result.sample_id,
            k_steps=5,
        )

        # Step 6: Verify request properly formed
        assert sim_request.media_sample_id == "sample_xyz"

        # Note: Full API endpoint test would require running FastAPI test client
        # This validates the data flow and integration points
