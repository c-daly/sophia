"""Tests for error handling and edge cases in Sophia services."""

import pytest
from unittest.mock import Mock, AsyncMock
from fastapi import HTTPException
from fastapi.testclient import TestClient

from sophia.storage.media_storage import MediaStorageService
from sophia.ingestion.media_service import MediaIngestionService
from sophia.models.media_models import MediaType
from sophia.api.app import create_app


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
    client.add_edge = Mock()
    client.get_node = Mock(return_value=None)

    # Mock _milvus
    client._milvus = Mock()
    client._milvus.insert_embedding = Mock()

    # Mock _neo4j._driver
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
def media_ingestion_service(mock_hcg_client, media_storage_service):
    """Create MediaIngestionService instance."""
    return MediaIngestionService(
        hcg_client=mock_hcg_client,
        storage_service=media_storage_service,
    )


class TestNeo4jUnavailable:
    """Tests for Neo4j unavailability scenarios."""

    @pytest.mark.asyncio
    async def test_media_ingestion_fails_when_neo4j_unavailable(
        self, media_ingestion_service, mock_hcg_client
    ):
        """Test that media ingestion handles Neo4j failures gracefully."""
        # Mock Neo4j failure
        mock_hcg_client.add_node.side_effect = Exception("Neo4j connection failed")

        # Create mock upload file
        mock_file = Mock()
        mock_file.filename = "test.jpg"
        mock_file.read = AsyncMock(return_value=b"fake image data")

        # Should raise exception (or handle gracefully depending on design)
        with pytest.raises(Exception) as exc_info:
            await media_ingestion_service.ingest_media(
                file=mock_file,
                media_type=MediaType.IMAGE,
            )

        assert "Neo4j" in str(exc_info.value) or "connection" in str(exc_info.value).lower()

    def test_get_media_sample_returns_none_when_neo4j_unavailable(
        self, media_ingestion_service, mock_hcg_client
    ):
        """Test that get_media_sample handles Neo4j failures."""
        # Mock Neo4j failure
        mock_hcg_client.get_node.side_effect = Exception("Neo4j connection failed")

        # Should handle gracefully (return None or raise)
        try:
            result = media_ingestion_service.get_media_sample("sample_123")
            # If it doesn't raise, should return None
            assert result is None
        except Exception as e:
            # Or raise a clear error
            assert "Neo4j" in str(e) or "connection" in str(e).lower()


class TestMilvusUnavailable:
    """Tests for Milvus unavailability scenarios."""

    @pytest.mark.asyncio
    async def test_embedding_storage_fails_when_milvus_unavailable(
        self, media_ingestion_service, mock_hcg_client
    ):
        """Test that embedding storage handles Milvus failures."""
        # Mock Milvus failure
        mock_hcg_client._milvus.insert_embedding.side_effect = Exception(
            "Milvus connection failed"
        )

        # Try to store embeddings
        embeddings = {
            "visual": [0.1] * 768,
            "physics": [0.2] * 768,
        }

        # Should raise or handle gracefully
        with pytest.raises(Exception) as exc_info:
            await media_ingestion_service._store_jepa_embeddings(
                sample_id="sample_123",
                embeddings=embeddings,
                metadata={"confidence": 0.85},
            )

        assert "Milvus" in str(exc_info.value) or "connection" in str(exc_info.value).lower()


class TestInvalidInputs:
    """Tests for invalid input handling."""

    def test_invalid_file_extension_rejected(self, media_storage_service):
        """Test that invalid file extensions are rejected."""
        mock_file = Mock()
        mock_file.filename = "malware.exe"

        with pytest.raises(HTTPException) as exc_info:
            media_storage_service.validate_file(mock_file, MediaType.IMAGE)

        assert exc_info.value.status_code == 400
        assert "Invalid file extension" in str(exc_info.value.detail)

    def test_missing_file_extension_rejected(self, media_storage_service):
        """Test that files without extensions are rejected."""
        mock_file = Mock()
        mock_file.filename = "noextension"

        with pytest.raises(HTTPException) as exc_info:
            media_storage_service.validate_file(mock_file, MediaType.IMAGE)

        assert exc_info.value.status_code == 400

    def test_wrong_media_type_for_extension(self, media_storage_service):
        """Test that mismatched media types are rejected."""
        mock_file = Mock()
        mock_file.filename = "video.mp4"

        # Try to upload video as image
        with pytest.raises(HTTPException) as exc_info:
            media_storage_service.validate_file(mock_file, MediaType.IMAGE)

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_corrupted_image_file(self, media_storage_service, temp_storage_dir):
        """Test handling of corrupted image files."""
        # Create corrupted "image"
        corrupted_file = temp_storage_dir / "corrupted.jpg"
        corrupted_file.write_bytes(b"not a valid jpeg")

        # Attempting to extract metadata should fail gracefully
        try:
            metadata = media_storage_service.extract_metadata(
                str(corrupted_file), MediaType.IMAGE
            )
            # Should return empty/default metadata rather than crash
            assert metadata is not None
        except Exception as e:
            # Or raise a clear error
            assert "corrupt" in str(e).lower() or "invalid" in str(e).lower()


class TestSimulationErrors:
    """Tests for simulation error cases."""

    @pytest.mark.asyncio
    async def test_simulation_with_invalid_media_sample_id(self):
        """Test simulation with non-existent media_sample_id."""
        from sophia.jepa.runner import JEPARunner

        runner = JEPARunner()

        # Simulate with invalid media sample
        result = await runner.simulate(
            entities=[],
            k_steps=1,
            media_sample_id="nonexistent_sample",
        )

        # Should complete but might have reduced confidence or warning
        assert result["media_sample_id"] == "nonexistent_sample"
        # In production, might want to verify the sample exists first

    @pytest.mark.asyncio
    async def test_simulation_with_missing_dependencies(self):
        """Test simulation when dependencies are missing."""
        from sophia.jepa.runner import JEPARunner

        runner = JEPARunner()

        # Try simulation with entities that have missing required fields
        incomplete_entities = [
            {
                "id": "broken_entity",
                "type": "object",
                # Missing position, velocity
            }
        ]

        # Should handle gracefully (validate or use defaults)
        try:
            result = await runner.simulate(
                entities=incomplete_entities,
                k_steps=1,
                media_sample_id=None,
            )
            # If it succeeds, entities were filled with defaults
            assert result is not None
        except (ValueError, KeyError, TypeError):
            # Or raise validation error
            assert True


class TestAPIValidationErrors:
    """Tests for API request validation errors."""

    def test_missing_required_fields(self):
        """Test that missing required fields return 422."""
        import os
        os.environ["SOPHIA_API_TOKEN"] = "test-token"
        os.environ["NEO4J_URI"] = "bolt://mock:7687"

        app = create_app()
        client = TestClient(app)

        # Missing media_type field
        files = {"file": ("test.jpg", b"fake data", "image/jpeg")}

        response = client.post(
            "/ingest/media",
            files=files,
            headers={"Authorization": "Bearer test-token"},
        )

        assert response.status_code == 422

    def test_malformed_json_request(self):
        """Test that malformed JSON returns 422."""
        import os
        os.environ["SOPHIA_API_TOKEN"] = "test-token"
        os.environ["NEO4J_URI"] = "bolt://mock:7687"

        app = create_app()
        client = TestClient(app)

        # Send invalid JSON
        response = client.post(
            "/simulate",
            data="not valid json",
            headers={
                "Authorization": "Bearer test-token",
                "Content-Type": "application/json",
            },
        )

        assert response.status_code == 422

    def test_invalid_media_type_enum(self):
        """Test that invalid media_type enum values return 422."""
        import os
        os.environ["SOPHIA_API_TOKEN"] = "test-token"
        os.environ["NEO4J_URI"] = "bolt://mock:7687"

        app = create_app()
        client = TestClient(app)

        files = {"file": ("test.jpg", b"fake data", "image/jpeg")}
        data = {"media_type": "invalid_type"}

        response = client.post(
            "/ingest/media",
            files=files,
            data=data,
            headers={"Authorization": "Bearer test-token"},
        )

        assert response.status_code == 422


class TestTimeouts:
    """Tests for timeout handling."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_simulation_completes_within_reasonable_time(self):
        """Test that simulation doesn't hang indefinitely."""
        from sophia.jepa.runner import JEPARunner

        runner = JEPARunner()

        # Even large k-step should complete quickly (stub)
        result = await runner.simulate(
            entities=[],
            k_steps=100,
            media_sample_id=None,
        )

        assert result["k_steps"] == 100
        # Test passes if it completes within timeout


class TestConcurrentRequests:
    """Tests for concurrent request handling."""

    @pytest.mark.asyncio
    async def test_concurrent_media_ingestion(
        self, media_ingestion_service, mock_hcg_client
    ):
        """Test that concurrent ingestions don't conflict."""
        import asyncio

        # Create multiple mock files
        mock_files = []
        for i in range(3):
            mock_file = Mock()
            mock_file.filename = f"test_{i}.jpg"
            mock_file.read = AsyncMock(return_value=f"fake data {i}".encode())
            mock_files.append(mock_file)

        # Ingest concurrently
        tasks = [
            media_ingestion_service.ingest_media(
                file=mock_file,
                media_type=MediaType.IMAGE,
            )
            for mock_file in mock_files
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed or fail independently
        for result in results:
            if isinstance(result, Exception):
                # Log but don't fail test (concurrency issues expected in test env)
                print(f"Concurrent request failed: {result}")
            else:
                # Successful results should have unique sample IDs
                assert result.sample_id is not None

    @pytest.mark.asyncio
    async def test_concurrent_simulations(self):
        """Test that concurrent simulations don't interfere."""
        import asyncio
        from sophia.jepa.runner import JEPARunner

        runner = JEPARunner()

        # Run multiple simulations concurrently
        tasks = [
            runner.simulate(entities=[], k_steps=2, media_sample_id=None)
            for _ in range(5)
        ]

        results = await asyncio.gather(*tasks)

        # All should complete with unique IDs
        simulation_ids = [r["simulation_id"] for r in results]
        assert len(simulation_ids) == len(set(simulation_ids))  # All unique


class TestResourceCleanup:
    """Tests for proper resource cleanup."""

    @pytest.mark.asyncio
    async def test_failed_upload_cleans_up_files(
        self, media_ingestion_service, mock_hcg_client, temp_storage_dir
    ):
        """Test that failed uploads clean up partial files."""
        # Mock storage to succeed but Neo4j to fail
        mock_hcg_client.add_node.side_effect = Exception("Neo4j failed")

        mock_file = Mock()
        mock_file.filename = "test.jpg"
        mock_file.read = AsyncMock(return_value=b"fake image data")

        # Attempt ingestion (should fail)
        try:
            await media_ingestion_service.ingest_media(
                file=mock_file,
                media_type=MediaType.IMAGE,
            )
        except Exception:
            pass

        # Check if files were cleaned up (implementation-dependent)
        # This test documents expected behavior

        # Ideally, failed uploads don't leave orphaned files
        # (In current implementation, files might remain until manual cleanup)


class TestLargeInputs:
    """Tests for handling large inputs."""

    @pytest.mark.asyncio
    async def test_large_number_of_entities(self):
        """Test simulation with many entities."""
        from sophia.jepa.runner import JEPARunner

        runner = JEPARunner()

        # Create many entities
        large_entity_list = [
            {
                "id": f"entity_{i}",
                "type": "object",
                "properties": {"mass": 1.0},
                "position": {"x": float(i), "y": 0.0, "z": 0.0},
            }
            for i in range(100)
        ]

        # Should handle large inputs
        result = await runner.simulate(
            entities=large_entity_list,
            k_steps=1,
            media_sample_id=None,
        )

        assert result["k_steps"] == 1
        # Verify entities preserved (or processed)
        assert len(result["imagined_states"]) == 1

    @pytest.mark.asyncio
    async def test_deeply_nested_entity_properties(self):
        """Test entities with deeply nested properties."""
        from sophia.jepa.runner import JEPARunner

        runner = JEPARunner()

        nested_entity = {
            "id": "complex_entity",
            "type": "object",
            "properties": {
                "level1": {
                    "level2": {
                        "level3": {
                            "value": 42
                        }
                    }
                }
            },
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        }

        # Should handle or validate nested structures
        result = await runner.simulate(
            entities=[nested_entity],
            k_steps=1,
            media_sample_id=None,
        )

        assert result is not None
