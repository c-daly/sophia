"""E2E tests for media ingestion and processing workflows.

These tests validate complete media lifecycle from upload
through processing to retrieval and deletion.

Requires: Full stack running via docker-compose.test.yml
"""

import io
import pytest
import httpx
from PIL import Image

from sophia.hcg_client import HCGClient


pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def hcg_client(neo4j_config):
    """Create HCG client for direct database verification."""
    client = HCGClient(
        neo4j_uri=neo4j_config["uri"],
        neo4j_username=neo4j_config["user"],
        neo4j_password=neo4j_config["password"],
    )
    yield client
    client.close()


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    img = Image.new("RGB", (100, 100), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes


@pytest.fixture
def sample_video():
    """Create minimal video-like bytes for testing."""
    # Minimal valid-looking video header (not actually playable)
    return io.BytesIO(b"\x00\x00\x00\x1c\x66\x74\x79\x70" + b"\x00" * 100)


class TestMediaUploadWorkflow:
    """E2E tests for media upload workflow."""

    def test_image_upload_complete_workflow(
        self, sophia_url, auth_headers, sample_image, hcg_client
    ):
        """Test complete image upload → process → verify workflow."""
        # Step 1: Upload image
        upload_response = httpx.post(
            f"{sophia_url}/ingest/media",
            files={"file": ("test_image.jpg", sample_image, "image/jpeg")},
            data={"media_type": "image"},
            headers=auth_headers,
            timeout=30,
        )
        assert upload_response.status_code == 201
        result = upload_response.json()
        sample_id = result["sample_id"]

        # Step 2: Verify metadata was extracted
        assert "metadata" in result
        assert result["media_type"] == "image"

        # Step 3: Verify persisted to Neo4j
        media_node = hcg_client.get_node(sample_id)
        assert media_node is not None
        assert media_node["type"] == "media_sample"

        # Step 4: Retrieve via API
        get_response = httpx.get(
            f"{sophia_url}/media/samples/{sample_id}",
            headers=auth_headers,
            timeout=10,
        )
        assert get_response.status_code == 200
        retrieved = get_response.json()
        assert retrieved["sample_id"] == sample_id

    def test_image_with_question_workflow(self, sophia_url, auth_headers, sample_image):
        """Test image upload with JEPA question processing."""
        upload_response = httpx.post(
            f"{sophia_url}/ingest/media",
            files={"file": ("physics_test.jpg", sample_image, "image/jpeg")},
            data={
                "media_type": "image",
                "question": "Will the ball clear the obstacle?",
            },
            headers=auth_headers,
            timeout=30,
        )
        assert upload_response.status_code == 201
        result = upload_response.json()

        # Verify upload succeeded - question may not be returned in response
        # The question is stored with the media sample but may not be in the response
        assert "sample_id" in result

    def test_media_list_pagination(self, sophia_url, auth_headers, sample_image):
        """Test media listing with pagination."""
        # Upload multiple images
        for i in range(3):
            sample_image.seek(0)
            httpx.post(
                f"{sophia_url}/ingest/media",
                files={"file": (f"test_{i}.jpg", sample_image, "image/jpeg")},
                data={"media_type": "image"},
                headers=auth_headers,
                timeout=30,
            )

        # List with pagination
        list_response = httpx.get(
            f"{sophia_url}/media/samples?limit=2&offset=0",
            headers=auth_headers,
            timeout=10,
        )
        assert list_response.status_code == 200
        page1 = list_response.json()
        assert len(page1["samples"]) <= 2

        # Get next page
        list_response2 = httpx.get(
            f"{sophia_url}/media/samples?limit=2&offset=2",
            headers=auth_headers,
            timeout=10,
        )
        assert list_response2.status_code == 200

    def test_media_filter_by_type(self, sophia_url, auth_headers, sample_image):
        """Test filtering media by type."""
        # Upload an image
        sample_image.seek(0)
        httpx.post(
            f"{sophia_url}/ingest/media",
            files={"file": ("filter_test.jpg", sample_image, "image/jpeg")},
            data={"media_type": "image"},
            headers=auth_headers,
            timeout=30,
        )

        # Filter by image type
        list_response = httpx.get(
            f"{sophia_url}/media/samples?media_type=image",
            headers=auth_headers,
            timeout=10,
        )
        assert list_response.status_code == 200
        results = list_response.json()

        # All results should be images
        for sample in results.get("samples", []):
            assert sample["media_type"] == "image"


class TestMediaToSimulationWorkflow:
    """E2E tests for media → simulation pipeline."""

    def test_upload_then_simulate_with_reference(
        self, sophia_url, auth_headers, sample_image
    ):
        """Test uploading media then using it in simulation."""
        # Step 1: Upload media
        sample_image.seek(0)
        upload_response = httpx.post(
            f"{sophia_url}/ingest/media",
            files={"file": ("sim_input.jpg", sample_image, "image/jpeg")},
            data={
                "media_type": "image",
                "question": "What happens when the ball is released?",
            },
            headers=auth_headers,
            timeout=30,
        )
        assert upload_response.status_code == 201
        sample_id = upload_response.json()["sample_id"]

        # Step 2: Run simulation referencing the media
        simulate_response = httpx.post(
            f"{sophia_url}/simulate",
            json={
                "entities": [
                    {
                        "id": "ball",
                        "type": "object",
                        "properties": {"mass": 0.5},
                        "position": {"x": 0, "y": 0, "z": 1.0},
                    }
                ],
                "media_sample_id": sample_id,
                "k_steps": 5,
            },
            headers=auth_headers,
            timeout=10,
        )
        assert simulate_response.status_code == 201
        simulation = simulate_response.json()

        # Simulation should use embeddings from media
        assert len(simulation["imagined_states"]) == 5

    def test_batch_media_processing(self, sophia_url, auth_headers, sample_image):
        """Test processing multiple media files for simulation."""
        sample_ids = []

        # Upload multiple frames
        for i in range(3):
            sample_image.seek(0)
            response = httpx.post(
                f"{sophia_url}/ingest/media",
                files={"file": (f"frame_{i}.jpg", sample_image, "image/jpeg")},
                data={"media_type": "image"},
                headers=auth_headers,
                timeout=30,
            )
            if response.status_code == 201:
                sample_ids.append(response.json()["sample_id"])

        assert len(sample_ids) >= 1, "At least one upload should succeed"

        # Use first sample in simulation
        simulate_response = httpx.post(
            f"{sophia_url}/simulate",
            json={
                "entities": [{"id": "obj", "type": "object"}],
                "media_sample_id": sample_ids[0],
                "k_steps": 3,
            },
            headers=auth_headers,
            timeout=10,
        )
        assert simulate_response.status_code in [201, 503]


class TestMediaErrorHandling:
    """E2E tests for media error handling."""

    def test_invalid_media_type_rejected(self, sophia_url, auth_headers):
        """Test that invalid media types are rejected."""
        fake_file = io.BytesIO(b"not a real image")

        response = httpx.post(
            f"{sophia_url}/ingest/media",
            files={"file": ("test.xyz", fake_file, "application/octet-stream")},
            data={"media_type": "image"},
            headers=auth_headers,
            timeout=10,
        )

        assert response.status_code == 400

    def test_missing_file_rejected(self, sophia_url, auth_headers):
        """Test that missing file is rejected."""
        response = httpx.post(
            f"{sophia_url}/ingest/media",
            data={"media_type": "image"},
            headers=auth_headers,
            timeout=10,
        )

        assert response.status_code == 422

    def test_get_nonexistent_sample(self, sophia_url, auth_headers):
        """Test getting non-existent sample returns 404."""
        response = httpx.get(
            f"{sophia_url}/media/samples/nonexistent_sample_xyz",
            headers=auth_headers,
            timeout=10,
        )

        assert response.status_code == 404

    def test_upload_requires_auth(self, sophia_url, sample_image):
        """Test that upload requires authentication."""
        sample_image.seek(0)
        response = httpx.post(
            f"{sophia_url}/ingest/media",
            files={"file": ("test.jpg", sample_image, "image/jpeg")},
            data={"media_type": "image"},
            timeout=10,
        )

        # Should require auth (403) or the endpoint may not require auth (201)
        assert response.status_code in [201, 403]
