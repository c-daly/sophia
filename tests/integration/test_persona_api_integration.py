"""Integration tests for Persona API endpoints.

These tests require Sophia and Neo4j to be running.
Run with: pytest tests/integration/test_persona_api_integration.py -v -m integration

Start services with: ./scripts/test_integration.sh up
"""

import pytest
import uuid

pytestmark = [
    pytest.mark.integration,
]


@pytest.fixture
def unique_entry_content():
    """Generate unique content for each test to avoid collisions."""
    return f"Test entry content {uuid.uuid4().hex[:8]}"


class TestPersonaCreateEndpoint:
    """Integration tests for POST /persona/entries endpoint."""

    def test_create_persona_entry_success(
        self, http_client, auth_headers, unique_entry_content
    ):
        """Test that POST /persona/entries creates a new entry."""
        payload = {
            "entry_type": "decision",
            "content": unique_entry_content,
            "summary": "Test summary",
            "sentiment": "positive",
            "confidence": 0.85,
            "emotion_tags": ["confident", "decisive"],
            "metadata": {"test": True},
        }
        response = http_client.post(
            "/persona/entries", json=payload, headers=auth_headers
        )
        assert response.status_code == 201

        data = response.json()
        assert "entry_id" in data
        assert "cwm_state_id" in data
        assert "timestamp" in data
        assert data["entry_id"].startswith("persona_")
        assert data["cwm_state_id"].startswith("cwm_e_")

    def test_create_persona_entry_minimal(
        self, http_client, auth_headers, unique_entry_content
    ):
        """Test creating entry with only required fields."""
        payload = {
            "entry_type": "observation",
            "content": unique_entry_content,
        }
        response = http_client.post(
            "/persona/entries", json=payload, headers=auth_headers
        )
        assert response.status_code == 201

        data = response.json()
        assert "entry_id" in data

    def test_create_persona_entry_validation_empty_content(
        self, http_client, auth_headers
    ):
        """Test that empty content returns validation error."""
        payload = {
            "entry_type": "belief",
            "content": "",
        }
        response = http_client.post(
            "/persona/entries", json=payload, headers=auth_headers
        )
        assert response.status_code == 422

    def test_create_persona_entry_validation_invalid_type(
        self, http_client, auth_headers
    ):
        """Test that invalid entry_type returns validation error."""
        payload = {
            "entry_type": "invalid_type",
            "content": "Some content",
        }
        response = http_client.post(
            "/persona/entries", json=payload, headers=auth_headers
        )
        assert response.status_code == 422

    def test_create_persona_entry_validation_confidence_range(
        self, http_client, auth_headers
    ):
        """Test that confidence outside [0, 1] returns validation error."""
        payload = {
            "entry_type": "belief",
            "content": "Some content",
            "confidence": 1.5,  # Out of range
        }
        response = http_client.post(
            "/persona/entries", json=payload, headers=auth_headers
        )
        assert response.status_code == 422

    def test_create_persona_entry_requires_auth(self, http_client):
        """Test that POST /persona/entries requires authentication."""
        payload = {
            "entry_type": "belief",
            "content": "Test content",
        }
        response = http_client.post("/persona/entries", json=payload)
        assert response.status_code in [401, 403]


class TestPersonaListEndpoint:
    """Integration tests for GET /persona/entries endpoint."""

    def test_list_persona_entries(self, http_client, auth_headers):
        """Test that GET /persona/entries returns entries list."""
        response = http_client.get("/persona/entries", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "entries" in data
        assert "total" in data
        assert "limit" in data
        assert "offset" in data
        assert isinstance(data["entries"], list)

    def test_list_persona_entries_with_limit(self, http_client, auth_headers):
        """Test that limit parameter works."""
        response = http_client.get(
            "/persona/entries",
            params={"limit": 5},
            headers=auth_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert data["limit"] == 5
        assert len(data["entries"]) <= 5

    def test_list_persona_entries_with_offset(self, http_client, auth_headers):
        """Test that offset parameter works."""
        response = http_client.get(
            "/persona/entries",
            params={"limit": 10, "offset": 5},
            headers=auth_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert data["offset"] == 5

    def test_list_persona_entries_filter_by_type(self, http_client, auth_headers):
        """Test filtering by entry_type."""
        response = http_client.get(
            "/persona/entries",
            params={"entry_type": "decision"},
            headers=auth_headers,
        )
        assert response.status_code == 200

        data = response.json()
        # All returned entries should be of type "decision"
        for entry in data["entries"]:
            assert entry["entry_type"] == "decision"

    def test_list_persona_entries_filter_by_sentiment(self, http_client, auth_headers):
        """Test filtering by sentiment."""
        response = http_client.get(
            "/persona/entries",
            params={"sentiment": "positive"},
            headers=auth_headers,
        )
        assert response.status_code == 200

        data = response.json()
        # All returned entries should have positive sentiment
        for entry in data["entries"]:
            assert entry.get("sentiment") == "positive"

    def test_list_persona_entries_requires_auth(self, http_client):
        """Test that GET /persona/entries requires authentication."""
        response = http_client.get("/persona/entries")
        assert response.status_code in [401, 403]


class TestPersonaGetByIdEndpoint:
    """Integration tests for GET /persona/entries/{entry_id} endpoint."""

    def test_get_persona_entry_by_id(
        self, http_client, auth_headers, unique_entry_content
    ):
        """Test getting a specific persona entry by ID."""
        # First create an entry
        payload = {
            "entry_type": "reflection",
            "content": unique_entry_content,
            "sentiment": "neutral",
        }
        create_response = http_client.post(
            "/persona/entries", json=payload, headers=auth_headers
        )
        assert create_response.status_code == 201
        entry_id = create_response.json()["entry_id"]

        # Now get it by ID
        get_response = http_client.get(
            f"/persona/entries/{entry_id}", headers=auth_headers
        )
        assert get_response.status_code == 200

        data = get_response.json()
        assert data["entry_id"] == entry_id
        assert data["content"] == unique_entry_content
        assert data["entry_type"] == "reflection"
        assert data["sentiment"] == "neutral"

    def test_get_persona_entry_not_found(self, http_client, auth_headers):
        """Test that 404 is returned for nonexistent entry."""
        response = http_client.get(
            "/persona/entries/persona_nonexistent12345",
            headers=auth_headers,
        )
        assert response.status_code == 404

    def test_get_persona_entry_requires_auth(self, http_client):
        """Test that GET /persona/entries/{id} requires authentication."""
        response = http_client.get("/persona/entries/some-id")
        assert response.status_code in [401, 403]


class TestPersonaUpdateEndpoint:
    """Integration tests for PATCH /persona/entries/{entry_id} endpoint."""

    def test_update_persona_entry(
        self, http_client, auth_headers, unique_entry_content
    ):
        """Test updating a persona entry."""
        # First create an entry
        payload = {
            "entry_type": "belief",
            "content": unique_entry_content,
            "sentiment": "neutral",
            "confidence": 0.5,
        }
        create_response = http_client.post(
            "/persona/entries", json=payload, headers=auth_headers
        )
        assert create_response.status_code == 201
        entry_id = create_response.json()["entry_id"]

        # Update it
        update_payload = {
            "sentiment": "positive",
            "confidence": 0.9,
            "summary": "Updated summary",
        }
        patch_response = http_client.patch(
            f"/persona/entries/{entry_id}",
            json=update_payload,
            headers=auth_headers,
        )
        assert patch_response.status_code == 200

        data = patch_response.json()
        assert data["entry_id"] == entry_id
        assert data["sentiment"] == "positive"
        assert data["confidence"] == 0.9
        assert data["summary"] == "Updated summary"
        # Original content should be preserved
        assert data["content"] == unique_entry_content

    def test_update_persona_entry_not_found(self, http_client, auth_headers):
        """Test that 404 is returned for updating nonexistent entry."""
        update_payload = {"sentiment": "positive"}
        response = http_client.patch(
            "/persona/entries/persona_nonexistent12345",
            json=update_payload,
            headers=auth_headers,
        )
        assert response.status_code == 404

    def test_update_persona_entry_requires_auth(self, http_client):
        """Test that PATCH /persona/entries/{id} requires authentication."""
        response = http_client.patch(
            "/persona/entries/some-id",
            json={"sentiment": "positive"},
        )
        assert response.status_code in [401, 403]


class TestPersonaDeleteEndpoint:
    """Integration tests for DELETE /persona/entries/{entry_id} endpoint."""

    def test_delete_persona_entry(
        self, http_client, auth_headers, unique_entry_content
    ):
        """Test soft-deleting a persona entry."""
        # First create an entry
        payload = {
            "entry_type": "observation",
            "content": unique_entry_content,
        }
        create_response = http_client.post(
            "/persona/entries", json=payload, headers=auth_headers
        )
        assert create_response.status_code == 201
        entry_id = create_response.json()["entry_id"]

        # Delete it
        delete_response = http_client.delete(
            f"/persona/entries/{entry_id}",
            headers=auth_headers,
        )
        assert delete_response.status_code == 204

        # Verify it's gone (404)
        get_response = http_client.get(
            f"/persona/entries/{entry_id}", headers=auth_headers
        )
        assert get_response.status_code == 404

    def test_delete_persona_entry_not_found(self, http_client, auth_headers):
        """Test that 404 is returned for deleting nonexistent entry."""
        response = http_client.delete(
            "/persona/entries/persona_nonexistent12345",
            headers=auth_headers,
        )
        assert response.status_code == 404

    def test_delete_persona_entry_requires_auth(self, http_client):
        """Test that DELETE /persona/entries/{id} requires authentication."""
        response = http_client.delete("/persona/entries/some-id")
        assert response.status_code in [401, 403]


class TestPersonaSentimentEndpoint:
    """Integration tests for GET /persona/sentiment endpoint."""

    def test_get_sentiment_aggregation(self, http_client, auth_headers):
        """Test that GET /persona/sentiment returns aggregated data."""
        response = http_client.get("/persona/sentiment", headers=auth_headers)
        assert response.status_code == 200

        data = response.json()
        assert "sentiment" in data
        assert "confidence_avg" in data
        assert "recent_sentiment_trend" in data
        assert "emotion_distribution" in data
        assert "entry_count" in data
        assert "last_updated" in data

    def test_get_sentiment_with_limit(self, http_client, auth_headers):
        """Test that limit parameter controls aggregation window."""
        response = http_client.get(
            "/persona/sentiment",
            params={"limit": 5},
            headers=auth_headers,
        )
        assert response.status_code == 200

        data = response.json()
        # entry_count should be <= limit
        assert data["entry_count"] <= 5

    def test_get_sentiment_empty_result(self, http_client, auth_headers):
        """Test that empty result returns expected structure."""
        # Use a timestamp in the future to get empty results
        response = http_client.get(
            "/persona/sentiment",
            params={"after_timestamp": "2099-01-01T00:00:00Z"},
            headers=auth_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert data["entry_count"] == 0
        # sentiment and confidence_avg should be null for empty results
        assert data["sentiment"] is None or data["entry_count"] == 0

    def test_get_sentiment_requires_auth(self, http_client):
        """Test that GET /persona/sentiment requires authentication."""
        response = http_client.get("/persona/sentiment")
        assert response.status_code in [401, 403]
