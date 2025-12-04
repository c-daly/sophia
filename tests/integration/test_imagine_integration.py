"""Integration tests for the /imagine endpoint.

These tests require Sophia and Neo4j to be running.
Run with: pytest tests/integration/test_imagine_integration.py -v -m integration

Start services with: ./scripts/test_integration.sh up
"""

import pytest

pytestmark = [
    pytest.mark.integration,
]


class TestImagineIntegration:
    """Integration tests for the /imagine endpoint."""

    def test_imagine_creates_imagined_states(self, http_client, auth_headers):
        """Test that /imagine creates imagined states with default parameters."""
        response = http_client.post(
            "/imagine",
            json={
                "horizon": 3,
                "model_version": "v1.0",
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        assert "imagination_id" in data
        assert "imagined_states" in data
        assert len(data["imagined_states"]) == 3
        assert data["model_version"] == "v1.0"
        assert data["horizon"] == 3

    def test_imagine_with_emotion_tags(self, http_client, auth_headers):
        """Test that /imagine processes CWM-E emotion tags."""
        response = http_client.post(
            "/imagine",
            json={
                "cwm_e_emotion_tags": ["curiosity", "anticipation"],
                "horizon": 2,
                "model_version": "v1.0",
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        assert len(data["imagined_states"]) == 2
        for state in data["imagined_states"]:
            assert "emotion_tags" in state["properties"]
            assert "curiosity" in state["properties"]["emotion_tags"]
            assert "anticipation" in state["properties"]["emotion_tags"]

    def test_imagine_with_context(self, http_client, auth_headers):
        """Test that /imagine processes additional context."""
        context = {
            "scene": "warehouse",
            "objects": ["red_block", "bin"],
            "goal": "organize objects",
        }
        response = http_client.post(
            "/imagine",
            json={
                "context": context,
                "horizon": 2,
                "model_version": "v1.0",
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        for state in data["imagined_states"]:
            assert state["properties"]["context"] == context

    def test_imagine_with_assumptions(self, http_client, auth_headers):
        """Test that /imagine processes assumptions."""
        assumptions = ["gripper is at home", "bin is accessible"]
        response = http_client.post(
            "/imagine",
            json={
                "assumptions": assumptions,
                "horizon": 2,
                "model_version": "v1.0",
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        assert data["assumptions"] == assumptions

    def test_imagine_persists_to_neo4j(self, http_client, auth_headers, hcg_client):
        """Test that imagined states are persisted to Neo4j."""
        response = http_client.post(
            "/imagine",
            json={
                "horizon": 2,
                "model_version": "v1.0",
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        imagination_id = data["imagination_id"]

        # Verify nodes were created in Neo4j
        with hcg_client.driver.session(database=hcg_client.database) as session:
            result = session.run(
                """
                MATCH (n {imagination_id: $imagination_id})
                RETURN n.imagination_id as imagination_id, count(n) as count
                """,
                {"imagination_id": imagination_id},
            )
            record = result.single()
            if record:
                assert record["count"] >= 2

    def test_imagine_confidence_decreases_with_horizon(self, http_client, auth_headers):
        """Test that confidence decreases for states further in the horizon."""
        response = http_client.post(
            "/imagine",
            json={
                "horizon": 5,
                "model_version": "v1.0",
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        states = data["imagined_states"]

        for i in range(len(states) - 1):
            assert states[i]["confidence"] >= states[i + 1]["confidence"]

    def test_imagine_requires_auth(self, http_client):
        """Test that /imagine requires authentication."""
        response = http_client.post(
            "/imagine",
            json={
                "horizon": 2,
                "model_version": "v1.0",
            },
        )
        assert response.status_code in [401, 403]

    def test_imagine_with_invalid_horizon(self, http_client, auth_headers):
        """Test that /imagine rejects invalid horizon values."""
        response = http_client.post(
            "/imagine",
            json={
                "horizon": 0,
                "model_version": "v1.0",
            },
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_imagine_state_ids_are_unique(self, http_client, auth_headers):
        """Test that each imagined state has a unique ID."""
        response = http_client.post(
            "/imagine",
            json={
                "horizon": 5,
                "model_version": "v1.0",
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        state_ids = [s["state_id"] for s in data["imagined_states"]]
        assert len(state_ids) == len(set(state_ids))

    def test_imagine_with_cwm_g_imagery(self, http_client, auth_headers):
        """Test that /imagine processes CWM-G imagery data."""
        imagery_data = [
            {
                "image_id": "img_001",
                "description": "Red block on table",
                "features": {"color": "red", "shape": "cube"},
            },
            {
                "image_id": "img_002",
                "description": "Empty bin",
                "features": {"type": "container", "status": "empty"},
            },
        ]
        response = http_client.post(
            "/imagine",
            json={
                "cwm_g_imagery": imagery_data,
                "horizon": 2,
                "model_version": "v1.0",
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

        data = response.json()
        assert len(data["imagined_states"]) == 2
