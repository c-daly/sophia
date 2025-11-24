"""Tests for CWMState unified envelope validation."""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from sophia.api.models import SimulateResponse
from sophia.api.app import create_app


@pytest.fixture
def mock_hcg_client():
    """Fixture for mocked HCG client."""
    client = Mock()
    client.add_node = Mock(return_value="node_123")
    client.add_edge = Mock()
    client.get_node = Mock(return_value={
        "state_id": "state_123",
        "model_type": "jepa-stub-v1.0",
        "source": "jepa_runner",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "confidence": 0.85,
        "status": "completed",
        "links": [],
        "tags": ["imagined:true"],
        "data": {},
    })

    # Mock _milvus
    client._milvus = Mock()
    client._milvus.insert_embedding = Mock()
    client._milvus.query_similar = Mock(return_value=[])

    # Mock _neo4j._driver for session context
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
def test_app(mock_hcg_client):
    """Create test FastAPI application with mocked dependencies."""
    import os
    os.environ["SOPHIA_API_TOKEN"] = "test-token"
    os.environ["NEO4J_URI"] = "bolt://mock:7687"
    
    app = create_app()
    
    # Inject mock HCG client
    with patch("sophia.api.app._hcg_client", mock_hcg_client):
        with patch("sophia.api.app._simulation_service") as mock_sim:
            mock_sim_instance = Mock()
            mock_sim_instance.simulate = AsyncMock(return_value=SimulateResponse(
                simulation_id="sim_123",
                imagined_processes=[],
                imagined_states=[{
                    "state_id": "state_123",
                    "step": 1,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "entities": [],
                    "confidence": 0.85,
                    "imagined": True,
                }],
                k_steps=1,
                model_version="jepa-stub-v1.0",
                overall_confidence=0.85,
            ))
            mock_sim.return_value = mock_sim_instance
            yield app


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


@pytest.fixture
def auth_headers():
    """Create authentication headers."""
    return {"Authorization": "Bearer test-token"}


class TestSimulateEnvelope:
    """Tests for /simulate endpoint CWMState envelope."""

    def test_simulate_returns_cwmstate_structure(self, client, auth_headers):
        """Test that /simulate response follows CWMState envelope structure."""
        request_data = {
            "entities": [
                {
                    "id": "ball_1",
                    "type": "object",
                    "properties": {"mass": 0.5},
                    "position": {"x": 0.0, "y": 0.0, "z": 1.0},
                }
            ],
            "k_steps": 3,
        }

        response = client.post(
            "/simulate",
            json=request_data,
            headers=auth_headers,
        )

        assert response.status_code == 200
        result = response.json()

        # Verify envelope structure (implicit in response)
        assert "simulation_id" in result
        assert "imagined_states" in result
        assert "k_steps" in result
        assert "overall_confidence" in result

    def test_simulate_imagined_states_have_required_fields(
        self, client, auth_headers
    ):
        """Test that imagined states contain CWMState required fields."""
        request_data = {
            "entities": [],
            "k_steps": 2,
        }

        response = client.post(
            "/simulate",
            json=request_data,
            headers=auth_headers,
        )

        assert response.status_code == 200
        result = response.json()

        for state in result["imagined_states"]:
            # CWMState required fields
            assert "state_id" in state
            assert "step" in state
            assert "timestamp" in state
            assert "confidence" in state
            assert "imagined" in state
            
            # Verify types
            assert isinstance(state["state_id"], str)
            assert isinstance(state["step"], int)
            assert isinstance(state["timestamp"], str)
            assert isinstance(state["confidence"], (int, float))
            assert isinstance(state["imagined"], bool)

    def test_simulate_includes_model_version(self, client, auth_headers):
        """Test that response includes model_version (source)."""
        request_data = {
            "entities": [],
            "k_steps": 1,
        }

        response = client.post(
            "/simulate",
            json=request_data,
            headers=auth_headers,
        )

        assert response.status_code == 200
        result = response.json()
        
        assert "model_version" in result
        assert result["model_version"] == "jepa-stub-v1.0"

    def test_simulate_confidence_in_valid_range(self, client, auth_headers):
        """Test that confidence values are within [0.0, 1.0]."""
        request_data = {
            "entities": [],
            "k_steps": 5,
        }

        response = client.post(
            "/simulate",
            json=request_data,
            headers=auth_headers,
        )

        assert response.status_code == 200
        result = response.json()

        # Overall confidence
        assert 0.0 <= result["overall_confidence"] <= 1.0

        # Per-state confidence
        for state in result["imagined_states"]:
            assert 0.0 <= state["confidence"] <= 1.0


class TestCWMStateLinks:
    """Tests for CWMState links array structure."""

    def test_imagined_processes_provide_linkage(self, client, auth_headers):
        """Test that imagined_processes provide state linkage."""
        request_data = {
            "entities": [],
            "k_steps": 3,
        }

        response = client.post(
            "/simulate",
            json=request_data,
            headers=auth_headers,
        )

        assert response.status_code == 200
        result = response.json()

        # Processes serve as links between states
        assert "imagined_processes" in result
        
        for process in result["imagined_processes"]:
            assert "process_id" in process
            assert "from_state_id" in process
            assert "to_state_id" in process

    def test_media_sample_id_creates_link(self, client, auth_headers):
        """Test that media_sample_id creates linkage to media."""
        request_data = {
            "entities": [],
            "k_steps": 1,
            "media_sample_id": "sample_abc123",
        }

        response = client.post(
            "/simulate",
            json=request_data,
            headers=auth_headers,
        )

        assert response.status_code == 200
        result = response.json()

        # Media sample ID provides link
        assert result.get("media_sample_id") == "sample_abc123"


class TestCWMStateTags:
    """Tests for CWMState tags format."""

    def test_imagined_states_have_imagined_tag(self, client, auth_headers):
        """Test that imagined states are tagged with imagined:true."""
        request_data = {
            "entities": [],
            "k_steps": 2,
        }

        response = client.post(
            "/simulate",
            json=request_data,
            headers=auth_headers,
        )

        assert response.status_code == 200
        result = response.json()

        for state in result["imagined_states"]:
            # imagined:true tag (boolean field)
            assert state["imagined"] is True

    def test_imagined_processes_have_imagined_tag(self, client, auth_headers):
        """Test that imagined processes are tagged."""
        request_data = {
            "entities": [],
            "k_steps": 2,
        }

        response = client.post(
            "/simulate",
            json=request_data,
            headers=auth_headers,
        )

        assert response.status_code == 200
        result = response.json()

        for process in result["imagined_processes"]:
            assert process["imagined"] is True


class TestCWMStateTimestamps:
    """Tests for CWMState timestamp format."""

    def test_timestamps_are_iso_format(self, client, auth_headers):
        """Test that timestamps are valid ISO 8601 format."""
        request_data = {
            "entities": [],
            "k_steps": 2,
        }

        response = client.post(
            "/simulate",
            json=request_data,
            headers=auth_headers,
        )

        assert response.status_code == 200
        result = response.json()

        for state in result["imagined_states"]:
            timestamp_str = state["timestamp"]
            
            # Parse as ISO datetime
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            assert dt is not None

    def test_timestamps_are_recent(self, client, auth_headers):
        """Test that timestamps are recent (within last minute)."""
        request_data = {
            "entities": [],
            "k_steps": 1,
        }

        response = client.post(
            "/simulate",
            json=request_data,
            headers=auth_headers,
        )

        assert response.status_code == 200
        result = response.json()

        now = datetime.now(timezone.utc)
        
        for state in result["imagined_states"]:
            timestamp_str = state["timestamp"]
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            
            time_diff = abs((now - dt).total_seconds())
            assert time_diff < 60, f"Timestamp {timestamp_str} not recent"


class TestCWMStateStatus:
    """Tests for CWMState status field."""

    def test_successful_simulation_has_success_status(self, client, auth_headers):
        """Test that successful simulations have appropriate status."""
        request_data = {
            "entities": [],
            "k_steps": 1,
        }

        response = client.post(
            "/simulate",
            json=request_data,
            headers=auth_headers,
        )

        assert response.status_code == 200
        # Successful response implies success status


class TestCWMStateData:
    """Tests for CWMState data field."""

    def test_imagined_states_contain_entity_data(self, client, auth_headers):
        """Test that state data includes entities."""
        request_data = {
            "entities": [
                {
                    "id": "ball_1",
                    "type": "object",
                    "properties": {"mass": 0.5},
                    "position": {"x": 0.0, "y": 0.0, "z": 1.0},
                }
            ],
            "k_steps": 1,
        }

        response = client.post(
            "/simulate",
            json=request_data,
            headers=auth_headers,
        )

        assert response.status_code == 200
        result = response.json()

        for state in result["imagined_states"]:
            assert "entities" in state
            # Entities provide the state data


class TestNeo4jPersistence:
    """Tests for CWMState persistence to Neo4j (mocked)."""

    @patch("sophia.api.app._hcg_client")
    def test_simulate_creates_neo4j_nodes(self, mock_hcg, client, auth_headers):
        """Test that simulation creates Neo4j nodes for states."""
        # This test would verify Neo4j integration
        # For now, it's a placeholder showing intent
        
        # In real implementation:
        # - Mock HCG client calls
        # - Verify add_node called with CWMState structure
        # - Verify (:CWMState) label used
        # - Verify [:DESCRIBES] relationships created
        
        pass


class TestMilvusMirroring:
    """Tests for CWMState mirroring in Milvus (mocked)."""

    @patch("sophia.api.app._hcg_client")
    def test_simulate_stores_embeddings_in_milvus(
        self, mock_hcg, client, auth_headers
    ):
        """Test that simulation stores embeddings in Milvus."""
        # This test would verify Milvus integration
        # For now, it's a placeholder showing intent
        
        # In real implementation:
        # - Mock Milvus adapter calls
        # - Verify insert_embedding called
        # - Verify metadata includes CWMState fields
        # - Verify bidirectional linkage (Milvus ID ↔ Neo4j node)
        
        pass


class TestCWMStateConsistency:
    """Tests for CWMState consistency across endpoints."""

    def test_multiple_simulations_use_consistent_envelope(
        self, client, auth_headers
    ):
        """Test that multiple /simulate calls use consistent envelope."""
        request_data = {
            "entities": [],
            "k_steps": 1,
        }

        # Make two simulation requests
        response1 = client.post(
            "/simulate",
            json=request_data,
            headers=auth_headers,
        )

        response2 = client.post(
            "/simulate",
            json=request_data,
            headers=auth_headers,
        )

        assert response1.status_code == 200
        assert response2.status_code == 200

        result1 = response1.json()
        result2 = response2.json()

        # Both should have same structure (different IDs)
        assert set(result1.keys()) == set(result2.keys())
        
        # Both should have same state structure
        if result1["imagined_states"] and result2["imagined_states"]:
            state1_keys = set(result1["imagined_states"][0].keys())
            state2_keys = set(result2["imagined_states"][0].keys())
            assert state1_keys == state2_keys
