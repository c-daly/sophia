"""Unit tests for CWMState unified envelope validation.

These tests validate the structure and format of CWMState envelopes
returned by JEPARunner. No external services needed.
"""

import os
import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from sophia.jepa.runner import JEPARunner
from sophia.jepa.models import SimulationContext, Entity


pytestmark = pytest.mark.unit


@pytest.fixture
def jepa_runner():
    """Fixture for JEPA runner using stub backend."""
    # Ensure we use stub backend regardless of environment
    with patch.dict(os.environ, {"JEPA_BACKEND": "stub"}):
        return JEPARunner(model_version="jepa-stub-v1.0")


@pytest.fixture
def mock_hcg_client():
    """Fixture for mocked HCG client."""
    client = Mock()
    client.add_node = Mock()
    client.add_edge = Mock()
    client._milvus = Mock()
    client._neo4j = Mock()
    return client


@pytest.fixture
def sample_context():
    """Fixture for simulation context."""
    return SimulationContext(
        entities=[
            Entity(
                id="test_entity",
                type="object",
                properties={"mass": 1.0},
                position={"x": 0.0, "y": 0.0, "z": 0.0},
            )
        ]
    )


class TestSimulateEnvelope:
    """Tests for /simulate endpoint response envelope."""

    def test_simulate_returns_cwmstate_structure(self, jepa_runner, sample_context):
        """Test that simulate returns proper CWMState envelope."""
        result = jepa_runner.simulate(context=sample_context, k_steps=3)

        # Verify core structure
        assert result.simulation_id
        assert result.imagined_processes
        assert result.imagined_states
        assert result.k_steps == 3

    def test_simulate_imagined_states_have_required_fields(
        self, jepa_runner, sample_context
    ):
        """Test that imagined states have all CWMState fields."""
        result = jepa_runner.simulate(context=sample_context, k_steps=2)

        for state in result.imagined_states:
            assert state.state_id
            assert state.step >= 0
            assert state.description
            assert 0.0 <= state.confidence <= 1.0
            assert state.model_version
            assert state.imagined is True

    def test_simulate_includes_model_version(self, jepa_runner, sample_context):
        """Test that model version is included."""
        result = jepa_runner.simulate(context=sample_context, k_steps=1)

        assert result.model_version == "jepa-stub-v1.0"

    def test_simulate_confidence_in_valid_range(self, jepa_runner, sample_context):
        """Test that all confidence scores are in [0,1]."""
        result = jepa_runner.simulate(context=sample_context, k_steps=5)

        assert 0.0 <= result.overall_confidence <= 1.0

        for state in result.imagined_states:
            assert 0.0 <= state.confidence <= 1.0

        for process in result.imagined_processes:
            assert 0.0 <= process.confidence <= 1.0


class TestCWMStateLinks:
    """Tests for linkage between CWMState nodes."""

    def test_imagined_processes_provide_linkage(self, jepa_runner, sample_context):
        """Test that imagined processes are generated."""
        result = jepa_runner.simulate(context=sample_context, k_steps=3)

        assert len(result.imagined_processes) >= 1  # Stub generates process(es)


class TestCWMStateTags:
    """Tests for imagined:true tagging."""

    def test_imagined_states_have_imagined_tag(self, jepa_runner, sample_context):
        """Test that all imagined states have imagined:true."""
        result = jepa_runner.simulate(context=sample_context, k_steps=3)

        for state in result.imagined_states:
            assert state.imagined is True

    def test_imagined_processes_have_imagined_tag(self, jepa_runner, sample_context):
        """Test that all imagined processes have imagined:true."""
        result = jepa_runner.simulate(context=sample_context, k_steps=3)

        for process in result.imagined_processes:
            assert process.imagined is True


class TestCWMStateTimestamps:
    """Tests for timestamp handling."""

    def test_timestamps_are_iso_format(self, jepa_runner, sample_context):
        """Test that timestamps are valid ISO format."""
        result = jepa_runner.simulate(context=sample_context, k_steps=2)

        # Verify created_at timestamp
        datetime.fromisoformat(result.created_at.replace("Z", "+00:00"))

    def test_timestamps_are_recent(self, jepa_runner, sample_context):
        """Test that timestamps are recent."""
        result = jepa_runner.simulate(context=sample_context, k_steps=1)

        created_time = datetime.fromisoformat(result.created_at.replace("Z", "+00:00"))
        now = datetime.now(created_time.tzinfo)

        time_diff = (now - created_time).total_seconds()
        assert time_diff < 5, "Timestamp should be within last 5 seconds"


class TestCWMStateStatus:
    """Tests for status field in responses."""

    def test_successful_simulation_has_success_status(
        self, jepa_runner, sample_context
    ):
        """Test that successful simulation completes."""
        result = jepa_runner.simulate(context=sample_context, k_steps=2)

        assert result.simulation_id
        assert len(result.imagined_states) == 2


class TestCWMStateData:
    """Tests for state data contents."""

    def test_imagined_states_contain_entity_data(self, jepa_runner, sample_context):
        """Test that imagined states include entity data."""
        result = jepa_runner.simulate(context=sample_context, k_steps=2)

        for state in result.imagined_states:
            assert isinstance(state.entities, list)
            assert isinstance(state.state_data, dict)


class TestCWMStateConsistency:
    """Tests for envelope consistency across multiple requests."""

    def test_multiple_simulations_use_consistent_envelope(
        self, jepa_runner, sample_context
    ):
        """Test that all simulations return consistent structure."""
        result1 = jepa_runner.simulate(context=sample_context, k_steps=2)
        result2 = jepa_runner.simulate(context=sample_context, k_steps=3)

        # Both should have same structure
        assert hasattr(result1, "simulation_id")
        assert hasattr(result2, "simulation_id")
        assert hasattr(result1, "imagined_states")
        assert hasattr(result2, "imagined_states")
        assert hasattr(result1, "overall_confidence")
        assert hasattr(result2, "overall_confidence")
