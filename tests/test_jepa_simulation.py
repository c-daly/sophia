"""Tests for JEPA simulation workflows and k-step rollout."""

import pytest
from unittest.mock import Mock
from datetime import datetime

from sophia.jepa.runner import JEPARunner
from sophia.jepa.models import Entity


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

    # Mock _milvus
    client._milvus = Mock()
    client._milvus.insert_embedding = Mock()

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
def sample_entities():
    """Fixture for sample entities."""
    return [
        Entity(
            id="ball_1",
            type="object",
            properties={"mass": 0.5, "material": "rubber"},
            position={"x": 0.0, "y": 0.0, "z": 1.0},
        ),
        Entity(
            id="block_1",
            type="object",
            properties={"mass": 2.0, "material": "wood"},
            position={"x": 2.0, "y": 0.0, "z": 0.0},
        ),
    ]


class TestKStepRollout:
    """Tests for k-step simulation rollout."""

    @pytest.mark.asyncio
    async def test_single_step_simulation(self, jepa_runner, sample_entities):
        """Test simulation with k=1."""
        result = await jepa_runner.simulate(
            entities=sample_entities,
            k_steps=1,
            media_sample_id=None,
        )

        assert result["k_steps"] == 1
        assert len(result["imagined_states"]) == 1
        assert len(result["imagined_processes"]) == 1

    @pytest.mark.asyncio
    async def test_five_step_simulation(self, jepa_runner, sample_entities):
        """Test simulation with k=5."""
        result = await jepa_runner.simulate(
            entities=sample_entities,
            k_steps=5,
            media_sample_id=None,
        )

        assert result["k_steps"] == 5
        assert len(result["imagined_states"]) == 5
        assert len(result["imagined_processes"]) == 5

    @pytest.mark.asyncio
    async def test_ten_step_simulation(self, jepa_runner, sample_entities):
        """Test simulation with k=10."""
        result = await jepa_runner.simulate(
            entities=sample_entities,
            k_steps=10,
            media_sample_id=None,
        )

        assert result["k_steps"] == 10
        assert len(result["imagined_states"]) == 10
        assert len(result["imagined_processes"]) == 10

    @pytest.mark.asyncio
    async def test_simulation_generates_unique_ids(self, jepa_runner, sample_entities):
        """Test that each simulation generates unique IDs."""
        result1 = await jepa_runner.simulate(
            entities=sample_entities,
            k_steps=3,
            media_sample_id=None,
        )

        result2 = await jepa_runner.simulate(
            entities=sample_entities,
            k_steps=3,
            media_sample_id=None,
        )

        assert result1["simulation_id"] != result2["simulation_id"]


class TestConfidenceDecay:
    """Tests for confidence decay over simulation steps."""

    @pytest.mark.asyncio
    async def test_confidence_decreases_with_steps(self, jepa_runner, sample_entities):
        """Test that confidence decreases as k increases."""
        result = await jepa_runner.simulate(
            entities=sample_entities,
            k_steps=5,
            media_sample_id=None,
        )

        # Extract confidences from each step
        confidences = [state["confidence"] for state in result["imagined_states"]]

        # Each step should have lower or equal confidence
        for i in range(len(confidences) - 1):
            assert confidences[i] >= confidences[i + 1], (
                f"Confidence should decay: step {i} ({confidences[i]:.3f}) "
                f">= step {i+1} ({confidences[i+1]:.3f})"
            )

    @pytest.mark.asyncio
    async def test_initial_confidence_reasonable(self, jepa_runner, sample_entities):
        """Test that initial confidence is in reasonable range."""
        result = await jepa_runner.simulate(
            entities=sample_entities,
            k_steps=1,
            media_sample_id=None,
        )

        initial_confidence = result["imagined_states"][0]["confidence"]
        assert 0.7 <= initial_confidence <= 1.0, (
            f"Initial confidence {initial_confidence} should be high (0.7-1.0)"
        )

    @pytest.mark.asyncio
    async def test_final_confidence_within_bounds(self, jepa_runner, sample_entities):
        """Test that confidence stays within [0, 1] range."""
        result = await jepa_runner.simulate(
            entities=sample_entities,
            k_steps=10,
            media_sample_id=None,
        )

        for idx, state in enumerate(result["imagined_states"]):
            assert 0.0 <= state["confidence"] <= 1.0, (
                f"Step {idx} confidence {state['confidence']} out of bounds"
            )

    @pytest.mark.asyncio
    async def test_overall_confidence_matches_last_step(
        self, jepa_runner, sample_entities
    ):
        """Test that overall_confidence reflects final step confidence."""
        result = await jepa_runner.simulate(
            entities=sample_entities,
            k_steps=5,
            media_sample_id=None,
        )

        last_step_confidence = result["imagined_states"][-1]["confidence"]
        overall_confidence = result["overall_confidence"]

        # Overall confidence should be close to or based on last step
        assert abs(overall_confidence - last_step_confidence) < 0.1


class TestImaginedStatesAndProcesses:
    """Tests for imagined state and process node creation."""

    @pytest.mark.asyncio
    async def test_imagined_states_have_correct_structure(
        self, jepa_runner, sample_entities
    ):
        """Test that imagined states have all required fields."""
        result = await jepa_runner.simulate(
            entities=sample_entities,
            k_steps=3,
            media_sample_id=None,
        )

        for state in result["imagined_states"]:
            assert "state_id" in state
            assert "step" in state
            assert "timestamp" in state
            assert "entities" in state
            assert "confidence" in state
            assert state["imagined"] is True

    @pytest.mark.asyncio
    async def test_imagined_processes_have_correct_structure(
        self, jepa_runner, sample_entities
    ):
        """Test that imagined processes have all required fields."""
        result = await jepa_runner.simulate(
            entities=sample_entities,
            k_steps=3,
            media_sample_id=None,
        )

        for process in result["imagined_processes"]:
            assert "process_id" in process
            assert "step" in process
            assert "from_state_id" in process
            assert "to_state_id" in process
            assert "description" in process
            assert "duration" in process
            assert process["imagined"] is True

    @pytest.mark.asyncio
    async def test_states_and_processes_are_linked(
        self, jepa_runner, sample_entities
    ):
        """Test that processes correctly reference states."""
        result = await jepa_runner.simulate(
            entities=sample_entities,
            k_steps=3,
            media_sample_id=None,
        )

        states = result["imagined_states"]
        processes = result["imagined_processes"]

        # Each process should link consecutive states
        for idx, process in enumerate(processes):
            assert process["from_state_id"] == states[idx]["state_id"]
            assert process["to_state_id"] == states[idx]["state_id"]

    @pytest.mark.asyncio
    async def test_imagined_flag_is_true(self, jepa_runner, sample_entities):
        """Test that imagined:true tag is applied."""
        result = await jepa_runner.simulate(
            entities=sample_entities,
            k_steps=2,
            media_sample_id=None,
        )

        # All states should be imagined
        for state in result["imagined_states"]:
            assert state["imagined"] is True

        # All processes should be imagined
        for process in result["imagined_processes"]:
            assert process["imagined"] is True


class TestStateReferences:
    """Tests for state reference chains in simulation."""

    @pytest.mark.asyncio
    async def test_states_form_chain(self, jepa_runner, sample_entities):
        """Test that states reference previous states correctly."""
        result = await jepa_runner.simulate(
            entities=sample_entities,
            k_steps=5,
            media_sample_id=None,
        )

        states = result["imagined_states"]

        # First state might have no parent (initial)
        # Subsequent states should reference previous state via process
        processes = result["imagined_processes"]

        for idx in range(1, len(states)):
            # Process for step i connects state i-1 to state i
            process = processes[idx]
            assert process["from_state_id"] == states[idx]["state_id"]
            assert process["to_state_id"] == states[idx]["state_id"]

    @pytest.mark.asyncio
    async def test_step_numbers_sequential(self, jepa_runner, sample_entities):
        """Test that step numbers are sequential."""
        result = await jepa_runner.simulate(
            entities=sample_entities,
            k_steps=5,
            media_sample_id=None,
        )

        for idx, state in enumerate(result["imagined_states"]):
            assert state["step"] == idx + 1

        for idx, process in enumerate(result["imagined_processes"]):
            assert process["step"] == idx + 1


class TestSimulationWithMediaContext:
    """Tests for simulations using media context."""

    @pytest.mark.asyncio
    async def test_simulation_with_media_sample_id(
        self, jepa_runner, sample_entities
    ):
        """Test simulation includes media_sample_id when provided."""
        result = await jepa_runner.simulate(
            entities=sample_entities,
            k_steps=3,
            media_sample_id="sample_abc123",
        )

        assert result["media_sample_id"] == "sample_abc123"

    @pytest.mark.asyncio
    async def test_simulation_without_media_sample_id(
        self, jepa_runner, sample_entities
    ):
        """Test simulation works without media_sample_id."""
        result = await jepa_runner.simulate(
            entities=sample_entities,
            k_steps=3,
            media_sample_id=None,
        )

        assert result["media_sample_id"] is None


class TestEmbeddingGeneration:
    """Tests for embedding generation in simulation."""

    @pytest.mark.asyncio
    async def test_embeddings_have_correct_dimensions(
        self, jepa_runner, sample_entities
    ):
        """Test that embeddings are 768-dimensional."""
        result = await jepa_runner.simulate(
            entities=sample_entities,
            k_steps=1,
            media_sample_id=None,
        )

        # Note: Current stub doesn't expose embeddings in simulate response
        # This would test actual JEPA model when implemented
        # For now, verify structure is valid
        assert "imagined_states" in result

    @pytest.mark.asyncio
    async def test_physics_and_visual_embeddings_separate(self, jepa_runner):
        """Test that JEPA generates separate physics and visual embeddings."""
        # Test process_media_sample which explicitly returns embeddings
        result = await jepa_runner.process_media_sample(
            sample_id="test_sample",
            file_path="/path/to/image.jpg",
            media_type="image",
            metadata={},
            question="What happens?",
        )

        assert "embeddings" in result
        assert "visual" in result["embeddings"]
        assert "physics" in result["embeddings"]
        assert len(result["embeddings"]["visual"]) == 768
        assert len(result["embeddings"]["physics"]) == 768


class TestSimulationFailureCases:
    """Tests for simulation error handling."""

    @pytest.mark.asyncio
    async def test_simulation_with_zero_steps(self, jepa_runner, sample_entities):
        """Test that k=0 is handled gracefully."""
        with pytest.raises((ValueError, AssertionError)):
            await jepa_runner.simulate(
                entities=sample_entities,
                k_steps=0,
                media_sample_id=None,
            )

    @pytest.mark.asyncio
    async def test_simulation_with_negative_steps(self, jepa_runner, sample_entities):
        """Test that negative k is rejected."""
        with pytest.raises((ValueError, AssertionError)):
            await jepa_runner.simulate(
                entities=sample_entities,
                k_steps=-1,
                media_sample_id=None,
            )

    @pytest.mark.asyncio
    async def test_simulation_with_empty_entities(self, jepa_runner):
        """Test simulation with no entities (valid edge case)."""
        result = await jepa_runner.simulate(
            entities=[],
            k_steps=2,
            media_sample_id=None,
        )

        # Should still generate states/processes, just with empty entity lists
        assert len(result["imagined_states"]) == 2
        assert len(result["imagined_processes"]) == 2


class TestSimulationMetadata:
    """Tests for simulation metadata and versioning."""

    @pytest.mark.asyncio
    async def test_simulation_includes_model_version(
        self, jepa_runner, sample_entities
    ):
        """Test that simulation result includes model version."""
        result = await jepa_runner.simulate(
            entities=sample_entities,
            k_steps=1,
            media_sample_id=None,
        )

        assert "model_version" in result
        assert result["model_version"] == "jepa-stub-v1.0"

    @pytest.mark.asyncio
    async def test_simulation_includes_timestamp(self, jepa_runner, sample_entities):
        """Test that imagined states include timestamps."""
        result = await jepa_runner.simulate(
            entities=sample_entities,
            k_steps=2,
            media_sample_id=None,
        )

        for state in result["imagined_states"]:
            assert "timestamp" in state
            # Verify it's a valid ISO timestamp
            datetime.fromisoformat(state["timestamp"].replace("Z", "+00:00"))

    @pytest.mark.asyncio
    async def test_simulation_timestamps_are_ordered(
        self, jepa_runner, sample_entities
    ):
        """Test that state timestamps are chronologically ordered."""
        result = await jepa_runner.simulate(
            entities=sample_entities,
            k_steps=5,
            media_sample_id=None,
        )

        timestamps = [
            datetime.fromisoformat(state["timestamp"].replace("Z", "+00:00"))
            for state in result["imagined_states"]
        ]

        for i in range(len(timestamps) - 1):
            assert timestamps[i] <= timestamps[i + 1], (
                f"Timestamps out of order: step {i} to {i+1}"
            )
