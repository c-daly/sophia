"""Tests for JEPA simulation workflows and k-step rollout."""

import pytest
from datetime import datetime

from sophia.jepa.runner import JEPARunner
from sophia.jepa.models import Entity, SimulationContext


@pytest.fixture
def jepa_runner():
    """Fixture for JEPA runner."""
    return JEPARunner(model_version="jepa-stub-v1.0")


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


@pytest.fixture
def simulation_context(sample_entities):
    """Fixture for simulation context."""
    return SimulationContext(entities=sample_entities)


def run_simulation(jepa_runner, context, k_steps, assumptions=None):
    """Helper to run simulation with consistent API."""
    return jepa_runner.simulate(
        context=context, k_steps=k_steps, assumptions=assumptions
    )


class TestKStepRollout:
    """Tests for k-step simulation rollout."""

    def test_single_step_simulation(self, jepa_runner, simulation_context):
        """Test simulation with k=1."""
        result = run_simulation(jepa_runner, simulation_context, k_steps=1)

        assert result.k_steps == 1
        assert len(result.imagined_states) == 1
        assert len(result.imagined_processes) == 1

    def test_five_step_simulation(self, jepa_runner, simulation_context):
        """Test simulation with k=5."""
        result = run_simulation(jepa_runner, simulation_context, k_steps=5)

        assert result.k_steps == 5
        assert len(result.imagined_states) == 5

    def test_ten_step_simulation(self, jepa_runner, simulation_context):
        """Test simulation with k=10."""
        result = run_simulation(jepa_runner, simulation_context, k_steps=10)

        assert result.k_steps == 10
        assert len(result.imagined_states) == 10

    def test_simulation_generates_unique_ids(self, jepa_runner, simulation_context):
        """Test that each simulation generates unique IDs."""
        result1 = run_simulation(jepa_runner, simulation_context, k_steps=3)
        result2 = run_simulation(jepa_runner, simulation_context, k_steps=3)

        assert result1.simulation_id != result2.simulation_id


class TestConfidenceDecay:
    """Tests for confidence decay over simulation steps."""

    def test_confidence_decreases_with_steps(self, jepa_runner, simulation_context):
        """Test that confidence decreases as k increases."""
        result = run_simulation(jepa_runner, simulation_context, k_steps=5)

        confidences = [state.confidence for state in result.imagined_states]

        for i in range(len(confidences) - 1):
            assert confidences[i] >= confidences[i + 1], (
                f"Confidence should decay: step {i} ({confidences[i]:.3f}) "
                f">= step {i + 1} ({confidences[i + 1]:.3f})"
            )

    def test_initial_confidence_reasonable(self, jepa_runner, simulation_context):
        """Test that initial confidence is in reasonable range."""
        result = run_simulation(jepa_runner, simulation_context, k_steps=1)

        initial_confidence = result.imagined_states[0].confidence
        assert (
            0.7 <= initial_confidence <= 1.0
        ), f"Initial confidence {initial_confidence} should be high (0.7-1.0)"

    def test_final_confidence_within_bounds(self, jepa_runner, simulation_context):
        """Test that confidence stays within [0, 1] range."""
        result = run_simulation(jepa_runner, simulation_context, k_steps=10)

        for idx, state in enumerate(result.imagined_states):
            assert (
                0.0 <= state.confidence <= 1.0
            ), f"Step {idx} confidence {state.confidence} out of bounds"

    def test_overall_confidence_matches_average(self, jepa_runner, simulation_context):
        """Test that overall_confidence is average of step confidences."""
        result = run_simulation(jepa_runner, simulation_context, k_steps=5)

        avg_confidence = sum(s.confidence for s in result.imagined_states) / len(
            result.imagined_states
        )
        assert abs(result.overall_confidence - avg_confidence) < 0.01


class TestImaginedStatesAndProcesses:
    """Tests for imagined state and process node creation."""

    def test_imagined_states_have_correct_structure(
        self, jepa_runner, simulation_context
    ):
        """Test that imagined states have all required fields."""
        result = run_simulation(jepa_runner, simulation_context, k_steps=3)

        for state in result.imagined_states:
            assert state.state_id
            assert state.step >= 0
            assert isinstance(state.entities, list)
            assert 0.0 <= state.confidence <= 1.0
            assert state.imagined is True

    def test_imagined_processes_have_correct_structure(
        self, jepa_runner, simulation_context
    ):
        """Test that imagined processes have all required fields."""
        result = run_simulation(jepa_runner, simulation_context, k_steps=3)

        for process in result.imagined_processes:
            assert process.process_id
            assert process.description
            assert 0.0 <= process.confidence <= 1.0
            assert process.imagined is True

    def test_imagined_flag_is_true(self, jepa_runner, simulation_context):
        """Test that imagined:true flag is applied."""
        result = run_simulation(jepa_runner, simulation_context, k_steps=2)

        for state in result.imagined_states:
            assert state.imagined is True

        for process in result.imagined_processes:
            assert process.imagined is True


class TestStateReferences:
    """Tests for state reference chains in simulation."""

    def test_step_numbers_sequential(self, jepa_runner, simulation_context):
        """Test that step numbers are sequential."""
        result = run_simulation(jepa_runner, simulation_context, k_steps=5)

        for idx, state in enumerate(result.imagined_states):
            assert state.step == idx


class TestEmbeddingGeneration:
    """Tests for embedding generation in simulation."""

    def test_simulation_produces_valid_output(self, jepa_runner, simulation_context):
        """Test that simulation completes successfully."""
        result = run_simulation(jepa_runner, simulation_context, k_steps=1)

        assert len(result.imagined_states) == 1
        assert result.overall_confidence > 0


class TestSimulationFailureCases:
    """Tests for simulation error handling."""

    def test_simulation_with_empty_entities(self, jepa_runner):
        """Test simulation with no entities (valid edge case)."""
        context = SimulationContext(entities=[])
        result = run_simulation(jepa_runner, context, k_steps=2)

        assert len(result.imagined_states) == 2
        assert len(result.imagined_processes) >= 1  # Stub generates 1 dynamics process


class TestSimulationMetadata:
    """Tests for simulation metadata and versioning."""

    def test_simulation_includes_model_version(self, jepa_runner, simulation_context):
        """Test that simulation result includes model version."""
        result = run_simulation(jepa_runner, simulation_context, k_steps=1)

        assert result.model_version == "jepa-stub-v1.0"

    def test_simulation_includes_timestamp(self, jepa_runner, simulation_context):
        """Test that simulation includes creation timestamp."""
        result = run_simulation(jepa_runner, simulation_context, k_steps=2)

        assert result.created_at
        # Verify it's a valid ISO timestamp
        datetime.fromisoformat(result.created_at.replace("Z", "+00:00"))
