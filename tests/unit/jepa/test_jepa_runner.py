"""Tests for JEPA runner module."""

import pytest
from sophia.jepa import JEPARunner
from sophia.jepa.models import (
    SimulationContext,
    Entity,
    SensorReference,
    TalosMetadata,
)


pytestmark = pytest.mark.unit


def test_jepa_runner_initialization():
    """Test creating a JEPA runner instance."""
    runner = JEPARunner(model_version="test-v1.0")
    assert runner.model_version == "test-v1.0"
    assert runner.confidence_decay == 0.05


def test_jepa_runner_custom_confidence_decay():
    """Test creating a JEPA runner with custom confidence decay."""
    runner = JEPARunner(confidence_decay=0.1)
    assert runner.confidence_decay == 0.1


def test_jepa_runner_simulate_basic():
    """Test basic simulation with entities."""
    runner = JEPARunner()

    # Create simple context
    entities = [
        Entity(
            id="block_1",
            type="object",
            properties={"mass": 0.5},
            position={"x": 0.0, "y": 0.0, "z": 0.1},
        )
    ]

    context = SimulationContext(entities=entities)

    # Run simulation
    result = runner.simulate(context, k_steps=3)

    assert result.simulation_id is not None
    assert result.k_steps == 3
    assert len(result.imagined_states) == 3
    assert len(result.imagined_processes) >= 1
    assert 0.0 <= result.overall_confidence <= 1.0


def test_jepa_runner_simulate_with_actions():
    """Test simulation with action sequence."""
    runner = JEPARunner()

    entities = [Entity(id="robot", type="agent", properties={"status": "idle"})]

    actions = [
        {"type": "MOVE", "target": "robot", "target_position": {"x": 1.0, "y": 0.0}},
        {"type": "GRASP", "target": "robot"},
    ]

    context = SimulationContext(entities=entities, actions=actions)

    result = runner.simulate(context, k_steps=2, assumptions=["robot is functional"])

    assert len(result.imagined_states) == 2
    # Should have main process + action processes
    assert len(result.imagined_processes) >= 2
    assert result.imagined_states[0].assumptions == ["robot is functional"]


def test_jepa_runner_simulate_with_sensors():
    """Test simulation with sensor references."""
    runner = JEPARunner()

    entities = [Entity(id="camera_target", type="object")]
    sensor_refs = [
        SensorReference(
            sensor_id="camera_1", sensor_type="camera", frame_id="base_link"
        )
    ]

    context = SimulationContext(entities=entities, sensor_refs=sensor_refs)

    result = runner.simulate(context, k_steps=2)

    assert result.context.sensor_refs == sensor_refs
    assert len(result.imagined_states) == 2


def test_jepa_runner_simulate_with_talos_metadata():
    """Test simulation with Talos metadata."""
    runner = JEPARunner()

    entities = [Entity(id="test_obj", type="object")]
    talos_metadata = TalosMetadata(
        simulator_version="talos-v2.0",
        physics_engine="ODE",
        time_step=0.02,
        use_hardware=True,
        robot_model="talos",
    )

    context = SimulationContext(entities=entities, talos_metadata=talos_metadata)

    result = runner.simulate(context, k_steps=1)

    assert result.context.talos_metadata.simulator_version == "talos-v2.0"
    assert result.context.talos_metadata.physics_engine == "ODE"
    assert result.context.talos_metadata.use_hardware is True


def test_jepa_runner_imagined_states_have_metadata():
    """Test that imagined states contain required metadata."""
    runner = JEPARunner(model_version="test-v1.0")

    entities = [Entity(id="obj", type="object")]
    context = SimulationContext(entities=entities)

    result = runner.simulate(context, k_steps=2, assumptions=["test assumption"])

    for state in result.imagined_states:
        assert state.imagined is True
        assert state.model_version == "test-v1.0"
        assert state.horizon == 2
        assert "test assumption" in state.assumptions
        assert state.confidence >= 0.0
        assert state.confidence <= 1.0


def test_jepa_runner_imagined_processes_have_metadata():
    """Test that imagined processes contain required metadata."""
    runner = JEPARunner(model_version="test-v1.0")

    entities = [Entity(id="obj", type="object")]
    context = SimulationContext(entities=entities)

    result = runner.simulate(context, k_steps=3, assumptions=["test assumption"])

    for process in result.imagined_processes:
        assert process.imagined is True
        assert process.model_version == "test-v1.0"
        assert process.horizon == 3
        assert "test assumption" in process.assumptions


def test_jepa_runner_confidence_decays():
    """Test that confidence decreases with step number."""
    runner = JEPARunner(confidence_decay=0.1)

    entities = [Entity(id="obj", type="object")]
    context = SimulationContext(entities=entities)

    result = runner.simulate(context, k_steps=5)

    confidences = [state.confidence for state in result.imagined_states]

    # Confidence should generally decrease
    for i in range(len(confidences) - 1):
        assert confidences[i] >= confidences[i + 1]


def test_jepa_runner_state_evolution():
    """Test that states evolve over steps."""
    runner = JEPARunner()

    entities = [
        Entity(id="moving_obj", type="object", position={"x": 0.0, "y": 0.0, "z": 0.0})
    ]

    context = SimulationContext(entities=entities)

    result = runner.simulate(context, k_steps=3)

    # Check that step numbers are correct
    for i, state in enumerate(result.imagined_states):
        assert state.step == i


def test_jepa_runner_action_application():
    """Test that actions are applied to entities."""
    runner = JEPARunner()

    entities = [
        Entity(
            id="target_obj",
            type="object",
            properties={"grasped": False},
            position={"x": 0.0, "y": 0.0, "z": 0.0},
        )
    ]

    actions = [{"type": "GRASP", "target": "target_obj"}]

    context = SimulationContext(entities=entities, actions=actions)

    result = runner.simulate(context, k_steps=2)

    # First state should show the grasp action applied
    assert len(result.imagined_states) == 2
