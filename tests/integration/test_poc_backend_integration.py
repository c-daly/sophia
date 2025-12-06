"""Integration test for V-JEPA PoC backend.

This test validates that the PoC backend works end-to-end with the same
API contracts as the stub backend.
"""

import os
import pytest
import tempfile
from PIL import Image

from sophia.jepa import JEPARunner
from sophia.jepa.models import SimulationContext, Entity


pytestmark = pytest.mark.integration


@pytest.fixture
def poc_runner():
    """Create a JEPARunner with PoC backend."""
    # Save original value
    original = os.environ.get("JEPA_BACKEND")

    # Set to PoC
    os.environ["JEPA_BACKEND"] = "poc"
    runner = JEPARunner()

    yield runner

    # Restore original
    if original:
        os.environ["JEPA_BACKEND"] = original
    else:
        os.environ.pop("JEPA_BACKEND", None)


@pytest.fixture
def sample_image():
    """Create a temporary sample image."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.new("RGB", (128, 128), color=(100, 150, 200))
        img.save(f.name)
        yield f.name
    # Cleanup
    try:
        os.unlink(f.name)
    except Exception:
        pass


@pytest.mark.asyncio
async def test_poc_media_processing_integration(poc_runner, sample_image):
    """Test end-to-end media processing with PoC backend."""
    result = await poc_runner.process_media_sample(
        sample_id="integration_test_sample",
        file_path=sample_image,
        media_type="image",
        metadata={"source": "integration_test"},
    )

    # Validate API contract
    assert result["sample_id"] == "integration_test_sample"
    assert result["embedding_dim"] == 768
    assert "visual" in result["embeddings"]
    assert "physics" in result["embeddings"]
    assert len(result["embeddings"]["visual"]) == 768
    assert len(result["embeddings"]["physics"]) == 768
    assert 0.0 < result["confidence"] <= 1.0

    # Validate PoC-specific metadata
    assert result["model_version"] == "jepa-stub-v1.0"
    assert result["metadata"]["device"] == "cpu"


def test_poc_simulation_integration(poc_runner):
    """Test end-to-end simulation with PoC backend."""
    entities = [
        Entity(
            id="test_block",
            type="object",
            properties={"mass": 1.0},
            position={"x": 0.0, "y": 0.0, "z": 0.5},
        )
    ]

    context = SimulationContext(entities=entities)
    result = poc_runner.simulate(context, k_steps=5)

    # Validate API contract
    assert result.k_steps == 5
    assert len(result.imagined_states) == 5
    assert len(result.imagined_processes) >= 1
    assert 0.0 < result.overall_confidence <= 1.0

    # Validate all states are marked as imagined
    for state in result.imagined_states:
        assert state.imagined is True
        assert state.model_version == "jepa-stub-v1.0"

    # Validate PoC-specific process metadata
    for process in result.imagined_processes:
        assert process.properties.get("backend") == "poc"


def test_poc_simulation_with_actions(poc_runner):
    """Test PoC simulation with action sequence."""
    entities = [
        Entity(
            id="robot",
            type="agent",
            properties={"status": "idle"},
            position={"x": 0.0, "y": 0.0, "z": 0.0},
        ),
        Entity(
            id="target",
            type="object",
            properties={"grasped": False},
            position={"x": 0.5, "y": 0.0, "z": 0.1},
        ),
    ]

    actions = [
        {"type": "MOVE", "target": "robot", "target_position": {"x": 0.5, "y": 0.0}},
        {"type": "GRASP", "target": "target"},
    ]

    context = SimulationContext(entities=entities, actions=actions)
    result = poc_runner.simulate(context, k_steps=3)

    # Validate action processes were created
    action_processes = [
        p for p in result.imagined_processes if p.properties.get("type") == "action"
    ]
    assert len(action_processes) == 2

    # Check that actions are reflected in processes
    assert any("MOVE" in p.description for p in action_processes)
    assert any("GRASP" in p.description for p in action_processes)


def test_poc_backend_selection():
    """Test that backend selection works via environment variable."""
    # Test default (stub)
    os.environ["JEPA_BACKEND"] = "stub"
    stub_runner = JEPARunner()
    assert stub_runner._backend.__class__.__name__ == "StubJEPABackend"

    # Test PoC
    os.environ["JEPA_BACKEND"] = "poc"
    poc_runner = JEPARunner()
    assert poc_runner._backend.__class__.__name__ == "PoCJEPABackend"

    # Clean up
    os.environ["JEPA_BACKEND"] = "stub"


def test_poc_confidence_pattern(poc_runner):
    """Test that PoC has expected confidence decay pattern."""
    entities = [Entity(id="obj", type="object")]
    context = SimulationContext(entities=entities)

    result = poc_runner.simulate(context, k_steps=10)

    confidences = [state.confidence for state in result.imagined_states]

    # PoC uses exponential decay, so check pattern
    assert confidences[0] > 0.9  # High initial confidence
    assert confidences[-1] < confidences[0]  # Decreasing

    # Check monotonic decrease
    for i in range(len(confidences) - 1):
        assert confidences[i] >= confidences[i + 1]


@pytest.mark.asyncio
async def test_poc_api_compatibility_with_stub(sample_image):
    """Verify PoC and stub produce compatible API structures."""
    # Create stub runner
    os.environ["JEPA_BACKEND"] = "stub"
    stub_runner = JEPARunner()

    # Create PoC runner
    os.environ["JEPA_BACKEND"] = "poc"
    poc_runner = JEPARunner()

    # Test media processing
    stub_result = await stub_runner.process_media_sample(
        sample_id="test",
        file_path=sample_image,
        media_type="image",
        metadata={},
    )

    poc_result = await poc_runner.process_media_sample(
        sample_id="test",
        file_path=sample_image,
        media_type="image",
        metadata={},
    )

    # Check both have same structure
    assert set(stub_result.keys()) == set(poc_result.keys())
    assert set(stub_result["embeddings"].keys()) == set(poc_result["embeddings"].keys())

    # Test simulation
    entities = [Entity(id="obj", type="object")]
    context = SimulationContext(entities=entities)

    stub_sim = stub_runner.simulate(context, k_steps=3)
    poc_sim = poc_runner.simulate(context, k_steps=3)

    # Check both have same structure
    assert stub_sim.k_steps == poc_sim.k_steps
    assert len(stub_sim.imagined_states) == len(poc_sim.imagined_states)
    assert all(
        hasattr(stub_sim, attr)
        for attr in ["simulation_id", "k_steps", "overall_confidence"]
    )
    assert all(
        hasattr(poc_sim, attr)
        for attr in ["simulation_id", "k_steps", "overall_confidence"]
    )

    # Clean up
    os.environ["JEPA_BACKEND"] = "stub"
