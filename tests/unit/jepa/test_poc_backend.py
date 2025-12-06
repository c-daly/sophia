"""Tests for PoC JEPA backend."""

import os
import pytest
import tempfile
from PIL import Image
import numpy as np

from sophia.jepa.poc_backend import PoCJEPABackend
from sophia.jepa.models import (
    SimulationContext,
    Entity,
    SensorReference,
)


pytestmark = pytest.mark.unit


@pytest.fixture
def poc_backend():
    """Create a PoC backend instance for testing."""
    return PoCJEPABackend(model_version="test-poc-v1.0")


@pytest.fixture
def sample_image():
    """Create a temporary sample image for testing."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.new("RGB", (100, 100), color=(73, 109, 137))
        img.save(f.name)
        yield f.name
    # Cleanup
    try:
        os.unlink(f.name)
    except Exception:
        pass


def test_poc_backend_initialization(poc_backend):
    """Test that PoC backend initializes correctly."""
    assert poc_backend.model_version == "test-poc-v1.0"
    assert poc_backend.embedding_dim == 768
    assert poc_backend.device == "cpu"
    assert poc_backend._initialized is True


def test_poc_backend_with_config():
    """Test PoC backend initialization with custom configuration."""
    backend = PoCJEPABackend(
        model_version="custom-v1.0",
        confidence_decay=0.1,
        device="cuda:0",
        dtype="fp16",
    )
    assert backend.model_version == "custom-v1.0"
    assert backend.confidence_decay == 0.1
    assert backend.device == "cuda:0"
    assert backend.dtype == "fp16"


@pytest.mark.asyncio
async def test_poc_media_processing(poc_backend, sample_image):
    """Test media sample processing with PoC backend."""
    result = await poc_backend.process_media_sample(
        sample_id="test_sample_1",
        file_path=sample_image,
        media_type="image",
        metadata={"width": 100, "height": 100},
        question="What objects are visible?",
    )
    
    # Check result structure
    assert result["sample_id"] == "test_sample_1"
    assert result["media_type"] == "image"
    assert result["model_version"] == "test-poc-v1.0"
    assert result["embedding_dim"] == 768
    
    # Check embeddings
    assert "embeddings" in result
    assert "visual" in result["embeddings"]
    assert "physics" in result["embeddings"]
    
    visual_emb = result["embeddings"]["visual"]
    physics_emb = result["embeddings"]["physics"]
    
    # Check embedding dimensions
    assert len(visual_emb) == 768
    assert len(physics_emb) == 768
    
    # Check embeddings are normalized (roughly unit length)
    visual_norm = np.linalg.norm(visual_emb)
    physics_norm = np.linalg.norm(physics_emb)
    assert 0.9 < visual_norm < 1.1
    assert 0.9 < physics_norm < 1.1
    
    # Check confidence
    assert 0.0 < result["confidence"] <= 1.0
    
    # Check metadata
    assert result["metadata"]["file_path"] == sample_image
    assert result["metadata"]["question"] == "What objects are visible?"


@pytest.mark.asyncio
async def test_poc_media_processing_missing_file(poc_backend):
    """Test media processing with missing file (should use fallback)."""
    result = await poc_backend.process_media_sample(
        sample_id="test_sample_missing",
        file_path="/nonexistent/path/image.jpg",
        media_type="image",
        metadata={},
    )
    
    # Should still produce valid embeddings using deterministic fallback
    assert result["sample_id"] == "test_sample_missing"
    assert len(result["embeddings"]["visual"]) == 768
    assert len(result["embeddings"]["physics"]) == 768


def test_poc_simulate_basic(poc_backend):
    """Test basic simulation with PoC backend."""
    entities = [
        Entity(
            id="block_1",
            type="object",
            properties={"mass": 0.5},
            position={"x": 0.0, "y": 0.0, "z": 0.1},
        )
    ]
    
    context = SimulationContext(entities=entities)
    result = poc_backend.simulate(context, k_steps=3)
    
    # Check basic structure
    assert result.simulation_id is not None
    assert result.k_steps == 3
    assert len(result.imagined_states) == 3
    assert len(result.imagined_processes) >= 1
    assert 0.0 <= result.overall_confidence <= 1.0
    
    # Check PoC-specific metadata
    for process in result.imagined_processes:
        assert process.properties.get("backend") == "poc"
        assert process.model_version == "test-poc-v1.0"


def test_poc_simulate_with_actions(poc_backend):
    """Test simulation with actions using PoC backend."""
    entities = [
        Entity(
            id="robot",
            type="agent",
            properties={"status": "idle"},
            position={"x": 0.0, "y": 0.0, "z": 0.0},
        )
    ]
    
    actions = [
        {"type": "MOVE", "target": "robot", "target_position": {"x": 1.0, "y": 0.0}},
        {"type": "GRASP", "target": "robot"},
    ]
    
    context = SimulationContext(entities=entities, actions=actions)
    result = poc_backend.simulate(context, k_steps=3, assumptions=["robot is functional"])
    
    # Check action processes were created
    action_processes = [p for p in result.imagined_processes if p.properties.get("type") == "action"]
    assert len(action_processes) == 2
    
    # Check assumptions propagated
    assert result.imagined_states[0].assumptions == ["robot is functional"]


def test_poc_confidence_decay(poc_backend):
    """Test that PoC backend has realistic confidence decay."""
    entities = [Entity(id="obj", type="object")]
    context = SimulationContext(entities=entities)
    
    result = poc_backend.simulate(context, k_steps=10)
    
    confidences = [state.confidence for state in result.imagined_states]
    
    # Check exponential-like decay
    for i in range(len(confidences) - 1):
        assert confidences[i] >= confidences[i + 1]
    
    # First step should be high confidence
    assert confidences[0] > 0.9
    
    # Later steps should show decay
    assert confidences[-1] < confidences[0]


def test_poc_physics_dynamics(poc_backend):
    """Test that PoC backend applies physics-like dynamics."""
    entities = [
        Entity(
            id="heavy_obj",
            type="object",
            properties={"mass": 2.0},
            position={"x": 0.0, "y": 0.0, "z": 1.0},
        )
    ]
    
    context = SimulationContext(entities=entities)
    result = poc_backend.simulate(context, k_steps=5)
    
    # Check that position changes over time
    initial_state = result.imagined_states[0]
    final_state = result.imagined_states[-1]
    
    initial_entity = initial_state.entities[0]
    final_entity = final_state.entities[0]
    
    # Position should change
    assert initial_entity.position != final_entity.position
    
    # Z should decrease (gravity effect)
    assert final_entity.position["z"] <= initial_entity.position["z"]


def test_poc_deterministic_features(poc_backend):
    """Test that deterministic feature generation is consistent."""
    # Same seed should produce same features
    features1 = poc_backend._generate_deterministic_features("test_seed")
    features2 = poc_backend._generate_deterministic_features("test_seed")
    
    assert np.allclose(features1, features2)
    
    # Different seeds should produce different features
    features3 = poc_backend._generate_deterministic_features("different_seed")
    assert not np.allclose(features1, features3)


def test_poc_projection_dimensions(poc_backend):
    """Test that projection produces correct dimensions."""
    features = np.random.randn(512).astype(np.float32)
    
    visual_emb = poc_backend._project_to_embedding(features, "visual")
    physics_emb = poc_backend._project_to_embedding(features, "physics")
    
    assert len(visual_emb) == 768
    assert len(physics_emb) == 768
    
    # Visual and physics should be different
    assert visual_emb != physics_emb


def test_poc_with_sensors(poc_backend):
    """Test simulation with sensor references."""
    entities = [Entity(id="target", type="object")]
    sensor_refs = [
        SensorReference(
            sensor_id="camera_1",
            sensor_type="camera",
            frame_id="base_link",
        )
    ]
    
    context = SimulationContext(entities=entities, sensor_refs=sensor_refs)
    result = poc_backend.simulate(context, k_steps=2)
    
    # Check sensor info in metadata
    for state in result.imagined_states:
        assert state.state_data["metadata"]["sensor_count"] == 1


def test_poc_backend_interface_compliance(poc_backend):
    """Test that PoC backend complies with JEPABackend protocol."""
    # Check required methods exist
    assert hasattr(poc_backend, "simulate")
    assert hasattr(poc_backend, "process_media_sample")
    
    # Check they are callable
    assert callable(poc_backend.simulate)
    assert callable(poc_backend.process_media_sample)
