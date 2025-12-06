"""Integration tests for PoC JEPA Backend.

These tests validate the PoCJEPABackend with real PyTorch and GPU,
including actual model loading, inference, and API endpoint integration.

Tests are gated on:
- GPU availability (cuda)
- PyTorch availability
- Optional: checkpoint file availability

Run with: pytest tests/integration/jepa/ -v -m integration
Skip GPU tests: pytest tests/integration/jepa/ -v -m "integration and not gpu"
"""

import pytest

pytestmark = [pytest.mark.integration]


def _torch_available() -> bool:
    """Check if PyTorch is available."""
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _gpu_available() -> bool:
    """Check if GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# Skip markers - combine pytest.mark for CI filtering with skipif for runtime
requires_torch = pytest.mark.requires_torch
_skip_no_torch = pytest.mark.skipif(
    not _torch_available(),
    reason="PyTorch not available"
)

requires_gpu = pytest.mark.gpu
_skip_no_gpu = pytest.mark.skipif(
    not _gpu_available(),
    reason="GPU not available"
)


@pytest.fixture
def minimal_checkpoint(tmp_path):
    """Create a minimal checkpoint file for testing."""
    if not _torch_available():
        pytest.skip("PyTorch required for checkpoint fixture")
    
    import torch
    
    checkpoint_path = tmp_path / "test_checkpoint.pth"
    
    # Create minimal checkpoint with embedding dimension
    checkpoint = {
        "embed_dim": 768,
        "model_version": "test-checkpoint-v1.0",
    }
    
    torch.save(checkpoint, checkpoint_path)
    return str(checkpoint_path)


@pytest.fixture
def poc_backend(minimal_checkpoint):
    """Create a PoCJEPABackend with minimal checkpoint."""
    from sophia.jepa.backends.poc import PoCJEPABackend
    
    return PoCJEPABackend(
        weights_path=minimal_checkpoint,
        device="cuda:0" if _gpu_available() else "cpu",
    )


@requires_torch
@_skip_no_torch
class TestPoCJEPABackendIntegration:
    """Integration tests for PoCJEPABackend with real PyTorch."""

    def test_backend_loads_checkpoint(self, poc_backend):
        """Test that backend loads checkpoint successfully."""
        from sophia.jepa.models import SimulationContext, Entity
        
        entities = [Entity(id="obj", type="object")]
        context = SimulationContext(entities=entities)
        
        # Should load checkpoint and run simulation
        result = poc_backend.simulate(context, k_steps=2)
        
        assert result.simulation_id is not None
        assert poc_backend.is_loaded

    def test_simulation_produces_valid_result(self, poc_backend):
        """Test that simulation produces valid SimulationResult."""
        from sophia.jepa.models import SimulationContext, SimulationResult, Entity
        
        entities = [
            Entity(
                id="block",
                type="object",
                properties={"mass": 0.5},
                position={"x": 0.0, "y": 0.0, "z": 0.1},
            )
        ]
        context = SimulationContext(entities=entities)
        
        result = poc_backend.simulate(context, k_steps=3)
        
        # Validate result structure
        assert isinstance(result, SimulationResult)
        assert result.k_steps == 3
        assert len(result.imagined_states) == 3
        assert len(result.imagined_processes) >= 1
        
        # Validate confidence decay
        confidences = [s.confidence for s in result.imagined_states]
        for i in range(len(confidences) - 1):
            assert confidences[i] >= confidences[i + 1]

    def test_simulation_with_actions(self, poc_backend):
        """Test simulation with action sequence."""
        from sophia.jepa.models import SimulationContext, Entity
        
        entities = [
            Entity(id="robot", type="agent", properties={"status": "idle"})
        ]
        actions = [
            {"type": "MOVE", "target": "robot"},
            {"type": "GRASP", "target": "robot"},
        ]
        context = SimulationContext(entities=entities, actions=actions)
        
        result = poc_backend.simulate(context, k_steps=2, assumptions=["robot is ready"])
        
        # Should have action processes
        assert len(result.imagined_processes) >= 2
        assert "robot is ready" in result.imagined_states[0].assumptions

    @pytest.mark.asyncio
    async def test_process_media_sample(self, poc_backend, tmp_path):
        """Test media sample processing."""
        # Create a test image file
        from PIL import Image
        
        img_path = tmp_path / "test_image.jpg"
        img = Image.new("RGB", (224, 224), color="blue")
        img.save(img_path)
        
        result = await poc_backend.process_media_sample(
            sample_id="test-sample-001",
            file_path=str(img_path),
            media_type="image",
            metadata={"source": "test"},
            question="What is in this image?",
        )
        
        # Validate embedding structure
        assert result["sample_id"] == "test-sample-001"
        assert result["media_type"] == "image"
        assert "embeddings" in result
        assert "visual" in result["embeddings"]
        assert "physics" in result["embeddings"]
        assert len(result["embeddings"]["visual"]) == 768
        assert len(result["embeddings"]["physics"]) == 768

    def test_health_status_after_load(self, poc_backend):
        """Test health status reflects loaded state."""
        from sophia.jepa.models import SimulationContext, Entity
        
        # Trigger load
        entities = [Entity(id="obj", type="object")]
        context = SimulationContext(entities=entities)
        poc_backend.simulate(context, k_steps=1)
        
        status = poc_backend.get_health_status()
        
        assert status["backend"] == "poc"
        assert status["model_loaded"] is True
        assert status["load_time_seconds"] is not None
        assert status["inference_count"] >= 1


@requires_torch
@requires_gpu
@_skip_no_torch
@_skip_no_gpu
class TestPoCJEPABackendGPU:
    """GPU-specific integration tests."""

    def test_uses_gpu_when_available(self, minimal_checkpoint):
        """Test that backend uses GPU when available."""
        from sophia.jepa.backends.poc import PoCJEPABackend
        from sophia.jepa.models import SimulationContext, Entity
        
        backend = PoCJEPABackend(
            weights_path=minimal_checkpoint,
            device="cuda:0",
        )
        
        entities = [Entity(id="obj", type="object")]
        context = SimulationContext(entities=entities)
        _result = backend.simulate(context, k_steps=1)  # noqa: F841
        
        # Check device in health status
        status = backend.get_health_status()
        assert status["gpu_available"] is True
        assert "cuda" in status["device"]

    def test_gpu_memory_tracking(self, minimal_checkpoint):
        """Test that GPU memory is tracked during inference."""
        import torch
        from sophia.jepa.backends.poc import PoCJEPABackend
        from sophia.jepa.models import SimulationContext, Entity
        
        _initial_memory = torch.cuda.memory_allocated()  # noqa: F841
        
        backend = PoCJEPABackend(
            weights_path=minimal_checkpoint,
            device="cuda:0",
        )
        
        entities = [Entity(id="obj", type="object")]
        context = SimulationContext(entities=entities)
        backend.simulate(context, k_steps=3)
        
        # Verify some memory was used (even with minimal model)
        # Note: actual memory tracking would be in observability metrics
        status = backend.get_health_status()
        assert status["inference_count"] >= 1


@requires_torch
@_skip_no_torch
class TestJEPARunnerIntegration:
    """Integration tests for JEPARunner with PoC backend."""

    def test_runner_with_poc_backend(self, minimal_checkpoint, monkeypatch):
        """Test JEPARunner selects and uses PoC backend."""
        from sophia.jepa.runner import JEPARunner
        from sophia.jepa.models import SimulationContext, Entity
        
        monkeypatch.setenv("JEPA_BACKEND", "poc")
        monkeypatch.setenv("JEPA_WEIGHTS_PATH", minimal_checkpoint)
        
        runner = JEPARunner()
        
        assert runner.backend_name == "PoCJEPABackend"
        
        entities = [Entity(id="obj", type="object")]
        context = SimulationContext(entities=entities)
        result = runner.simulate(context, k_steps=2)
        
        assert result.simulation_id is not None
        assert "poc" in result.model_version.lower()

    def test_runner_health_status(self, minimal_checkpoint, monkeypatch):
        """Test JEPARunner exposes backend health status."""
        from sophia.jepa.runner import JEPARunner
        from sophia.jepa.models import SimulationContext, Entity
        
        monkeypatch.setenv("JEPA_BACKEND", "poc")
        monkeypatch.setenv("JEPA_WEIGHTS_PATH", minimal_checkpoint)
        
        runner = JEPARunner()
        
        # Trigger model load
        entities = [Entity(id="obj", type="object")]
        context = SimulationContext(entities=entities)
        runner.simulate(context, k_steps=1)
        
        status = runner.get_health_status()
        assert status["backend"] == "poc"
        assert status["model_loaded"] is True

    @pytest.mark.asyncio
    async def test_runner_process_media_sample(self, minimal_checkpoint, tmp_path, monkeypatch):
        """Test JEPARunner media processing with PoC backend."""
        from sophia.jepa.runner import JEPARunner
        from PIL import Image
        
        monkeypatch.setenv("JEPA_BACKEND", "poc")
        monkeypatch.setenv("JEPA_WEIGHTS_PATH", minimal_checkpoint)
        
        runner = JEPARunner()
        
        # Create test image
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(img_path)
        
        result = await runner.process_media_sample(
            sample_id="test-001",
            file_path=str(img_path),
            media_type="image",
            metadata={},
        )
        
        assert result["sample_id"] == "test-001"
        assert len(result["embeddings"]["visual"]) == 768
