"""Tests for PoC JEPA Backend.

These tests validate the PoCJEPABackend implementation including:
- Interface contract compliance with JEPABackend protocol
- Shape and key assertions for embeddings
- Projection head correctness
- Configuration loading
- Graceful degradation when GPU/weights unavailable

Tests are gated on GPU/weights availability where appropriate.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from sophia.jepa.models import (
    SimulationContext,
    SimulationResult,
    Entity,
)

pytestmark = pytest.mark.unit


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


# Skip markers for torch/GPU-dependent tests
# Use pytest.mark for CI filtering (-m "not requires_torch")
# Use skipif for runtime skipping when package not available
requires_torch = pytest.mark.requires_torch
_skip_no_torch = pytest.mark.skipif(
    not _torch_available(), reason="PyTorch not available"
)

requires_gpu = pytest.mark.gpu
_skip_no_gpu = pytest.mark.skipif(not _gpu_available(), reason="GPU not available")


class TestPoCJEPABackendProtocol:
    """Test that PoCJEPABackend implements JEPABackend protocol correctly."""

    def test_implements_simulate_method(self):
        """Verify simulate method exists with correct signature."""
        from sophia.jepa.backends.poc import PoCJEPABackend

        backend = PoCJEPABackend()
        assert hasattr(backend, "simulate")
        assert callable(backend.simulate)

    def test_implements_process_media_sample_method(self):
        """Verify process_media_sample method exists with correct signature."""
        from sophia.jepa.backends.poc import PoCJEPABackend

        backend = PoCJEPABackend()
        assert hasattr(backend, "process_media_sample")
        assert callable(backend.process_media_sample)

    def test_is_runtime_checkable_protocol(self):
        """Verify PoCJEPABackend satisfies JEPABackend protocol."""
        from sophia.jepa.runner import JEPABackend
        from sophia.jepa.backends.poc import PoCJEPABackend

        backend = PoCJEPABackend()
        # Protocol check - this uses runtime_checkable decorator
        assert isinstance(backend, JEPABackend)


class TestPoCJEPABackendInitialization:
    """Test PoCJEPABackend initialization and configuration."""

    def test_default_initialization(self):
        """Test initialization with default values."""
        from sophia.jepa.backends.poc import PoCJEPABackend

        backend = PoCJEPABackend()
        assert backend.model_version == "jepa-poc-v1.0"
        assert backend.confidence_decay == 0.05
        assert backend.dtype == "fp16"

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        from sophia.jepa.backends.poc import PoCJEPABackend

        backend = PoCJEPABackend(
            model_version="custom-v2.0",
            confidence_decay=0.1,
            device="cpu",
            dtype="fp32",
        )
        assert backend.model_version == "custom-v2.0"
        assert backend.confidence_decay == 0.1
        assert backend.dtype == "fp32"

    def test_env_var_configuration(self):
        """Test configuration via environment variables."""
        from sophia.jepa.backends.poc import PoCJEPABackend

        with patch.dict(
            os.environ,
            {
                "JEPA_WEIGHTS_PATH": "/test/path/checkpoint.pth",
                "JEPA_DEVICE": "cuda:1",
                "JEPA_DTYPE": "bf16",
            },
        ):
            backend = PoCJEPABackend()
            assert backend.weights_path == "/test/path/checkpoint.pth"
            assert backend._requested_device == "cuda:1"
            assert backend.dtype == "bf16"

    def test_explicit_params_override_env_vars(self):
        """Test that explicit parameters override environment variables."""
        from sophia.jepa.backends.poc import PoCJEPABackend

        with patch.dict(
            os.environ,
            {
                "JEPA_WEIGHTS_PATH": "/env/path.pth",
                "JEPA_DEVICE": "cuda:0",
            },
        ):
            backend = PoCJEPABackend(
                weights_path="/explicit/path.pth",
                device="cpu",
            )
            assert backend.weights_path == "/explicit/path.pth"
            assert backend._requested_device == "cpu"


class TestPoCJEPABackendHealthStatus:
    """Test health status reporting."""

    def test_get_health_status_unloaded(self):
        """Test health status when model is not loaded."""
        from sophia.jepa.backends.poc import PoCJEPABackend

        backend = PoCJEPABackend()
        status = backend.get_health_status()

        assert status["backend"] == "poc"
        assert status["model_loaded"] is False
        assert "gpu_available" in status
        assert "weights_path" in status

    def test_health_status_includes_metrics(self):
        """Test that health status includes performance metrics."""
        from sophia.jepa.backends.poc import PoCJEPABackend

        backend = PoCJEPABackend()
        status = backend.get_health_status()

        assert "inference_count" in status
        assert "avg_inference_time_ms" in status


class TestPoCJEPABackendWithMockedTorch:
    """Test PoCJEPABackend with mocked PyTorch for CI environments."""

    def test_simulate_requires_weights_path_or_torch(self):
        """Test that simulate fails gracefully without weights or torch."""
        from sophia.jepa.backends.poc import PoCJEPABackend

        backend = PoCJEPABackend()  # No weights_path set

        entities = [Entity(id="obj", type="object")]
        context = SimulationContext(entities=entities)

        # Should fail with either "JEPA_WEIGHTS_PATH" or "PyTorch" error
        with pytest.raises(RuntimeError, match="(JEPA_WEIGHTS_PATH|PyTorch)"):
            backend.simulate(context, k_steps=2)

    def test_simulate_fails_on_missing_checkpoint(self):
        """Test that simulate fails gracefully with non-existent checkpoint."""
        from sophia.jepa.backends.poc import PoCJEPABackend

        backend = PoCJEPABackend(weights_path="/nonexistent/path.pth")

        entities = [Entity(id="obj", type="object")]
        context = SimulationContext(entities=entities)

        with pytest.raises((FileNotFoundError, RuntimeError)):
            backend.simulate(context, k_steps=2)


class TestPoCJEPABackendEmbeddingShapes:
    """Test embedding dimensions and key stability."""

    @requires_torch
    @_skip_no_torch
    def test_embedding_dimension_is_768(self):
        """Verify embeddings are 768-dimensional."""
        from sophia.jepa.backends.poc import PoCJEPABackend

        backend = PoCJEPABackend()

        # Mock weights loading
        backend._model = MagicMock()
        backend._projection_head = MagicMock()

        # Generate embedding via the helper method directly
        embedding = backend._generate_embedding("test_input", "visual")

        assert len(embedding) == 768
        assert all(isinstance(x, float) for x in embedding)

    @requires_torch
    @_skip_no_torch
    def test_embeddings_are_deterministic(self):
        """Verify same input produces same embedding."""
        from sophia.jepa.backends.poc import PoCJEPABackend

        backend = PoCJEPABackend()
        backend._model = MagicMock()

        embedding1 = backend._generate_embedding("test_input", "visual")
        embedding2 = backend._generate_embedding("test_input", "visual")

        assert embedding1 == embedding2

    @requires_torch
    @_skip_no_torch
    def test_different_types_produce_different_embeddings(self):
        """Verify visual and physics embeddings differ."""
        from sophia.jepa.backends.poc import PoCJEPABackend

        backend = PoCJEPABackend()
        backend._model = MagicMock()

        visual_embedding = backend._generate_embedding("test", "visual")
        physics_embedding = backend._generate_embedding("test", "physics")

        assert visual_embedding != physics_embedding


class TestPoCJEPABackendSimulationOutput:
    """Test simulation output structure and contract."""

    @requires_torch
    @_skip_no_torch
    def test_simulation_result_structure(self):
        """Verify SimulationResult has all required fields."""
        from sophia.jepa.backends.poc import PoCJEPABackend
        import tempfile
        import torch

        # Create minimal checkpoint
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            torch.save({"embed_dim": 768}, f.name)

            try:
                backend = PoCJEPABackend(weights_path=f.name, device="cpu")

                entities = [Entity(id="obj", type="object")]
                context = SimulationContext(entities=entities)

                result = backend.simulate(context, k_steps=3)

                assert isinstance(result, SimulationResult)
                assert result.simulation_id is not None
                assert result.k_steps == 3
                assert len(result.imagined_states) == 3
                assert len(result.imagined_processes) >= 1
                assert 0.0 <= result.overall_confidence <= 1.0
                assert result.model_version.startswith("jepa-poc")
            finally:
                os.unlink(f.name)

    @requires_torch
    @_skip_no_torch
    def test_imagined_states_have_correct_metadata(self):
        """Verify imagined states contain PoC backend metadata."""
        from sophia.jepa.backends.poc import PoCJEPABackend
        import tempfile
        import torch

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            torch.save({"embed_dim": 768}, f.name)

            try:
                backend = PoCJEPABackend(weights_path=f.name, device="cpu")

                entities = [Entity(id="obj", type="object")]
                context = SimulationContext(entities=entities)

                result = backend.simulate(context, k_steps=2, assumptions=["test"])

                for state in result.imagined_states:
                    assert state.imagined is True
                    assert "poc" in state.model_version.lower()
                    assert "test" in state.assumptions
                    assert state.state_data.get("backend") == "poc"
            finally:
                os.unlink(f.name)


class TestJEPARunnerBackendSelection:
    """Test that JEPARunner correctly selects backends."""

    def test_default_selects_stub(self):
        """Verify default backend is stub."""
        from sophia.jepa.runner import JEPARunner

        with patch.dict(os.environ, {}, clear=True):
            # Ensure JEPA_BACKEND is not set
            os.environ.pop("JEPA_BACKEND", None)
            runner = JEPARunner()
            assert runner.backend_name == "StubJEPABackend"

    def test_stub_env_var_selects_stub(self):
        """Verify JEPA_BACKEND=stub selects stub backend."""
        from sophia.jepa.runner import JEPARunner

        with patch.dict(os.environ, {"JEPA_BACKEND": "stub"}):
            runner = JEPARunner()
            assert runner.backend_name == "StubJEPABackend"

    def test_poc_env_var_selects_poc(self):
        """Verify JEPA_BACKEND=poc selects PoC backend."""
        from sophia.jepa.runner import JEPARunner

        with patch.dict(os.environ, {"JEPA_BACKEND": "poc"}):
            runner = JEPARunner()
            assert runner.backend_name == "PoCJEPABackend"

    def test_real_env_var_selects_poc(self):
        """Verify JEPA_BACKEND=real selects PoC backend (alias)."""
        from sophia.jepa.runner import JEPARunner

        with patch.dict(os.environ, {"JEPA_BACKEND": "real"}):
            runner = JEPARunner()
            assert runner.backend_name == "PoCJEPABackend"

    def test_unknown_env_var_falls_back_to_stub(self):
        """Verify unknown JEPA_BACKEND value falls back to stub."""
        from sophia.jepa.runner import JEPARunner

        with patch.dict(os.environ, {"JEPA_BACKEND": "invalid"}):
            runner = JEPARunner()
            assert runner.backend_name == "StubJEPABackend"

    def test_get_health_status_for_stub(self):
        """Test health status for stub backend."""
        from sophia.jepa.runner import JEPARunner

        runner = JEPARunner()
        status = runner.get_health_status()

        assert "backend" in status
        assert "model_version" in status

    def test_get_health_status_for_poc(self):
        """Test health status for PoC backend."""
        from sophia.jepa.runner import JEPARunner

        with patch.dict(os.environ, {"JEPA_BACKEND": "poc"}):
            runner = JEPARunner()
            status = runner.get_health_status()

            assert status["backend"] == "poc"
            assert "model_loaded" in status
            assert "gpu_available" in status
