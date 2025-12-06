"""Unit tests for JEPA OpenTelemetry metrics module.

Tests the JEPAMetrics class without requiring external OTEL collectors.
"""

import pytest
from unittest.mock import MagicMock, patch


def _otel_available() -> bool:
    """Check if OpenTelemetry is available."""
    try:
        from logos_observability import get_meter  # noqa: F401

        return True
    except ImportError:
        return False


class TestJEPAMetricsImport:
    """Test metrics module imports correctly."""

    def test_import_jepa_metrics_class(self):
        """Test JEPAMetrics class can be imported."""
        from sophia.jepa.metrics import JEPAMetrics

        assert JEPAMetrics is not None

    def test_import_get_jepa_metrics_function(self):
        """Test get_jepa_metrics function can be imported."""
        from sophia.jepa.metrics import get_jepa_metrics

        assert callable(get_jepa_metrics)


class TestJEPAMetricsInitialization:
    """Test JEPAMetrics initialization."""

    def test_default_initialization(self):
        """Test JEPAMetrics initializes with default backend name."""
        from sophia.jepa.metrics import JEPAMetrics

        metrics = JEPAMetrics()
        assert metrics.backend_name == "unknown"
        assert metrics._initialized is False

    def test_custom_backend_name(self):
        """Test JEPAMetrics accepts custom backend name."""
        from sophia.jepa.metrics import JEPAMetrics

        metrics = JEPAMetrics(backend_name="poc")
        assert metrics.backend_name == "poc"

    def test_lazy_initialization(self):
        """Test metrics are not initialized until first use."""
        from sophia.jepa.metrics import JEPAMetrics

        metrics = JEPAMetrics(backend_name="stub")
        assert metrics._initialized is False
        assert metrics._inference_counter is None


class TestJEPAMetricsRecording:
    """Test JEPAMetrics recording methods."""

    def test_record_inference_without_otel(self):
        """Test record_inference works even without OTEL available."""
        from sophia.jepa.metrics import JEPAMetrics

        # Patch the meter getter to simulate OTEL not available
        with patch("sophia.jepa.metrics._get_meter", return_value=None):
            metrics = JEPAMetrics(backend_name="test")

            # Should not raise even without OTEL
            metrics.record_inference(
                duration_seconds=0.05,
                success=True,
                operation="simulate",
            )

            assert metrics._initialized is True

    def test_record_model_load_without_otel(self):
        """Test record_model_load works without OTEL."""
        from sophia.jepa.metrics import JEPAMetrics

        with patch("sophia.jepa.metrics._get_meter", return_value=None):
            metrics = JEPAMetrics(backend_name="poc")

            metrics.record_model_load(load_time_seconds=2.5, embedding_dim=768)

            assert metrics._last_load_time == 2.5
            assert metrics._last_model_ready == 1
            assert metrics._last_embedding_dim == 768

    def test_record_gpu_memory_without_otel(self):
        """Test record_gpu_memory works without OTEL."""
        from sophia.jepa.metrics import JEPAMetrics

        with patch("sophia.jepa.metrics._get_meter", return_value=None):
            metrics = JEPAMetrics(backend_name="poc")
            metrics._ensure_initialized()

            metrics.record_gpu_memory(bytes_used=1024 * 1024 * 500, device="cuda:0")

            assert metrics._last_gpu_memory == 1024 * 1024 * 500

    def test_record_model_unload(self):
        """Test record_model_unload resets state."""
        from sophia.jepa.metrics import JEPAMetrics

        with patch("sophia.jepa.metrics._get_meter", return_value=None):
            metrics = JEPAMetrics(backend_name="poc")

            # First load the model
            metrics.record_model_load(load_time_seconds=1.0, embedding_dim=768)
            metrics.record_gpu_memory(bytes_used=1024 * 1024)

            assert metrics._last_model_ready == 1
            assert metrics._last_gpu_memory is not None

            # Now unload
            metrics.record_model_unload()

            assert metrics._last_model_ready == 0
            assert metrics._last_gpu_memory is None


class TestJEPAMetricsWithMockedOTEL:
    """Test JEPAMetrics with mocked OpenTelemetry."""

    def test_creates_counter_for_inference(self):
        """Test that a counter is created for inference total."""
        from sophia.jepa.metrics import JEPAMetrics

        mock_meter = MagicMock()
        mock_counter = MagicMock()
        mock_meter.create_counter.return_value = mock_counter

        with patch("sophia.jepa.metrics._get_meter", return_value=mock_meter):
            metrics = JEPAMetrics(backend_name="poc")
            metrics._ensure_initialized()

            mock_meter.create_counter.assert_called_once_with(
                name="jepa_inference_total",
                description="Total number of JEPA inference operations",
                unit="1",
            )

    def test_creates_histogram_for_duration(self):
        """Test that a histogram is created for inference duration."""
        from sophia.jepa.metrics import JEPAMetrics

        mock_meter = MagicMock()
        mock_histogram = MagicMock()
        mock_meter.create_histogram.return_value = mock_histogram

        with patch("sophia.jepa.metrics._get_meter", return_value=mock_meter):
            metrics = JEPAMetrics(backend_name="poc")
            metrics._ensure_initialized()

            mock_meter.create_histogram.assert_called_once_with(
                name="jepa_inference_duration_seconds",
                description="Duration of JEPA inference operations in seconds",
                unit="s",
            )

    def test_record_inference_increments_counter(self):
        """Test that record_inference increments the counter."""
        from sophia.jepa.metrics import JEPAMetrics

        mock_meter = MagicMock()
        mock_counter = MagicMock()
        mock_histogram = MagicMock()
        mock_meter.create_counter.return_value = mock_counter
        mock_meter.create_histogram.return_value = mock_histogram
        mock_meter.create_observable_gauge.return_value = MagicMock()

        with patch("sophia.jepa.metrics._get_meter", return_value=mock_meter):
            metrics = JEPAMetrics(backend_name="poc")

            metrics.record_inference(
                duration_seconds=0.1,
                success=True,
                operation="simulate",
            )

            mock_counter.add.assert_called_once()
            call_args = mock_counter.add.call_args
            assert call_args[0][0] == 1
            assert call_args[0][1]["backend"] == "poc"
            assert call_args[0][1]["operation"] == "simulate"
            assert call_args[0][1]["success"] == "true"

    def test_record_inference_records_histogram(self):
        """Test that record_inference records to histogram."""
        from sophia.jepa.metrics import JEPAMetrics

        mock_meter = MagicMock()
        mock_counter = MagicMock()
        mock_histogram = MagicMock()
        mock_meter.create_counter.return_value = mock_counter
        mock_meter.create_histogram.return_value = mock_histogram
        mock_meter.create_observable_gauge.return_value = MagicMock()

        with patch("sophia.jepa.metrics._get_meter", return_value=mock_meter):
            metrics = JEPAMetrics(backend_name="poc")

            metrics.record_inference(
                duration_seconds=0.05,
                success=True,
                operation="embed",
            )

            mock_histogram.record.assert_called_once()
            call_args = mock_histogram.record.call_args
            assert call_args[0][0] == 0.05


class TestGetJEPAMetricsSingleton:
    """Test get_jepa_metrics singleton behavior."""

    def test_returns_jepa_metrics_instance(self):
        """Test get_jepa_metrics returns JEPAMetrics instance."""
        from sophia.jepa.metrics import get_jepa_metrics, JEPAMetrics

        metrics = get_jepa_metrics("test")
        assert isinstance(metrics, JEPAMetrics)

    def test_returns_same_instance_for_same_backend(self):
        """Test get_jepa_metrics returns same instance for same backend."""
        from sophia.jepa import metrics as metrics_module

        # Reset the singleton
        metrics_module._default_metrics = None

        metrics1 = metrics_module.get_jepa_metrics("poc")
        metrics2 = metrics_module.get_jepa_metrics("poc")

        assert metrics1 is metrics2

    def test_creates_new_instance_for_different_backend(self):
        """Test get_jepa_metrics creates new instance for different backend."""
        from sophia.jepa import metrics as metrics_module

        # Reset the singleton
        metrics_module._default_metrics = None

        metrics1 = metrics_module.get_jepa_metrics("stub")
        metrics2 = metrics_module.get_jepa_metrics("poc")

        assert metrics1 is not metrics2
        assert metrics2.backend_name == "poc"


class TestJEPAMetricsIntegration:
    """Integration tests with actual OpenTelemetry (if available)."""

    @pytest.mark.skipif(
        not _otel_available(),
        reason="OpenTelemetry not available",
    )
    def test_metrics_with_real_otel(self):
        """Test metrics work with real OpenTelemetry SDK."""
        from sophia.jepa.metrics import JEPAMetrics

        metrics = JEPAMetrics(backend_name="integration-test")

        # Should not raise
        metrics.record_inference(duration_seconds=0.01, success=True, operation="test")
        metrics.record_model_load(load_time_seconds=1.0, embedding_dim=768)
        metrics.record_gpu_memory(bytes_used=1024)
        metrics.record_model_unload()
