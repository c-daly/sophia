"""OpenTelemetry Metrics for JEPA Backend.

Provides standardized metrics instrumentation for the JEPA simulation
system. Metrics are exported via OpenTelemetry to configured collectors.

Metrics:
    jepa_inference_total: Counter of total inference operations
    jepa_inference_duration_seconds: Histogram of inference latencies
    jepa_model_load_time_seconds: Gauge for model load time
    jepa_gpu_memory_bytes: Gauge for GPU memory usage
    jepa_embedding_dimension: Gauge for embedding dimensionality
    jepa_model_ready: Gauge (1 if model loaded, 0 otherwise)

Usage:
    from sophia.jepa.metrics import JEPAMetrics

    metrics = JEPAMetrics()
    metrics.record_inference(duration_seconds=0.05, backend="poc", success=True)
    metrics.record_model_load(load_time_seconds=2.5, backend="poc")
    metrics.record_gpu_memory(bytes_used=1024*1024*500, device="cuda:0")
"""

import logging
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

# Lazy import opentelemetry metrics to avoid startup cost
_meter = None
_metrics_available: Optional[bool] = None


def _get_meter() -> Optional[Any]:
    """Lazy import and get meter for JEPA metrics."""
    global _meter, _metrics_available
    if _metrics_available is None:
        try:
            from logos_observability import get_meter

            _meter = get_meter("sophia.jepa", version="1.0.0")
            _metrics_available = True
            logger.debug("OpenTelemetry metrics initialized for JEPA")
        except ImportError:
            _metrics_available = False
            logger.warning(
                "logos_observability not available - metrics will not be exported"
            )
    return _meter if _metrics_available else None


class JEPAMetrics:
    """OpenTelemetry metrics instrumentation for JEPA backend.

    Provides standardized metrics for monitoring JEPA operations including
    inference latency, throughput, GPU memory usage, and model status.

    Metrics are lazily initialized on first use to avoid startup overhead
    if metrics are not needed.
    """

    def __init__(self, backend_name: str = "unknown"):
        """Initialize JEPA metrics.

        Args:
            backend_name: Name of the backend (stub, poc, real) for labels
        """
        self.backend_name = backend_name
        self._initialized = False
        self._inference_counter = None
        self._inference_histogram = None
        self._load_time_gauge = None
        self._gpu_memory_gauge = None
        self._model_ready_gauge = None
        self._embedding_dim_gauge = None

    def _ensure_initialized(self) -> None:
        """Lazy initialization of metric instruments."""
        if self._initialized:
            return

        meter = _get_meter()
        if meter is None:
            self._initialized = True
            return

        # Counter: Total inference operations
        self._inference_counter = meter.create_counter(
            name="jepa_inference_total",
            description="Total number of JEPA inference operations",
            unit="1",
        )

        # Histogram: Inference duration distribution
        self._inference_histogram = meter.create_histogram(
            name="jepa_inference_duration_seconds",
            description="Duration of JEPA inference operations in seconds",
            unit="s",
        )

        # Observable Gauges via callback - we'll track last values
        self._last_load_time: Optional[float] = None
        self._last_gpu_memory: Optional[int] = None
        self._last_model_ready: int = 0
        self._last_embedding_dim: int = 768

        # Register observable gauges with callbacks
        self._load_time_gauge = meter.create_observable_gauge(
            name="jepa_model_load_time_seconds",
            callbacks=[self._observe_load_time],
            description="Time taken to load the JEPA model in seconds",
            unit="s",
        )

        self._gpu_memory_gauge = meter.create_observable_gauge(
            name="jepa_gpu_memory_bytes",
            callbacks=[self._observe_gpu_memory],
            description="GPU memory used by JEPA model in bytes",
            unit="By",
        )

        self._model_ready_gauge = meter.create_observable_gauge(
            name="jepa_model_ready",
            callbacks=[self._observe_model_ready],
            description="Whether the JEPA model is loaded and ready (1=ready, 0=not ready)",
            unit="1",
        )

        self._embedding_dim_gauge = meter.create_observable_gauge(
            name="jepa_embedding_dimension",
            callbacks=[self._observe_embedding_dim],
            description="Dimension of JEPA embeddings",
            unit="1",
        )

        self._initialized = True
        logger.info(f"JEPA metrics initialized for backend: {self.backend_name}")

    def _observe_load_time(self, options: Any) -> Iterator[Any]:
        """Callback for load time gauge."""
        if self._last_load_time is not None:
            yield metrics.Observation(
                self._last_load_time,
                {"backend": self.backend_name},
            )

    def _observe_gpu_memory(self, options: Any) -> Iterator[Any]:
        """Callback for GPU memory gauge."""
        if self._last_gpu_memory is not None:
            yield metrics.Observation(
                self._last_gpu_memory,
                {"backend": self.backend_name},
            )

    def _observe_model_ready(self, options: Any) -> Iterator[Any]:
        """Callback for model ready gauge."""
        yield metrics.Observation(
            self._last_model_ready,
            {"backend": self.backend_name},
        )

    def _observe_embedding_dim(self, options: Any) -> Iterator[Any]:
        """Callback for embedding dimension gauge."""
        yield metrics.Observation(
            self._last_embedding_dim,
            {"backend": self.backend_name},
        )

    def record_inference(
        self,
        duration_seconds: float,
        success: bool = True,
        operation: str = "simulate",
    ) -> None:
        """Record an inference operation.

        Args:
            duration_seconds: Time taken for the inference in seconds
            success: Whether the inference succeeded
            operation: Type of operation (simulate, embed, process_media)
        """
        self._ensure_initialized()

        attributes = {
            "backend": self.backend_name,
            "operation": operation,
            "success": str(success).lower(),
        }

        if self._inference_counter is not None:
            self._inference_counter.add(1, attributes)

        if self._inference_histogram is not None:
            self._inference_histogram.record(duration_seconds, attributes)

        logger.debug(
            f"Recorded inference: {operation}, {duration_seconds*1000:.2f}ms, "
            f"success={success}"
        )

    def record_model_load(self, load_time_seconds: float, embedding_dim: int = 768) -> None:
        """Record model load event.

        Args:
            load_time_seconds: Time taken to load the model
            embedding_dim: Dimension of model embeddings
        """
        self._ensure_initialized()
        self._last_load_time = load_time_seconds
        self._last_model_ready = 1
        self._last_embedding_dim = embedding_dim
        logger.info(
            f"Recorded model load: {load_time_seconds:.2f}s, embed_dim={embedding_dim}"
        )

    def record_gpu_memory(self, bytes_used: int, device: str = "cuda:0") -> None:
        """Record GPU memory usage.

        Args:
            bytes_used: GPU memory in bytes
            device: CUDA device identifier
        """
        self._ensure_initialized()
        self._last_gpu_memory = bytes_used
        logger.debug(
            f"Recorded GPU memory: {bytes_used / (1024*1024):.1f}MB on {device}"
        )

    def record_model_unload(self) -> None:
        """Record model unload event."""
        self._ensure_initialized()
        self._last_model_ready = 0
        self._last_gpu_memory = None
        logger.info("Recorded model unload")


# Import metrics module for Observable callback
try:
    from opentelemetry import metrics
except ImportError:
    metrics = None  # type: ignore


# Default singleton instance
_default_metrics: Optional[JEPAMetrics] = None


def get_jepa_metrics(backend_name: str = "unknown") -> JEPAMetrics:
    """Get or create JEPA metrics instance.

    Args:
        backend_name: Name of the backend for metric labels

    Returns:
        JEPAMetrics instance
    """
    global _default_metrics
    if _default_metrics is None or _default_metrics.backend_name != backend_name:
        _default_metrics = JEPAMetrics(backend_name)
    return _default_metrics
