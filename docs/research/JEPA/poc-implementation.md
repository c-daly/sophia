# V-JEPA PoC Implementation

## Overview

This document describes the Proof-of-Concept (PoC) implementation of a real V-JEPA (Vision Joint-Embedding Predictive Architecture) backend for Sophia's simulation and perception systems.

## Purpose

The PoC demonstrates:
1. **Backend Integration Pattern**: How a real V-JEPA model can be integrated via the pluggable backend interface
2. **API Compatibility**: That existing `/simulate` and `/ingest/media` endpoints work unchanged
3. **Configuration**: Environment-based configuration for backend selection, weights, device, and dtype
4. **Embedding Generation**: Realistic 768-dimensional visual and physics embeddings
5. **Rollout Mechanics**: K-step simulation with confidence tracking and uncertainty propagation

## Architecture

### Backend Selection

The JEPA runner supports multiple backends via the `JEPA_BACKEND` environment variable:
- `stub` (default): CPU-friendly stub for development and CI
- `poc`: Proof-of-Concept backend with V-JEPA-like operations
- `real`: Reserved for production V-JEPA implementation (future)

```python
from sophia.jepa import JEPARunner

# Stub backend (default)
runner = JEPARunner()

# PoC backend (via environment)
import os
os.environ["JEPA_BACKEND"] = "poc"
runner = JEPARunner()
```

### PoC Backend Implementation

The PoC backend (`src/sophia/jepa/poc_backend.py`) implements the `JEPABackend` protocol:

```python
class PoCJEPABackend:
    def __init__(
        self,
        model_version: str = "v-jepa-poc-v1.0",
        confidence_decay: float = 0.08,
        weights_path: str | None = None,
        device: str = "cpu",
        dtype: str = "fp32",
    ):
        # Initialize model, projection matrices, etc.
        ...
    
    async def process_media_sample(...) -> Dict[str, Any]:
        # Generate visual and physics embeddings
        ...
    
    def simulate(...) -> SimulationResult:
        # Run k-step rollout with confidence tracking
        ...
```

Key features:
- **Deterministic embeddings**: Uses pseudo-random projections for reproducibility
- **Feature extraction**: Basic image statistics as features (real V-JEPA would use vision encoder)
- **Projection head**: Linear projection to 768-dim target space
- **Physics-aware**: Separate visual and physics embeddings with different modulation
- **Confidence modeling**: Exponential decay accounting for uncertainty accumulation
- **Configuration support**: Accepts weights path, device, dtype from environment

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `JEPA_BACKEND` | `stub` | Backend selection: `stub`, `poc`, or `real` |
| `JEPA_WEIGHTS_PATH` | None | Path to model weights (optional for PoC) |
| `JEPA_DEVICE` | `cpu` | Device for inference: `cpu`, `cuda:0`, etc. |
| `JEPA_DTYPE` | `fp32` | Data type: `fp32`, `fp16`, `bf16` |

### Example Usage

```bash
# Run with PoC backend
export JEPA_BACKEND=poc
export JEPA_DEVICE=cuda:0
export JEPA_DTYPE=fp16
python examples/poc_jepa_demo.py

# Run with stub backend (default)
python examples/poc_jepa_demo.py
```

## API Compatibility

The PoC backend maintains full API compatibility with existing endpoints:

### Media Processing

```python
from sophia.jepa import JEPARunner
import os

os.environ["JEPA_BACKEND"] = "poc"
runner = JEPARunner()

result = await runner.process_media_sample(
    sample_id="sample_1",
    file_path="/path/to/image.jpg",
    media_type="image",
    metadata={"width": 1920, "height": 1080},
    question="What physical properties can be inferred?",
)

# Result structure (identical to stub):
{
    "sample_id": "sample_1",
    "media_type": "image",
    "embeddings": {
        "visual": [0.123, ..., 0.456],    # 768-dim
        "physics": [0.789, ..., 0.321]    # 768-dim
    },
    "embedding_dim": 768,
    "model_version": "v-jepa-poc-v1.0",
    "confidence": 0.85,
    "metadata": {...}
}
```

### Simulation

```python
from sophia.jepa.models import SimulationContext, Entity

entities = [
    Entity(
        id="block",
        type="object",
        properties={"mass": 0.5},
        position={"x": 0.0, "y": 0.0, "z": 0.1}
    )
]

context = SimulationContext(entities=entities)
result = runner.simulate(context, k_steps=5)

# Result structure (identical to stub):
{
    "simulation_id": "uuid",
    "imagined_processes": [...],
    "imagined_states": [...],
    "k_steps": 5,
    "model_version": "v-jepa-poc-v1.0",
    "overall_confidence": 0.88
}
```

## Differences from Stub

| Aspect | Stub Backend | PoC Backend |
|--------|-------------|------------|
| **Purpose** | Fast development/CI | V-JEPA integration demo |
| **Embeddings** | Hash-based deterministic | Projection-based with image features |
| **Confidence** | Linear decay | Exponential decay with uncertainty |
| **Physics** | Simple position updates | Mass-dependent dynamics |
| **Configuration** | None | Weights path, device, dtype |
| **Performance** | <100ms | ~100-200ms (with image loading) |

## Testing

### Unit Tests

Run PoC backend tests:
```bash
poetry run pytest tests/unit/jepa/test_poc_backend.py -v
```

Test coverage includes:
- Backend initialization and configuration
- Media processing with real and missing images
- Simulation with various scenarios
- Confidence decay patterns
- Physics dynamics
- Backend interface compliance

### Integration Testing

The PoC backend can be used in existing integration tests by setting the environment variable:

```bash
JEPA_BACKEND=poc poetry run pytest tests/integration/test_jepa_simulation.py
```

### Demo Script

Run the interactive demo:
```bash
# With PoC backend
JEPA_BACKEND=poc python examples/poc_jepa_demo.py

# With stub backend (shows comparison)
python examples/poc_jepa_demo.py
```

## Performance

### PoC Backend Benchmarks

Approximate timings on CPU (Intel Xeon):
- **Model initialization**: ~50ms
- **Media processing**: ~100-150ms per image
- **Simulation (k=5)**: ~20-30ms
- **Total end-to-end**: ~200ms for media + simulation

GPU acceleration (when configured):
- Expected 2-3x speedup for media processing
- Negligible benefit for simulation (stub dynamics)

## Limitations

The PoC backend is for **demonstration and validation only**:

1. **Not Production-Ready**: Uses simplified operations, not real V-JEPA model
2. **No Actual Learning**: Embeddings are deterministic projections, not learned
3. **Limited Feature Extraction**: Basic image statistics, not deep visual features
4. **Stub Weights**: Doesn't require or use real V-JEPA checkpoint
5. **No GPU Optimization**: Runs on CPU with NumPy, not optimized kernels

## Migration Path to Real V-JEPA

To replace the PoC with a real V-JEPA model:

1. **Load Real Weights**:
   ```python
   # In _load_model():
   checkpoint = torch.load(self.weights_path)
   self.model = VJEPA.from_config(checkpoint["config"])
   self.model.load_state_dict(checkpoint["model"])
   self.model.to(self.device)
   self.model.eval()
   ```

2. **Replace Feature Extraction**:
   ```python
   # In _extract_image_features():
   img_tensor = preprocess_image(image_path)
   with torch.no_grad():
       features = self.model.encode_image(img_tensor)
   return features.cpu().numpy()
   ```

3. **Update Rollout Logic**:
   ```python
   # In _generate_state_rollout():
   latent = self.model.encode_context(context)
   for step in range(k_steps):
       latent = self.model.predict_next(latent)
       state = self.model.decode_state(latent)
   ```

4. **Add GPU Optimization**:
   - Use CUDA streams for async processing
   - Enable FlashAttention/SDPA
   - Add TensorRT compilation (optional)

5. **Update Tests**:
   - Add GPU-gated tests
   - Use mini checkpoint for CI
   - Record fixture outputs for fast testing

## Security Considerations

The PoC backend follows security best practices:

1. **No Secrets**: No credentials or API keys required
2. **Input Validation**: Validates paths and configurations
3. **Safe Defaults**: Falls back to safe CPU/fp32 settings
4. **No Logging of Content**: Doesn't log image data or embeddings
5. **File Access**: Only reads files explicitly provided

## Future Work

1. **Real V-JEPA Integration**: Load actual Meta V-JEPA checkpoint
2. **Mini Model**: Create distilled version for CI/testing
3. **GPU Optimization**: Add batching, CUDA streams, TensorRT
4. **Uncertainty Estimation**: Use ensemble or Bayesian methods
5. **Multi-Modal**: Extend to video and other sensor modalities
6. **Benchmark Suite**: Compare PoC vs real model vs hardware simulator

## References

- Integration spec: `docs/research/JEPA/real-v-jepa-integration.md`
- Issue tracker: [#75 - Define process to run real V-JEPA model](https://github.com/c-daly/sophia/issues/75)
- Stub implementation: `src/sophia/jepa/runner.py`
- JEPA simulation docs: `docs/JEPA_SIMULATION.md`

## Support

For questions or issues:
1. Review this document and the integration spec
2. Check test examples in `tests/unit/jepa/test_poc_backend.py`
3. Run the demo script: `examples/poc_jepa_demo.py`
4. Open an issue on GitHub with `capability:perception` label
