# JEPA Runner & Simulation Documentation

## Overview

The JEPA (Joint-Embedding Predictive Architecture) runner provides dynamics simulation and media perception capabilities for the Sophia cognitive system. It performs:
- K-step forward prediction of system states
- Media sample processing (images, video) for physical world understanding
- Cross-modal embedding generation for semantic reasoning

The runner supports multiple backends:
- **Stub backend** (default): CPU-friendly stub for development and CI
- **PoC backend**: Real V-JEPA model support with GPU acceleration

## Backend Selection

The backend is selected via the `JEPA_BACKEND` environment variable:

| Value | Backend | Description |
|-------|---------|-------------|
| `stub` | StubJEPABackend | CPU-friendly stub (default) |
| `poc` | PoCJEPABackend | PoC with real V-JEPA model |
| `real` | PoCJEPABackend | Alias for `poc` |

### Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `JEPA_BACKEND` | `stub` | Backend selection |
| `JEPA_WEIGHTS_PATH` | — | Local path to checkpoint file (required for poc/real) |
| `JEPA_WEIGHTS_URI` | — | Remote URI for checkpoint download (fallback) |
| `JEPA_DEVICE` | `cuda:0` | Device for inference |
| `JEPA_DTYPE` | `fp16` | Data type for inference |

### Using the PoC Backend

```bash
# Install with ML dependencies
poetry install --with ml

# Configure and run
export JEPA_BACKEND=poc
export JEPA_WEIGHTS_PATH=/path/to/checkpoint.pth
export JEPA_DEVICE=cuda:0

# Start Sophia service
poetry run uvicorn sophia.api.app:app --reload
```

### Health Status

The `/health` endpoint includes JEPA backend status:

```json
{
  "components": {
    "jepa": {
      "backend": "poc",
      "model_loaded": true,
      "gpu_available": true,
      "device": "cuda:0",
      "inference_count": 42,
      "avg_inference_time_ms": 15.3
    }
  }
}
```

## Architecture

### Components

1. **JEPA Runner** (`src/sophia/jepa/runner.py`)
   - Pluggable backend architecture with `JEPABackend` protocol
   - Backend selection via environment variable
   - Performs k-step rollouts with confidence decay
   - Processes media samples to generate visual and physics embeddings

2. **Backends** (`src/sophia/jepa/backends/`)
   - `StubJEPABackend`: CPU-friendly stub for tests/CI
   - `PoCJEPABackend`: PoC with real model, GPU support, metrics

3. **Simulation Models** (`src/sophia/jepa/models.py`)
   - Context schema for simulation requests
   - Entity, sensor, and metadata models
   - Imagined process and state models

4. **API Endpoints**
   - `/simulate` - RESTful endpoint for running simulations
   - `/ingest/media` - Media upload with automatic JEPA processing
   - Results stored in Neo4j HCG with embeddings in Milvus

## Context Schema

### SimulationContext

The simulation context defines the environment and parameters for a simulation:

```python
{
  "entities": [
    {
      "id": "red_block",
      "type": "object",
      "properties": {"mass": 0.5, "shape": "cube"},
      "position": {"x": 0.0, "y": 0.0, "z": 0.1}
    }
  ],
  "sensor_refs": [
    {
      "sensor_id": "camera_1",
      "sensor_type": "camera",
      "frame_id": "base_link",
      "last_reading": {...}
    }
  ],
  "talos_metadata": {
    "simulator_version": "stub-v1.0",
    "physics_engine": "none",
    "time_step": 0.01,
    "use_hardware": false,
    "robot_model": null
  },
  "initial_state": {...},
  "actions": [
    {"type": "MOVE", "target": "red_block"},
    {"type": "GRASP", "target": "red_block"}
  ]
}
```

### Entity Schema

Entities represent objects, agents, or locations in the simulation:

- **id**: Unique identifier
- **type**: Entity type (`object`, `agent`, `location`)
- **properties**: Dictionary of entity attributes
- **position**: Optional spatial coordinates (x, y, z)

### Sensor Reference Schema

Sensor references provide perception data context:

- **sensor_id**: Unique sensor identifier
- **sensor_type**: Type of sensor (`camera`, `lidar`, `force`, `proprioception`)
- **frame_id**: Reference frame for sensor data
- **last_reading**: Most recent sensor reading

### Talos Metadata Schema

Metadata for integration with Talos/Gazebo simulators:

- **simulator_version**: Version of simulator being used
- **physics_engine**: Physics engine (`ODE`, `Bullet`, `none`)
- **time_step**: Simulation time step in seconds
- **use_hardware**: Whether using hardware simulator or CPU stub
- **robot_model**: Robot model identifier (e.g., `talos`, `ur5`)

## JEPA Runner

### Initialization

```python
from sophia.jepa import JEPARunner

runner = JEPARunner(
    model_version="jepa-stub-v1.0",
    confidence_decay=0.05  # Decay rate per step
)
```

### Running Simulations

```python
from sophia.jepa.models import SimulationContext, Entity

# Create simulation context
entities = [
    Entity(
        id="block_1",
        type="object",
        properties={"mass": 0.5},
        position={"x": 0.0, "y": 0.0, "z": 0.1}
    )
]

context = SimulationContext(entities=entities)

# Run k-step simulation
result = runner.simulate(
    context=context,
    k_steps=5,
    assumptions=["objects are graspable"]
)

# Access results
print(f"Simulation ID: {result.simulation_id}")
print(f"Overall confidence: {result.overall_confidence:.2f}")
print(f"Imagined states: {len(result.imagined_states)}")
print(f"Imagined processes: {len(result.imagined_processes)}")
```

### Output Structure

The simulation result contains:

- **simulation_id**: Unique identifier
- **imagined_processes**: List of imagined processes with metadata
- **imagined_states**: K-step rollout of states
- **k_steps**: Number of prediction steps
- **model_version**: JEPA model version
- **overall_confidence**: Average confidence across states

## Media Processing with JEPA

### Overview

The JEPA runner can process uploaded media (images, video) to generate embeddings for physical world understanding. This enables:
- Grounding language in visual observations
- Cross-modal semantic reasoning (match thoughts to images)
- Context-aware simulations using uploaded media

### Process Flow

1. User uploads media via `/ingest/media` endpoint
2. Media stored to disk + indexed in Neo4j as `MediaSample` node
3. JEPA runner automatically processes the media
4. Generates **visual_embedding** (768-dim) and **physics_embedding** (768-dim)
5. Embeddings stored in Milvus vector database
6. Neo4j relationships created: `MediaSample -[:has_embedding]-> Embedding`
7. User can reference media in simulations via `media_sample_id`

### Media Processing API

#### Method: `process_media_sample()`

```python
from sophia.jepa import JEPARunner

runner = JEPARunner()

result = await runner.process_media_sample(
    sample_id="ms_abc123",
    file_path="/app/media_storage/image/ms_abc123.jpg",
    media_type="image",
    question="Will this stack collapse?",
    metadata={"width": 1920, "height": 1080}
)

# Result structure
{
    "sample_id": "ms_abc123",
    "media_type": "image",
    "embeddings": {
        "visual": [0.123, 0.456, ...],      # 768-dim vector
        "physics": [0.789, 0.321, ...]      # 768-dim vector
    },
    "embedding_dim": 768,
    "model_version": "jepa-stub-v1.0",
    "confidence": 0.85,
    "metadata": {
        "file_path": "/app/media_storage/image/ms_abc123.jpg",
        "question": "Will this stack collapse?",
        "media_metadata": {"width": 1920, "height": 1080}
    }
}
```

### Embedding Types

**visual_embedding (768-dim)**
- Captures visual features: objects, spatial layout, appearance
- Generated from image/video frames
- Enables visual similarity search

**physics_embedding (768-dim)**
- Captures physical properties: stability, dynamics, affordances
- Predicts physical outcomes
- Grounds language in physics understanding

### Using Media in Simulations

Reference uploaded media in simulation requests:

```python
# Upload media first
media_response = await upload_media("physics_scene.jpg", media_type="image")
sample_id = media_response["sample_id"]

# Use in simulation
simulate_response = await simulate(
    entities=[...],
    k_steps=5,
    media_sample_id=sample_id  # Links simulation to uploaded media
)

# Response includes media context
{
    "simulation_id": "sim_xyz789",
    "media_sample_id": "ms_abc123",
    "media_embeddings": ["emb_visual_abc123", "emb_physics_abc123"],
    "imagined_states": [...],
    ...
}
```

### Neo4j Schema for Media

```cypher
# MediaSample node
(m:MediaSample {
  sample_id: "ms_abc123",
  media_type: "image",
  file_path: "/app/media_storage/image/ms_abc123.jpg",
  file_size: 245678,
  file_hash: "a1b2c3...",
  timestamp: "2025-01-15T10:30:00Z",
  question: "Will this stack collapse?",
  metadata_width: 1920,
  metadata_height: 1080
})

# Embedding relationship
(m)-[:has_embedding]->(e:Embedding {
  id: "emb_visual_abc123",
  vector_type: "visual",
  dimension: 768
})

# Simulation usage
(s:Simulation {simulation_id: "sim_xyz789"})-[:uses_media]->(m)
```

### Cross-Modal Reasoning

JEPA embeddings enable semantic search across modalities:

```python
# Find images similar to text description
query = "unstable block tower about to fall"
text_embedding = embed_text(query)  # via Hermes

# Search Milvus for similar visual embeddings
similar_images = milvus.search(
    collection="media_embeddings",
    query_embedding=text_embedding,
    filter={"vector_type": "visual"},
    top_k=10
)

# Returns media samples visually similar to the text description
```

### Stub Implementation

The current CPU-friendly stub:
- Uses hash-based deterministic embedding generation
- No GPU required, fast processing (<100ms)
- Maintains correct data structures and API contracts
- **Phase 3 Ready**: Real JEPA model can be swapped in with zero API changes

```python
# Current stub logic (simplified)
def generate_visual_embedding(sample_id, embedding_dim=768):
    return [
        float(hash(f"{sample_id}_visual_{i}") % 1000) / 1000.0
        for i in range(embedding_dim)
    ]
```

## Imagined Nodes

### ImaginedProcess

Processes created during simulation have the following metadata:

- **imagined**: `true` (always)
- **model_version**: Version of JEPA model used
- **horizon**: Planning horizon (k_steps)
- **assumptions**: List of assumptions
- **confidence**: Confidence score (0.0-1.0)
- **properties**: Additional process-specific data

### ImaginedState

States created during simulation have the following metadata:

- **imagined**: `true` (always)
- **model_version**: Version of JEPA model used
- **horizon**: Planning horizon (k_steps)
- **assumptions**: List of assumptions
- **step**: Step number in rollout (0 to k-1)
- **confidence**: Confidence score (0.0-1.0)
- **state_data**: Complete state data

## API Endpoint: /simulate

### Request

```bash
POST /simulate
Authorization: Bearer <token>
Content-Type: application/json

{
  "entities": [
    {
      "id": "red_block",
      "type": "object",
      "properties": {"mass": 0.5},
      "position": {"x": 0.0, "y": 0.0, "z": 0.1}
    }
  ],
  "sensor_refs": [
    {
      "sensor_id": "camera_1",
      "sensor_type": "camera",
      "frame_id": "base_link"
    }
  ],
  "media_sample_id": "ms_abc123",  // Optional: link to uploaded media
  "talos_metadata": {
    "simulator_version": "stub-v1.0",
    "physics_engine": "none",
    "use_hardware": false
  },
  "initial_state": {},
  "actions": [
    {"type": "MOVE", "target": "red_block"}
  ],
  "k_steps": 5,
  "assumptions": ["block is graspable"]
}
```

### Response

```json
{
  "simulation_id": "550e8400-e29b-41d4-a716-446655440000",
  "imagined_processes": [
    {
      "process_id": "550e8400-e29b-41d4-a716-446655440000_process_dynamics",
      "description": "Forward dynamics prediction process",
      "confidence": 0.85,
      "model_version": "jepa-stub-v1.0",
      "horizon": 5,
      "assumptions": ["block is graspable"],
      "imagined": true,
      "properties": {
        "type": "dynamics",
        "context_entities": 1,
        "context_sensors": 1
      }
    }
  ],
  "imagined_states": [
    {
      "state_id": "550e8400-e29b-41d4-a716-446655440000_state_0",
      "step": 0,
      "description": "Imagined state at step 0",
      "confidence": 0.95,
      "model_version": "jepa-stub-v1.0",
      "horizon": 5,
      "assumptions": ["block is graspable"],
      "imagined": true,
      "state_data": {...},
      "entities": [...]
    },
    ...
  ],
  "k_steps": 5,
  "model_version": "jepa-stub-v1.0",
  "overall_confidence": 0.88,
  "created_at": "2025-11-20T16:30:00.000Z"
}
```

### Persistence

All simulation results are automatically persisted to Neo4j HCG:

1. **Simulation node**: Metadata about the simulation
2. **ImaginedProcess nodes**: Each process with `imagined:true`
3. **ImaginedState nodes**: Each state with `imagined:true`
4. **Edges**: Links between simulation and states (`produces` relation)

## Swapping in Hardware Simulators

The JEPA runner is designed to be easily swappable with hardware simulators like Talos or Gazebo.

### Current Implementation (CPU Stub)

The current implementation is a CPU-friendly stub that:
- Requires no GPU or external dependencies
- Performs basic state evolution
- Applies simple action effects
- Uses confidence decay for uncertainty modeling

### Integrating Talos/Gazebo

To integrate a hardware simulator:

1. **Implement the JEPARunner Interface**

```python
class TalosJEPARunner(JEPARunner):
    def __init__(self, talos_config, **kwargs):
        super().__init__(**kwargs)
        self.talos_client = TalosClient(talos_config)
    
    def simulate(self, context, k_steps, assumptions=None):
        # Send context to Talos simulator
        sim_id = self.talos_client.create_simulation(context)
        
        # Run k-step rollout using Talos physics
        talos_result = self.talos_client.run_steps(sim_id, k_steps)
        
        # Convert Talos result to SimulationResult format
        return self._convert_talos_result(talos_result)
```

2. **Update Configuration**

```python
# In src/sophia/api/app.py
if os.getenv("USE_TALOS_SIMULATOR", "false").lower() == "true":
    from sophia.jepa.talos_runner import TalosJEPARunner
    _jepa_runner = TalosJEPARunner(
        talos_config=get_talos_config(),
        model_version="talos-v1.0"
    )
else:
    _jepa_runner = JEPARunner(model_version="jepa-stub-v1.0")
```

3. **Update Talos Metadata**

When using Talos, update the metadata in requests:

```json
{
  "talos_metadata": {
    "simulator_version": "talos-v2.0",
    "physics_engine": "ODE",
    "time_step": 0.001,
    "use_hardware": true,
    "robot_model": "talos"
  }
}
```

### Configuration Variables

Add to `.env`:

```bash
# JEPA Simulator Configuration
USE_TALOS_SIMULATOR=false
TALOS_URI=http://localhost:11345
TALOS_TIMEOUT=30
GAZEBO_URI=http://localhost:11346
```

## Testing

### Unit Tests

```bash
# Run JEPA runner tests
poetry run pytest tests/jepa/ -v

# Run simulation API tests
poetry run pytest tests/api/test_api.py::TestSimulateEndpoint -v
```

### Integration Tests

Integration tests with live Neo4j can be run with:

```bash
poetry run pytest -m integration tests/integration/test_prototype_integration.py
```

### Example Test

```python
def test_simulation_with_actions():
    runner = JEPARunner()
    
    entities = [Entity(id="robot", type="agent")]
    actions = [{"type": "MOVE", "target": "robot"}]
    context = SimulationContext(entities=entities, actions=actions)
    
    result = runner.simulate(context, k_steps=3)
    
    assert len(result.imagined_states) == 3
    assert all(s.imagined for s in result.imagined_states)
    assert all(s.model_version == runner.model_version 
               for s in result.imagined_states)
```

## Performance Considerations

### CPU Stub Performance

The CPU stub is optimized for:
- Fast startup (no model loading)
- Low memory usage
- Quick iteration during development
- Testing without external dependencies

Typical performance:
- 1-5 steps: <100ms
- 10 steps: <500ms
- 50 steps: <2s

### Hardware Simulator Performance

When using Talos/Gazebo:
- Physics simulation adds latency
- GPU acceleration available
- Higher fidelity predictions
- Resource requirements depend on scene complexity

## Future Enhancements

1. **Learned Dynamics Models**
   - Train neural network models on real data
   - Replace stub with learned predictor
   - Improve confidence calibration

2. **Uncertainty Quantification**
   - Ensemble predictions
   - Bayesian inference
   - Probabilistic rollouts

3. **Multi-Modal Simulation**
   - Visual rendering
   - Haptic feedback
   - Audio simulation

4. **Parallel Rollouts**
   - Explore multiple trajectories
   - Compare outcomes
   - Select best paths

## References

- PHASE2_SPEC.md: Imagination/simulation pipeline section
- API Documentation: `/docs` and `/redoc` endpoints
- Test Suite: `tests/jepa/` and `tests/api/test_api.py`

## Support

For questions or issues:
1. Check existing tests for examples
2. Review API documentation at `/docs`
3. Open an issue on GitHub
