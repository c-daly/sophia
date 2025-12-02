# Sophia

[![CI](https://github.com/c-daly/sophia/actions/workflows/ci.yml/badge.svg)](https://github.com/c-daly/sophia/actions/workflows/ci.yml)

**Non-linguistic cognitive core for Project LOGOS**

Sophia is a foundational infrastructure for building knowledge graphs and managing cognitive data structures. It provides a flexible, extensible framework for representing and storing knowledge in a graph-based format.

## Features

- **Knowledge Graph**: In-memory graph structure using NetworkX for efficient node and edge management
- **Persistent Storage**: SQLAlchemy-based database abstraction for storing knowledge graphs
- **Type-Safe Models**: Pydantic-based data models for nodes and edges
- **Configuration Management**: Flexible settings management for deployment
- **Extensible Architecture**: Clean, modular design for easy extension
- **FastAPI Service**: RESTful API with `/plan`, `/imagine`, `/execute`, `/simulate`, and `/ingest` endpoints
- **JEPA Integration**: Joint-Embedding Predictive Architecture for dynamics simulation and media perception
- **Media Ingestion**: Upload and process images/video with automatic JEPA embedding generation
- **Neo4j + SHACL**: Integration with Neo4j graph database and SHACL validation
- **Milvus Vector Store**: 768-dimensional embeddings for semantic search and cross-modal reasoning
- **Authentication**: Token-based authentication middleware for secure API access
- **Docker Support**: Containerized deployment with Docker Compose

## Installation

### Prerequisites

- Python >=3.11
- Poetry (for dependency management)

### Install Poetry

If you don't have Poetry installed, install it using:

```bash
curl -sSL https://install.python-poetry.org | python3 -
# or
pip install poetry
```

### Install Sophia

```bash
# Clone the repository
git clone https://github.com/c-daly/sophia.git
cd sophia

# Note: logos_hcg and logos_sophia_sdk are currently vendored inside src/
# to keep installation self-contained until the shared packages are published.

# Install dependencies (includes both runtime and development dependencies)
poetry install

# Activate the virtual environment
poetry shell
```

Alternatively, run commands without activating the shell:

```bash
poetry run python your_script.py
```

### Using Docker

Sophia is available as a pre-built container image for easy deployment:

```bash
# Pull the latest Sophia image
docker pull ghcr.io/c-daly/sophia:latest

# Run Sophia service
docker run -d \
  -p 8000:8000 \
  -e NEO4J_URI=bolt://neo4j:7687 \
  -e NEO4J_USER=neo4j \
  -e NEO4J_PASSWORD=your_password \
  -e MILVUS_HOST=milvus \
  -e MILVUS_PORT=19530 \
  ghcr.io/c-daly/sophia:latest
```

The container includes all Python dependencies and the Sophia API service. For development and testing, Sophia uses the `logos-foundry` base image which includes all LOGOS shared packages.

## Integration Tests

Most tests run without external services. The integration suite under
`tests/integration/` requires Neo4j and Milvus. These tests are skipped by
default; use the helper script to run them with Docker:

```bash
./scripts/run_integration_stack.sh
```

The script starts the Neo4j and Milvus services from
`tests/e2e/stack/sophia/docker-compose.test.yml`, waits for them to become
healthy, and runs the integration tests with `RUN_SOPHIA_INTEGRATION=1`.

### Stack Configuration

The test stack uses non-conflicting ports (37xxx/39xxx range) to allow
multiple repo stacks to run concurrently:
- Neo4j: 37474 (HTTP), 37687 (Bolt)
- Milvus: 39530 (gRPC), 39091 (Health)

Stack files are generated from LOGOS. To regenerate:
```bash
cd /path/to/logos
poetry run render-test-stacks --repo sophia
```

## Quick Start

```python
from sophia import KnowledgeGraph, Database
from sophia.knowledge_graph import Node, Edge

# Create a knowledge graph
kg = KnowledgeGraph()

# Add nodes
concept1 = Node(type="concept", properties={"name": "Learning"})
concept2 = Node(type="concept", properties={"name": "Intelligence"})

kg.add_node(concept1)
kg.add_node(concept2)

# Add edges
relation = Edge(
    source=concept1.id,
    target=concept2.id,
    relation="enables",
    properties={"strength": 0.9}
)
kg.add_edge(relation)

# Persist to database
db = Database("sqlite:///my_knowledge.db")
db.store_node(concept1.id, concept1.type, concept1.properties)
db.store_node(concept2.id, concept2.type, concept2.properties)
db.store_edge(relation.id, relation.source, relation.target, 
              relation.relation, relation.properties)
```

## Sophia API Service 🚀

Sophia provides a FastAPI service with endpoints for planning, imagination, and execution.

### Running with Docker Compose

The easiest way to run Sophia is using Docker Compose:

```bash
# Set your API token (required for authentication)
export SOPHIA_API_TOKEN=your-secure-token-here

# Start all services (Sophia API, Neo4j, Milvus)
docker-compose up -d

# Check service health
curl http://localhost:8000/health

# View logs
docker-compose logs -f sophia
```

The service will be available at `http://localhost:8000` with the following endpoints:
- `GET /health` - Health check (no auth required)
- `GET /state` - Read current world state from Neo4j HCG
- `POST /state` - Update world state in Neo4j with SHACL validation
- `POST /plan` - Generate a plan from a goal (reads/writes to Neo4j)
- `POST /imagine` - Generate imagined future states
- `POST /simulate` - Run JEPA-based k-step dynamics simulation (Phase 2)
- `POST /execute` - Execute a plan
- `POST /ingest/hermes_proposal` - Ingest LLM proposals from Hermes with provenance tracking (no auth for local dev)

### Prototype: Minimal Plan/State API over HCG

See [PROTOTYPE_README.md](PROTOTYPE_README.md) for a comprehensive guide to the prototype implementation, which demonstrates:
- ✅ Reading goal/state from Neo4j HCG
- ✅ Generating MOVE→GRASP→MOVE→RELEASE plans via backward chaining
- ✅ Writing plans back to Neo4j with SHACL validation
- ✅ State management with SHACL gating
- ✅ Pick-and-place scenario with auto-seeding

Quick start:
```bash
# Run the prototype demo
export SOPHIA_API_TOKEN=test-token
./examples/prototype_demo.sh
```

### API Documentation

Once the service is running, access the interactive API docs at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Configuration

Configure the service using environment variables (see `.env.example`):

```bash
# Required
SOPHIA_API_TOKEN=your-secure-token-here

# Optional (defaults shown)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4jtest
MILVUS_HOST=localhost
MILVUS_PORT=19530
CORS_ORIGINS=*
```

### Example API Usage

```bash
# Set your token
TOKEN="your-secure-token-here"

# Generate a plan
curl -X POST http://localhost:8000/plan \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "goal": {
      "description": "red block in bin",
      "target_state": "red_block_in_bin"
    }
  }'

# Generate imagined states
curl -X POST http://localhost:8000/imagine \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "cwm_g_imagery": [{"type": "visual", "content": "red block"}],
    "cwm_e_emotion_tags": ["curious", "focused"],
    "model_version": "v1.0",
    "horizon": 3,
    "assumptions": ["block is graspable"]
  }'

# Run JEPA-based simulation (Phase 2)
curl -X POST http://localhost:8000/simulate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "entities": [
      {
        "id": "red_block",
        "type": "object",
        "properties": {"mass": 0.5},
        "position": {"x": 0.0, "y": 0.0, "z": 0.1}
      }
    ],
    "k_steps": 5,
    "assumptions": ["block is graspable"]
  }'

# Execute a plan
curl -X POST http://localhost:8000/execute \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "plan_id": "plan-uuid",
    "dry_run": true
  }'
```

## Local Testing (CI Parity)

Sophia uses the shared LOGOS workflow template. Run the same commands locally before opening a PR:

```bash
poetry install --with dev
poetry run ruff check src tests
poetry run black --check src tests
poetry run mypy src
poetry run pytest tests/ -v -m "not integration" --cov=sophia --cov-report=term --cov-report=xml
```

If these commands pass locally, they will pass the GitHub Actions gate defined in `.github/workflows/ci.yml`.

### Running Locally (Development)

For development without Docker:

```bash
# Install dependencies
poetry install

# Set environment variables
export SOPHIA_API_TOKEN=dev-token
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=neo4jtest

# Start Neo4j and Milvus (using docker-compose for dependencies only)
docker-compose up -d neo4j milvus-standalone

# Run the API server
poetry run uvicorn sophia.api.app:app --reload --host 0.0.0.0 --port 8000
```

## Milestone M3: Sophia can plan simple actions ✅

**Status**: Complete (Epoch 3: Cognitive Core & Reasoning)

Sophia now has a complete cognitive architecture with:
- 🧠 **Planning**: Backward chaining for goal decomposition
- 🗺️ **World Modeling**: Knowledge graphs for representing domain knowledge
- 🎯 **Reasoning**: Causal relationships and action sequencing
- 💾 **State Management**: Tracking and updating world state

### Try the Demo

```bash
# Run the end-to-end milestone demonstration
poetry run python examples/milestone_m3_demo.py
```

This demonstrates:
1. Building a world model (knowledge graph)
2. Planning a pick-and-place task
3. Simulating execution with state updates
4. Integrating all cognitive components

See [Milestone M3 Verification](docs/MILESTONE_M3_VERIFICATION.md) for detailed documentation.

## Phase 2: JEPA-Based Simulation 🔮

**Status**: Implemented (CPU-friendly stub)

Sophia now includes JEPA (Joint-Embedding Predictive Architecture) simulation capabilities:
- 🎯 **k-step rollouts**: Forward prediction of system dynamics
- 🧠 **Imagined states**: States with `imagined:true`, model version, and confidence scores
- 📊 **Confidence tracking**: Decreasing confidence over prediction horizon
- 🔄 **Action simulation**: Apply action sequences and predict outcomes
- 💾 **Persistence**: All imagined nodes stored in Neo4j with metadata
- 🔧 **Swappable**: Can be replaced with Talos/Gazebo hardware simulators

### Try the Simulation

```bash
# Run a simple 5-step simulation
curl -X POST http://localhost:8000/simulate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "entities": [
      {
        "id": "block",
        "type": "object",
        "properties": {"mass": 0.5},
        "position": {"x": 0.0, "y": 0.0, "z": 0.1}
      }
    ],
    "k_steps": 5,
    "assumptions": ["block is graspable"]
  }'
```

See [JEPA Simulation Documentation](docs/JEPA_SIMULATION.md) for detailed information on:
- Context schema (entities, sensors, Talos metadata)
- Imagined node metadata
- Swapping in hardware simulators
- API usage examples



## Development

### Running Tests

```bash
poetry run pytest
```

**Test Results**: 65/65 tests passing, 98% coverage

#### Functional Tests

Sophia includes functional tests for the plan API with a pick-and-place scenario:

```bash
# Run functional tests
poetry run pytest tests/test_plan_api_pick_and_place.py -v
```

These tests validate:
- Plan generation for goal-directed tasks
- MOVE→GRASP→MOVE→RELEASE action sequences
- State management and updates
- Validation of state changes

### Code Quality

```bash
# Format code
poetry run black src tests

# Lint code
poetry run ruff check src tests

# Type checking
poetry run mypy src
```

### Adding Dependencies

```bash
# Add a runtime dependency
poetry add package-name

# Add a development dependency
poetry add --group dev package-name

# Update dependencies
poetry update
```

## Project Structure

```
sophia/
├── src/sophia/           # Main package
│   ├── knowledge_graph/  # Knowledge graph implementation
│   ├── storage/          # Database abstraction
│   ├── planner/          # Planning and goal decomposition
│   ├── executor/         # Action execution management
│   ├── orchestrator/     # Cognitive process coordination
│   ├── cwm_a/            # Continuous Working Memory - Associative
│   ├── cwm_g/            # Continuous Working Memory - Generative
│   └── config/           # Configuration management
├── docs/                 # Documentation and research
│   ├── research/         # Research surveys and design documents
│   └── MILESTONE_M3_VERIFICATION.md  # M3 verification document
├── examples/             # Example scripts and demonstrations
│   └── milestone_m3_demo.py  # End-to-end M3 demo
├── tests/                # Test suite
└── pyproject.toml        # Project configuration
```

## Milestones & Roadmap

### ✅ Epoch 1: Infrastructure & Knowledge Foundation

- ✅ Core knowledge graph data structures (Node, Edge, KnowledgeGraph)
- ✅ Persistent storage layer with database abstraction
- ✅ Configuration management system
- ✅ Type-safe, well-documented codebase

### ✅ Epoch 3: Cognitive Core & Reasoning (M3)

- ✅ Planning component with backward chaining
- ✅ Goal decomposition and action sequencing
- ✅ World modeling with causal relationships
- ✅ State management and updates
- ✅ Cognitive architecture integration (Orchestrator, Executor, CWM)
- ✅ Comprehensive test coverage (98%, 65 tests)

### 🚀 Future: Advanced Planning & Reasoning

- ⏳ Causal enhancement with strength annotations
- ⏳ Forward chaining for reactive planning
- ⏳ Counterfactual reasoning
- ⏳ Multi-goal planning
- ⏳ Hierarchical task decomposition

## Research & Documentation

Research documents and design notes are available in the `docs/research/` directory:

- [Causal Reasoning Methods Survey](docs/research/causal-reasoning-methods.md) - Comprehensive survey of backward/forward chaining, causal graphs, and counterfactual reasoning for HCG planning
- [Planner Applicability Notes](docs/research/planner-applicability-notes.md) - Quick reference guide for implementing causal reasoning in the Planner component
- [GNN Integration Assessment](docs/research/gnn-integration-assessment.md) - Analysis of Graph Neural Network approaches for knowledge graph enhancement, integration risks/benefits, and recommendations

## License

MIT
