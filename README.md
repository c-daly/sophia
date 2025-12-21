# Sophia

[![CI](https://github.com/c-daly/sophia/actions/workflows/ci.yml/badge.svg)](https://github.com/c-daly/sophia/actions/workflows/ci.yml)

**Non-linguistic cognitive core for [Project LOGOS](https://github.com/c-daly/logos)**

Sophia is the cognitive core: planning, execution, world models, and knowledge graph operations. Reasoning happens in graph structures, not natural language.

## Quick Start

```bash
# Install
poetry install

# Run
poetry run uvicorn sophia.main:app --host 0.0.0.0 --port 8000 --reload

# Test
poetry run pytest tests/unit/ -v
```

### Docker

```bash
docker pull ghcr.io/c-daly/sophia:latest
docker run -p 8000:8000 \
  -e NEO4J_URI=bolt://localhost:47687 \
  -e NEO4J_USER=neo4j \
  -e NEO4J_PASSWORD=neo4jtest \
  -e MILVUS_HOST=localhost \
  -e MILVUS_PORT=47530 \
  ghcr.io/c-daly/sophia:latest
```

For ML-enabled builds, use `ghcr.io/c-daly/sophia:ml-latest` (or set `SOPHIA_IMAGE_TAG=ml-latest` in compose files that reference Sophia).

> **Note:** The example above uses default Milvus port 19530. For test stacks, Sophia uses port 47530 (the 47xxx prefix) to avoid conflicts with other services.

The container includes all Python dependencies and the Sophia API service. For development and testing, Sophia uses the `logos-foundry` base image which includes all LOGOS shared packages.

## Integration Tests

Most tests run without external services. The integration suite under
`tests/integration/` requires Neo4j and Milvus. These tests are skipped by
default; use the helper script to run them with Docker:

```bash
./scripts/run_integration_stack.sh
```

The script starts the Neo4j and Milvus services from
`docker-compose.test.yml`, waits for them to become
healthy, and runs the integration tests with `RUN_SOPHIA_INTEGRATION=1`.

### Stack Configuration

The test stack uses non-conflicting ports (47xxx range) to allow
multiple repo stacks to run concurrently:
- Neo4j: 47474 (HTTP), 47687 (Bolt)
- Milvus: 47530 (gRPC), 47091 (Health)

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
## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/plan` | POST | Generate execution plan from goal |
| `/execute` | POST | Execute a plan step |
| `/simulate` | POST | Dry-run simulation of a plan |
| `/imagine` | POST | Counterfactual reasoning |
| `/ingest` | POST | Media ingestion with JEPA embeddings |
| `/health` | GET | Health check |

📖 API docs: `http://localhost:8000/docs` (when running)

## Integration Tests

```bash
./scripts/run_integration_stack.sh
```

Uses port 47xxx range (Neo4j 47474/47687, Milvus 47530).

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | bolt://localhost:47687 | Neo4j connection |
| `NEO4J_USER` | neo4j | Neo4j username |
| `NEO4J_PASSWORD` | neo4jtest | Neo4j password |
| `MILVUS_HOST` | localhost | Milvus server host |
| `MILVUS_PORT` | 47530 | Milvus gRPC port |

## Documentation

- [LOGOS Getting Started](https://github.com/c-daly/logos/blob/main/docs/guides/GETTING_STARTED.md)
- [Architecture Overview](https://github.com/c-daly/logos/blob/main/docs/architecture/ARCHITECTURE.md)
- [Testing Guide](https://github.com/c-daly/logos/blob/main/docs/guides/TESTING.md)
- [SDK Guide](https://github.com/c-daly/logos/blob/main/docs/sdk/SDK_GUIDE.md)

## License

MIT - see [LICENSE](LICENSE)
