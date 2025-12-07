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

- [LOGOS Getting Started](https://github.com/c-daly/logos/blob/main/docs/GETTING_STARTED.md)
- [Architecture Overview](https://github.com/c-daly/logos/blob/main/docs/ARCHITECTURE.md)
- [Testing Guide](https://github.com/c-daly/logos/blob/main/docs/TESTING.md)
- [SDK Guide](https://github.com/c-daly/logos/blob/main/docs/SDK_GUIDE.md)

## License

MIT - see [LICENSE](LICENSE)
