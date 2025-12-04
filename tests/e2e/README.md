# End-to-End Tests

This directory contains Sophia's end-to-end (e2e) integration tests.

## Stack Configuration

The test stack is configured via compose files at the repository root:
- `docker-compose.test.yml` - Base infrastructure (Neo4j, Milvus)
- `docker-compose.test.sophia.yml` - Sophia API service overlay

### Services

Sophia requires:
- **Neo4j** (ports 37474/37687) - Knowledge graph storage
- **Milvus** (ports 39530/39091) - Vector similarity search
- **Sophia API** (port 38001) - The service under test

### Port Allocation

Sophia uses the 37xxx/38xxx/39xxx port range to avoid conflicts:
| Service | Host Port | Container Port |
|---------|-----------|----------------|
| Neo4j HTTP | 37474 | 7474 |
| Neo4j Bolt | 37687 | 7687 |
| Milvus gRPC | 39530 | 19530 |
| Milvus Health | 39091 | 9091 |

## Running Integration Tests

### Using the Helper Scripts

```bash
# Start the test stack
./scripts/start_services.sh

# Run integration tests
./scripts/test_integration.sh test

# Run e2e tests
./scripts/test_e2e.sh

# Stop the stack
./scripts/stop_services.sh
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SOPHIA_API_TOKEN` | API authentication token | `test-token-for-sophia` |
| `NEO4J_URI` | Neo4j connection URI | `bolt://localhost:37687` |
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | `neo4jtest` |
| `MILVUS_HOST` | Milvus host | `localhost` |
| `MILVUS_PORT` | Milvus port | `39530` |
| `SOPHIA_URL` | Sophia API URL | `http://localhost:38001` |

## Running Locally

```bash
# Start all services
docker compose -f docker-compose.test.yml -f docker-compose.test.sophia.yml up -d

# Run tests
poetry run pytest tests/ -v

# Stop services
docker compose -f docker-compose.test.yml -f docker-compose.test.sophia.yml down
```
