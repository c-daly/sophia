# End-to-End Tests

This directory contains Sophia's end-to-end (e2e) integration tests and their supporting infrastructure.

## Directory Structure

```
tests/e2e/
├── README.md           # This file
└── stack/
    └── sophia/
        ├── .env.test           # Environment variables for the stack
        ├── STACK_VERSION       # Git commit hash of logos that generated these files
        └── docker-compose.test.yml  # Neo4j + Milvus test stack
```

## Stack Configuration

The test stack is **generated from LOGOS** using the `render-test-stacks` command. This ensures consistency across all repos.

### Services

Sophia requires:
- **Neo4j** (ports 37474/37687) - Knowledge graph storage
- **Milvus** (ports 39530/39091) - Vector similarity search

### Port Allocation

Sophia uses the 37xxx/39xxx port range to avoid conflicts:
| Service | Host Port | Container Port |
|---------|-----------|----------------|
| Neo4j HTTP | 37474 | 7474 |
| Neo4j Bolt | 37687 | 7687 |
| Milvus gRPC | 39530 | 19530 |
| Milvus Health | 39091 | 9091 |

## Running Integration Tests

### Using the Helper Script

```bash
# Run all integration tests
./scripts/run_integration_stack.sh

# Run specific tests
./scripts/run_integration_stack.sh tests/integration/test_specific.py -v
```

The script will:
1. Start the Neo4j + Milvus stack
2. Wait for all services to be healthy
3. Run pytest with the specified arguments
4. Clean up containers on exit

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SOPHIA_REPO_ROOT` | Override repo root detection | Auto-detected |
| `HEALTH_TIMEOUT` | Seconds to wait for services | 180 |
| `COMPOSE_CMD` | Docker compose command | `docker compose` |

## Regenerating Stack Files

If you need to update the stack configuration:

```bash
# From the LOGOS repo
cd /path/to/logos
poetry run render-test-stacks --repo sophia

# Copy the generated files to sophia
cp tests/e2e/stack/sophia/* /path/to/sophia/tests/e2e/stack/sophia/
```

The `STACK_VERSION` file contains the LOGOS commit hash used to generate the files.
