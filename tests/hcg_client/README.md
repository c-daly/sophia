# HCG Client Integration Tests

This directory contains tests for the HCG (Hierarchical Cognitive Graph) client with Neo4j and Milvus adapters.

## Test Types

### Unit Tests
Unit tests use mocks and don't require external services:
- `test_shacl_validator.py` - SHACL validation tests
- `test_neo4j_adapter.py` - Neo4j adapter tests (mocked)
- `test_milvus_adapter.py` - Milvus adapter tests (mocked)
- `test_client.py` - HCG client tests (mocked)

Run unit tests:
```bash
poetry run pytest tests/hcg_client/ -v -m "not integration"
```

### Integration Tests
Integration tests require live Neo4j and Milvus services running via docker-compose:
- `test_integration.py` - Full integration tests with live services

## Running Integration Tests

### 1. Start Services

First, start Neo4j and Milvus using docker-compose:

```bash
docker-compose -f docker-compose.hcg.dev.yml up -d
```

Wait for services to be ready (about 30-60 seconds):
```bash
# Check Neo4j
curl http://localhost:7474

# Check Milvus
curl http://localhost:9091/healthz
```

### 2. Run Integration Tests

Run all tests including integration:
```bash
poetry run pytest tests/hcg_client/ -v
```

Run only integration tests:
```bash
poetry run pytest tests/hcg_client/test_integration.py -v
```

### 3. Stop Services

After testing:
```bash
docker-compose -f docker-compose.hcg.dev.yml down
```

To remove volumes as well:
```bash
docker-compose -f docker-compose.hcg.dev.yml down -v
```

## Service URLs

When services are running:
- **Neo4j Browser**: http://localhost:7474
  - Username: `neo4j`
  - Password: `sophiadev`
- **Neo4j Bolt**: bolt://localhost:7687
- **Milvus gRPC**: localhost:19530
- **Milvus Metrics**: http://localhost:9091

## Troubleshooting

### Services won't start
- Ensure Docker is running
- Check ports are not already in use: `lsof -i :7474,7687,19530,9091`
- Check logs: `docker-compose -f docker-compose.hcg.dev.yml logs`

### Tests fail with connection errors
- Wait longer for services to fully start
- Check health: `docker-compose -f docker-compose.hcg.dev.yml ps`
- Restart services: `docker-compose -f docker-compose.hcg.dev.yml restart`

### Tests fail with validation errors
- This is expected behavior for negative test cases
- Check that positive test cases pass

## CI/CD

In CI environments, integration tests are typically:
1. Skipped by default using `-m "not integration"`
2. Run in a separate job with docker-compose services
3. May use test databases separate from production
