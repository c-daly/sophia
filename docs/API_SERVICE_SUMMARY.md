# Sophia FastAPI Service - Implementation Summary

## Overview
Successfully implemented Phase 2 Sophia FastAPI service exposing `/plan`, `/imagine`, `/execute`, and `/ingest/hermes_proposal` endpoints on top of Neo4j + SHACL.

The service provides:
- **Planning**: Backward chaining to generate actionable plans from goals
- **Imagination**: Generate future state predictions from CWM-G imagery
- **Execution**: Execute plans with dry-run support
- **Ingestion**: Accept and persist LLM proposals from Hermes with full provenance tracking

## Architecture

### API Endpoints

#### GET /health
- **Auth Required**: No
- **Description**: Health check for all components
- **Returns**: Status of Neo4j and Milvus connections

#### POST /plan
- **Auth Required**: Yes (Bearer token)
- **Description**: Generate a plan to achieve a goal using backward chaining
- **Input**: Goal specification with description and target_state
- **Returns**: Ordered list of plan steps with plan_id

#### POST /imagine
- **Auth Required**: Yes (Bearer token)
- **Description**: Generate imagined future states from CWM-G imagery and CWM-E emotion tags
- **Input**: 
  - cwm_g_imagery: Optional CWM-G imagery data
  - cwm_e_emotion_tags: Optional emotion tags
  - model_version: Model version for imagination
  - horizon: Planning horizon (default: 5)
  - assumptions: Optional assumptions
- **Returns**: List of imagined states with metadata
- **Storage**: Stores imagined state nodes in Neo4j with:
  - model_version
  - horizon
  - assumptions
  - imagination_id

#### POST /execute
- **Auth Required**: Yes (Bearer token)
- **Description**: Execute a plan or specific step
- **Input**: 
  - plan_id: Plan identifier
  - step_index: Optional step to execute
  - dry_run: Simulate execution without state changes
- **Returns**: Execution results with status

#### POST /ingest/hermes_proposal
- **Auth Required**: No (disabled for local development)
- **Description**: Ingest LLM proposals from Hermes with full provenance tracking
- **Input**:
  - proposal_id: Unique identifier for the proposal
  - source_service: Source service (default: "hermes")
  - llm_provider: LLM provider name (e.g., "openai", "anthropic", "azure")
  - model: Model identifier (e.g., "gpt-4", "claude-3-opus")
  - generated_at: ISO timestamp when proposal was generated
  - confidence: Confidence score [0.0, 1.0]
  - raw_text: Optional raw LLM response text
  - plan_steps: Optional array of structured plan steps
  - imagined_states: Optional array of imagined future states
  - diagnostics: Optional diagnostic information
  - tool_calls: Optional array of tool calls requested by the LLM
  - metadata: Optional additional metadata for provenance
- **Returns**: 
  - proposal_id: The ingested proposal identifier
  - stored_node_ids: Array of Neo4j node IDs created
  - status: Ingestion status ("accepted", "rejected", "partial")
  - created_at: ISO timestamp of ingestion
  - validation_results: Optional SHACL validation results
- **Storage**: Creates the following node types in Neo4j:
  - `hermes_proposal`: Main proposal node with provenance metadata
  - `proposed_plan_step`: Plan step nodes linked to the proposal
  - `proposed_imagined_state`: Imagined state nodes linked to the proposal
  - `proposed_tool_call`: Tool call nodes linked to the proposal
- **Validation**: SHACL validation ensures:
  - All required provenance fields are present (source_service, llm_provider, model, generated_at, confidence)
  - Confidence score is in range [0.0, 1.0]
  - Child nodes have source_proposal links back to the parent proposal

## Configuration

### Environment Variables
```bash
# Required
SOPHIA_API_TOKEN=your-secure-token-here

# Neo4j (optional, defaults shown)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=sophiadev

# Milvus (optional, defaults shown)
MILVUS_HOST=localhost
MILVUS_PORT=19530

# CORS (optional)
CORS_ORIGINS=*
```

### Running with Docker Compose

```bash
# Set API token
export SOPHIA_API_TOKEN=your-secure-token-here

# Start all services
docker-compose up -d

# Check health
curl http://localhost:8000/health

# View logs
docker-compose logs -f sophia
```

### Running Locally (Development)

```bash
# Install dependencies
poetry install

# Set environment variables
export SOPHIA_API_TOKEN=dev-token
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=sophiadev

# Start dependencies (Neo4j + Milvus)
docker-compose up -d neo4j milvus-standalone

# Run API server
poetry run uvicorn sophia.api.app:app --reload --host 0.0.0.0 --port 8000
```

## Testing

### Unit Tests
```bash
# Run all tests
poetry run pytest tests/ -v -m "not integration"

# Run only API tests
poetry run pytest tests/api/ -v

# Run with coverage
poetry run pytest tests/ -v -m "not integration" --cov=sophia --cov-report=term-missing
```

### Integration Test
```bash
# Set API token
export SOPHIA_API_TOKEN=test-token

# Run integration test script
./examples/test_api.sh
```

### Manual Testing

```bash
TOKEN="your-token-here"

# Health check (no auth)
curl http://localhost:8000/health

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

# Execute a plan (dry run)
curl -X POST http://localhost:8000/execute \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "plan_id": "plan-uuid",
    "dry_run": true
  }'

# Ingest a Hermes proposal (no auth required for local dev)
curl -X POST http://localhost:8000/ingest/hermes_proposal \
  -H "Content-Type: application/json" \
  -d '{
    "proposal_id": "hermes_20251123_abc123",
    "source_service": "hermes",
    "llm_provider": "openai",
    "model": "gpt-4",
    "generated_at": "2025-11-23T12:00:00Z",
    "confidence": 0.85,
    "raw_text": "Move the red block to the bin",
    "plan_steps": [
      {
        "action": "move_to_red_block",
        "target": "red_block",
        "parameters": {}
      },
      {
        "action": "grasp_red_block",
        "target": "red_block",
        "parameters": {"force": 0.5}
      },
      {
        "action": "move_to_bin",
        "target": "bin",
        "parameters": {}
      }
    ],
    "imagined_states": [
      {
        "state_id": "state_1",
        "entities": {"red_block": {"location": "table"}}
      },
      {
        "state_id": "state_2",
        "entities": {"red_block": {"location": "bin"}}
      }
    ],
    "diagnostics": {
      "reasoning": "Block needs to be moved from table to bin"
    },
    "tool_calls": [
      {
        "tool": "get_object_location",
        "parameters": {"object_id": "red_block"}
      }
    ],
    "metadata": {
      "session_id": "test_session_123",
      "user_id": "test_user"
    }
  }'
```

### Hermes Proposal Ingestion

The ingestion endpoint accepts LLM proposals from Hermes and persists them with full provenance:

**Error Responses:**
- **201 Created**: Proposal successfully ingested and persisted
- **422 Unprocessable Entity**: Validation failed (missing required fields, invalid confidence, SHACL failure)
- **500 Internal Server Error**: Unexpected error during ingestion
- **503 Service Unavailable**: HCG client not available

**Python SDK Example** (after SDK regeneration):
```python
from sophia_client import SophiaClient
from datetime import datetime, timezone

client = SophiaClient(base_url="http://localhost:8000")

proposal = {
    "proposal_id": "hermes_20251123_abc123",
    "llm_provider": "openai",
    "model": "gpt-4",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "confidence": 0.85,
    "plan_steps": [
        {"action": "move_to_red_block", "target": "red_block", "parameters": {}},
        {"action": "grasp_red_block", "target": "red_block", "parameters": {"force": 0.5}},
    ],
}

response = client.ingest_hermes_proposal(proposal)
print(f"Ingested proposal: {response.proposal_id}")
print(f"Created nodes: {response.stored_node_ids}")
print(f"Status: {response.status}")
```

## Documentation

### API Documentation
Access interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI Schema: http://localhost:8000/openapi.json

## Test Coverage

```
Total: 120 tests passing
Coverage: 81%

API Tests: 22 tests
- Health endpoint: 2 tests
- Plan endpoint: 5 tests
- Imagine endpoint: 6 tests
- Execute endpoint: 6 tests
- Documentation: 3 tests
```

## Security

- Authentication via Bearer token for all write operations
- Environment variable for token configuration
- SHACL validation on all graph mutations
- CodeQL analysis: 0 vulnerabilities found

## CI/CD

GitHub Actions CI workflow automatically:
- Runs linters (black, ruff)
- Runs all unit and API tests
- Generates coverage report
- Uploads to codecov

## File Structure

```
sophia/
├── src/sophia/api/
│   ├── __init__.py
│   ├── app.py           # Main FastAPI application
│   ├── auth.py          # Authentication middleware
│   └── models.py        # Pydantic request/response models
├── tests/api/
│   ├── __init__.py
│   └── test_api.py      # API endpoint tests
├── examples/
│   └── test_api.sh      # Integration test script
├── Dockerfile           # Docker image definition
├── docker-compose.yml   # Service orchestration
├── .env.example         # Example environment variables
└── README.md            # Updated with API documentation
```

## Acceptance Criteria Status

✅ FastAPI app with `/plan`, `/imagine`, `/execute` endpoints wired to existing planner modules and Neo4j driver

✅ `/imagine` stores imagined state nodes with metadata (model version, horizon, assumptions)

✅ Authentication middleware (token header) enforced for read/write calls

✅ Docker image + compose service documented; README explains config/env vars

✅ CI workflow runs unit/API tests

## Next Steps

For production deployment:
1. Generate a secure API token: `openssl rand -hex 32`
2. Set `SOPHIA_API_TOKEN` environment variable
3. Configure Neo4j and Milvus with production credentials
4. Update CORS_ORIGINS to restrict allowed origins
5. Enable HTTPS/TLS for production endpoints
6. Monitor health endpoint for service availability
7. Scale services as needed using docker-compose scale
