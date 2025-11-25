# Sophia Prototype: Minimal Plan/State API over HCG

This prototype demonstrates Sophia's minimal planning and state management API operating on Neo4j with SHACL validation.

## Overview

The prototype implements:

1. **`/state` API** - Read/write world state from/to Neo4j HCG
2. **`/plan` API** - Generate plans using backward chaining over HCG
3. **SHACL Validation** - All graph mutations are validated
4. **Pick-and-Place Scenario** - Seeded test data for robotics planning

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              Sophia FastAPI Service                 │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │ /plan (POST) │  │/state (GET)  │  │  /state   │ │
│  │              │  │              │  │  (POST)   │ │
│  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘ │
│         │                 │                 │       │
│         └─────────────────┴─────────────────┘       │
│                           │                         │
│                    ┌──────▼──────┐                  │
│                    │ HCG Client  │                  │
│                    │ (+ SHACL)   │                  │
│                    └──────┬──────┘                  │
└───────────────────────────┼─────────────────────────┘
                            │
                     ┌──────▼──────┐
                     │   Neo4j     │
                     │   (Graph)   │
                     └─────────────┘
```

## Prerequisites

- Docker and Docker Compose
- Python >=3.11 with Poetry
- Neo4j 5.x (via Docker)
- Milvus 2.x (via Docker)

## CI Coverage

- **Standard CI** runs on every PR/push, covering lint, unit tests, and non-integration API suites.
- **Prototype Integration** lives in `.github/workflows/prototype-integration.yml` and runs nightly, on demand (`workflow_dispatch`), and when prototype-specific files change on `main`. It executes `scripts/run_prototype_integration.sh`, which now fails fast if Neo4j/Milvus containers cannot become healthy within 15 minutes, capturing recent Docker logs for debugging.

## Quick Start

### 1. Start Services

```bash
# Set API token
export SOPHIA_API_TOKEN=your-secure-token-here

# Start Neo4j and Milvus
docker-compose up -d neo4j milvus-standalone

# Wait for services to be ready (check logs)
docker-compose logs -f neo4j
```

### 2. Run Sophia API

The API automatically seeds the Neo4j database with pick-and-place data on startup.

```bash
# Install dependencies
poetry install

# Set environment variables
export SOPHIA_API_TOKEN=test-token
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=sophiadev
export SEED_PICK_AND_PLACE_DATA=true  # Enable data seeding

# Run the API server
poetry run uvicorn sophia.api.app:app --reload --host 0.0.0.0 --port 8000
```

### 3. Run the Prototype Demo

```bash
# In a new terminal
export SOPHIA_API_TOKEN=test-token
./examples/prototype_demo.sh
```

## API Endpoints

### GET /state

Read the current world state from Neo4j.

**Request:**
```bash
curl -X GET http://localhost:8000/state \
  -H "Authorization: Bearer $TOKEN"
```

**Response:**
```json
{
  "state": {
    "red_block": {"location": "table", "grasped": false},
    "blue_block": {"location": "table", "grasped": false},
    "gripper": {"position": "home", "holding": null}
  },
  "state_id": "current_state",
  "timestamp": "2025-11-20T01:00:00.000Z"
}
```

### POST /state

Update the world state in Neo4j with SHACL validation.

**Request:**
```bash
curl -X POST http://localhost:8000/state \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "state": {
      "red_block": {"location": "bin", "grasped": false},
      "blue_block": {"location": "table", "grasped": false},
      "gripper": {"position": "bin", "holding": null}
    }
  }'
```

**Response:**
```json
{
  "state_id": "current_state",
  "updated_at": "2025-11-20T01:00:00.000Z",
  "validation_passed": true
}
```

### POST /plan

Generate a plan for a given goal using backward chaining.

**Request:**
```bash
curl -X POST http://localhost:8000/plan \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "goal": {
      "description": "red block in bin",
      "target_state": "red_block_in_bin"
    }
  }'
```

**Response:**
```json
{
  "plan": [
    {
      "id": "move_to_red_block",
      "name": "Move to Red Block",
      "type": "action",
      "action_type": "MOVE",
      "target": "red_block"
    },
    {
      "id": "grasp_red_block",
      "name": "Grasp Red Block",
      "type": "action",
      "action_type": "GRASP",
      "target": "red_block"
    },
    {
      "id": "move_to_bin",
      "name": "Move to Bin",
      "type": "action",
      "action_type": "MOVE",
      "target": "bin"
    },
    {
      "id": "release_red_block",
      "name": "Release Red Block",
      "type": "action",
      "action_type": "RELEASE",
      "target": "red_block"
    }
  ],
  "goal": {
    "description": "red block in bin",
    "target_state": "red_block_in_bin"
  },
  "plan_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2025-11-20T01:00:00.000Z"
}
```

## Sample Workflow

### 1. Read Initial State

```bash
curl -X GET http://localhost:8000/state \
  -H "Authorization: Bearer test-token"
```

**Output:**
```json
{
  "state": {
    "red_block": {"location": "table", "grasped": false},
    "blue_block": {"location": "table", "grasped": false},
    "gripper": {"position": "home", "holding": null}
  },
  "state_id": "current_state",
  "timestamp": "2025-11-20T01:05:00.123Z"
}
```

### 2. Generate Plan

```bash
curl -X POST http://localhost:8000/plan \
  -H "Authorization: Bearer test-token" \
  -H "Content-Type: application/json" \
  -d '{
    "goal": {
      "description": "red block in bin",
      "target_state": "red_block_in_bin"
    }
  }'
```

**Output:**
```json
{
  "plan": [
    {"id": "move_to_red_block", "name": "Move to Red Block", "type": "action", "action_type": "MOVE", "target": "red_block"},
    {"id": "grasp_red_block", "name": "Grasp Red Block", "type": "action", "action_type": "GRASP", "target": "red_block"},
    {"id": "move_to_bin", "name": "Move to Bin", "type": "action", "action_type": "MOVE", "target": "bin"},
    {"id": "release_red_block", "name": "Release Red Block", "type": "action", "action_type": "RELEASE", "target": "red_block"}
  ],
  "goal": {"description": "red block in bin", "target_state": "red_block_in_bin"},
  "plan_id": "abc-123-def-456",
  "created_at": "2025-11-20T01:05:01.234Z"
}
```

✅ **Plan follows MOVE→GRASP→MOVE→RELEASE pattern**
✅ **Plan written to Neo4j with SHACL validation**

### 3. Update State

```bash
curl -X POST http://localhost:8000/state \
  -H "Authorization: Bearer test-token" \
  -H "Content-Type: application/json" \
  -d '{
    "state": {
      "red_block": {"location": "bin", "grasped": false},
      "blue_block": {"location": "table", "grasped": false},
      "gripper": {"position": "bin", "holding": null}
    }
  }'
```

**Output:**
```json
{
  "state_id": "current_state",
  "updated_at": "2025-11-20T01:05:02.345Z",
  "validation_passed": true
}
```

✅ **State updated in Neo4j with SHACL validation**

### 4. Verify Updated State

```bash
curl -X GET http://localhost:8000/state \
  -H "Authorization: Bearer test-token"
```

**Output:**
```json
{
  "state": {
    "red_block": {"location": "bin", "grasped": false},
    "blue_block": {"location": "table", "grasped": false},
    "gripper": {"position": "bin", "holding": null}
  },
  "state_id": "current_state",
  "timestamp": "2025-11-20T01:05:03.456Z"
}
```

✅ **State persisted in Neo4j HCG**

## SHACL Validation

All graph mutations (adding/updating nodes and edges) are validated with SHACL constraints.

### Valid State Update
```bash
curl -X POST http://localhost:8000/state \
  -H "Authorization: Bearer test-token" \
  -H "Content-Type: application/json" \
  -d '{
    "state": {
      "red_block": {"location": "bin", "grasped": false}
    }
  }'
```
✅ **Passes validation** (HTTP 200)

### Invalid State Update
```bash
curl -X POST http://localhost:8000/state \
  -H "Authorization: Bearer test-token" \
  -H "Content-Type: application/json" \
  -d '{
    "state": "invalid_format"
  }'
```
❌ **Fails validation** (HTTP 422)

## Testing

### Run Unit Tests
```bash
poetry run pytest tests/test_plan_api_pick_and_place.py -v
```

### Run Integration Tests (Requires Neo4j)
```bash
# Start services
docker-compose up -d neo4j milvus-standalone

# Run API tests
poetry run pytest tests/api/ -v -m integration

# Or use the shell script
export SOPHIA_API_TOKEN=test-token
./examples/prototype_demo.sh
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SOPHIA_API_TOKEN` | (required) | API authentication token |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `sophiadev` | Neo4j password |
| `MILVUS_HOST` | `localhost` | Milvus host |
| `MILVUS_PORT` | `19530` | Milvus port |
| `SEED_PICK_AND_PLACE_DATA` | `true` | Auto-seed pick-and-place data |
| `CLEAR_BEFORE_SEED` | `false` | Clear HCG before seeding |

## Key Features

✅ **Neo4j HCG Backend** - All state and planning data stored in Neo4j
✅ **SHACL Validation** - Graph constraints enforced on all mutations
✅ **Backward Chaining** - Goal decomposition using causal relationships
✅ **Pick-and-Place Domain** - Template-based MOVE→GRASP→MOVE→RELEASE plans
✅ **RESTful API** - Clean, documented endpoints
✅ **Authentication** - Token-based API security

## Files

- `src/sophia/api/app.py` - Main FastAPI application with `/plan` and `/state` endpoints
- `src/sophia/api/models.py` - Pydantic models for requests/responses
- `src/sophia/hcg_client/seeder.py` - Pick-and-place data seeding utility
- `examples/prototype_demo.sh` - Complete demonstration script
- `tests/test_plan_api_pick_and_place.py` - Functional tests

## Troubleshooting

### Neo4j Connection Issues
```bash
# Check if Neo4j is running
docker-compose ps neo4j

# View Neo4j logs
docker-compose logs neo4j

# Verify credentials
docker exec -it sophia-neo4j-1 cypher-shell -u neo4j -p sophiadev
```

### SHACL Validation Errors
- Ensure all nodes have a `type` field
- Ensure all edges have `source`, `target`, and `relation` fields
- Check the SHACL shapes in `src/sophia/hcg_client/shacl_validator.py`

### API Authentication Errors
```bash
# Set the API token
export SOPHIA_API_TOKEN=your-token-here

# Include Bearer token in requests
curl -H "Authorization: Bearer $SOPHIA_API_TOKEN" ...
```

## Next Steps

1. **Scale to Complex Domains** - Extend beyond pick-and-place
2. **Hierarchical Planning** - Multi-level goal decomposition
3. **Execution Monitoring** - Real-time state tracking
4. **Causal Reasoning** - Enhanced forward/backward chaining
5. **Multi-Agent Coordination** - Distributed planning

## License

MIT
