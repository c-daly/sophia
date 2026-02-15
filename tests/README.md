# Sophia Test Guide

The `tests/` tree is organized into three tiers following the test pyramid:

| Directory | Marker | Tests | Dependencies | Run Time |
|-----------|--------|-------|--------------|----------|
| `tests/unit/` | `@pytest.mark.unit` | 132 | None | ~7s |
| `tests/integration/` | `@pytest.mark.integration` | 109 | Neo4j, Milvus | ~15s |
| `tests/e2e/` | `@pytest.mark.e2e` | 41 | Full stack + Sophia API | ~15s |

## Running Tests by Marker

```bash
# Run only unit tests (fast, no services needed)
poetry run pytest -m unit

# Run only integration tests (requires Neo4j + Milvus)
poetry run pytest -m integration

# Run only e2e tests (requires full stack)
poetry run pytest -m e2e

# Run everything except e2e (for local dev)
poetry run pytest -m "not e2e"

# Run everything except slow tests
poetry run pytest -m "not slow"
```

## Test Organization

### Unit Tests (`tests/unit/`)

Fast tests that verify isolated logic. Mocking external services is appropriate here.

```bash
poetry run pytest tests/unit/ -v
```

| Module | What it tests |
|--------|---------------|
| `unit/media/` | File validation, endpoint routing (mocked) |
| `unit/api/` | API endpoint behavior (mocked) |
| `unit/jepa/` | JEPA runner output format, model construction |
| `unit/cwm_state/` | CWMState envelope format validation |
| `unit/error_handling/` | Service failure scenarios (mocked) |
| `unit/hcg_client/` | HCG client wrapper logic |
| Other unit dirs | Orchestrator, executor, planner, cwm_a, cwm_g |

### Integration Tests (`tests/integration/`)

Tests that run against real Neo4j and Milvus. **No mocking of these services.**

```bash
# Start services first
./scripts/run_tests.sh up

# Run integration tests
poetry run pytest tests/integration/ -v
```

| File | What it tests |
|------|---------------|
| `test_prototype_integration.py` | Full plan/state API flow |
| `test_media_ingestion_integration.py` | Media ingest with real Neo4j |
| `test_hermes_ingestion_integration.py` | Hermes proposal ingest |
| `test_planner_integration.py` | Planner with real graph |
| `test_execute_integration.py` | Execution with real state |
| `test_media_storage.py` | Real file operations |

### E2E Tests (`tests/e2e/`)

Full workflow tests requiring the complete stack.

```bash
poetry run pytest tests/e2e/ -v
```

## CI Testing Behavior

### Coverage Requirement

All CI jobs enforce a **60% minimum coverage** threshold. Tests will fail if coverage drops below 60%.

### CI Jobs Overview

| Job | Triggers | Services | Purpose |
|-----|----------|----------|---------|
| **standard** | All PRs/pushes | ✅ Neo4j, Milvus | Lint + type check + full test suite (via reusable workflow) |

The standard job:
- Starts Neo4j and Milvus via docker-compose
- Runs ALL tests including integration tests
- Enforces 60% minimum coverage
- Reports skip reasons with `-r sS`

### Weekly Scheduled Run

Full integration tests run automatically every **Sunday at 4 AM UTC** to catch any regressions.

---

| Category | Paths | What it exercises | Typical command |
| --- | --- | --- | --- |
| API surface & Hermes ingestion | `tests/unit/api/` | FastAPI endpoints, validation, routing, Hermes payload handling, and HTTP error semantics. | `poetry run pytest tests/unit/api/` |
| Planning, orchestration & execution | `tests/unit/planner/`, `tests/unit/orchestrator/`, `tests/unit/executor/` | Backward-chaining planner logic, orchestrator state machine, and executor callbacks. | `poetry run pytest tests/unit/planner tests/unit/orchestrator tests/unit/executor` |
| Knowledge graph & storage | `tests/unit/knowledge_graph/`, `tests/unit/storage/` | Graph node/edge invariants, Neo4j/Milvus persistence adapters, and configuration defaults. | `poetry run pytest tests/unit/knowledge_graph tests/unit/storage` |
| HCG client & envelopes | `tests/unit/hcg_client/`, `tests/unit/cwm_state/` | Low-level client wrapper around Neo4j/Milvus plus the schema used to exchange state with CWM. | `poetry run pytest tests/unit/hcg_client tests/unit/cwm_state` |
| CWM connectors | `tests/unit/cwm_a/`, `tests/unit/cwm_g/` | Serializers/parsers that map Sophia data structures to CWM-A/G representations. | `poetry run pytest tests/unit/cwm_a tests/unit/cwm_g` |
| Media ingestion & JEPA perception | `tests/unit/media/`, `tests/unit/jepa/` | Upload pipeline, metadata extraction, JEPA runner behavior, and simulation coupling. | `poetry run pytest tests/unit/media tests/unit/jepa` |
| Integration tests | `tests/integration/` | Full plan/state API flow backed by Neo4j + Milvus via standardized stack. | `./scripts/run_tests.sh integration` |
| E2E tests | `tests/e2e/` | Full workflow tests requiring the complete stack. | `./scripts/run_tests.sh e2e` |
| Data fixtures | `tests/data/` | Validates that the Cypher seed used for pick-and-place stays in sync with tests and docs. | N/A |

## Tips

- Use `pytest -k <keyword>` to filter within any category.
- When editing multiple areas, run `poetry run pytest` from the repo root to
  execute the entire suite.
- The integration suite requires Docker; see `./scripts/test.sh integration`
  and `tests/e2e/README.md` for stack details and environment variables.

## Environment Variables

### Repository Root Configuration

The `SOPHIA_REPO_ROOT` environment variable can be set to override automatic
repository root detection. This is useful when:

- Running tests from a relocated repository
- Running in CI environments where the checkout path differs
- Testing with the repo installed as a package

**Priority order for repo root resolution:**
1. `SOPHIA_REPO_ROOT` environment variable (if set and path exists)
2. `GITHUB_WORKSPACE` (automatically set by GitHub Actions)
3. Fallback to parent directory of `src/sophia/env.py`

**Example usage:**
```bash
# Override repo root for a specific test run
SOPHIA_REPO_ROOT=/custom/path/to/sophia poetry run pytest tests/

# Or export for the session
export SOPHIA_REPO_ROOT=/custom/path/to/sophia
poetry run pytest tests/
```

### Stack Environment Variables

The following variables are used by integration tests and helper scripts:

| Variable | Default | Description |
| --- | --- | --- |
| `SOPHIA_REPO_ROOT` | (auto-detected) | Repository root directory |
| `RUN_SOPHIA_INTEGRATION` | `0` | Set to `1` to enable integration tests |
| `NEO4J_URI` | `bolt://localhost:47687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `neo4jtest` | Neo4j password |
| `MILVUS_HOST` | `localhost` | Milvus host |
| `MILVUS_PORT` | `47530` | Milvus port |

These can be set in `.env.test` or exported directly. The helper module
`sophia.env` provides functions to load and access these values:

```python
from sophia.env import get_repo_root, get_neo4j_config, get_milvus_config

# Get repository root path
repo_root = get_repo_root()

# Get Neo4j connection config
neo4j = get_neo4j_config()
# Returns: {'uri': '...', 'user': '...', 'password': '...'}

# Get Milvus connection config
milvus = get_milvus_config()
# Returns: {'host': '...', 'port': '...', 'healthcheck': '...'}
```
