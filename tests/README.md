# Sophia Test Guide

The `tests/` tree is organized by the parts of the system it protects instead of
milestones or project phases. Use this guide to find the suite that exercises
the code you are changing.

Most suites can be run with `poetry run pytest <path>`. Long-running tests (for
example, the prototype integration flow) are called out explicitly.

| Category | Paths | What it exercises | Typical command |
| --- | --- | --- | --- |
| API surface & Hermes ingestion | `tests/api/test_api.py`, `tests/api/test_hermes_ingestion.py`, `tests/test_plan_api_pick_and_place.py`, `tests/test_error_handling.py` | FastAPI endpoints, validation, routing, Hermes payload handling, and HTTP error semantics. | `poetry run pytest tests/api tests/test_plan_api_pick_and_place.py tests/test_error_handling.py` |
| Planning, orchestration & execution | `tests/planner/test_planner.py`, `tests/orchestrator/test_orchestrator.py`, `tests/executor/test_executor.py` | Backward-chaining planner logic, orchestrator state machine, and executor callbacks. | `poetry run pytest tests/planner tests/orchestrator tests/executor` |
| Knowledge graph & storage | `tests/knowledge_graph/`, `tests/storage/test_database.py`, `tests/test_sophia.py`, `tests/test_config.py` | Graph node/edge invariants, Neo4j/Milvus persistence adapters, and configuration defaults. | `poetry run pytest tests/knowledge_graph tests/storage tests/test_sophia.py tests/test_config.py` |
| HCG client & envelopes | `tests/hcg_client/test_client_wrapper.py`, `tests/test_cwmstate_envelope.py` | Low-level client wrapper around Neo4j/Milvus plus the schema used to exchange state with CWM. | `poetry run pytest tests/hcg_client tests/test_cwmstate_envelope.py` |
| CWM connectors | `tests/cwm_a/test_cwm_a.py`, `tests/cwm_g/test_cwm_g.py` | Serializers/parsers that map Sophia data structures to CWM-A/G representations. | `poetry run pytest tests/cwm_a tests/cwm_g` |
| Media ingestion & JEPA perception | `tests/test_media_ingestion.py`, `tests/jepa/test_jepa_runner.py`, `tests/test_jepa_integration.py`, `tests/test_jepa_simulation.py` | Upload pipeline, metadata extraction, JEPA runner behavior, and simulation coupling. | `poetry run pytest tests/test_media_ingestion.py tests/jepa tests/test_jepa_integration.py tests/test_jepa_simulation.py` |
| Prototype end-to-end integration | `tests/integration/test_prototype_integration.py` | Full plan/state API flow backed by Neo4j + Milvus seeded via `run_prototype_integration.sh`. | `./scripts/run_prototype_integration.sh` |
| Data fixtures | `tests/data/test_data_pick_and_place.cypher` | Validates that the Cypher seed used for pick-and-place stays in sync with tests and docs. | `poetry run pytest tests/data` |

## Tips

- Use `pytest -k <keyword>` to filter within any category.
- When editing multiple areas, run `poetry run pytest` from the repo root to
  execute the entire suite.
- The integration suite requires Docker; see `scripts/run_prototype_integration.sh`
  for the exact compose command and environment variables.

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
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `neo4jtest` | Neo4j password |
| `MILVUS_HOST` | `localhost` | Milvus host |
| `MILVUS_PORT` | `19530` | Milvus port |

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
