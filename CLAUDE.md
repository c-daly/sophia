# CLAUDE.md — sophia

## What This Is

Sophia is the non-linguistic cognitive core for LOGOS. It provides planning,
simulation, world-state modelling (CWM), and plan execution. Sophia consumes
contracts from `logos` (Foundry) and exposes a FastAPI service consumed by
Apollo and other LOGOS components.

## Dependencies

| Service | Default Port | Notes |
|---------|-------------|-------|
| Neo4j | 7474 (HTTP), 7687 (Bolt) | Graph storage, shared instance |
| Milvus | 19530 (gRPC) | Vector storage, shared instance |
| Redis | 6379 | Caching/state, shared instance |

Infrastructure runs on **default ports** (shared across all LOGOS repos).
The Sophia API itself listens on port **47000**.

Python: `>=3.12`. Package manager: **Poetry**.
Key upstream dependency: `logos-foundry` (provides `logos_config`, `logos_hcg`,
`logos_test_utils`, `logos_tools`).

Optional: `poetry install --with ml` for PyTorch/JEPA features.

## Key Commands

```bash
# Install
poetry install                         # core deps
poetry install --with ml               # + PyTorch/JEPA

# Test
./scripts/run_tests.sh unit            # unit tests (no services)
./scripts/run_tests.sh integration     # needs Neo4j + Milvus
./scripts/run_tests.sh e2e             # full stack
./scripts/run_tests.sh all             # everything
./scripts/run_tests.sh ci              # CI parity (lint + type + test)

# Test infra
./scripts/run_tests.sh up              # start containers
./scripts/run_tests.sh down            # stop containers
./scripts/run_tests.sh status          # health check
./scripts/run_tests.sh seed            # seed test data

# Lint & format
poetry run ruff check --fix .          # lint (auto-fix)
poetry run ruff format .               # ruff formatter
poetry run black .                     # black formatter
poetry run mypy src/                   # type check

# Run server
uvicorn sophia.api.app:create_app --factory --port 47000
```

## Architecture

```
src/sophia/
  api/              FastAPI app, auth, request/response models
  config/           Pydantic Settings (db_url, data_dir, log_level)
  cwm/              Causal World Model — unified state engine
  cwm_a/            CWM-Active — current world state
  cwm_g/            CWM-Generative — imagined/predicted states
  executor/         Plan execution and monitoring
  experiments/      Experiment tracking and evaluation
  feedback/         Feedback loop system
  hcg_client/       Hybrid Causal Graph client (wraps logos_hcg)
  ingestion/        Media and proposal ingestion pipelines
  jepa/             Joint-Embedding Predictive Architecture (perception)
  knowledge_graph/  KG construction and query
  maintenance/      Background maintenance tasks
  models/           Shared Pydantic domain models
  orchestrator/     Coordinates cognitive processes
  planner/          Goal-directed planning and reasoning
  storage/          Persistence layer
  env.py            Environment helpers (Neo4j/Milvus config resolution)
```

## Endpoints (28 total on :47000)

| Prefix | Count | Methods | Description |
|--------|-------|---------|-------------|
| `/health` | 1 | GET | Service + dependency health |
| `/state` | 2 | GET, POST | World state read/update |
| `/state/cwm`, `/cwm` | 2 | GET | CWM state views (current, persisted) |
| `/plan` | 1 | POST | Generate action plans |
| `/imagine` | 1 | POST | CWM-G imagined states |
| `/simulate` | 1 | POST | Run simulations |
| `/execute` | 1 | POST | Execute a plan |
| `/ingest/*` | 2 | POST | Hermes proposal + media ingestion |
| `/hcg/*` | 7 | GET | Snapshot, entities, edges, states, processes, plans, history, health |
| `/persona/*` | 5 | CRUD | Persona entries + sentiment |
| `/media/*` | 2 | GET | List/get media samples |

Auth: Bearer token via `SOPHIA_API_TOKEN` env var. Some endpoints (e.g.
`/ingest/hermes_proposal`) are unauthenticated.

## Conventions & Gotchas

- **E402 suppressed in app.py** — `load_dotenv()` must run before
  pydantic-settings imports; ruff is configured to allow this.
- **`asyncio_mode = "auto"`** — all async tests run automatically; no need for
  `@pytest.mark.asyncio`.
- **Test markers**: `unit`, `integration`, `e2e`, `slow`, `gpu`,
  `requires_torch`, `requires_weights`. Use `-m "not integration"` to skip.
- **logos_sophia_sdk** is excluded from ruff/black/mypy (auto-generated).
- **mypy `ignore_missing_imports = true`** globally; tests exempt from
  `disallow_untyped_defs`.
- **Test containers** map Milvus to 47530 and Redis to 46379 to avoid
  conflicts; Neo4j stays on default 7474/7687. The `run_tests.sh` script sets
  env vars automatically.
- **Port 47000** is Sophia's API port. Do not hardcode infrastructure ports
  with 47xxx offsets — infrastructure is shared on default ports.
- Commit `poetry.lock` alongside `pyproject.toml` — always together.
- Cross-repo contract changes flow: `logos -> sophia -> apollo`. Check
  downstream impact before modifying APIs.

## Env Vars

| Variable | Default | Purpose |
|----------|---------|---------|
| `SOPHIA_API_TOKEN` | — | Bearer auth token |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection |
| `NEO4J_USER` / `NEO4J_PASSWORD` | `neo4j` / `logosdev` | Neo4j auth |
| `MILVUS_HOST` / `MILVUS_PORT` | `localhost` / `19530` | Milvus connection |
| `REDIS_HOST` / `REDIS_PORT` | `localhost` / `6379` | Redis connection |
| `MEDIA_STORAGE_ROOT` | `./media_storage` | Media file storage path |

## Docs

- `README.md` — installation, features, API overview
- `docs/JEPA_SIMULATION.md` — JEPA perception pipeline
- `docs/MEDIA_INGESTION.md` — media ingestion pipeline
- `SOPHIA_TEST_GUIDE.md` — testing infrastructure details
- `logos/docs/TESTING_STANDARDS.md` — ecosystem testing standards
- `logos/docs/GIT_PROJECT_STANDARDS.md` — git/project workflow

## Issue Templates

| Template | Use For |
|----------|---------|
| `sophia-task.yml` | Sophia-specific tasks |
| `infrastructure-task.yml` | HCG, ontology, CI/CD |
| `research-task.yml` | Research/investigation |
| `documentation-task.yml` | Docs updates |
