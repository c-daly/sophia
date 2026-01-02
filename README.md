# Sophia

[![CI](https://github.com/c-daly/sophia/actions/workflows/ci.yml/badge.svg)](https://github.com/c-daly/sophia/actions/workflows/ci.yml)

**Non-linguistic cognitive core for [Project LOGOS](https://github.com/c-daly/logos)**

Sophia handles cognition without language: planning, execution, world models, and perception (JEPA). Hermes handles all linguistic I/O.

## Quick Start

```bash
# Install
poetry install

# Run tests
./scripts/run_tests.sh unit

# Start services
./scripts/run_tests.sh up

# Run API
poetry run uvicorn sophia.api.app:app --host 0.0.0.0 --port 47000
```

## API

Interactive docs at `http://localhost:47000/docs` when running.

Key endpoints:
- `POST /plan` - Generate plan from goal
- `POST /execute` - Execute a plan
- `GET /cwm` - Query cognitive world model
- `GET /health` - Health check

## Architecture

Sophia provides:
- **Planner** - Backward-chaining goal decomposition
- **Executor** - Plan execution with state tracking
- **CWM** - Cognitive World Model (goals, actions, effects)
- **JEPA** - Visual perception via embeddings (not language)

Sophia is non-linguistic. For any text I/O, she relies on Hermes.

## Configuration

Uses `logos_config` for ports/settings. Sophia port range: 47xxx.

| Service | Port |
|---------|------|
| API | 47000 |
| Neo4j HTTP | 47474 |
| Neo4j Bolt | 47687 |
| Milvus | 47530 |

## Development

See [AGENTS.md](AGENTS.md) for development guidelines.

```bash
# Lint
poetry run ruff check --fix . && poetry run ruff format .

# Type check
poetry run mypy src/

# All tests
./scripts/run_tests.sh all
```

## License

MIT
