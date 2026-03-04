# Sophia

[![CI](https://github.com/c-daly/sophia/actions/workflows/ci.yml/badge.svg)](https://github.com/c-daly/sophia/actions/workflows/ci.yml)

**Non-linguistic cognitive core for [Project LOGOS](https://github.com/c-daly/logos)**

Sophia handles cognition without language: planning, execution, world models, and perception (JEPA). Hermes handles all linguistic I/O.

## Quick Start

```bash
# Install
poetry install

# Run tests
./scripts/test.sh

# Start dev server
./scripts/dev.sh

# Lint
./scripts/lint.sh
```

## API

Interactive docs at `http://localhost:47000/docs` when running.

See interactive docs for the full endpoint list.

## Architecture

Sophia provides:
- **Planner** - Backward-chaining goal decomposition
- **Executor** - Plan execution with state tracking
- **CWM** - Cognitive World Model (goals, actions, effects)
- **JEPA** - Visual perception via embeddings (not language)

Sophia is non-linguistic. For any text I/O, she relies on Hermes.

## Configuration

Uses `logos_config` for ports/settings. Sophia API runs on port 47000; infrastructure services are shared across all LOGOS repos.

| Service | Port |
|---------|------|
| API | 47000 |
| Neo4j HTTP | 7474 |
| Neo4j Bolt | 7687 |
| Milvus | 19530 |
| Redis | 6379 |

## Development

See [AGENTS.md](AGENTS.md) for development guidelines.

```bash
# Lint
./scripts/lint.sh

# Type check
poetry run mypy src/

# All tests
./scripts/run_tests.sh all

# Unit / integration / e2e
./scripts/test_unit.sh
./scripts/test_integration.sh
./scripts/test_e2e.sh
```

## License

MIT
