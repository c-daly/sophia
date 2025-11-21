# Copilot Instructions — Sophia

Focused guidance for AI coding agents working in `sophia/` (the cognitive core).

Big picture
- Sophia is the non-linguistic cognitive core responsible for planning, execution, and direct HCG (Neo4j + Milvus) updates. This repo contains the planner, executor, and CWM adapters.

Key files & locations
- `pyproject.toml` — package metadata, dev deps, and scripts
- `docs/` — architecture and design notes
- `tests/` — unit and integration tests (some tests may require Neo4j + Milvus)
- `planner_stub/` or `planner/` — planner-related code (if present)

Developer workflows
- Install dev deps: `pip install -e ".[dev]"` (run from `sophia/`).
- Start required infra locally via `logos/infra/docker-compose.hcg.dev.yml` (Neo4j + Milvus) when running integration tests that interact with the HCG.
- Opt-in integration tests: use `RUN_NEO4J_SHACL=1` or see `logos/` docs for environment variables required to run Neo4j-backed SHACL tests.
- Run unit tests: `pytest`; run slow/integration tests separately (markings in test suite indicate `slow` or `integration`).

Safety & patterns
- Sophia writes to the HCG. Exercise caution: prefer API-driven flows and use test/staging clusters when possible. Avoid performing destructive schema changes in automated tests.
- Use parameterized Cypher queries and follow ontology constraints in `logos/ontology/` (UUID uniqueness, relationship patterns `:CAUSES`, `:HAS_STATE`, etc.).

GitHub, tickets & PRs
- Follow the workspace-wide rules in `logos/.github/copilot-instructions.md` for issue titles, labels, branch naming, and PR requirements. PRs that change ontology, SHACL, or infra must include references to `logos/` artifacts and update validation tests.
- When you start work on an issue that lives on the LOGOS workspace project, move its card to *In Progress* (and adjust any `status/*` label). Once the change lands, move it to *Done* so the shared board remains accurate.

Examples
- Add a planner endpoint: add handler in `src/`, add `tests/` covering planner logic, document API in `docs/`, and add the PR checklist linking to the issue that requested the feature.

If you want, I can extract exact service start commands or env var names from `pyproject.toml` or `Dockerfile` and update this file with precise commands.
