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
