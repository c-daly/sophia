# Agent Instructions

This guidance applies to the Sophia repository and governs how AI agents interact with the codebase.

## Repository context

### Ecosystem overview
Sophia is one of **five tightly coupled repositories** that compose the LOGOS cognitive architecture:

| Repo | Purpose |
|------|---------||
| **logos** | Foundry—canonical contracts, ontology, SDKs, shared tooling |
| **sophia** (this repo) | Non-linguistic cognitive core (Orchestrator, CWM-A/G/E, Planner, Executor) |
| **hermes** | Stateless language & embedding utility (STT, TTS, NLP, embeddings) |
| **talos** | Hardware abstraction layer for sensors/actuators |
| **apollo** | Thin client UI and command layer |

Sophia is a **core downstream consumer** of logos contracts and a **provider** to Apollo. Changes here can break Apollo and must respect upstream contracts from logos.

### This repository
Sophia provides the non-linguistic cognitive core for LOGOS:
- **Orchestrator** – Coordinates cognitive processes
- **CWM-A / CWM-G / CWM-E** – Causal World Models (Active/Generative/Emotional)
- **Planner** – Goal-directed planning and reasoning
- **Executor** – Plan execution and monitoring
- **JEPA Integration** – Joint-Embedding Predictive Architecture for perception

Key directories:
- `src/sophia/` – Core cognitive components
- `src/sophia/api/` – FastAPI service endpoints (`/plan`, `/imagine`, `/execute`, `/simulate`, `/ingest`)
- `tests/` – Unit and integration tests

### Dependencies
- Consumes contracts from `logos/contracts/sophia.openapi.yaml`
- Uses `logos_hcg` for Hybrid Causal Graph operations
- Integrates with Neo4j (graph) and Milvus (vectors)
- Calls Hermes for language/embedding services

### Key documentation
- `README.md` – Installation, features, API overview
- `CONTRIBUTING.md` – PR process and coding standards
- `SOPHIA_TEST_GUIDE.md` – Testing infrastructure details
- `.github/copilot-instructions.md` – Detailed development guidance

---

## Communication and transparency

### Announce intent before acting
Do not take impactful actions—large refactors, dependency bumps, new features, API changes—without first describing your intent and waiting for acknowledgment. Explain *what* you plan to change and *why*.

### Surface uncertainty early
If a task is ambiguous, ask clarifying questions rather than guessing. When multiple reasonable interpretations exist, list them and ask which to pursue.

### No silent side effects
If your change will affect behavior, logging, error handling, or external APIs, call it out explicitly before proceeding.

---

## Workflow safety

### Never work directly on `main`
Always create a feature branch before making any changes. Branch naming convention:
```
{kind}/{repo}{issue-number}-{short-kebab}
# e.g., feature/sophia1234-planner-retry-logic
# e.g., fix/sophia42-connection-timeout
# e.g., chore/sophia63-test-reorganization
```

The repo prefix (`sophia`) makes branches identifiable across the ecosystem.

### Never push without a pull request
All changes—no matter how small—must go through a PR. Direct pushes to any shared branch are forbidden.

### Cross-repository changes
When a change spans multiple repositories:
1. **Create a tracking issue** in the most relevant repo (usually `logos` for ecosystem-wide changes)
2. **Reference it from each repo's PR** using `Part of c-daly/logos#N`
3. **Use consistent branch names** across repos: `chore/logos420-testing-standards`
4. **Close the tracking issue** only after all PRs merge

See `logos/docs/GIT_PROJECT_STANDARDS.md` for full cross-repo workflow documentation.

### Respect cross-repo dependencies
Before shipping a change that modifies shared contracts, APIs, or data structures:
1. Identify downstream repos (Apollo) that depend on this API.
2. If the change is breaking, **stop** and create a ticket describing the required upstream change in logos first.
3. Coordinate migrations via issues so dependent repos can adapt.

---

## Code quality and professional practices

### Elevate code you touch
When modifying existing code, lift the surrounding area toward current best practices—improved typing, clearer error handling, better logging, more readable structure. Do not blindly copy nearby patterns that look stale or inconsistent.

### Small, composable functions
Prefer small, focused functions over monolithic blocks. Each function should do one thing well. Compose larger behaviors from smaller, testable units.

### Type hints and docstrings
Add or update type hints and docstrings whenever you introduce or modify public functions, classes, or methods. Prefer explicit types over `Any`.

### Cognitive architecture considerations
- Maintain causal coherence in all HCG operations
- Respect the non-linguistic cognition philosophy—language is I/O, not substrate
- Keep cognitive components loosely coupled via well-defined interfaces

### Backward compatibility
Maintain backward compatibility unless the task explicitly calls for a breaking change. If you must break compatibility:
- Call it out clearly in your summary.
- Ensure tests cover the migration path.
- File tickets for Apollo if API changes affect it.

### Defensive coding
- Validate inputs; handle edge cases.
- Avoid silent failures—log or raise when something unexpected occurs.
- If skipping handling is intentional, document why with a comment.

### Purposeful comments
Explain *intent* or *non-obvious decisions*. Do not restate what the code already expresses. Keep comments current when you change logic.

### Security and privacy hygiene
- Never log secrets, tokens, or PII.
- Use parameterized queries for Neo4j to prevent injection.
- Sanitize user inputs; assume external data is hostile.
- When touching auth or data-handling code, review for least-privilege and error hygiene.

---

## Reflection and course correction

### Pause when things aren't working
If you encounter:
- Repeated errors or test failures
- Persistent friction or unexpected behavior
- Uncertainty about the right approach

**Stop.** Do not push forward blindly.

### Reassess and gather context
- Reread relevant files, docs, or specs.
- Search for related patterns in the codebase.
- Check if assumptions you made earlier are still valid.
- Ask for clarification or additional context if needed.

### Adjust your approach
If the same strategy keeps failing, try a different angle. Consider whether:
- The problem is elsewhere (e.g., upstream data, configuration).
- You're missing context from another repo (logos contracts, Hermes API).
- The task needs to be broken into smaller steps.

Document what you tried and why it didn't work so you (or another agent) don't repeat the same mistakes.

---

## Do's and Don'ts

### Definitely Do
- **Create a branch before any changes** – Never work directly on `main`
- **Run tests before pushing** – At minimum: `./scripts/run_tests.sh unit`
- **Ask before large refactors** – Describe intent, wait for acknowledgment
- **Reference issues in commits/PRs** – Use `Closes #N` or `Part of #N`
- **Update tests when changing behavior** – Tests document expectations
- **Check downstream impact** – Apollo depends on Sophia's API
- **Use the standard scripts** – `./scripts/run_tests.sh` handles env setup
- **Read the error message** – Most issues are explained in the output
- **Commit `poetry.lock` with `pyproject.toml`** – Always together

### Definitely Don't
- **Don't push directly to `main`** – All changes require a PR
- **Don't ignore failing tests** – Fix them or explain why they're skipped
- **Don't make unrelated changes in a PR** – Keep PRs focused; file follow-up tickets
- **Don't hardcode ports** – Use environment variables; ports differ per repo
- **Don't commit secrets or tokens** – Ever. Check your diffs.
- **Don't skip the PR description** – Reviewers need context
- **Don't merge without CI passing** – If CI is broken, fix it first
- **Don't guess at cross-repo impacts** – Check contracts in logos
- **Don't leave zombie containers running** – `./scripts/run_tests.sh down`
- **Don't copy-paste code without understanding it** – Especially test fixtures

---

## How to work

### Searching
Prefer `rg` (ripgrep) for fast text searches. Avoid slow recursive `grep` or `find` commands when ripgrep can do the job.

### Dependency management
Use **Poetry** for all Python dependency work.
- Add dependencies: `poetry add <pkg>` or `poetry add --group dev <pkg>`
- Update lock file: `poetry update`
- Always commit **both** `pyproject.toml` and `poetry.lock` together.

### Keep diffs minimal
Stay focused on the task. Avoid drive-by refactors, unrelated formatting changes, or scope creep. If you notice something worth fixing outside the current task, note it and suggest a follow-up ticket instead of bundling it in.

---

## Testing and linting

### Linting and formatting

All Python code must pass ruff and mypy before merge.

**Ruff** (linting + formatting):
```bash
# Check for issues
poetry run ruff check .

# Auto-fix what's possible
poetry run ruff check --fix .

# Format code
poetry run ruff format .

# Check formatting without changing files
poetry run ruff format --check .
```

**Mypy** (type checking):
```bash
poetry run mypy src/
```

**Pre-commit workflow**:
```bash
# Before committing, run:
poetry run ruff check --fix .
poetry run ruff format .
poetry run mypy src/
poetry run pytest tests/unit/
```

**Common issues and fixes**:
- `F401 imported but unused` → Remove the import or add `# noqa: F401` if re-exported
- `E501 line too long` → Ruff format usually fixes this; if not, break the line manually
- `I001 import order` → `ruff check --fix` will reorder imports
- Mypy `missing-imports` → Add type stubs or `# type: ignore[import-untyped]`

### Test infrastructure
Sophia follows the LOGOS ecosystem testing standards (see `logos/docs/TESTING_STANDARDS.md`):

| Test Type | Location | Services Required | Command |
|-----------|----------|-------------------|---------|
| Unit | `tests/unit/` | None | `./scripts/run_tests.sh unit` |
| Integration | `tests/integration/` | Neo4j, Milvus | `./scripts/run_tests.sh integration` |
| E2E | `tests/e2e/` | Neo4j, Milvus, Sophia API | `./scripts/run_tests.sh e2e` |

Port allocation (+40000 offset per ecosystem standard):
- Neo4j: 47474 (HTTP), 47687 (Bolt)
- Milvus: 47530 (gRPC), 47091 (health)
- Sophia API: 48000

### Quick commands
```bash
# Run all tests (starts services automatically)
./scripts/run_tests.sh all

# Run specific test tier
./scripts/run_tests.sh unit
./scripts/run_tests.sh integration
./scripts/run_tests.sh e2e

# Service management
./scripts/run_tests.sh up      # Start infrastructure
./scripts/run_tests.sh down    # Stop infrastructure
./scripts/run_tests.sh status  # Check health
./scripts/run_tests.sh seed    # Seed test data

# Full CI parity
./scripts/run_tests.sh ci
```

### Local CI parity
For full CI parity, run:
```bash
./scripts/run_tests.sh ci
# or
./.github/workflows/run_ci.sh
```
This wraps Ruff, Black, mypy, and pytest with the same arguments as the GitHub Actions workflow.

### Narrower checks
For scoped changes, run the smallest relevant subset:
```bash
poetry run pytest <path>
poetry run ruff check <path>
poetry run mypy src/
```

### Always note what you ran
In your summary, explicitly list which checks you executed. If none were run (e.g., documentation-only change), state that clearly.

---

## Pull request and summary expectations

### Issue format

**Title:** `[sophia] Short imperative description`
```
[sophia] Add retry logic for Neo4j connection failures
[sophia] Fix planner timeout on complex goals
```

**Body:**
```markdown
## Summary
One or two sentences describing the problem or feature.

## Context
Why this matters. Link to related issues, specs, or discussions if relevant.

## Acceptance criteria
- [ ] Testable criterion 1
- [ ] Testable criterion 2
- [ ] Tests pass, no regressions

## Notes (optional)
Implementation hints, open questions, or out-of-scope items.
```

**Labels:** At minimum: `component:sophia`, `type:*`, `priority:*`

---

### Labels, projects, and status

**Required labels for issues:**

| Category | Options | Notes |
|----------|---------|-------|
| **Component** | `component:logos`, `component:sophia`, `component:hermes`, `component:talos`, `component:apollo`, `component:infrastructure` | Which repo/area is affected |
| **Type** | `type:bug`, `type:feature`, `type:documentation`, `type:refactor`, `type:testing`, `type:research` | Nature of work |
| **Priority** | `priority:high`, `priority:medium`, `priority:low` | Urgency (`priority:critical` for blockers) |

**Optional but recommended:**

| Category | Options | Notes |
|----------|---------|-------|
| **Status** | `status:in-progress`, `status:review`, `status:blocked`, `status:on-hold` | Current state |
| **Phase** | `phase:1`, `phase:2` | Project phase scope |
| **Workstream** | `workstream:B` (Sophia) | Which workstream |
| **Domain** | `domain:hcg`, `domain:planner`, `domain:diagnostics` | Technical domain |
| **Capability** | `capability:perception`, `capability:actuation`, `capability:explainability` | System capability |

**Project board:**
- **Every issue and PR must be added to the `Project LOGOS` GitHub Project.** This is required, not optional.
- **Every issue and PR must have a status, and the status must be kept current.** This is also required.
- When creating an issue, immediately add it to `Project LOGOS` and set the appropriate status column.
- When opening a PR, add it to `Project LOGOS` as well.
- Move cards between columns as work progresses; keep `status:*` labels in sync with the column.
- When you start work on an issue, move it to *In Progress* and apply `status:in-progress`.
- When the PR is ready for review, apply `status:review`.
- When the PR merges, move the issue to *Done*.

**Cross-repo issues:**
- If an issue spans multiple repos, apply multiple `component:*` labels.
- Note affected repos explicitly in the issue body.
- Create linked issues in sibling repos when coordination is required.

---

### Pull request format

**Title:** `[sophia] Short imperative description (#issue)`
```
[sophia] Add retry logic for Neo4j connection failures (#427)
```

**Body:**
```markdown
## Summary
Brief description of what this PR does.

Closes #427

## Changes
- Added `RetryPolicy` class to `src/sophia/hcg/connection.py`
- Updated `Neo4jClient` to use exponential backoff
- Added unit tests for retry behavior

## Testing
- `./scripts/run_tests.sh unit` – ✅
- `poetry run ruff check src/sophia/` – ✅

## Notes (optional)
Anything reviewers should know—tradeoffs, follow-up work, etc.
```

**For cross-repo changes**, add a section:
```markdown
## Cross-Repository Change
This PR is part of a multi-repo change tracked in c-daly/logos#420.

### Related PRs
- c-daly/logos#421 - Standards doc (merged)
- c-daly/sophia#64 - This PR

Part of c-daly/logos#420
```

---

### Concise bullet summaries
- Highlight key changes, scoped to the packages you touched.
- Note any behavioral changes, deprecations, or migration steps.
- If the change affects sibling repos (especially Apollo), call that out.

### Testing section
Include a bullet list of tests/checks executed using the exact commands you ran:
```
- `poetry run pytest tests/integration/` – ✅ passed
- `poetry run ruff check src/sophia/` – ✅ no issues
```
Or, for documentation-only work:
```
- ⚠️ Not run (documentation-only change)
```

### Link related issues
Use `Closes #<issue-number>` or `Refs #<issue-number>` to connect PRs to their tracking issues.

---

## GitHub MCP integration

### Using GitHub tools
This workspace has access to GitHub via the MCP (Model Context Protocol) server. Use the `mcp_io_github_git_*` tools to:
- Search issues and pull requests
- Create branches, commits, and PRs
- Read file contents from remote repos
- Manage labels and reviews

### Authentication troubleshooting
If GitHub MCP tools fail with authentication errors:
1. Run `~/mcp` in the terminal to refresh the `GITHUB_MCP_PAT` environment variable.
2. Retry the operation.

The script populates `GITHUB_MCP_PAT` with a fresh token. You may need to restart the MCP server or your session after running it.

### Best practices for GitHub operations
- Use `get_me` first to verify authentication and understand the current user context.
- Prefer `search_*` tools for targeted queries; use `list_*` for broad enumeration.
- When creating PRs, search for PR templates in `.github/PULL_REQUEST_TEMPLATE.md` first.
- Always link PRs to issues with `Closes #<number>` in the description.

---

## Troubleshooting

### Port conflicts
If services fail to start with "port already in use":
```bash
# Check what's using the port
lsof -i :47474

# Stop any existing test containers
./scripts/run_tests.sh down
docker ps | grep sophia-test | awk '{print $1}' | xargs docker stop
```

### Neo4j won't start
```bash
# Check container logs
docker logs sophia-test-neo4j

# Common fix: clear volumes and restart
./scripts/run_tests.sh clean
./scripts/run_tests.sh up
```

### Milvus health check failing
Milvus can take 60-90 seconds to become healthy. If it times out:
```bash
# Check etcd and minio are healthy first
docker logs sophia-test-milvus-etcd
docker logs sophia-test-milvus-minio

# Then check milvus
docker logs sophia-test-milvus
```

### Tests can't connect to services
Ensure environment variables are set:
```bash
export NEO4J_URI=bolt://localhost:47687
export MILVUS_HOST=localhost
export MILVUS_PORT=47530
```
Or just use `./scripts/run_tests.sh` which sets these automatically.

### GitHub MCP authentication errors
```bash
# Refresh the GitHub token
~/mcp
# Then retry the operation
```

### Import errors after pulling
```bash
poetry install
```

### Coverage not collecting in e2e tests
E2E tests run sophia in a subprocess. Coverage should still work, but if it doesn't:
```bash
# Run with explicit coverage
poetry run coverage run -m pytest tests/e2e/
poetry run coverage report
```

---

## GitHub Access

**You have full access to GitHub.** Do not claim otherwise. If one method doesn't work, try another.

| Method | Best For | Example |
|--------|----------|---------|
| **MCP tools** | Most operations | `mcp_github_list_issues`, `mcp_github_create_pull_request` |
| **GitHub CLI** | Complex queries, projects | `gh issue list`, `gh project item-add` |
| **GraphQL API** | Projects, advanced queries | `gh api graphql -f query='...'` |
| **REST API** | Simple operations | `gh api repos/c-daly/sophia/issues` |

### Quick examples
```bash
# Add issue to project 10
gh project item-add 10 --owner c-daly --url https://github.com/c-daly/sophia/issues/63

# Add labels
gh issue edit 63 --repo c-daly/sophia --add-label "testing,phase:3"

# Search across repos
gh search issues "testing" --owner c-daly --state open
```

For complete documentation, see `logos/AGENTS.md` (GitHub Access section).

---

## Quick reference

| Task | Command / Location |
|------|-------------------|
| Install deps | `poetry install` |
| Run all tests | `./scripts/run_tests.sh all` |
| Run unit tests | `./scripts/run_tests.sh unit` |
| Run integration tests | `./scripts/run_tests.sh integration` |
| Run e2e tests | `./scripts/run_tests.sh e2e` |
| Full CI locally | `./scripts/run_tests.sh ci` |
| Start test infrastructure | `./scripts/run_tests.sh up` |
| Stop test infrastructure | `./scripts/run_tests.sh down` |
| Check service health | `./scripts/run_tests.sh status` |
| Seed test data | `./scripts/run_tests.sh seed` |
| Refresh GitHub token | `~/mcp` |

### Ecosystem standards
| Document | Location |
|----------|----------|
| Testing standards | `logos/docs/TESTING_STANDARDS.md` |
| Git/project standards | `logos/docs/GIT_PROJECT_STANDARDS.md` |
