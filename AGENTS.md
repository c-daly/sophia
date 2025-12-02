# Agent Instructions

This guidance applies to the Sophia repository and governs how AI agents interact with the codebase.

## Repository context

### Ecosystem overview
Sophia is one of **five tightly coupled repositories** that compose the LOGOS cognitive architecture:

| Repo | Purpose |
|------|---------|
| **logos** | Foundry—canonical contracts, ontology, SDKs, shared tooling |
| **sophia** (this repo) | Non-linguistic cognitive core (Orchestrator, CWM-A/G, Planner, Executor) |
| **hermes** | Stateless language & embedding utility (STT, TTS, NLP, embeddings) |
| **talos** | Hardware abstraction layer for sensors/actuators |
| **apollo** | Thin client UI and command layer |

Sophia is a **core downstream consumer** of logos contracts and a **provider** to Apollo. Changes here can break Apollo and must respect upstream contracts from logos.

### This repository
Sophia provides the non-linguistic cognitive core for LOGOS:
- **Orchestrator** – Coordinates cognitive processes
- **CWM-A / CWM-G** – Causal World Models (Active/Generative)
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
{kind}/{issue-number}-{short-kebab}
# e.g., feature/1234-planner-retry-logic
```

### Never push without a pull request
All changes—no matter how small—must go through a PR. Direct pushes to any shared branch are forbidden.

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

### Local CI parity
For full CI parity, run:
```bash
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

### Integration tests
Sophia has integration tests requiring Neo4j and Milvus:
```bash
# Start the test stack
./tests/e2e/run_e2e.sh up

# Run integration tests
poetry run pytest tests/integration/

# Tear down
./tests/e2e/run_e2e.sh down
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
- `poetry run pytest tests/unit/test_connection.py` – ✅
- `poetry run ruff check src/sophia/` – ✅

## Notes (optional)
Anything reviewers should know—tradeoffs, follow-up work, etc.
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

## Quick reference

| Task | Command / Location |
|------|-------------------|
| Install deps | `poetry install` |
| Run all tests | `poetry run pytest` |
| Full CI locally | `./.github/workflows/run_ci.sh` |
| Start test stack | `./tests/e2e/run_e2e.sh up` |
| Stop test stack | `./tests/e2e/run_e2e.sh down` |
| Run integration tests | `poetry run pytest tests/integration/` |
| Refresh GitHub token | `~/mcp` |
