# Provenance Design Decisions (Issue #15)

## Core Principles
1. **Provenance lives on nodes, not envelope** - CWMState is thin transport wrapper
2. **Envelope is cheap** - wraps node verbatim, no computation required
3. **Flexible extensions via tags/links** - avoid rigid schemas for evolving needs
4. **Source is granular** - module/job level (e.g., `jepa_runner`, `planner`), not service level (e.g., `sophia`)

## Base Node Schema (7 new fields)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `source` | string | `"unknown"` | Module/job that created it |
| `derivation` | string | `"observed"` | How derived: `observed`, `imagined`, `reflected` |
| `confidence` | float | `null` | 0.0-1.0 certainty (optional) |
| `created` | ISO8601 | now | Creation timestamp |
| `updated` | ISO8601 | now | Last modified timestamp |
| `tags` | list | `[]` | Free-form labels |
| `links` | dict | `{}` | Related entity IDs |

## CWMState Envelope (simplified)

| Field | Description |
|-------|-------------|
| `state_id` | Node UUID |
| `model_type` | CWM_A, CWM_G, CWM_E |
| `timestamp` | Response time |
| `data` | Verbatim node properties |

Removed from envelope: `source`, `confidence`, `status`, `links`, `tags` (all now on node)

## Key Implementation Notes

### add_node() signature
```python
def add_node(
    self,
    name: str,
    node_type: str,
    uuid: Optional[str] = None,
    ancestors: Optional[List[str]] = None,
    is_type_definition: bool = False,
    properties: Optional[Dict[str, Any]] = None,
    *,
    source: str = "unknown",
    derivation: str = "observed",
    confidence: Optional[float] = None,
    tags: Optional[List[str]] = None,
    links: Optional[Dict[str, Any]] = None,
) -> str:
```

### Source values (examples)
- `jepa_runner` - JEPA video processing
- `planner` - Planning module
- `ingestion` - Hermes ingestion
- `orchestrator` - Sophia orchestration
- `reflection_job` - Reflection/introspection
- `human` - Manual entry
- `bootstrap` - Ontology/system init

### Derivation values
- `observed` - Real sensor/perception data
- `imagined` - Predicted/simulated
- `reflected` - Introspection/reasoning output

### Links structure (examples)
```python
links = {
    "process_ids": ["sim_abc"],
    "plan_id": "plan_123",
    "media_sample_id": "vid_xyz",
    "talos_run_id": "run_456",
    "persona_entry_id": "entry_789"
}
```

## Design Rationale

### Why `derivation` not `status`?
`status` is too generic. `derivation` explicitly describes how the knowledge was derived.

### Why `source` is granular?
Service-level (`sophia`, `hermes`) doesn't tell you enough for debugging. Module-level (`jepa_runner`, `planner`) lets you trace back to specific code.

### Why `links` as dict not edges?
Graph edges are the right long-term solution, but node properties are simpler to query and don't require graph traversal. Can migrate to edges later.

### Why `confidence` is separate from `tags`?
Confidence is used frequently enough for filtering/sorting that it deserves a dedicated field rather than parsing from tags.
