# Memory & Planning Model

Design notes from 2026-01-02 session. To be incorporated into wiki.

## Two-Layer Architecture

### Layer 1: HCG Nodes (Persistent Storage)

Graph nodes in Neo4j representing persistent knowledge:

- **Entity** (`entity-*`) - concrete objects
- **Concept** (`concept-*`) - abstract categories
- **State** (`state-*`) - temporal snapshots
- **Process** (`process-*`) - actions/transformations
- **Capability** (`capability-*`) - tools for planning
- **CWM-A nodes** - Fact, Association, Abstraction, Rule
- **CWM-G nodes** - PerceptionFrame, ImaginedProcess, ImaginedState
- **CWM-E nodes** - EmotionState, PersonaEntry, Preference

### Layer 2: CWMState Envelope (Transport)

Notification/transport envelope for state changes:

```python
{
    "state_id": "cwm_<model>_<uuid>",
    "model_type": "CWM_A | CWM_G | CWM_E",
    "source": "<subsystem>",           # talos, jepa_runner, orchestrator, etc.
    "timestamp": "<ISO-8601>",
    "confidence": 0.0-1.0,
    "status": "observed | imagined | reflected",
    "links": {
        "process_ids": [],
        "plan_id": "",
        "entity_ids": [],
        "media_sample_id": "",
        "persona_entry_id": "",
        "talos_run_id": ""
    },
    "tags": [],
    "data": {}  # model-specific payload
}
```

**Model payloads:**
- CWM-A: `{entities, relations, violations, validation}`
- CWM-G: `{imagined, horizon_steps, frames, embeddings, assumptions}`
- CWM-E: `{sentiment, confidence_delta, caution_delta, narrative}`

## Memory Tiers

### Ephemeral Memory (Working Memory)

- In-memory graph
- Current perceptions + relevant HCG knowledge merged
- Planning operates here
- Most observations stay here and evaporate

### Persistent Memory (HCG)

- Neo4j + Milvus
- Significant knowledge that persists
- Plan templates, learned rules, facts, preferences

### Unified Working Graph

Planner sees one graph - doesn't distinguish source:

```
Perception ──────────────────►┐
                              │
HCG query (relevant facts) ──►├──► Ephemeral (unified) ──► Planner
                              │
```

## Promotion Criteria

**Promote to HCG when:**
- Observation was used in a plan
- Observation contradicts existing knowledge
- Observation is novel (no matching concept)
- Observation relates to active goal
- User explicitly referenced it
- Successful strategy worth reusing
- Learned preference or constraint

**Stay ephemeral when:**
- Just background context
- Redundant with existing knowledge
- Not relevant to anything active
- Mundane plan execution (nothing learned)

## Planning Model

### Plans with Prerequisites

```python
{
    "plan": "manipulate_large_object",
    "prereqs": [
        {"condition": "distance_to_object < 0.5m"},
        {"condition": "gripper_available"},
        {"condition": "clear_approach_path"}
    ],
    "steps": ["grip", "lift", "move"]
}
```

### Recursive Goal Satisfaction

1. Check prereqs
2. If prereq not satisfied → find subplan to achieve it
3. Recurse until all prereqs met by current state
4. Execute bottom-up

Example:
```
Goal: manipulate large object
  └── prereq: within 0.5m → not satisfied
        └── find plan → navigation plan
              └── prereq: path exists → satisfied
                    └── execute navigation
                          └── now within 0.5m
                                └── continue manipulation
```

### Late Binding

Abstract plans defer decisions to subplans:

- Stored plan says *what* (manipulate large object)
- Subplans determine *how* (approach from left)
- Details resolved at execution time based on current context

## Abstraction via Tagging

### Tag Successful Plans

When plan succeeds, tag with abstract features (not specific instances):

**Too specific (doesn't transfer):**
```
["object:chair", "approach:left", "grip:wide"]
```

**Better (transfers):**
```
["activity:manipulation", "object_size:large", "object_material:rigid"]
```

### Query by Tag Similarity

Later, facing new situation:
```cypher
MATCH (p:Plan)
WHERE p.outcome = 'success'
  AND 'activity:manipulation' IN p.tags
  AND 'object_size:large' IN p.tags
RETURN p
ORDER BY tag_overlap DESC
```

Retrieved plan is a candidate, not a guarantee. May need adaptation.

## Learning Loop

```
Execute plan ──► Success? ──► Tag with abstract features ──► Store in HCG
                                                                   │
                                                                   ▼
                                                          Future plans can query
```

**What gets promoted:**

| Type | Example | HCG Target |
|------|---------|------------|
| Novel entity | First chair seen | Entity + Concept |
| Successful strategy | Approach from open side | Plan with tags |
| Learned preference | Slow approach for glass | Preference (CWM-E) |
| Failure lesson | Grip doesn't work on smooth | Rule (constraint) |

## Type System

### Dynamic Types

- Sophia creates new types as needed
- Types stored in HCG with full flattened JSON schema
- Inheritance via `parent_ref` edge (schema merged at creation time)
- Graph is the index - no redundant child lists

### Type Creation Flow

1. Sophia encounters something that doesn't fit existing types
2. Identifies distinguishing properties
3. Asks Hermes to name the type and fields
4. Creates type node with full schema (inherited + new fields)
5. Creates instance nodes of that type

## Open Questions

- Exact salience scoring mechanism (what triggers promotion?)
- Session boundaries (what defines a session?)
- How does Sophia decide "this strategy is generalizable"?
- Tag vocabulary - predefined or emergent?

## Related Documents

- `logos/docs/hcg/CWM_STATE.md` - CWM overview
- `logos/docs/hcg/INGESTION.md` - How data enters CWM
- `logos/docs/hcg/HCG_DATA_LAYER.md` - Core ontology
- Phase 2 spec - CWMState contract details
