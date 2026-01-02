# Research: State Architecture in LOGOS

**Date:** 2026-01-01
**Depth:** Deep architectural analysis
**Status:** Resolved - see design doc

## Summary

**Core insight: There is one graph (HCG). CWM is content within it.**

- CWM (Causal World Model) nodes are stored in the HCG alongside everything else
- CWMState is a **transport/event envelope** for HCG change notifications (name is historical)
- Provenance lives on HCG nodes, not the transport envelope
- No Redis tier - mid-term vs long-term is just `expires_at` field

---

## Key Findings

### 1. Two Systems, One Graph

| System | What It Stores | Authority |
|--------|----------------|-----------|
| **Talos** | World State (physical reality) | Ground truth from sensors |
| **HCG** | Everything Sophia knows | CWM nodes, processes, types, plans |

### 2. Memory Tiers (Simplified from PHASE3_SPEC)

```
┌─────────────────────────────────────────┐
│     LONG-TERM MEMORY                    │
│     (HCG - Neo4j, no expires_at)        │
│     Lifetime: indefinite                │
└─────────────────────────────────────────┘
              ▲ TTL removed
              │
┌─────────────────────────────────────────┐
│     MID-TERM MEMORY                     │
│     (HCG - Neo4j + expires_at)          │
│     Lifetime: configurable TTL          │
└─────────────────────────────────────────┘
              ▲ Promotion (salience score)
              │
┌─────────────────────────────────────────┐
│     EPHEMERAL MEMORY                    │
│     (In-memory, not in HCG yet)         │
│     Lifetime: session duration          │
└─────────────────────────────────────────┘
```

**Key change from PHASE3_SPEC:** No Redis tier. Mid-term vs long-term is just presence/absence of `expires_at` field on HCG nodes.

### 3. CWM: Node Categories Within HCG

CWM = Causal World Model. These are **node types within the HCG**, not separate systems:

| Category | What It Represents | Node Type |
|----------|-------------------|-----------|
| **CWM-G** (Grounded) | Physical beliefs, predictions | `cwm_grounded` |
| **CWM-A** (Abstract) | Conceptual knowledge, goals | `cwm_abstract` |
| **CWM-E** (Affective) | Emotional/reflective states | `cwm_affective` |

Cross-category edges allowed (G observation → A goal it supports).

### 4. CWMState: Transport Envelope

`CWMState` is an **event format** for notifying about HCG changes:

```python
CWMState:
  state_id: str          # "cwm_<model>_<uuid>"
  model_type: str        # "CWM_A" | "CWM_G" | "CWM_E"
  source: str            # "sophia_api", "jepa_runner", etc.
  timestamp: datetime
  confidence: float      # 0.0-1.0
  status: str            # "observed" | "imagined" | "reflected" | "ephemeral"
  links: CWMStateLinks   # process_ids, plan_id, entity_ids, etc.
  tags: list[str]
  data: CWMStateData     # OneOf: CWMAGraphData | CWMGImaginedData | CWMESentimentData
```

**Important:** CWMState is transport, not storage. Provenance lives on the HCG nodes it references.

### 5. HCG `State` Model (Planning)

The `logos_hcg/models.py` `State` class is for **planning preconditions/effects**:

```python
class State(BaseModel):
    """Temporal snapshot of entity properties."""
    uuid: UUID
    timestamp: datetime | None
    position_x, position_y, position_z: float | None
    is_grasped, is_closed, is_empty: bool | None
    # ... etc
```

This is distinct from CWM nodes - it's for symbolic planning state.

---

## Architecture Diagram (Corrected)

```
Physical World
      │
      ▼ (sensors)
┌─────────────┐
│    Talos    │  ← Ground truth (separate system)
│ World State │
└─────────────┘
      │
      ▼ (observation)
┌─────────────────────────────────────────────────────┐
│                        HCG                          │
│                  (Sophia's Graph)                   │
│                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │  CWM-G   │◄─►│  CWM-A   │◄─►│  CWM-E   │        │
│  │ Grounded │   │ Abstract │   │Affective │        │
│  └──────────┘   └──────────┘   └──────────┘        │
│                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │ Process  │   │   Type   │   │   Plan   │        │
│  │  Nodes   │   │   Defs   │   │  Nodes   │        │
│  └──────────┘   └──────────┘   └──────────┘        │
│                                                     │
│  All nodes can have: expires_at, provenance        │
└─────────────────────────────────────────────────────┘
      │
      ▼ (events)
┌─────────────┐
│  CWMState   │  ← Transport envelope (not storage)
│  (events)   │     Name is historical
└─────────────┘
      │
      ▼
   Apollo (display)
```

---

## Resolved Questions

### Q1: What is "state" canonically?

**Resolved:** Two locations:
1. **Talos** - World state (physical reality, ground truth)
2. **HCG** - Sophia's knowledge (CWM nodes, plans, processes, types)

### Q2: How does `/state` endpoint relate to CWM?

**Resolved:** `/state` should create **CWM-G nodes** (grounded beliefs), not CWM-A. Physical observations are grounded, not abstract.

### Q3: What about ephemeral vs persisted?

**Resolved:**
- Ephemeral = in-memory, not in HCG yet
- Mid-term = HCG nodes with `expires_at`
- Long-term = HCG nodes without `expires_at`
- **No Redis tier** - just the `expires_at` field

### Q4: Where does provenance go?

**Resolved:** On HCG nodes. CWMState is transport - it references node IDs, doesn't duplicate provenance.

---

## Updated Recommendations

### For Issue #15 (Provenance)

1. ~~Add full Provenance to CWMState envelope~~ **No - provenance on HCG nodes**
2. Ensure Plan/Goal/PlanStep nodes populate `Provenance` (already have the field)
3. `/plan`, `/execute` endpoints populate provenance from request context
4. Scope to **Plan nodes only** - CWM nodes can get provenance later

### For Issue #101 (Session Boundaries)

1. Define "session" = correlation_id scope or process lifetime
2. Ephemeral data = in-memory CWM not yet in HCG
3. Session end triggers promotion evaluation (salience score)
4. **No Redis cleanup** - there is no Redis tier

### For Naming Clarity

1. Keep `CWMState` name (too much churn to change), document it's transport
2. Rename HCG `State` → `PlanState` or `TemporalState` (optional, low priority)
3. `/state` endpoint should create CWM-G nodes, consider renaming to `/observe`

---

## Relevant Files

### Sophia
| File | Purpose |
|------|---------|
| `cwm_a/memory.py` | Associative key-value memory (in-memory buffer) |
| `cwm_a/state_service.py` | CWMState emission, entity diffs |
| `cwm_g/memory.py` | Generative buffer for predictions |
| `cwm/persistence.py` | Neo4j persistence for CWM nodes |
| `api/app.py` | `/state`, `/cwm` endpoints |

### Logos
| File | Purpose |
|------|---------|
| `logos_hcg/models.py` | HCG models including State (for planning) |
| `sdk/.../cwm_state.py` | CWMState envelope definition |
| `docs/hcg/CWM_STATE.md` | CWM state specification (needs update) |
| `docs/architecture/PHASE3_SPEC.md` | Memory hierarchy spec (needs update re: Redis) |

---

## Next Steps

1. ~~Present findings to user for architectural decision~~ **Done**
2. ~~Create design doc with clear state definitions~~ **Done: `logos/docs/plans/2026-01-01-state-architecture-design.md`**
3. Update PHASE3_SPEC to remove Redis tier mention
4. Update CWM_STATE.md to clarify it's transport, provenance on HCG nodes
5. Scope #15 intake to Plan provenance only
