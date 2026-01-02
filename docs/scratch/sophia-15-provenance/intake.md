# Intake Summary - State Architecture Design

**Task:** Define state architecture for LOGOS before implementing provenance (#15)
**Date:** 2026-01-01
**Classification:** Complex (Design/Architecture)

## What
Design a coherent "state" model for LOGOS that clarifies the relationship between CWM states, entity states, world states, and plan precondition/effect states. This provides the foundation for issue #15 (provenance) and #101 (session boundaries).

## Why
- "State" is overloaded with 5+ meanings in the codebase
- Issue #15 requires provenance on "state nodes" but it's unclear which
- Issue #101 requires session boundaries for "ephemeral state" but definition is fuzzy
- Cannot implement either cleanly without architectural clarity

## Current State Concepts

| Concept | Location | Purpose | Persistence |
|---------|----------|---------|-------------|
| **CWM-A State** | `sophia/cwm_a/` | Attention/working memory | In-memory |
| **CWM-G State** | `sophia/cwm_g/` | Ground truth state | In-memory |
| **CWMState envelope** | `logos_cwm_e/` | State wrapper with source/status/confidence | Neo4j |
| **`/state` endpoint** | `sophia/api/app.py` | World state CRUD | Neo4j |
| **`State` model** | `logos_hcg/models.py` | Entity property snapshot | Neo4j |
| **PlanStep states** | `logos_hcg/models.py` | Preconditions/effects | Neo4j |

## Questions to Answer

1. **What is the canonical definition of "state" in LOGOS?**
   - Is it a property of an entity at a point in time?
   - Is it a snapshot of the entire world?
   - Is it both, and if so, how do they relate?

2. **How do CWM-A, CWM-G, and persisted states relate?**
   - CWM-A = attention buffer (what's being considered)
   - CWM-G = ground truth (believed reality)
   - Neo4j = persistent history
   - What flows where and when?

3. **What is "ephemeral" vs "persisted" state?**
   - When does state get promoted from ephemeral to persisted?
   - Session boundaries (#101) depend on this

4. **What provenance belongs on which state type?**
   - All states need source_service?
   - Only persisted states need full audit trail?

## Success Criteria
1. Single coherent "state" definition documented
2. Clear data flow diagram: CWM-A ↔ CWM-G ↔ Neo4j
3. Ephemeral vs persisted lifecycle defined
4. Provenance requirements per state type identified
5. Foundation enables #15 and #101 implementation

## Constraints
- Must align with existing HCG ontology (flexible Node system)
- Must support planning (preconditions/effects)
- Must support real-time updates (CWM)
- Should enable future temporal queries (state history)

## Workflow
**Complex** - Design phase will explore approaches, propose architecture, checkpoint for approval.

## Related Issues
- #15: Provenance on state nodes (blocked by this)
- #101: Session boundaries and ephemeral lifecycle (related)
