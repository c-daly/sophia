# Procedural Memory & Non-Linguistic Thought

**Date:** 2026-02-22
**Component:** Sophia (cognitive core)
**Status:** Design — awaiting implementation planning

---

## 1. Motivation

Sophia's existing cognitive infrastructure (CWM-A graph reasoning, CWM-G JEPA simulation, CWM-E affect) operates on structured representations but lacks a mechanism for **procedural cognition** — the ability to learn, store, mentally rehearse, and abstract over action sequences without language.

Humans reason procedurally: mentally rehearsing a reach before executing it, "feeling out" whether an action sequence will work, forming abstract concepts like "graspable" from repeated motor experience. This design adds that capability to Sophia.

### Design Principles

1. **No language at any point.** Schemas, rehearsal, prediction error, and concept formation operate entirely on graph structures, embeddings, and sensory patterns.
2. **Sophia owns all cognition.** Talos is a sensor/motor bus with no intelligence. Sophia handles spatial reasoning, trajectory planning, prediction, and evaluation.
3. **Same loop for thinking and doing.** Whether Sophia is rehearsing (stub backend), simulating (Gazebo), or executing (real hardware), the cognitive process is identical. Only the Talos backend changes.
4. **Everything lives in the HCG.** Schemas are graph nodes. Rehearsals produce ImaginedProcess/ImaginedState nodes. Concepts emerge from graph pattern matching.

---

## 2. Architecture Overview

```
                    ┌─────────────┐
                    │   Planner   │
                    └──────┬──────┘
                           │ selects schemas for goal
                    ┌──────▼──────┐
                    │ Procedural  │  ← stores, retrieves, rehearses,
                    │   Memory    │     and abstracts action schemas
                    └──┬──────┬──┘
           rehearse/   │      │  compare predictions
           execute     │      │  to observations
                ┌──────▼──┐ ┌─▼──────────┐
                │  Motor  │ │  Prediction │
                │  Exec   │ │  Evaluator  │
                │  Layer  │ │             │
                └────┬────┘ └─────────────┘
                     │ motor commands / sensor reads
                ┌────▼────┐
                │  Talos  │  ← pure sensor/motor bus
                │  (API)  │     backend: real | gazebo | stub
                └─────────┘
```

### Component Responsibilities

| Component | Responsibility | Does NOT do |
|-----------|---------------|-------------|
| **Planner** | Goal decomposition, schema selection | Motor control, spatial reasoning |
| **Procedural Memory** | Schema storage, retrieval, rehearsal orchestration, abstraction | Execution, kinematics |
| **Motor Execution Layer** | Spatial reasoning, trajectory planning, motor command generation | Schema selection, goal reasoning |
| **Prediction Evaluator** | Compare expected vs. observed, compute prediction error, flag anomalies | Action selection, execution |
| **Talos** | Send motor commands, read sensors, provide uniform API | Any cognition whatsoever |

---

## 3. Action Schema Model

An action schema is a first-class cognitive object representing a learned way of doing something.

### Schema Structure

```
ActionSchema:
  schema_id: str                          # unique identifier
  name: str                               # e.g., "reach_and_grasp"

  # Trigger: when is this schema relevant?
  trigger_conditions:
    required_entities: [EntityPattern]     # what must exist in the world
    required_state: StatePattern           # what the world must look like

  # Body: the procedure
  steps: [
    SchemaStep:
      action: ActionPrimitive             # MOVE, GRASP, ROTATE, etc.
      parameters: dict                    # action-specific params (relative, not absolute)
      spatial_intent: SpatialIntent       # approach direction, force category, grip strategy
      expected_state_change: StateDelta   # what should change in the world
      expected_sensory: SensoryPattern    # what we expect to observe
      confidence: float                   # how sure we are about this step
  ]

  # Outcomes
  success_criteria: StatePattern          # what success looks like (sensory)
  failure_modes: [FailurePattern]         # known ways this can go wrong

  # Metadata
  source: str                             # "video_observation" | "rehearsal" | "execution" | "abstracted"
  execution_count: int
  success_rate: float
  abstracted_from: [schema_id]            # if generalized from other schemas
  learned_from: [media_sample_id]         # if learned from video
  embedding_id: str                       # for similarity search in Milvus
```

### Spatial Intent (Open Question)

The boundary between schema-level spatial knowledge and the motor execution layer's spatial reasoning is deliberately left flexible:

- **Early schemas** (e.g., learned from video) may encode detailed spatial information: exact approach vectors, trajectory shapes, force profiles observed in the source video.
- **Mature schemas** (e.g., after abstraction) encode spatial *intent*: "approach from above," "gentle grip," "slow withdrawal." The motor execution layer fills in specifics.
- The system should naturally evolve from detailed to abstract as Sophia accumulates experience.

### Parameters Are Relative

Schema parameters reference entities and relationships, never absolute coordinates:

```python
# Correct: relative to entities
{"target": "object.position", "offset": {"z": +0.05}}   # "above the object"
{"approach_from": "above", "grip_width": "object.width"}

# Wrong: absolute coordinates
{"x": 0.3, "y": -0.1, "z": 0.4}
```

Spatial adaptation happens because parameters resolve against the *current* world state, not the state when the schema was learned.

---

## 4. HCG Integration

### Schemas as Graph Nodes

Schemas are `Process`-typed nodes in the HCG, participating in all existing infrastructure: SHACL validation, causal edges, embedding search, CWM state envelopes.

```cypher
// Schema node (a type/template)
(s:Schema:Process {
  schema_id: "reach_and_grasp_001",
  name: "reach_and_grasp",
  source: "video_observation",
  success_rate: 0.87,
  execution_count: 14
})

// Steps as ordered nodes
(s)-[:HAS_STEP {order: 0}]->(step0:SchemaStep {
  action_type: "MOVE",
  spatial_intent: "approach_from_above",
  expected_delta: "gripper_near_object"
})
(s)-[:HAS_STEP {order: 1}]->(step1:SchemaStep {
  action_type: "GRASP",
  spatial_intent: "close_around_object",
  expected_delta: "object_held"
})

// Causal ordering between steps
(step0)-[:ENABLES]->(step1)

// Schema applies to entity types
(s)-[:APPLIES_TO]->(concept:Entity:Concept {name: "graspable_object"})

// Schema requires/produces states
(s)-[:REQUIRES_STATE]->(pre:State {description: "gripper_open"})
(s)-[:PRODUCES_STATE]->(post:State {description: "object_held"})
```

### Type/Instance Relationship

Schemas are *types*. Executions (real or imagined) are *instances*:

```cypher
// Schema (type)
(s:Schema {name: "reach_and_grasp"})

// Real execution (instance)
(s)<-[:INSTANCE_OF]-(p1:Process {imagined: false, timestamp: "..."})

// Rehearsal (imagined instance)
(s)<-[:INSTANCE_OF]-(p2:Process {imagined: true, timestamp: "..."})

// Each instance produces states
(p2)-[:PRODUCED]->(state0:State {imagined: true, step: 0})
(p2)-[:PRODUCED]->(state1:State {imagined: true, step: 1})
```

### What "Thought" Looks Like in the Graph

When Sophia is "thinking" (rehearsing schemas against a stub backend), the HCG grows a subgraph:

- `ImaginedProcess` nodes linked to their source `Schema`
- `ImaginedState` nodes at each step
- `PredictionError` nodes where expected ≠ observed
- Causal edges connecting the sequence

The thought *is* the subgraph. It's inspectable, queryable, and causally connected to the rest of Sophia's knowledge. No language representation exists at any point.

---

## 5. The Rehearsal-Prediction Loop

### Core Loop

```
1. SELECT schema (embedding similarity: current situation → schema triggers)
2. For each step in schema:
   a. PREDICT: expected state change + expected sensory feedback
   b. EXECUTE: send action to Talos (real, gazebo, or stub — identical API)
   c. OBSERVE: read sensory result from Talos
   d. COMPARE: prediction_error = |predicted - observed|
   e. UPDATE:
      - Small error → continue, schema is working
      - Large error → flag for re-evaluation, record PredictionError in HCG
      - Record prediction accuracy for this step
3. EVALUATE: did we reach success_criteria?
   - Yes → increment success_rate, store as Process in HCG
   - No → record FailurePattern, link to Schema
```

### Prediction Error as Learning Signal

| Error Pattern | Meaning | Sophia's Response |
|--------------|---------|-------------------|
| Consistent small errors | Schema is approximately right | Gradual schema refinement |
| Sudden large error | Something unexpected happened | Attention/investigation; PredictionError node becomes causal starting point |
| Systematic bias | Schema consistently over/under-predicts | Schema modification |
| Zero error on stub, high on real | Simulation doesn't match reality | Flag calibration issue |

### No Difference Between Thinking and Doing

The cognitive loop is identical regardless of Talos backend:

- **Stub:** "Mental rehearsal" — fast, cheap, no physical consequences
- **Gazebo:** "Simulation" — physics-accurate, moderate cost
- **Real hardware:** "Execution" — actual physical consequences

Sophia doesn't have an "imagination mode." She has one mode: run schemas through Talos. The fidelity varies with the backend. The `imagined` flag on resulting Process/State nodes is set based on backend type.

---

## 6. Learning from Video

### Two Pathways

Video feeds two parallel channels:

1. **Schema extraction:** video → action segments → ActionSchema → Procedural Memory
2. **World grounding:** video → visual/physics embeddings → CWM-G (Milvus)

Both pathways link back to the source video, so schemas carry provenance to their perceptual origins.

### Schema Extraction Pipeline

```
Video ingested
    │
    ▼
JEPA generates per-frame embeddings
    │
    ▼
Temporal clustering segments video into action boundaries
(embedding similarity between consecutive frames;
 cluster boundaries = action transitions)
    │
    ▼
Each segment classified by action type
(embedding similarity to known action prototypes)
    │
    ▼
Segments assembled into ActionSchema
(ordered steps with inferred parameters,
 spatial intent extracted from frame-to-frame deltas)
    │
    ▼
Schema rehearsed against Talos-stub
(does this sequence make physical sense?)
    │
    ▼
If plausible → stored in Procedural Memory (HCG)
If implausible → flagged for review, stored with low confidence
```

### HCG Representation

```cypher
// Video source
(v:MediaSample {media_type: "video", sample_id: "vid_001"})

// Schema learned from video
(s:Schema {schema_id: "learned_001", source: "video_observation"})
(s)-[:LEARNED_FROM]->(v)

// Video segments
(v)-[:HAS_SEGMENT]->(seg0:VideoSegment {start_ms: 0, end_ms: 1200, action: "REACH"})
(v)-[:HAS_SEGMENT]->(seg1:VideoSegment {start_ms: 1200, end_ms: 2100, action: "GRASP"})

// Steps reference source segments
(step0:SchemaStep)-[:OBSERVED_IN]->(seg0)
(step1:SchemaStep)-[:OBSERVED_IN]->(seg1)

// Visual embeddings link to both video and schema
(v)-[:HAS_EMBEDDING]->(emb:Embedding {vector_type: "visual"})
(s)-[:GROUNDED_BY]->(emb)
```

### The Non-Linguistic Learning Loop

1. Sophia **watches** video (ingested through Talos sensor pipeline)
2. JEPA **segments** the action sequence (temporal embedding clustering — no language)
3. Sophia **constructs** a schema from segments (graph construction)
4. Sophia **rehearses** the schema against Talos-stub (does it make physical sense?)
5. If plausible → schema enters Procedural Memory
6. Later, similar situation arises → embedding similarity retrieves this schema
7. Sophia executes through Talos (any backend)

No language at any point in the chain.

---

## 7. Concept Formation via Schema Abstraction

### How Concepts Emerge from Procedural Experience

**Step 1: Detect structural similarity**

Multiple schemas share step-level structure. Graph pattern matching (subgraph isomorphism) identifies shared patterns:

- "pick_up_cup" and "pick_up_ball" both have: APPROACH → GRASP → LIFT
- The shared substructure is detected through graph queries, not textual comparison

**Step 2: Abstract the shared structure**

A new abstract schema is created:
- "pick_up_object" with the shared step pattern
- Parameters generalized: `target` instead of `cup` / `ball`
- Links: `pick_up_cup -[:ABSTRACTED_INTO]-> pick_up_object`

**Step 3: Form the concept**

The entities that all concrete instances operated on share properties (small, solid, within reach). These shared properties define a concept:
- New Entity node: `(:Entity:Concept {name: "graspable_object", source: "abstracted"})`
- Abstract schema's `APPLIES_TO` edge points to this concept

**Step 4: Use for transfer**

Sophia encounters a new object (bottle). Embedding similarity finds "bottle" is close to "graspable_object." She retrieves "pick_up_object" schema, rehearses with "bottle" as target. If rehearsal succeeds — she knows how to pick up bottles without ever having been taught.

### Spatial Concepts

Spatial patterns that recur across schemas also become concepts:

- "approach_from_above" appears in many pick-up schemas
- "follow_contour" appears in wiping/tracing schemas
- "maintain_orientation" appears in pouring/carrying schemas

These become spatial concept nodes in the HCG, linked to the schemas that exhibit them. They're not verbal labels — they're graph-structural patterns with associated embeddings.

---

## 8. Research Context

This design draws on several research traditions:

| Tradition | Key Ideas Used | Key References |
|-----------|---------------|----------------|
| **Procedural memory** (cognitive architectures) | Schemas as executable templates; procedural/declarative distinction | ACT-R (Anderson), Soar (Laird) |
| **Perceptual Symbol Systems** | Concepts grounded in sensorimotor patterns, not language | Barsalou (1999) |
| **Predictive processing** | Prediction error as primary learning signal; perception as prediction | Friston (Free Energy), Clark (2013) |
| **Motor simulation theory** | Understanding actions by internally simulating them | Gallese & Rizzolatti (mirror neurons) |
| **World models in RL** | Learning internal models, imagining rollouts for planning | Ha & Schmidhuber (2018), DreamerV3 |
| **V-JEPA** | Self-supervised video understanding via joint-embedding prediction | LeCun, Assran et al. (2024) |
| **Conceptual Spaces** | Concepts as regions in geometric spaces | Gärdenfors (2000) |

### Approaches Considered and Not Chosen

1. **Active Inference Engine (Friston):** Hierarchical prediction with free energy minimization. More theoretically complete but computationally heavier and harder to implement incrementally. Could be layered on later.

2. **Embodied Concept Spaces (Gärdenfors + Barsalou):** Pure geometric reasoning over sensorimotor embedding manifolds. Elegant but requires massive grounding data to build meaningful spaces. Elements incorporated into concept formation.

3. **Continuous dynamical systems:** Thought as trajectories through state space with attractors. Theoretically appealing but hard to inspect/debug and doesn't leverage existing HCG infrastructure.

---

## 9. Open Questions

1. **Schema-to-motor boundary:** How much spatial detail belongs in the schema vs. a separate motor execution layer in Sophia? Recommended default: early schemas are detailed (copying video observations), mature schemas are abstract (motor layer fills in). Boundary shifts naturally with experience.

2. **Action segmentation fidelity:** V-JEPA temporal clustering for action boundaries is unproven at production quality. May need alternative approaches (supervised segmentation models, human-labeled bootstrapping).

3. **Subgraph isomorphism performance:** Schema abstraction requires detecting shared step structures across potentially many schemas. May need approximate methods for large schema libraries.

4. **Prediction error thresholds:** What counts as "small" vs. "large" prediction error? Likely domain-specific and needs calibration.

5. **Schema lifecycle:** When should low-confidence or rarely-used schemas be pruned? What's the retention policy?

6. **Multi-step rehearsal cost:** Rehearsing complex schemas against Talos-stub is cheap per step, but long schemas with branching could become expensive. May need rehearsal budgets or depth limits.

---

## 10. Success Criteria

The system is working when:

1. Sophia can learn an action schema from video observation without any language input
2. Sophia can rehearse a schema against Talos-stub and evaluate whether it achieves a goal
3. Sophia can execute a learned schema through Talos (any backend) using the same cognitive loop as rehearsal
4. Sophia can form abstract concepts by detecting shared structure across multiple schemas
5. Sophia can transfer a schema to a novel situation via embedding similarity + rehearsal validation
6. The entire chain (video → schema → rehearsal → execution → abstraction) operates without language at any point
7. All cognitive artifacts (schemas, rehearsals, predictions, errors, concepts) are represented as HCG nodes and edges
