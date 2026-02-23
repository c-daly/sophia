# Procedural Memory Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the Procedural Memory system described in `docs/plans/2026-02-22-procedural-memory-design.md` — enabling non-linguistic thought through action schemas, rehearsal, prediction error, video learning, and concept formation.

**Architecture:** New `sophia.procedural` package with Pydantic models, a memory store backed by the HCG (Neo4j), a prediction evaluator, a rehearsal loop that drives Talos the same way for thinking and doing, a video-to-schema extraction pipeline, and a schema abstraction engine for concept formation.

**Tech Stack:** Python 3.12, Pydantic v2, Neo4j (via `sophia.hcg_client`), Milvus (via `logos_hcg`), existing JEPA runner, pytest with monkeypatch/MagicMock.

**Key Codebase Context:**
- HCG client: `src/sophia/hcg_client/client.py` — `add_node(name, node_type, uuid, properties, source, derivation, confidence, tags)` returns uuid. `add_edge(source_uuid, target_uuid, relation, properties)` creates reified edge nodes. Properties use `__LOGOS_JSON__` sentinel for complex types.
- JEPA models: `src/sophia/jepa/models.py` — `Entity`, `SimulationContext`, `ImaginedProcess`, `ImaginedState`, `SimulationResult`.
- JEPA runner: `src/sophia/jepa/runner.py` — `JEPABackend` protocol with `simulate()` and `process_media_sample()`. `StubJEPABackend` provides CPU-friendly stubs.
- Planner: `src/sophia/planner/planner.py` — `plan(goal)` does backward chaining over KnowledgeGraph.
- Executor: `src/sophia/executor/executor.py` — simple action queue, no execution logic yet.
- Tests mock Neo4j driver via `monkeypatch.setattr("logos_hcg.client.GraphDatabase.driver", ...)`.
- Foundry models: `logos_hcg.models` — `Entity`, `Concept`, `State`, `Process`, `Edge` with embedding metadata fields.

---

## Task 1: Core Pydantic Models

**Files:**
- Create: `src/sophia/procedural/__init__.py`
- Create: `src/sophia/procedural/models.py`
- Create: `tests/unit/procedural/__init__.py`
- Create: `tests/unit/procedural/test_models.py`

### Step 1: Write the failing tests

```python
# tests/unit/procedural/test_models.py
"""Tests for procedural memory Pydantic models."""

import pytest
from sophia.procedural.models import (
    ActionPrimitive,
    SpatialIntent,
    SensoryPattern,
    StateDelta,
    EntityPattern,
    StatePattern,
    FailurePattern,
    TriggerConditions,
    SchemaStep,
    ActionSchema,
    PredictionError,
    RehearsalResult,
)


class TestActionPrimitive:
    def test_known_primitives(self):
        for name in ("MOVE", "GRASP", "RELEASE", "ROTATE", "LIFT", "PLACE", "PUSH", "PULL"):
            p = ActionPrimitive(name=name)
            assert p.name == name

    def test_custom_primitive(self):
        p = ActionPrimitive(name="CUSTOM_ACTION")
        assert p.name == "CUSTOM_ACTION"


class TestSpatialIntent:
    def test_minimal(self):
        si = SpatialIntent()
        assert si.approach_direction is None

    def test_full(self):
        si = SpatialIntent(
            approach_direction="above",
            force_category="gentle",
            grip_strategy="pinch",
            trajectory_hint="linear",
        )
        assert si.approach_direction == "above"
        assert si.force_category == "gentle"


class TestSchemaStep:
    def test_create_step(self):
        step = SchemaStep(
            step_id="step_0",
            action=ActionPrimitive(name="MOVE"),
            parameters={"target": "object.position", "offset": {"z": 0.05}},
            spatial_intent=SpatialIntent(approach_direction="above"),
            expected_state_change=StateDelta(changes={"gripper_pos": "near_object"}),
            expected_sensory=SensoryPattern(patterns={"proximity": "close"}),
            confidence=0.9,
        )
        assert step.action.name == "MOVE"
        assert step.confidence == 0.9

    def test_default_confidence(self):
        step = SchemaStep(
            step_id="step_0",
            action=ActionPrimitive(name="GRASP"),
            parameters={},
        )
        assert step.confidence == 1.0


class TestActionSchema:
    def test_create_minimal_schema(self):
        schema = ActionSchema(
            name="reach_and_grasp",
            steps=[
                SchemaStep(
                    step_id="s0",
                    action=ActionPrimitive(name="MOVE"),
                    parameters={"target": "object.position"},
                ),
                SchemaStep(
                    step_id="s1",
                    action=ActionPrimitive(name="GRASP"),
                    parameters={"target": "object"},
                ),
            ],
        )
        assert schema.schema_id  # auto-generated
        assert schema.name == "reach_and_grasp"
        assert len(schema.steps) == 2
        assert schema.source == "unknown"
        assert schema.execution_count == 0
        assert schema.success_rate == 0.0

    def test_schema_with_trigger_conditions(self):
        schema = ActionSchema(
            name="pick_up",
            steps=[
                SchemaStep(
                    step_id="s0",
                    action=ActionPrimitive(name="GRASP"),
                    parameters={},
                ),
            ],
            trigger_conditions=TriggerConditions(
                required_entities=[EntityPattern(entity_type="object", properties={"graspable": True})],
                required_state=StatePattern(conditions={"gripper": "open"}),
            ),
            success_criteria=StatePattern(conditions={"holding": True}),
            source="video_observation",
        )
        assert schema.trigger_conditions.required_entities[0].entity_type == "object"
        assert schema.source == "video_observation"

    def test_schema_id_auto_generated(self):
        s1 = ActionSchema(name="a", steps=[])
        s2 = ActionSchema(name="a", steps=[])
        assert s1.schema_id != s2.schema_id

    def test_schema_with_failure_modes(self):
        schema = ActionSchema(
            name="pour",
            steps=[],
            failure_modes=[
                FailurePattern(
                    name="spill",
                    description="Liquid spills during pour",
                    detection=SensoryPattern(patterns={"liquid_sensor": "wet"}),
                ),
            ],
        )
        assert len(schema.failure_modes) == 1
        assert schema.failure_modes[0].name == "spill"


class TestPredictionError:
    def test_create(self):
        pe = PredictionError(
            step_id="s0",
            predicted=StateDelta(changes={"pos": "near"}),
            observed=StateDelta(changes={"pos": "far"}),
            magnitude=0.8,
        )
        assert pe.magnitude == 0.8
        assert pe.step_id == "s0"


class TestRehearsalResult:
    def test_success(self):
        result = RehearsalResult(
            schema_id="schema_1",
            success=True,
            steps_completed=3,
            steps_total=3,
            prediction_errors=[],
            overall_confidence=0.92,
        )
        assert result.success

    def test_failure_with_errors(self):
        result = RehearsalResult(
            schema_id="schema_1",
            success=False,
            steps_completed=1,
            steps_total=3,
            prediction_errors=[
                PredictionError(
                    step_id="s1",
                    predicted=StateDelta(changes={}),
                    observed=StateDelta(changes={}),
                    magnitude=0.9,
                ),
            ],
            overall_confidence=0.3,
            failure_mode="collision_detected",
        )
        assert not result.success
        assert result.failure_mode == "collision_detected"
```

### Step 2: Run tests to verify they fail

Run: `cd /Users/cdaly/projects/LOGOS/sophia && python -m pytest tests/unit/procedural/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sophia.procedural'`

### Step 3: Write minimal implementation

```python
# src/sophia/procedural/__init__.py
"""Procedural Memory — non-linguistic thought through action schemas."""

# tests/unit/procedural/__init__.py
# (empty)
```

```python
# src/sophia/procedural/models.py
"""Pydantic models for procedural memory.

Action schemas, steps, spatial intent, prediction errors, and rehearsal results.
All models are non-linguistic — no natural language fields drive cognition.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ActionPrimitive(BaseModel):
    """An atomic action type (MOVE, GRASP, RELEASE, etc.)."""

    name: str


class SpatialIntent(BaseModel):
    """Spatial intent for a schema step — what kind of spatial action, not exact coordinates."""

    approach_direction: Optional[str] = None  # e.g., "above", "front", "side"
    force_category: Optional[str] = None  # e.g., "gentle", "firm", "precise"
    grip_strategy: Optional[str] = None  # e.g., "pinch", "power", "wrap"
    trajectory_hint: Optional[str] = None  # e.g., "linear", "arc", "follow_contour"


class SensoryPattern(BaseModel):
    """Expected sensory observation pattern."""

    patterns: Dict[str, Any] = Field(default_factory=dict)


class StateDelta(BaseModel):
    """Expected state change — what should be different after an action."""

    changes: Dict[str, Any] = Field(default_factory=dict)


class EntityPattern(BaseModel):
    """Pattern for matching entities in trigger conditions."""

    entity_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class StatePattern(BaseModel):
    """Pattern for matching world state in triggers and success criteria."""

    conditions: Dict[str, Any] = Field(default_factory=dict)


class FailurePattern(BaseModel):
    """A known failure mode for a schema."""

    name: str
    description: str = ""
    detection: SensoryPattern = Field(default_factory=SensoryPattern)


class TriggerConditions(BaseModel):
    """When a schema becomes relevant."""

    required_entities: List[EntityPattern] = Field(default_factory=list)
    required_state: StatePattern = Field(default_factory=StatePattern)


class SchemaStep(BaseModel):
    """A single step in an action schema."""

    step_id: str
    action: ActionPrimitive
    parameters: Dict[str, Any] = Field(default_factory=dict)
    spatial_intent: SpatialIntent = Field(default_factory=SpatialIntent)
    expected_state_change: StateDelta = Field(default_factory=StateDelta)
    expected_sensory: SensoryPattern = Field(default_factory=SensoryPattern)
    confidence: float = 1.0


class ActionSchema(BaseModel):
    """A learned action procedure — first-class cognitive object.

    Schemas are non-linguistic: they encode HOW to do something as a sequence
    of steps with spatial intent, expected state changes, and sensory predictions.
    """

    schema_id: str = Field(default_factory=lambda: f"schema_{uuid4().hex[:12]}")
    name: str
    steps: List[SchemaStep] = Field(default_factory=list)
    trigger_conditions: TriggerConditions = Field(default_factory=TriggerConditions)
    success_criteria: StatePattern = Field(default_factory=StatePattern)
    failure_modes: List[FailurePattern] = Field(default_factory=list)
    source: str = "unknown"  # "video_observation" | "rehearsal" | "execution" | "abstracted"
    execution_count: int = 0
    success_rate: float = 0.0
    abstracted_from: List[str] = Field(default_factory=list)  # schema_ids
    learned_from: List[str] = Field(default_factory=list)  # media_sample_ids
    embedding_id: Optional[str] = None


class PredictionError(BaseModel):
    """A mismatch between predicted and observed state at a schema step."""

    step_id: str
    predicted: StateDelta
    observed: StateDelta
    magnitude: float  # 0.0 = perfect match, 1.0 = completely wrong


class RehearsalResult(BaseModel):
    """Result of rehearsing (or executing) a schema."""

    schema_id: str
    success: bool
    steps_completed: int
    steps_total: int
    prediction_errors: List[PredictionError] = Field(default_factory=list)
    overall_confidence: float
    failure_mode: Optional[str] = None
    imagined: bool = True  # True for rehearsal, False for real execution
```

### Step 4: Run tests to verify they pass

Run: `cd /Users/cdaly/projects/LOGOS/sophia && python -m pytest tests/unit/procedural/test_models.py -v`
Expected: All tests PASS

### Step 5: Commit

```bash
cd /Users/cdaly/projects/LOGOS/sophia
git add src/sophia/procedural/__init__.py src/sophia/procedural/models.py tests/unit/procedural/__init__.py tests/unit/procedural/test_models.py
git commit -m "feat(procedural): add core Pydantic models for action schemas"
```

---

## Task 2: Procedural Memory — In-Memory Store + CRUD

**Files:**
- Create: `src/sophia/procedural/memory.py`
- Create: `tests/unit/procedural/test_memory.py`

### Step 1: Write the failing tests

```python
# tests/unit/procedural/test_memory.py
"""Tests for ProceduralMemory in-memory store."""

import pytest
from sophia.procedural.models import (
    ActionPrimitive,
    ActionSchema,
    SchemaStep,
    StatePattern,
    TriggerConditions,
    EntityPattern,
)
from sophia.procedural.memory import ProceduralMemory


@pytest.fixture()
def memory():
    return ProceduralMemory()


@pytest.fixture()
def grasp_schema():
    return ActionSchema(
        name="reach_and_grasp",
        steps=[
            SchemaStep(step_id="s0", action=ActionPrimitive(name="MOVE"), parameters={"target": "object.position"}),
            SchemaStep(step_id="s1", action=ActionPrimitive(name="GRASP"), parameters={"target": "object"}),
        ],
        trigger_conditions=TriggerConditions(
            required_entities=[EntityPattern(entity_type="object", properties={"graspable": True})],
        ),
        success_criteria=StatePattern(conditions={"holding": True}),
        source="execution",
    )


class TestProceduralMemoryStore:
    def test_store_and_retrieve(self, memory, grasp_schema):
        memory.store(grasp_schema)
        retrieved = memory.get(grasp_schema.schema_id)
        assert retrieved is not None
        assert retrieved.name == "reach_and_grasp"
        assert len(retrieved.steps) == 2

    def test_get_nonexistent_returns_none(self, memory):
        assert memory.get("nonexistent") is None

    def test_store_overwrites_existing(self, memory, grasp_schema):
        memory.store(grasp_schema)
        updated = grasp_schema.model_copy(update={"success_rate": 0.95})
        memory.store(updated)
        retrieved = memory.get(grasp_schema.schema_id)
        assert retrieved.success_rate == 0.95

    def test_delete(self, memory, grasp_schema):
        memory.store(grasp_schema)
        deleted = memory.delete(grasp_schema.schema_id)
        assert deleted is True
        assert memory.get(grasp_schema.schema_id) is None

    def test_delete_nonexistent(self, memory):
        assert memory.delete("nonexistent") is False

    def test_list_all(self, memory, grasp_schema):
        s2 = ActionSchema(name="lift", steps=[], source="rehearsal")
        memory.store(grasp_schema)
        memory.store(s2)
        all_schemas = memory.list_all()
        assert len(all_schemas) == 2

    def test_size(self, memory, grasp_schema):
        assert memory.size() == 0
        memory.store(grasp_schema)
        assert memory.size() == 1


class TestProceduralMemoryQuery:
    def test_find_by_name(self, memory, grasp_schema):
        memory.store(grasp_schema)
        results = memory.find_by_name("reach_and_grasp")
        assert len(results) == 1
        assert results[0].schema_id == grasp_schema.schema_id

    def test_find_by_name_no_match(self, memory, grasp_schema):
        memory.store(grasp_schema)
        assert memory.find_by_name("nonexistent") == []

    def test_find_by_source(self, memory, grasp_schema):
        s2 = ActionSchema(name="lift", steps=[], source="video_observation")
        memory.store(grasp_schema)
        memory.store(s2)
        results = memory.find_by_source("execution")
        assert len(results) == 1
        assert results[0].name == "reach_and_grasp"

    def test_find_by_action_type(self, memory, grasp_schema):
        memory.store(grasp_schema)
        results = memory.find_by_action_type("GRASP")
        assert len(results) == 1

    def test_find_by_action_type_no_match(self, memory, grasp_schema):
        memory.store(grasp_schema)
        assert memory.find_by_action_type("POUR") == []


class TestProceduralMemoryUpdate:
    def test_record_execution_success(self, memory, grasp_schema):
        memory.store(grasp_schema)
        memory.record_execution(grasp_schema.schema_id, success=True)
        updated = memory.get(grasp_schema.schema_id)
        assert updated.execution_count == 1
        assert updated.success_rate == 1.0

    def test_record_execution_failure(self, memory, grasp_schema):
        memory.store(grasp_schema)
        memory.record_execution(grasp_schema.schema_id, success=False)
        updated = memory.get(grasp_schema.schema_id)
        assert updated.execution_count == 1
        assert updated.success_rate == 0.0

    def test_record_multiple_executions(self, memory, grasp_schema):
        memory.store(grasp_schema)
        memory.record_execution(grasp_schema.schema_id, success=True)
        memory.record_execution(grasp_schema.schema_id, success=True)
        memory.record_execution(grasp_schema.schema_id, success=False)
        updated = memory.get(grasp_schema.schema_id)
        assert updated.execution_count == 3
        assert abs(updated.success_rate - 2.0 / 3.0) < 0.01

    def test_record_execution_nonexistent_raises(self, memory):
        with pytest.raises(KeyError):
            memory.record_execution("nonexistent", success=True)
```

### Step 2: Run tests to verify they fail

Run: `cd /Users/cdaly/projects/LOGOS/sophia && python -m pytest tests/unit/procedural/test_memory.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sophia.procedural.memory'`

### Step 3: Write minimal implementation

```python
# src/sophia/procedural/memory.py
"""Procedural Memory — in-memory store for action schemas.

Provides CRUD, query, and execution tracking for schemas.
Designed for later extension with HCG persistence.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from sophia.procedural.models import ActionSchema


class ProceduralMemory:
    """In-memory store for action schemas."""

    def __init__(self) -> None:
        self._schemas: Dict[str, ActionSchema] = {}
        self._success_counts: Dict[str, int] = {}
        self._total_counts: Dict[str, int] = {}

    def store(self, schema: ActionSchema) -> None:
        """Store or overwrite a schema."""
        self._schemas[schema.schema_id] = schema

    def get(self, schema_id: str) -> Optional[ActionSchema]:
        """Retrieve a schema by ID, or None."""
        return self._schemas.get(schema_id)

    def delete(self, schema_id: str) -> bool:
        """Delete a schema. Returns True if it existed."""
        if schema_id in self._schemas:
            del self._schemas[schema_id]
            self._success_counts.pop(schema_id, None)
            self._total_counts.pop(schema_id, None)
            return True
        return False

    def list_all(self) -> List[ActionSchema]:
        """Return all stored schemas."""
        return list(self._schemas.values())

    def size(self) -> int:
        """Number of stored schemas."""
        return len(self._schemas)

    def find_by_name(self, name: str) -> List[ActionSchema]:
        """Find schemas by exact name match."""
        return [s for s in self._schemas.values() if s.name == name]

    def find_by_source(self, source: str) -> List[ActionSchema]:
        """Find schemas by source type."""
        return [s for s in self._schemas.values() if s.source == source]

    def find_by_action_type(self, action_type: str) -> List[ActionSchema]:
        """Find schemas that contain a step with the given action type."""
        return [
            s
            for s in self._schemas.values()
            if any(step.action.name == action_type for step in s.steps)
        ]

    def record_execution(self, schema_id: str, *, success: bool) -> None:
        """Record an execution result, updating success_rate and execution_count."""
        if schema_id not in self._schemas:
            raise KeyError(f"Schema {schema_id} not found")

        self._total_counts[schema_id] = self._total_counts.get(schema_id, 0) + 1
        if success:
            self._success_counts[schema_id] = self._success_counts.get(schema_id, 0) + 1

        total = self._total_counts[schema_id]
        successes = self._success_counts.get(schema_id, 0)

        schema = self._schemas[schema_id]
        self._schemas[schema_id] = schema.model_copy(
            update={
                "execution_count": total,
                "success_rate": successes / total,
            }
        )
```

### Step 4: Run tests to verify they pass

Run: `cd /Users/cdaly/projects/LOGOS/sophia && python -m pytest tests/unit/procedural/test_memory.py -v`
Expected: All tests PASS

### Step 5: Commit

```bash
cd /Users/cdaly/projects/LOGOS/sophia
git add src/sophia/procedural/memory.py tests/unit/procedural/test_memory.py
git commit -m "feat(procedural): add ProceduralMemory in-memory store with CRUD and queries"
```

---

## Task 3: HCG Persistence — Schema ↔ Neo4j Serialization

**Files:**
- Create: `src/sophia/procedural/hcg_persistence.py`
- Create: `tests/unit/procedural/test_hcg_persistence.py`
- Modify: `src/sophia/procedural/__init__.py`

**Context:** Schemas become `Schema`-typed nodes. Steps become `SchemaStep`-typed nodes connected via reified `HAS_STEP` edges. The HCG client's `add_node()` accepts `name`, `node_type`, `uuid`, `properties`, `source`, `derivation`, `confidence`. `add_edge()` creates reified edge nodes.

### Step 1: Write the failing tests

```python
# tests/unit/procedural/test_hcg_persistence.py
"""Tests for HCG persistence of action schemas."""

from unittest.mock import MagicMock, call
import pytest

from sophia.procedural.models import (
    ActionPrimitive,
    ActionSchema,
    SchemaStep,
    SpatialIntent,
    StateDelta,
    SensoryPattern,
    TriggerConditions,
    EntityPattern,
    StatePattern,
)
from sophia.procedural.hcg_persistence import SchemaHCGPersistence


@pytest.fixture()
def mock_hcg_client():
    client = MagicMock()
    client.add_node.return_value = "mock-uuid"
    client.add_edge.return_value = "mock-edge-uuid"
    return client


@pytest.fixture()
def persistence(mock_hcg_client):
    return SchemaHCGPersistence(hcg_client=mock_hcg_client)


@pytest.fixture()
def two_step_schema():
    return ActionSchema(
        schema_id="test_schema_001",
        name="reach_and_grasp",
        steps=[
            SchemaStep(
                step_id="s0",
                action=ActionPrimitive(name="MOVE"),
                parameters={"target": "object.position"},
                spatial_intent=SpatialIntent(approach_direction="above"),
                expected_state_change=StateDelta(changes={"gripper_pos": "near"}),
                confidence=0.95,
            ),
            SchemaStep(
                step_id="s1",
                action=ActionPrimitive(name="GRASP"),
                parameters={"target": "object"},
                confidence=0.9,
            ),
        ],
        source="video_observation",
        success_criteria=StatePattern(conditions={"holding": True}),
        execution_count=5,
        success_rate=0.8,
    )


class TestSaveSchema:
    def test_saves_schema_node(self, persistence, mock_hcg_client, two_step_schema):
        persistence.save(two_step_schema)

        # First add_node call should be the schema node
        schema_call = mock_hcg_client.add_node.call_args_list[0]
        assert schema_call.kwargs["name"] == "reach_and_grasp"
        assert schema_call.kwargs["node_type"] == "Schema"
        assert schema_call.kwargs["uuid"] == "test_schema_001"
        assert schema_call.kwargs["source"] == "procedural_memory"

    def test_saves_step_nodes(self, persistence, mock_hcg_client, two_step_schema):
        persistence.save(two_step_schema)

        # Should create 3 nodes total: 1 schema + 2 steps
        assert mock_hcg_client.add_node.call_count == 3

        step_calls = mock_hcg_client.add_node.call_args_list[1:]
        assert step_calls[0].kwargs["node_type"] == "SchemaStep"
        assert step_calls[1].kwargs["node_type"] == "SchemaStep"

    def test_creates_has_step_edges(self, persistence, mock_hcg_client, two_step_schema):
        persistence.save(two_step_schema)

        # Should create HAS_STEP edges from schema to each step
        has_step_calls = [
            c for c in mock_hcg_client.add_edge.call_args_list if c.kwargs.get("relation") == "HAS_STEP"
        ]
        assert len(has_step_calls) == 2
        # Check order property
        assert has_step_calls[0].kwargs["properties"]["order"] == 0
        assert has_step_calls[1].kwargs["properties"]["order"] == 1

    def test_creates_enables_edges_between_steps(self, persistence, mock_hcg_client, two_step_schema):
        persistence.save(two_step_schema)

        enables_calls = [
            c for c in mock_hcg_client.add_edge.call_args_list if c.kwargs.get("relation") == "ENABLES"
        ]
        assert len(enables_calls) == 1  # step0 ENABLES step1

    def test_schema_properties_include_metadata(self, persistence, mock_hcg_client, two_step_schema):
        persistence.save(two_step_schema)

        schema_call = mock_hcg_client.add_node.call_args_list[0]
        props = schema_call.kwargs["properties"]
        assert props["schema_source"] == "video_observation"
        assert props["execution_count"] == 5
        assert props["success_rate"] == 0.8


class TestLoadSchema:
    def test_load_reconstructs_schema(self, persistence, mock_hcg_client):
        # Mock get_node for schema
        mock_hcg_client.get_node.return_value = {
            "uuid": "test_schema_001",
            "name": "reach_and_grasp",
            "type": "Schema",
            "properties": {
                "schema_source": "execution",
                "execution_count": 3,
                "success_rate": 0.67,
                "trigger_conditions": '{"required_entities": [], "required_state": {"conditions": {}}}',
                "success_criteria": '{"conditions": {"holding": true}}',
                "failure_modes": "[]",
                "abstracted_from": "[]",
                "learned_from": "[]",
            },
        }

        # Mock edge query for steps
        mock_hcg_client.list_all_edges.return_value = [
            {
                "id": "edge_1",
                "source": "test_schema_001",
                "target": "step_0",
                "relation": "HAS_STEP",
                "properties": {"order": 0},
            },
            {
                "id": "edge_2",
                "source": "test_schema_001",
                "target": "step_1",
                "relation": "HAS_STEP",
                "properties": {"order": 1},
            },
        ]

        # Mock get_node for steps
        def get_node_side_effect(uuid):
            nodes = {
                "test_schema_001": mock_hcg_client.get_node.return_value,
                "step_0": {
                    "uuid": "step_0",
                    "name": "step_0",
                    "type": "SchemaStep",
                    "properties": {
                        "action_name": "MOVE",
                        "parameters": '{"target": "object.position"}',
                        "spatial_intent": "{}",
                        "expected_state_change": '{"changes": {}}',
                        "expected_sensory": '{"patterns": {}}',
                        "confidence": 0.95,
                    },
                },
                "step_1": {
                    "uuid": "step_1",
                    "name": "step_1",
                    "type": "SchemaStep",
                    "properties": {
                        "action_name": "GRASP",
                        "parameters": '{"target": "object"}',
                        "spatial_intent": "{}",
                        "expected_state_change": '{"changes": {}}',
                        "expected_sensory": '{"patterns": {}}',
                        "confidence": 0.9,
                    },
                },
            }
            return nodes.get(uuid)

        mock_hcg_client.get_node.side_effect = get_node_side_effect

        schema = persistence.load("test_schema_001")
        assert schema is not None
        assert schema.name == "reach_and_grasp"
        assert schema.source == "execution"
        assert len(schema.steps) == 2
        assert schema.steps[0].action.name == "MOVE"
        assert schema.steps[1].action.name == "GRASP"

    def test_load_nonexistent_returns_none(self, persistence, mock_hcg_client):
        mock_hcg_client.get_node.return_value = None
        assert persistence.load("nonexistent") is None
```

### Step 2: Run tests to verify they fail

Run: `cd /Users/cdaly/projects/LOGOS/sophia && python -m pytest tests/unit/procedural/test_hcg_persistence.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'sophia.procedural.hcg_persistence'`

### Step 3: Write minimal implementation

```python
# src/sophia/procedural/hcg_persistence.py
"""HCG persistence for action schemas.

Serializes ActionSchema ↔ Neo4j graph nodes (Schema + SchemaStep nodes,
connected by reified HAS_STEP and ENABLES edges).
"""

from __future__ import annotations

import json
from typing import Any, Optional

from sophia.procedural.models import (
    ActionPrimitive,
    ActionSchema,
    FailurePattern,
    SchemaStep,
    SensoryPattern,
    SpatialIntent,
    StateDelta,
    StatePattern,
    TriggerConditions,
)


class SchemaHCGPersistence:
    """Reads and writes ActionSchemas to/from the HCG via the HCGClient."""

    def __init__(self, hcg_client: Any) -> None:
        self._hcg = hcg_client

    def save(self, schema: ActionSchema) -> str:
        """Persist a schema and its steps to the HCG. Returns schema uuid."""
        # 1. Create the Schema node
        schema_props = {
            "schema_source": schema.source,
            "execution_count": schema.execution_count,
            "success_rate": schema.success_rate,
            "trigger_conditions": schema.trigger_conditions.model_dump_json(),
            "success_criteria": schema.success_criteria.model_dump_json(),
            "failure_modes": json.dumps([fm.model_dump() for fm in schema.failure_modes]),
            "abstracted_from": json.dumps(schema.abstracted_from),
            "learned_from": json.dumps(schema.learned_from),
        }
        if schema.embedding_id:
            schema_props["embedding_id"] = schema.embedding_id

        self._hcg.add_node(
            name=schema.name,
            node_type="Schema",
            uuid=schema.schema_id,
            properties=schema_props,
            source="procedural_memory",
            derivation="observed",
            confidence=schema.success_rate if schema.execution_count > 0 else None,
        )

        # 2. Create SchemaStep nodes and HAS_STEP edges
        step_uuids = []
        for i, step in enumerate(schema.steps):
            step_props = {
                "action_name": step.action.name,
                "parameters": json.dumps(step.parameters),
                "spatial_intent": step.spatial_intent.model_dump_json(),
                "expected_state_change": step.expected_state_change.model_dump_json(),
                "expected_sensory": step.expected_sensory.model_dump_json(),
                "confidence": step.confidence,
            }
            self._hcg.add_node(
                name=step.step_id,
                node_type="SchemaStep",
                uuid=step.step_id,
                properties=step_props,
                source="procedural_memory",
                derivation="observed",
            )
            step_uuids.append(step.step_id)

            # HAS_STEP edge from schema to step
            self._hcg.add_edge(
                source_uuid=schema.schema_id,
                target_uuid=step.step_id,
                relation="HAS_STEP",
                properties={"order": i},
            )

        # 3. ENABLES edges between consecutive steps
        for i in range(len(step_uuids) - 1):
            self._hcg.add_edge(
                source_uuid=step_uuids[i],
                target_uuid=step_uuids[i + 1],
                relation="ENABLES",
            )

        return schema.schema_id

    def load(self, schema_id: str) -> Optional[ActionSchema]:
        """Load a schema from the HCG by ID. Returns None if not found."""
        node = self._hcg.get_node(schema_id)
        if node is None:
            return None

        props = node.get("properties", {})

        # Load steps via HAS_STEP edges
        has_step_edges = self._hcg.list_all_edges(
            relation_type="HAS_STEP",
            source_uuid=schema_id,
        )
        # Sort by order
        has_step_edges.sort(key=lambda e: e.get("properties", {}).get("order", 0))

        steps = []
        for edge in has_step_edges:
            step_node = self._hcg.get_node(edge["target"])
            if step_node is None:
                continue
            step_props = step_node.get("properties", {})
            steps.append(
                SchemaStep(
                    step_id=step_node["uuid"],
                    action=ActionPrimitive(name=step_props.get("action_name", "")),
                    parameters=json.loads(step_props.get("parameters", "{}")),
                    spatial_intent=SpatialIntent(**json.loads(step_props.get("spatial_intent", "{}"))),
                    expected_state_change=StateDelta(**json.loads(step_props.get("expected_state_change", '{"changes": {}}'))),
                    expected_sensory=SensoryPattern(**json.loads(step_props.get("expected_sensory", '{"patterns": {}}'))),
                    confidence=step_props.get("confidence", 1.0),
                )
            )

        trigger_conditions = TriggerConditions(**json.loads(props.get("trigger_conditions", '{"required_entities": [], "required_state": {"conditions": {}}}')))
        success_criteria = StatePattern(**json.loads(props.get("success_criteria", '{"conditions": {}}')))
        failure_modes_raw = json.loads(props.get("failure_modes", "[]"))
        failure_modes = [FailurePattern(**fm) for fm in failure_modes_raw]

        return ActionSchema(
            schema_id=node["uuid"],
            name=node["name"],
            steps=steps,
            trigger_conditions=trigger_conditions,
            success_criteria=success_criteria,
            failure_modes=failure_modes,
            source=props.get("schema_source", "unknown"),
            execution_count=props.get("execution_count", 0),
            success_rate=props.get("success_rate", 0.0),
            abstracted_from=json.loads(props.get("abstracted_from", "[]")),
            learned_from=json.loads(props.get("learned_from", "[]")),
            embedding_id=props.get("embedding_id"),
        )
```

### Step 4: Run tests to verify they pass

Run: `cd /Users/cdaly/projects/LOGOS/sophia && python -m pytest tests/unit/procedural/test_hcg_persistence.py -v`
Expected: All tests PASS

### Step 5: Commit

```bash
cd /Users/cdaly/projects/LOGOS/sophia
git add src/sophia/procedural/hcg_persistence.py tests/unit/procedural/test_hcg_persistence.py
git commit -m "feat(procedural): add HCG persistence for schemas and steps"
```

---

## Task 4: Prediction Evaluator

**Files:**
- Create: `src/sophia/procedural/evaluator.py`
- Create: `tests/unit/procedural/test_evaluator.py`

### Step 1: Write the failing tests

```python
# tests/unit/procedural/test_evaluator.py
"""Tests for PredictionEvaluator."""

import pytest
from sophia.procedural.models import StateDelta, SensoryPattern, PredictionError
from sophia.procedural.evaluator import PredictionEvaluator


@pytest.fixture()
def evaluator():
    return PredictionEvaluator(error_threshold=0.3)


class TestComputeError:
    def test_identical_states_zero_error(self, evaluator):
        predicted = StateDelta(changes={"pos": "near", "holding": True})
        observed = StateDelta(changes={"pos": "near", "holding": True})
        error = evaluator.compute_error("s0", predicted, observed)
        assert error.magnitude == 0.0

    def test_completely_different_states(self, evaluator):
        predicted = StateDelta(changes={"pos": "near", "holding": True})
        observed = StateDelta(changes={"pos": "far", "holding": False})
        error = evaluator.compute_error("s0", predicted, observed)
        assert error.magnitude == 1.0

    def test_partial_mismatch(self, evaluator):
        predicted = StateDelta(changes={"pos": "near", "holding": True, "force": 0.5})
        observed = StateDelta(changes={"pos": "near", "holding": False, "force": 0.5})
        error = evaluator.compute_error("s0", predicted, observed)
        assert 0.0 < error.magnitude < 1.0

    def test_extra_observed_keys_count_as_partial_mismatch(self, evaluator):
        predicted = StateDelta(changes={"pos": "near"})
        observed = StateDelta(changes={"pos": "near", "unexpected": True})
        error = evaluator.compute_error("s0", predicted, observed)
        assert error.magnitude > 0.0  # unexpected field is a partial surprise

    def test_missing_observed_keys_count_as_mismatch(self, evaluator):
        predicted = StateDelta(changes={"pos": "near", "holding": True})
        observed = StateDelta(changes={"pos": "near"})
        error = evaluator.compute_error("s0", predicted, observed)
        assert error.magnitude > 0.0  # missing expected field

    def test_empty_states(self, evaluator):
        predicted = StateDelta(changes={})
        observed = StateDelta(changes={})
        error = evaluator.compute_error("s0", predicted, observed)
        assert error.magnitude == 0.0


class TestIsSignificant:
    def test_below_threshold(self, evaluator):
        error = PredictionError(
            step_id="s0",
            predicted=StateDelta(changes={}),
            observed=StateDelta(changes={}),
            magnitude=0.1,
        )
        assert not evaluator.is_significant(error)

    def test_above_threshold(self, evaluator):
        error = PredictionError(
            step_id="s0",
            predicted=StateDelta(changes={}),
            observed=StateDelta(changes={}),
            magnitude=0.5,
        )
        assert evaluator.is_significant(error)

    def test_at_threshold(self, evaluator):
        error = PredictionError(
            step_id="s0",
            predicted=StateDelta(changes={}),
            observed=StateDelta(changes={}),
            magnitude=0.3,
        )
        assert evaluator.is_significant(error)


class TestComputeSensoryError:
    def test_matching_patterns(self, evaluator):
        predicted = SensoryPattern(patterns={"proximity": "close", "force": 0.5})
        observed = SensoryPattern(patterns={"proximity": "close", "force": 0.5})
        error = evaluator.compute_sensory_error("s0", predicted, observed)
        assert error.magnitude == 0.0

    def test_mismatched_patterns(self, evaluator):
        predicted = SensoryPattern(patterns={"proximity": "close"})
        observed = SensoryPattern(patterns={"proximity": "far"})
        error = evaluator.compute_sensory_error("s0", predicted, observed)
        assert error.magnitude > 0.0
```

### Step 2: Run tests to verify they fail

Run: `cd /Users/cdaly/projects/LOGOS/sophia && python -m pytest tests/unit/procedural/test_evaluator.py -v`
Expected: FAIL — `ModuleNotFoundError`

### Step 3: Write minimal implementation

```python
# src/sophia/procedural/evaluator.py
"""Prediction Evaluator — compares expected vs. observed state.

Produces PredictionError objects with a magnitude indicating how wrong
the prediction was. Uses a simple symmetric-difference metric over
state/sensory dictionaries.
"""

from __future__ import annotations

from sophia.procedural.models import PredictionError, SensoryPattern, StateDelta


class PredictionEvaluator:
    """Compares predicted state/sensory patterns against observations."""

    def __init__(self, error_threshold: float = 0.3) -> None:
        self._threshold = error_threshold

    def compute_error(
        self,
        step_id: str,
        predicted: StateDelta,
        observed: StateDelta,
    ) -> PredictionError:
        """Compute prediction error between predicted and observed state changes."""
        magnitude = self._dict_divergence(predicted.changes, observed.changes)
        return PredictionError(
            step_id=step_id,
            predicted=predicted,
            observed=observed,
            magnitude=magnitude,
        )

    def compute_sensory_error(
        self,
        step_id: str,
        predicted: SensoryPattern,
        observed: SensoryPattern,
    ) -> PredictionError:
        """Compute prediction error between predicted and observed sensory patterns."""
        magnitude = self._dict_divergence(predicted.patterns, observed.patterns)
        return PredictionError(
            step_id=step_id,
            predicted=StateDelta(changes=predicted.patterns),
            observed=StateDelta(changes=observed.patterns),
            magnitude=magnitude,
        )

    def is_significant(self, error: PredictionError) -> bool:
        """Is this error above the significance threshold?"""
        return error.magnitude >= self._threshold

    @staticmethod
    def _dict_divergence(expected: dict, actual: dict) -> float:
        """Compute a 0-1 divergence score between two flat dicts.

        0 = identical, 1 = completely different.
        """
        all_keys = set(expected.keys()) | set(actual.keys())
        if not all_keys:
            return 0.0

        mismatches = 0
        for key in all_keys:
            if key not in expected or key not in actual:
                mismatches += 1
            elif expected[key] != actual[key]:
                mismatches += 1

        return mismatches / len(all_keys)
```

### Step 4: Run tests to verify they pass

Run: `cd /Users/cdaly/projects/LOGOS/sophia && python -m pytest tests/unit/procedural/test_evaluator.py -v`
Expected: All tests PASS

### Step 5: Commit

```bash
cd /Users/cdaly/projects/LOGOS/sophia
git add src/sophia/procedural/evaluator.py tests/unit/procedural/test_evaluator.py
git commit -m "feat(procedural): add PredictionEvaluator for state/sensory comparison"
```

---

## Task 5: Rehearsal Loop

**Files:**
- Create: `src/sophia/procedural/rehearsal.py`
- Create: `tests/unit/procedural/test_rehearsal.py`

**Context:** The rehearsal loop drives Talos identically for thinking and doing. It iterates through schema steps, sends actions, reads observations, computes prediction errors. Talos is abstracted as a `TalosInterface` protocol — the caller provides a real or stub implementation.

### Step 1: Write the failing tests

```python
# tests/unit/procedural/test_rehearsal.py
"""Tests for the RehearsalLoop."""

from typing import Any, Dict
import pytest
from sophia.procedural.models import (
    ActionPrimitive,
    ActionSchema,
    SchemaStep,
    StateDelta,
    SensoryPattern,
    StatePattern,
)
from sophia.procedural.evaluator import PredictionEvaluator
from sophia.procedural.rehearsal import RehearsalLoop, TalosInterface


class StubTalos:
    """Test stub implementing TalosInterface."""

    def __init__(self, observations: list[Dict[str, Any]] | None = None):
        self._observations = observations or []
        self._step = 0
        self.commands_received: list[Dict[str, Any]] = []

    def send_action(self, action: Dict[str, Any]) -> None:
        self.commands_received.append(action)

    def read_observation(self) -> Dict[str, Any]:
        if self._step < len(self._observations):
            obs = self._observations[self._step]
            self._step += 1
            return obs
        return {}


@pytest.fixture()
def evaluator():
    return PredictionEvaluator(error_threshold=0.3)


@pytest.fixture()
def two_step_schema():
    return ActionSchema(
        schema_id="test_001",
        name="reach_and_grasp",
        steps=[
            SchemaStep(
                step_id="s0",
                action=ActionPrimitive(name="MOVE"),
                parameters={"target": "object.position"},
                expected_state_change=StateDelta(changes={"pos": "near"}),
                expected_sensory=SensoryPattern(patterns={"proximity": "close"}),
                confidence=0.95,
            ),
            SchemaStep(
                step_id="s1",
                action=ActionPrimitive(name="GRASP"),
                parameters={"target": "object"},
                expected_state_change=StateDelta(changes={"holding": True}),
                expected_sensory=SensoryPattern(patterns={"force": "contact"}),
                confidence=0.9,
            ),
        ],
        success_criteria=StatePattern(conditions={"holding": True}),
    )


class TestRehearsalSuccess:
    def test_successful_rehearsal(self, evaluator, two_step_schema):
        talos = StubTalos(observations=[
            {"state": {"pos": "near"}, "sensory": {"proximity": "close"}},
            {"state": {"holding": True}, "sensory": {"force": "contact"}},
        ])
        loop = RehearsalLoop(evaluator=evaluator, talos=talos)
        result = loop.run(two_step_schema)

        assert result.success
        assert result.steps_completed == 2
        assert result.steps_total == 2
        assert result.overall_confidence > 0.0
        assert len(result.prediction_errors) == 0

    def test_sends_correct_commands(self, evaluator, two_step_schema):
        talos = StubTalos(observations=[
            {"state": {"pos": "near"}, "sensory": {"proximity": "close"}},
            {"state": {"holding": True}, "sensory": {"force": "contact"}},
        ])
        loop = RehearsalLoop(evaluator=evaluator, talos=talos)
        loop.run(two_step_schema)

        assert len(talos.commands_received) == 2
        assert talos.commands_received[0]["action"] == "MOVE"
        assert talos.commands_received[1]["action"] == "GRASP"


class TestRehearsalFailure:
    def test_prediction_error_recorded(self, evaluator, two_step_schema):
        talos = StubTalos(observations=[
            {"state": {"pos": "far"}, "sensory": {"proximity": "far"}},  # wrong!
            {"state": {"holding": True}, "sensory": {"force": "contact"}},
        ])
        loop = RehearsalLoop(evaluator=evaluator, talos=talos)
        result = loop.run(two_step_schema)

        # Step 0 has a significant error
        significant_errors = [e for e in result.prediction_errors if e.magnitude >= 0.3]
        assert len(significant_errors) >= 1
        assert significant_errors[0].step_id == "s0"

    def test_failure_when_success_criteria_not_met(self, evaluator, two_step_schema):
        talos = StubTalos(observations=[
            {"state": {"pos": "near"}, "sensory": {"proximity": "close"}},
            {"state": {"holding": False}, "sensory": {"force": "none"}},  # didn't grasp
        ])
        loop = RehearsalLoop(evaluator=evaluator, talos=talos)
        result = loop.run(two_step_schema)

        assert not result.success


class TestRehearsalEdgeCases:
    def test_empty_schema(self, evaluator):
        schema = ActionSchema(name="empty", steps=[])
        talos = StubTalos()
        loop = RehearsalLoop(evaluator=evaluator, talos=talos)
        result = loop.run(schema)

        assert result.success
        assert result.steps_completed == 0
        assert result.steps_total == 0

    def test_imagined_flag_default_true(self, evaluator, two_step_schema):
        talos = StubTalos(observations=[
            {"state": {"pos": "near"}, "sensory": {"proximity": "close"}},
            {"state": {"holding": True}, "sensory": {"force": "contact"}},
        ])
        loop = RehearsalLoop(evaluator=evaluator, talos=talos)
        result = loop.run(two_step_schema)
        assert result.imagined is True

    def test_imagined_flag_can_be_false(self, evaluator, two_step_schema):
        talos = StubTalos(observations=[
            {"state": {"pos": "near"}, "sensory": {"proximity": "close"}},
            {"state": {"holding": True}, "sensory": {"force": "contact"}},
        ])
        loop = RehearsalLoop(evaluator=evaluator, talos=talos, imagined=False)
        result = loop.run(two_step_schema)
        assert result.imagined is False
```

### Step 2: Run tests to verify they fail

Run: `cd /Users/cdaly/projects/LOGOS/sophia && python -m pytest tests/unit/procedural/test_rehearsal.py -v`
Expected: FAIL — `ModuleNotFoundError`

### Step 3: Write minimal implementation

```python
# src/sophia/procedural/rehearsal.py
"""Rehearsal Loop — drives Talos the same way for thinking and doing.

Iterates through schema steps, sends actions via TalosInterface,
reads observations, computes prediction errors.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable

from sophia.procedural.evaluator import PredictionEvaluator
from sophia.procedural.models import (
    ActionSchema,
    PredictionError,
    RehearsalResult,
    SensoryPattern,
    StateDelta,
)


@runtime_checkable
class TalosInterface(Protocol):
    """Talos sensor/motor bus interface.

    Same API regardless of backend (real hardware, Gazebo, stub).
    """

    def send_action(self, action: Dict[str, Any]) -> None:
        """Send a motor command to Talos."""
        ...

    def read_observation(self) -> Dict[str, Any]:
        """Read current sensory observation from Talos.

        Returns dict with 'state' and 'sensory' keys.
        """
        ...


class RehearsalLoop:
    """Orchestrates schema execution step-by-step through Talos."""

    def __init__(
        self,
        evaluator: PredictionEvaluator,
        talos: TalosInterface,
        imagined: bool = True,
    ) -> None:
        self._evaluator = evaluator
        self._talos = talos
        self._imagined = imagined

    def run(self, schema: ActionSchema) -> RehearsalResult:
        """Run a schema through Talos, collecting prediction errors."""
        errors: List[PredictionError] = []
        steps_completed = 0
        final_state: Dict[str, Any] = {}

        for step in schema.steps:
            # PREDICT + ACT
            command = {
                "action": step.action.name,
                "parameters": step.parameters,
            }
            self._talos.send_action(command)

            # OBSERVE
            observation = self._talos.read_observation()
            observed_state = observation.get("state", {})
            observed_sensory = observation.get("sensory", {})
            final_state = observed_state

            # COMPARE state
            state_error = self._evaluator.compute_error(
                step.step_id,
                step.expected_state_change,
                StateDelta(changes=observed_state),
            )
            if self._evaluator.is_significant(state_error):
                errors.append(state_error)

            # COMPARE sensory
            sensory_error = self._evaluator.compute_sensory_error(
                step.step_id,
                step.expected_sensory,
                SensoryPattern(patterns=observed_sensory),
            )
            if self._evaluator.is_significant(sensory_error):
                errors.append(sensory_error)

            steps_completed += 1

        # EVALUATE success criteria
        success = self._check_success(schema, final_state)

        # Compute overall confidence
        if schema.steps:
            step_confidences = [s.confidence for s in schema.steps]
            error_penalty = sum(e.magnitude for e in errors) / len(schema.steps) if errors else 0.0
            overall = (sum(step_confidences) / len(step_confidences)) - error_penalty
            overall_confidence = max(0.0, min(1.0, overall))
        else:
            overall_confidence = 1.0

        return RehearsalResult(
            schema_id=schema.schema_id,
            success=success,
            steps_completed=steps_completed,
            steps_total=len(schema.steps),
            prediction_errors=errors,
            overall_confidence=overall_confidence,
            imagined=self._imagined,
        )

    @staticmethod
    def _check_success(schema: ActionSchema, final_state: Dict[str, Any]) -> bool:
        """Check if final state satisfies the schema's success criteria."""
        if not schema.success_criteria.conditions:
            return True  # No criteria = success by default
        for key, expected in schema.success_criteria.conditions.items():
            if final_state.get(key) != expected:
                return False
        return True
```

### Step 4: Run tests to verify they pass

Run: `cd /Users/cdaly/projects/LOGOS/sophia && python -m pytest tests/unit/procedural/test_rehearsal.py -v`
Expected: All tests PASS

### Step 5: Commit

```bash
cd /Users/cdaly/projects/LOGOS/sophia
git add src/sophia/procedural/rehearsal.py tests/unit/procedural/test_rehearsal.py
git commit -m "feat(procedural): add RehearsalLoop with TalosInterface protocol"
```

---

## Task 6: Video Segmentation

**Files:**
- Create: `src/sophia/procedural/video_extraction.py`
- Create: `tests/unit/procedural/test_video_extraction.py`

**Context:** Video segmentation uses temporal embedding clustering. JEPA generates per-frame embeddings; we detect action boundaries where consecutive embeddings diverge. This task builds the segmentation + schema construction pipeline.

### Step 1: Write the failing tests

```python
# tests/unit/procedural/test_video_extraction.py
"""Tests for video → schema extraction pipeline."""

import pytest
import numpy as np
from sophia.procedural.video_extraction import (
    VideoSegment,
    segment_by_embedding_distance,
    classify_segment,
    extract_schema_from_segments,
)
from sophia.procedural.models import ActionSchema


class TestSegmentByEmbeddingDistance:
    def test_detects_boundary_at_large_distance(self):
        # 10 frames: first 5 similar, then shift, last 5 similar
        embeddings = []
        for i in range(5):
            embeddings.append([0.1, 0.1, 0.1])  # cluster A
        for i in range(5):
            embeddings.append([0.9, 0.9, 0.9])  # cluster B

        segments = segment_by_embedding_distance(embeddings, threshold=0.5)
        assert len(segments) == 2
        assert segments[0].start_frame == 0
        assert segments[0].end_frame == 4
        assert segments[1].start_frame == 5
        assert segments[1].end_frame == 9

    def test_single_segment_when_uniform(self):
        embeddings = [[0.1, 0.1, 0.1]] * 10
        segments = segment_by_embedding_distance(embeddings, threshold=0.5)
        assert len(segments) == 1

    def test_empty_embeddings(self):
        segments = segment_by_embedding_distance([], threshold=0.5)
        assert segments == []

    def test_single_frame(self):
        segments = segment_by_embedding_distance([[0.1, 0.2]], threshold=0.5)
        assert len(segments) == 1

    def test_multiple_boundaries(self):
        embeddings = (
            [[0.0, 0.0]] * 3 +  # segment 0
            [[1.0, 1.0]] * 3 +  # segment 1
            [[0.0, 0.0]] * 3    # segment 2
        )
        segments = segment_by_embedding_distance(embeddings, threshold=0.5)
        assert len(segments) == 3


class TestClassifySegment:
    def test_classify_returns_action_type(self):
        # With known prototypes
        prototypes = {
            "REACH": [0.1, 0.0, 0.0],
            "GRASP": [0.0, 0.1, 0.0],
            "LIFT": [0.0, 0.0, 0.1],
        }
        segment_embedding = [0.09, 0.01, 0.01]  # closest to REACH
        action_type = classify_segment(segment_embedding, prototypes)
        assert action_type == "REACH"

    def test_classify_no_prototypes_returns_unknown(self):
        action_type = classify_segment([0.1, 0.2], {})
        assert action_type == "UNKNOWN"


class TestExtractSchemaFromSegments:
    def test_builds_schema_from_segments(self):
        segments = [
            VideoSegment(start_frame=0, end_frame=4, action_type="MOVE", embedding=[0.1, 0.1]),
            VideoSegment(start_frame=5, end_frame=9, action_type="GRASP", embedding=[0.2, 0.2]),
        ]
        schema = extract_schema_from_segments(
            segments=segments,
            name="learned_reach_grasp",
            media_sample_id="vid_001",
        )

        assert isinstance(schema, ActionSchema)
        assert schema.name == "learned_reach_grasp"
        assert len(schema.steps) == 2
        assert schema.steps[0].action.name == "MOVE"
        assert schema.steps[1].action.name == "GRASP"
        assert schema.source == "video_observation"
        assert "vid_001" in schema.learned_from

    def test_empty_segments_produces_empty_schema(self):
        schema = extract_schema_from_segments([], name="empty", media_sample_id="vid_002")
        assert len(schema.steps) == 0
```

### Step 2: Run tests to verify they fail

Run: `cd /Users/cdaly/projects/LOGOS/sophia && python -m pytest tests/unit/procedural/test_video_extraction.py -v`
Expected: FAIL — `ModuleNotFoundError`

### Step 3: Write minimal implementation

```python
# src/sophia/procedural/video_extraction.py
"""Video → Schema extraction pipeline.

Segments video by temporal embedding distance, classifies segments
into action types, and constructs ActionSchemas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from sophia.procedural.models import ActionPrimitive, ActionSchema, SchemaStep


@dataclass
class VideoSegment:
    """A temporal segment of video corresponding to one action."""

    start_frame: int
    end_frame: int
    action_type: str = "UNKNOWN"
    embedding: List[float] = field(default_factory=list)


def _euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two vectors."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def segment_by_embedding_distance(
    embeddings: List[List[float]],
    threshold: float = 0.5,
) -> List[VideoSegment]:
    """Segment a sequence of frame embeddings by detecting large jumps.

    When consecutive frame embeddings differ by more than threshold (L2 distance),
    a new segment boundary is placed.
    """
    if not embeddings:
        return []

    segments: List[VideoSegment] = []
    current_start = 0

    for i in range(1, len(embeddings)):
        dist = _euclidean_distance(embeddings[i - 1], embeddings[i])
        if dist > threshold:
            # Compute mean embedding for the segment
            segment_frames = embeddings[current_start:i]
            mean_emb = [
                sum(frame[d] for frame in segment_frames) / len(segment_frames)
                for d in range(len(segment_frames[0]))
            ]
            segments.append(
                VideoSegment(
                    start_frame=current_start,
                    end_frame=i - 1,
                    embedding=mean_emb,
                )
            )
            current_start = i

    # Final segment
    segment_frames = embeddings[current_start:]
    mean_emb = [
        sum(frame[d] for frame in segment_frames) / len(segment_frames)
        for d in range(len(segment_frames[0]))
    ]
    segments.append(
        VideoSegment(
            start_frame=current_start,
            end_frame=len(embeddings) - 1,
            embedding=mean_emb,
        )
    )

    return segments


def classify_segment(
    segment_embedding: List[float],
    prototypes: Dict[str, List[float]],
) -> str:
    """Classify a segment by nearest prototype embedding.

    Returns action type string, or "UNKNOWN" if no prototypes given.
    """
    if not prototypes:
        return "UNKNOWN"

    best_action = "UNKNOWN"
    best_distance = float("inf")

    for action_type, proto_emb in prototypes.items():
        dist = _euclidean_distance(segment_embedding, proto_emb)
        if dist < best_distance:
            best_distance = dist
            best_action = action_type

    return best_action


def extract_schema_from_segments(
    segments: List[VideoSegment],
    name: str,
    media_sample_id: str,
) -> ActionSchema:
    """Construct an ActionSchema from classified video segments."""
    steps = []
    for i, seg in enumerate(segments):
        steps.append(
            SchemaStep(
                step_id=f"vs_{i}",
                action=ActionPrimitive(name=seg.action_type),
                parameters={
                    "source_start_frame": seg.start_frame,
                    "source_end_frame": seg.end_frame,
                },
            )
        )

    return ActionSchema(
        name=name,
        steps=steps,
        source="video_observation",
        learned_from=[media_sample_id],
    )
```

### Step 4: Run tests to verify they pass

Run: `cd /Users/cdaly/projects/LOGOS/sophia && python -m pytest tests/unit/procedural/test_video_extraction.py -v`
Expected: All tests PASS

### Step 5: Commit

```bash
cd /Users/cdaly/projects/LOGOS/sophia
git add src/sophia/procedural/video_extraction.py tests/unit/procedural/test_video_extraction.py
git commit -m "feat(procedural): add video segmentation and schema extraction pipeline"
```

---

## Task 7: Schema Abstraction — Structural Similarity + Concept Formation

**Files:**
- Create: `src/sophia/procedural/abstraction.py`
- Create: `tests/unit/procedural/test_abstraction.py`

### Step 1: Write the failing tests

```python
# tests/unit/procedural/test_abstraction.py
"""Tests for schema abstraction and concept formation."""

import pytest
from sophia.procedural.models import (
    ActionPrimitive,
    ActionSchema,
    SchemaStep,
    StateDelta,
)
from sophia.procedural.abstraction import (
    extract_step_signature,
    compute_structural_similarity,
    find_shared_step_pattern,
    abstract_schemas,
)


@pytest.fixture()
def pick_up_cup():
    return ActionSchema(
        schema_id="pick_cup",
        name="pick_up_cup",
        steps=[
            SchemaStep(step_id="c0", action=ActionPrimitive(name="MOVE"), parameters={"target": "cup.position"}),
            SchemaStep(step_id="c1", action=ActionPrimitive(name="GRASP"), parameters={"target": "cup"}),
            SchemaStep(step_id="c2", action=ActionPrimitive(name="LIFT"), parameters={"target": "cup"}),
        ],
        source="execution",
    )


@pytest.fixture()
def pick_up_ball():
    return ActionSchema(
        schema_id="pick_ball",
        name="pick_up_ball",
        steps=[
            SchemaStep(step_id="b0", action=ActionPrimitive(name="MOVE"), parameters={"target": "ball.position"}),
            SchemaStep(step_id="b1", action=ActionPrimitive(name="GRASP"), parameters={"target": "ball"}),
            SchemaStep(step_id="b2", action=ActionPrimitive(name="LIFT"), parameters={"target": "ball"}),
        ],
        source="execution",
    )


@pytest.fixture()
def pour_water():
    return ActionSchema(
        schema_id="pour_water",
        name="pour_water",
        steps=[
            SchemaStep(step_id="p0", action=ActionPrimitive(name="MOVE"), parameters={"target": "glass.position"}),
            SchemaStep(step_id="p1", action=ActionPrimitive(name="ROTATE"), parameters={"angle": 90}),
        ],
        source="execution",
    )


class TestStepSignature:
    def test_extracts_action_sequence(self, pick_up_cup):
        sig = extract_step_signature(pick_up_cup)
        assert sig == ("MOVE", "GRASP", "LIFT")


class TestStructuralSimilarity:
    def test_identical_structure(self, pick_up_cup, pick_up_ball):
        sim = compute_structural_similarity(pick_up_cup, pick_up_ball)
        assert sim == 1.0

    def test_different_structure(self, pick_up_cup, pour_water):
        sim = compute_structural_similarity(pick_up_cup, pour_water)
        assert sim < 1.0

    def test_empty_schemas(self):
        a = ActionSchema(name="a", steps=[])
        b = ActionSchema(name="b", steps=[])
        sim = compute_structural_similarity(a, b)
        assert sim == 1.0  # Both empty = identical structure

    def test_one_empty_one_not(self, pick_up_cup):
        empty = ActionSchema(name="empty", steps=[])
        sim = compute_structural_similarity(pick_up_cup, empty)
        assert sim == 0.0


class TestFindSharedPattern:
    def test_finds_common_subsequence(self, pick_up_cup, pick_up_ball):
        pattern = find_shared_step_pattern([pick_up_cup, pick_up_ball])
        assert pattern == ("MOVE", "GRASP", "LIFT")

    def test_partial_overlap(self, pick_up_cup, pour_water):
        pattern = find_shared_step_pattern([pick_up_cup, pour_water])
        assert "MOVE" in pattern  # At least MOVE is shared

    def test_no_shared_pattern(self):
        a = ActionSchema(name="a", steps=[SchemaStep(step_id="a0", action=ActionPrimitive(name="PUSH"), parameters={})])
        b = ActionSchema(name="b", steps=[SchemaStep(step_id="b0", action=ActionPrimitive(name="PULL"), parameters={})])
        pattern = find_shared_step_pattern([a, b])
        assert pattern == ()


class TestAbstractSchemas:
    def test_creates_abstract_schema(self, pick_up_cup, pick_up_ball):
        abstract = abstract_schemas(
            schemas=[pick_up_cup, pick_up_ball],
            abstract_name="pick_up_object",
        )

        assert abstract.name == "pick_up_object"
        assert abstract.source == "abstracted"
        assert len(abstract.steps) == 3
        assert abstract.steps[0].action.name == "MOVE"
        assert abstract.steps[1].action.name == "GRASP"
        assert abstract.steps[2].action.name == "LIFT"
        assert set(abstract.abstracted_from) == {"pick_cup", "pick_ball"}

    def test_abstract_params_are_generic(self, pick_up_cup, pick_up_ball):
        abstract = abstract_schemas(
            schemas=[pick_up_cup, pick_up_ball],
            abstract_name="pick_up_object",
        )
        # Parameters should reference generic "target" not specific objects
        for step in abstract.steps:
            for value in step.parameters.values():
                assert "cup" not in str(value)
                assert "ball" not in str(value)
```

### Step 2: Run tests to verify they fail

Run: `cd /Users/cdaly/projects/LOGOS/sophia && python -m pytest tests/unit/procedural/test_abstraction.py -v`
Expected: FAIL — `ModuleNotFoundError`

### Step 3: Write minimal implementation

```python
# src/sophia/procedural/abstraction.py
"""Schema Abstraction — detect structural similarity and form abstract schemas.

Concept formation through shared procedural patterns: when multiple schemas
share the same step structure, abstract them into a generic schema and
generalize parameters.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from sophia.procedural.models import ActionPrimitive, ActionSchema, SchemaStep


def extract_step_signature(schema: ActionSchema) -> Tuple[str, ...]:
    """Extract the action-type sequence from a schema (its structural signature)."""
    return tuple(step.action.name for step in schema.steps)


def compute_structural_similarity(a: ActionSchema, b: ActionSchema) -> float:
    """Compute structural similarity between two schemas.

    Uses longest common subsequence (LCS) of action-type signatures.
    Returns 0.0 (no overlap) to 1.0 (identical structure).
    """
    sig_a = extract_step_signature(a)
    sig_b = extract_step_signature(b)

    if not sig_a and not sig_b:
        return 1.0
    if not sig_a or not sig_b:
        return 0.0

    lcs_len = _lcs_length(sig_a, sig_b)
    max_len = max(len(sig_a), len(sig_b))
    return lcs_len / max_len


def find_shared_step_pattern(schemas: List[ActionSchema]) -> Tuple[str, ...]:
    """Find the longest common action-type subsequence across all schemas."""
    if not schemas:
        return ()

    signatures = [extract_step_signature(s) for s in schemas]
    shared = signatures[0]

    for sig in signatures[1:]:
        shared = _lcs(shared, sig)
        if not shared:
            return ()

    return shared


def abstract_schemas(
    schemas: List[ActionSchema],
    abstract_name: str,
) -> ActionSchema:
    """Create an abstract schema from the shared structure of multiple schemas.

    Steps have generalized parameters (entity-specific references replaced
    with generic 'target' references).
    """
    shared_pattern = find_shared_step_pattern(schemas)

    steps = []
    for i, action_name in enumerate(shared_pattern):
        steps.append(
            SchemaStep(
                step_id=f"abs_{i}",
                action=ActionPrimitive(name=action_name),
                parameters={"target": "object"},
            )
        )

    return ActionSchema(
        name=abstract_name,
        steps=steps,
        source="abstracted",
        abstracted_from=[s.schema_id for s in schemas],
    )


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def _lcs(a: Tuple[str, ...], b: Sequence[str]) -> Tuple[str, ...]:
    """Reconstruct the longest common subsequence."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Backtrack to find the actual subsequence
    result: list[str] = []
    i, j = m, n
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            result.append(a[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return tuple(reversed(result))
```

### Step 4: Run tests to verify they pass

Run: `cd /Users/cdaly/projects/LOGOS/sophia && python -m pytest tests/unit/procedural/test_abstraction.py -v`
Expected: All tests PASS

### Step 5: Commit

```bash
cd /Users/cdaly/projects/LOGOS/sophia
git add src/sophia/procedural/abstraction.py tests/unit/procedural/test_abstraction.py
git commit -m "feat(procedural): add schema abstraction and concept formation"
```

---

## Task 8: API Endpoints for Procedural Memory

**Files:**
- Create: `src/sophia/procedural/api_models.py`
- Create: `tests/unit/procedural/test_api_models.py`
- Modify: `src/sophia/api/app.py` — add procedural memory endpoints
- Modify: `src/sophia/procedural/__init__.py` — export public API

**Context:** Add endpoints following existing FastAPI patterns in `app.py`. Sophia's app uses dependency injection via module-level singletons. Existing endpoints follow patterns like `@app.post("/simulate", response_model=SimulateResponse)`.

### Step 1: Write the failing tests for API models

```python
# tests/unit/procedural/test_api_models.py
"""Tests for procedural memory API models."""

import pytest
from sophia.procedural.api_models import (
    SchemaCreateRequest,
    SchemaStepRequest,
    SchemaResponse,
    SchemaListResponse,
    RehearsalRequest,
    RehearsalResponse,
    VideoExtractRequest,
    AbstractionRequest,
)


class TestSchemaCreateRequest:
    def test_minimal(self):
        req = SchemaCreateRequest(
            name="test_schema",
            steps=[
                SchemaStepRequest(
                    action="MOVE",
                    parameters={"target": "object.position"},
                ),
            ],
        )
        assert req.name == "test_schema"
        assert len(req.steps) == 1

    def test_with_source(self):
        req = SchemaCreateRequest(
            name="test",
            steps=[],
            source="video_observation",
        )
        assert req.source == "video_observation"


class TestSchemaResponse:
    def test_from_dict(self):
        resp = SchemaResponse(
            schema_id="s1",
            name="test",
            step_count=3,
            source="execution",
            execution_count=5,
            success_rate=0.8,
        )
        assert resp.schema_id == "s1"
        assert resp.step_count == 3


class TestRehearsalRequest:
    def test_defaults(self):
        req = RehearsalRequest(schema_id="s1")
        assert req.schema_id == "s1"
        assert req.imagined is True


class TestVideoExtractRequest:
    def test_required_fields(self):
        req = VideoExtractRequest(
            media_sample_id="vid_001",
            name="learned_grasp",
        )
        assert req.media_sample_id == "vid_001"


class TestAbstractionRequest:
    def test_required_fields(self):
        req = AbstractionRequest(
            schema_ids=["s1", "s2"],
            abstract_name="generic_pick_up",
        )
        assert len(req.schema_ids) == 2
```

### Step 2: Run tests to verify they fail

Run: `cd /Users/cdaly/projects/LOGOS/sophia && python -m pytest tests/unit/procedural/test_api_models.py -v`
Expected: FAIL — `ModuleNotFoundError`

### Step 3: Write the API models

```python
# src/sophia/procedural/api_models.py
"""API request/response models for procedural memory endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SchemaStepRequest(BaseModel):
    """Step in a schema creation request."""

    action: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    spatial_intent: Dict[str, Any] = Field(default_factory=dict)
    expected_state_change: Dict[str, Any] = Field(default_factory=dict)
    expected_sensory: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0


class SchemaCreateRequest(BaseModel):
    """Request to create an action schema."""

    name: str
    steps: List[SchemaStepRequest]
    source: str = "unknown"
    trigger_conditions: Dict[str, Any] = Field(default_factory=dict)
    success_criteria: Dict[str, Any] = Field(default_factory=dict)
    failure_modes: List[Dict[str, Any]] = Field(default_factory=list)


class SchemaResponse(BaseModel):
    """Summary response for a schema."""

    schema_id: str
    name: str
    step_count: int
    source: str
    execution_count: int = 0
    success_rate: float = 0.0


class SchemaDetailResponse(BaseModel):
    """Full detail response for a schema."""

    schema_id: str
    name: str
    steps: List[Dict[str, Any]]
    source: str
    execution_count: int = 0
    success_rate: float = 0.0
    trigger_conditions: Dict[str, Any] = Field(default_factory=dict)
    success_criteria: Dict[str, Any] = Field(default_factory=dict)
    failure_modes: List[Dict[str, Any]] = Field(default_factory=list)
    abstracted_from: List[str] = Field(default_factory=list)
    learned_from: List[str] = Field(default_factory=list)


class SchemaListResponse(BaseModel):
    """Response listing multiple schemas."""

    schemas: List[SchemaResponse]
    total: int


class RehearsalRequest(BaseModel):
    """Request to rehearse a schema."""

    schema_id: str
    imagined: bool = True


class RehearsalResponse(BaseModel):
    """Response from schema rehearsal."""

    schema_id: str
    success: bool
    steps_completed: int
    steps_total: int
    overall_confidence: float
    prediction_errors: List[Dict[str, Any]] = Field(default_factory=list)
    imagined: bool = True
    failure_mode: Optional[str] = None


class VideoExtractRequest(BaseModel):
    """Request to extract a schema from video."""

    media_sample_id: str
    name: str
    segmentation_threshold: float = 0.5


class AbstractionRequest(BaseModel):
    """Request to abstract common structure from multiple schemas."""

    schema_ids: List[str]
    abstract_name: str
```

### Step 4: Run tests to verify they pass

Run: `cd /Users/cdaly/projects/LOGOS/sophia && python -m pytest tests/unit/procedural/test_api_models.py -v`
Expected: All tests PASS

### Step 5: Update `__init__.py` exports

```python
# src/sophia/procedural/__init__.py
"""Procedural Memory — non-linguistic thought through action schemas."""

from sophia.procedural.models import (
    ActionPrimitive,
    ActionSchema,
    SchemaStep,
    SpatialIntent,
    StateDelta,
    SensoryPattern,
    EntityPattern,
    StatePattern,
    FailurePattern,
    TriggerConditions,
    PredictionError,
    RehearsalResult,
)
from sophia.procedural.memory import ProceduralMemory
from sophia.procedural.evaluator import PredictionEvaluator
from sophia.procedural.rehearsal import RehearsalLoop, TalosInterface

__all__ = [
    "ActionPrimitive",
    "ActionSchema",
    "SchemaStep",
    "SpatialIntent",
    "StateDelta",
    "SensoryPattern",
    "EntityPattern",
    "StatePattern",
    "FailurePattern",
    "TriggerConditions",
    "PredictionError",
    "RehearsalResult",
    "ProceduralMemory",
    "PredictionEvaluator",
    "RehearsalLoop",
    "TalosInterface",
]
```

### Step 6: Commit

```bash
cd /Users/cdaly/projects/LOGOS/sophia
git add src/sophia/procedural/api_models.py src/sophia/procedural/__init__.py tests/unit/procedural/test_api_models.py
git commit -m "feat(procedural): add API request/response models and public exports"
```

---

## Task 9: Wire API Endpoints into FastAPI App

**Files:**
- Modify: `src/sophia/api/app.py` — add procedural memory routes
- Create: `tests/unit/procedural/test_api_endpoints.py`

**Context:** The app.py file is large. Add a new section with procedural memory endpoints. Follow the existing pattern: module-level singleton, endpoint functions with type-hinted request/response models.

### Step 1: Write the failing tests

```python
# tests/unit/procedural/test_api_endpoints.py
"""Tests for procedural memory API endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    from sophia.api.app import app
    return TestClient(app)


class TestSchemaEndpoints:
    def test_create_schema(self, client):
        response = client.post(
            "/procedural/schemas",
            json={
                "name": "test_grasp",
                "steps": [
                    {"action": "MOVE", "parameters": {"target": "object.position"}},
                    {"action": "GRASP", "parameters": {"target": "object"}},
                ],
                "source": "execution",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "test_grasp"
        assert data["step_count"] == 2

    def test_list_schemas(self, client):
        # Create one first
        client.post(
            "/procedural/schemas",
            json={"name": "test", "steps": [], "source": "execution"},
        )
        response = client.get("/procedural/schemas")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1

    def test_get_schema_not_found(self, client):
        response = client.get("/procedural/schemas/nonexistent")
        assert response.status_code == 404

    def test_get_schema_detail(self, client):
        create_resp = client.post(
            "/procedural/schemas",
            json={"name": "detail_test", "steps": [{"action": "MOVE", "parameters": {}}]},
        )
        schema_id = create_resp.json()["schema_id"]
        response = client.get(f"/procedural/schemas/{schema_id}")
        assert response.status_code == 200
        assert response.json()["name"] == "detail_test"

    def test_delete_schema(self, client):
        create_resp = client.post(
            "/procedural/schemas",
            json={"name": "to_delete", "steps": []},
        )
        schema_id = create_resp.json()["schema_id"]
        response = client.delete(f"/procedural/schemas/{schema_id}")
        assert response.status_code == 204

    def test_delete_nonexistent(self, client):
        response = client.delete("/procedural/schemas/nonexistent")
        assert response.status_code == 404
```

### Step 2: Run tests to verify they fail

Run: `cd /Users/cdaly/projects/LOGOS/sophia && python -m pytest tests/unit/procedural/test_api_endpoints.py -v`
Expected: FAIL — 404 on `/procedural/schemas` (route doesn't exist yet)

### Step 3: Add endpoints to app.py

Add the following section to `src/sophia/api/app.py`. Insert it after the existing endpoint sections (e.g., after the simulation endpoints). The exact insertion point depends on the file structure — place it near the bottom, before any final health/status endpoints.

```python
# --- Procedural Memory Endpoints ---

from sophia.procedural import ProceduralMemory, ActionSchema, SchemaStep, ActionPrimitive
from sophia.procedural.models import SpatialIntent, StateDelta, SensoryPattern, TriggerConditions, EntityPattern, StatePattern, FailurePattern
from sophia.procedural.api_models import (
    SchemaCreateRequest,
    SchemaResponse,
    SchemaDetailResponse,
    SchemaListResponse,
)

_procedural_memory = ProceduralMemory()


@app.post("/procedural/schemas", response_model=SchemaResponse, status_code=201)
def create_schema(request: SchemaCreateRequest):
    """Create a new action schema."""
    steps = [
        SchemaStep(
            step_id=f"step_{i}",
            action=ActionPrimitive(name=s.action),
            parameters=s.parameters,
            spatial_intent=SpatialIntent(**s.spatial_intent) if s.spatial_intent else SpatialIntent(),
            expected_state_change=StateDelta(**s.expected_state_change) if s.expected_state_change else StateDelta(),
            expected_sensory=SensoryPattern(**s.expected_sensory) if s.expected_sensory else SensoryPattern(),
            confidence=s.confidence,
        )
        for i, s in enumerate(request.steps)
    ]

    schema = ActionSchema(
        name=request.name,
        steps=steps,
        source=request.source,
    )
    _procedural_memory.store(schema)

    return SchemaResponse(
        schema_id=schema.schema_id,
        name=schema.name,
        step_count=len(schema.steps),
        source=schema.source,
        execution_count=schema.execution_count,
        success_rate=schema.success_rate,
    )


@app.get("/procedural/schemas", response_model=SchemaListResponse)
def list_schemas():
    """List all stored schemas."""
    schemas = _procedural_memory.list_all()
    return SchemaListResponse(
        schemas=[
            SchemaResponse(
                schema_id=s.schema_id,
                name=s.name,
                step_count=len(s.steps),
                source=s.source,
                execution_count=s.execution_count,
                success_rate=s.success_rate,
            )
            for s in schemas
        ],
        total=len(schemas),
    )


@app.get("/procedural/schemas/{schema_id}", response_model=SchemaDetailResponse)
def get_schema(schema_id: str):
    """Get detailed schema by ID."""
    from fastapi import HTTPException

    schema = _procedural_memory.get(schema_id)
    if schema is None:
        raise HTTPException(status_code=404, detail="Schema not found")

    return SchemaDetailResponse(
        schema_id=schema.schema_id,
        name=schema.name,
        steps=[
            {
                "step_id": step.step_id,
                "action": step.action.name,
                "parameters": step.parameters,
                "confidence": step.confidence,
            }
            for step in schema.steps
        ],
        source=schema.source,
        execution_count=schema.execution_count,
        success_rate=schema.success_rate,
        abstracted_from=schema.abstracted_from,
        learned_from=schema.learned_from,
    )


@app.delete("/procedural/schemas/{schema_id}", status_code=204)
def delete_schema(schema_id: str):
    """Delete a schema by ID."""
    from fastapi import HTTPException

    if not _procedural_memory.delete(schema_id):
        raise HTTPException(status_code=404, detail="Schema not found")
```

### Step 4: Run tests to verify they pass

Run: `cd /Users/cdaly/projects/LOGOS/sophia && python -m pytest tests/unit/procedural/test_api_endpoints.py -v`
Expected: All tests PASS

### Step 5: Commit

```bash
cd /Users/cdaly/projects/LOGOS/sophia
git add src/sophia/api/app.py tests/unit/procedural/test_api_endpoints.py
git commit -m "feat(procedural): wire procedural memory CRUD endpoints into FastAPI app"
```

---

## Task 10: Run Full Test Suite + Final Commit

**Files:**
- No new files

### Step 1: Run the full procedural test suite

Run: `cd /Users/cdaly/projects/LOGOS/sophia && python -m pytest tests/unit/procedural/ -v`
Expected: All tests PASS

### Step 2: Run linting

Run: `cd /Users/cdaly/projects/LOGOS/sophia && python -m ruff check src/sophia/procedural/`
Expected: No errors (fix any that appear)

### Step 3: Run type checking (if configured)

Run: `cd /Users/cdaly/projects/LOGOS/sophia && python -m mypy src/sophia/procedural/ --ignore-missing-imports`
Expected: No errors (fix any that appear)

### Step 4: Run the broader test suite to catch regressions

Run: `cd /Users/cdaly/projects/LOGOS/sophia && python -m pytest tests/unit/ -v --timeout=60`
Expected: No regressions in existing tests

### Step 5: Final commit if any fixes were needed

```bash
cd /Users/cdaly/projects/LOGOS/sophia
git add -A
git commit -m "chore(procedural): fix lint and type issues from full suite run"
```

---

## Summary

| Task | Component | New Files |
|------|-----------|-----------|
| 1 | Core Pydantic models | `src/sophia/procedural/models.py` |
| 2 | ProceduralMemory store | `src/sophia/procedural/memory.py` |
| 3 | HCG persistence | `src/sophia/procedural/hcg_persistence.py` |
| 4 | PredictionEvaluator | `src/sophia/procedural/evaluator.py` |
| 5 | RehearsalLoop | `src/sophia/procedural/rehearsal.py` |
| 6 | Video segmentation | `src/sophia/procedural/video_extraction.py` |
| 7 | Schema abstraction | `src/sophia/procedural/abstraction.py` |
| 8 | API models | `src/sophia/procedural/api_models.py` |
| 9 | API endpoints | Modify `src/sophia/api/app.py` |
| 10 | Full suite validation | No new files |

**Total:** 8 new source files, 1 modified file, 8 test files, ~10 commits.
