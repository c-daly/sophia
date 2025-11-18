# Test Data

This directory contains test data and utilities for Sophia functional tests.

## Files

### test_data_pick_and_place.cypher
Cypher-like representation of the Hierarchical Cognitive Graph (HCG) for a simple pick-and-place robotics scenario. This data structure includes:
- Spatial entities (table, bin)
- Objects (red block, blue block)
- Action primitives (MOVE, GRASP, RELEASE)
- Causal relationships between actions
- Goal definitions

### __init__.py
Python utilities for loading test scenarios into Sophia's knowledge graph:
- `load_pick_and_place_scenario()`: Loads the pick-and-place HCG into a KnowledgeGraph
- `get_initial_state()`: Returns the initial state for the pick-and-place scenario

## Usage

```python
from tests.data import load_pick_and_place_scenario, get_initial_state
from sophia.planner import Planner

# Load scenario into knowledge graph
kg = load_pick_and_place_scenario()

# Create planner with the scenario
planner = Planner(knowledge_graph=kg)

# Initialize state
planner.update_state(get_initial_state())

# Plan for goal
goal = {"description": "red block in bin", "target_state": "red_block_in_bin"}
plan = planner.plan(goal)
```

## Test Coverage

These test data are used in functional tests located in:
- `tests/test_plan_api_pick_and_place.py`
