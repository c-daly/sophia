# LOGOS HCG Query Utilities

Python client library for accessing the Hybrid Causal Graph (HCG) stored in Neo4j.

## Overview

This package provides foundational infrastructure for HCG operations:
- **Connection management** with pooling and automatic retry
- **Common Cypher queries** for graph traversal and operations
- **Type-safe data models** for Entity, Concept, State, and Process nodes
- **Comprehensive error handling** for production use

See Project LOGOS spec: Section 4.1 (Core Ontology and Data Model)

## Installation

The package is included in the LOGOS Foundry distribution:

```bash
pip install -e .
```

## Quick Start

### Basic Usage

```python
from logos_hcg import HCGClient

# Create a client
client = HCGClient(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="logosdev"
)

# Find entities
entities = client.find_all_entities(limit=10)
for entity in entities:
    print(f"Entity: {entity.name} ({entity.uuid})")

# Get entity type
entity_type = client.get_entity_type(entities[0].uuid)
print(f"Type: {entity_type.name}")

# Close the client
client.close()
```

### Context Manager

```python
from logos_hcg import HCGClient

with HCGClient(uri="bolt://localhost:7687", user="neo4j", password="logosdev") as client:
    # Find concepts
    concepts = client.find_all_concepts()
    for concept in concepts:
        print(f"Concept: {concept.name}")
```

## Core Operations

### Entity Operations

```python
# Find entity by UUID
entity = client.find_entity_by_uuid("uuid-here")

# Find entities by name (partial match)
entities = client.find_entities_by_name("robot")

# Get all entities with pagination
entities = client.find_all_entities(skip=0, limit=100)
```

### Concept Operations

```python
# Find concept by name
concept = client.find_concept_by_name("Manipulator")

# Get all concepts
concepts = client.find_all_concepts()
```

### State Operations

```python
from datetime import datetime, timedelta

# Find states in time range
now = datetime.now()
past = now - timedelta(days=1)
states = client.find_states_by_timestamp_range(past, now)

# Get entity's current state
state = client.get_entity_current_state(entity_uuid)

# Get all entity states
states = client.get_entity_states(entity_uuid)
```

### Process Operations

```python
# Find processes in time range
processes = client.find_processes_by_time_range(start_time, end_time)

# Get states caused by a process
caused_states = client.get_process_causes(process_uuid)

# Get process preconditions
required_states = client.get_process_requirements(process_uuid)
```

### Relationship Queries

```python
# Get entity type (IS_A relationship)
concept = client.get_entity_type(entity_uuid)

# Get entity parts (PART_OF relationship)
parts = client.get_entity_parts(entity_uuid)

# Get entity parent
parent = client.get_entity_parent(entity_uuid)
```

### Causal Traversal

```python
# Traverse causality forward from a state
results = client.traverse_causality_forward(
    state_uuid="state-uuid",
    max_depth=5
)
for result in results:
    process = result["process"]
    state = result["state"]
    depth = result["depth"]
    print(f"Depth {depth}: Process {process.name} caused State {state.uuid}")

# Traverse causality backward
results = client.traverse_causality_backward(
    state_uuid="state-uuid",
    max_depth=5
)
```

## Configuration

### Connection Pooling

```python
client = HCGClient(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="logosdev",
    max_connection_pool_size=50,        # Max connections in pool
    max_connection_lifetime=3600,       # Connection lifetime in seconds
    connection_acquisition_timeout=60,  # Timeout for acquiring connection
    max_retry_attempts=3,               # Retry attempts for transient errors
)
```

### Environment Variables

Tests can be configured via environment variables:

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="logosdev"
```

## Data Models

The package provides Pydantic models for all HCG node types:

### Entity

```python
from logos_hcg import Entity

entity = Entity(
    uuid="entity-uuid",
    name="Robot Arm",
    description="6-DOF manipulator",
    # Spatial properties
    width=0.5,
    height=1.2,
    depth=0.3,
)
```

### Concept

```python
from logos_hcg import Concept

concept = Concept(
    uuid="concept-uuid",
    name="Manipulator",
    description="Abstract concept of robotic manipulator"
)
```

### State

```python
from logos_hcg import State
from datetime import datetime

state = State(
    uuid="state-uuid",
    timestamp=datetime.now(),
    position_x=1.0,
    position_y=2.0,
    position_z=0.5,
    is_grasped=True,
)
```

### Process

```python
from logos_hcg import Process
from datetime import datetime

process = Process(
    uuid="process-uuid",
    start_time=datetime.now(),
    name="Grasp",
    duration_ms=500,
)
```

## Error Handling

The client provides comprehensive error handling:

```python
from logos_hcg import HCGClient
from logos_hcg.client import HCGConnectionError, HCGQueryError

try:
    client = HCGClient(uri="bolt://localhost:7687", user="neo4j", password="wrong")
except HCGConnectionError as e:
    print(f"Connection failed: {e}")

try:
    with HCGClient(...) as client:
        entity = client.find_entity_by_uuid("some-uuid")
except HCGQueryError as e:
    print(f"Query failed: {e}")
```

The client automatically retries transient errors (network issues, deadlocks, etc.) up to `max_retry_attempts` times.

## Testing

Integration tests require a running Neo4j instance with the core ontology loaded:

```bash
# Start Neo4j
cd infra
docker compose -f docker-compose.hcg.dev.yml up -d

# Load core ontology
cd ../ontology
cat core_ontology.cypher | docker exec -i logos-hcg-neo4j cypher-shell -u neo4j -p logosdev

# Run tests
cd ..
pytest tests/infra/test_hcg_client.py -v
```

Tests are automatically skipped if Neo4j is not available.

## Architecture

### Connection Management

- Uses Neo4j Python driver's built-in connection pooling
- Automatically manages connection lifecycle
- Supports context manager pattern for resource cleanup

### Query Execution

- All queries use parameterized Cypher to prevent injection
- Automatic retry on transient errors (network issues, deadlocks)
- Type-safe result parsing with Pydantic models

### Relationship Types

The client supports all HCG relationship types:

- `IS_A` - Entity to Concept (type membership)
- `HAS_STATE` - Entity to State (temporal snapshots)
- `CAUSES` - Process to State (causal effects)
- `PART_OF` - Entity to Entity (compositional hierarchy)
- `LOCATED_AT` - Entity to Entity (spatial relationships)
- `ATTACHED_TO` - Entity to Entity (physical attachment)
- `PRECEDES` - State to State (temporal ordering)
- `REQUIRES` - Process to State (preconditions)

## See Also

- [Core Ontology](../ontology/core_ontology.cypher) - HCG schema definition
- [SHACL Shapes](../ontology/shacl_shapes.ttl) - Validation constraints
- [Project LOGOS Spec](../docs/spec/project_logos_full.md) - Full specification
