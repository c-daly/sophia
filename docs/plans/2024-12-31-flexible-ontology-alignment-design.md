# Sophia Flexible Ontology Alignment Design

## Overview

Migrate sophia's HCGClient and related code to use the logos flexible ontology standard properties: `uuid`, `name`, `is_type_definition`, `type`, `ancestors`. This ensures sophia nodes are fully compatible with logos ontology patterns.

## Approach

**Selected: Rename `id` → `uuid` and add missing properties**

Update sophia to use `uuid` (matching logos standard) instead of `id`, and add missing required properties (`name`, `is_type_definition`, `ancestors`).

**Rationale:**
- logos uses `uuid` as the Neo4j property name; sophia currently uses `id`
- sophia's CWMPersistence already uses HCGQueries which use `uuid`
- Full alignment ensures consistency within sophia and with logos
- Deprecated `add_node_legacy()` allows gradual migration of callers

## Components

### 1. HCGClient.add_node() - Updated Method

**Location:** `src/sophia/hcg_client/client.py`

**Responsibility:** Create/update Neo4j nodes with logos-standard properties.

**New Interface:**
```python
def add_node(
    self,
    uuid: str,
    name: str,
    node_type: str,
    ancestors: list[str],
    is_type_definition: bool = False,
    properties: dict[str, Any] | None = None,
) -> str:
    """Create or update a node with logos-standard properties.

    Args:
        uuid: Unique identifier (e.g., "entity-123", UUID string)
        name: Human-readable name (e.g., "red_block", "pick_action")
        node_type: Semantic type (e.g., "object", "action", "location")
        ancestors: Type inheritance chain (e.g., ["physical_entity", "entity"])
        is_type_definition: True if this node defines a type, False for instances
        properties: Additional custom properties

    Returns:
        The uuid of the created/updated node
    """
```

**Dependencies:** neo4j driver, SHACLValidator

### 2. HCGClient.add_node() - Deprecated Bridge

**Location:** `src/sophia/hcg_client/client.py`

**Responsibility:** Backward-compatible wrapper during migration.

**Interface:**
```python
def add_node_legacy(
    self,
    node_id: str,
    node_type: str,
    properties: dict[str, Any] | None = None,
) -> str:
    """DEPRECATED: Use add_node() with logos-standard properties.

    Maps old signature to new:
    - node_id → uuid
    - name = node_id (fallback)
    - ancestors = [] (empty, caller should provide)
    - is_type_definition = False
    """
```

### 3. Updated Cypher Query

**Current:**
```cypher
MERGE (n:Node {id: $id})
SET n.type = $type
SET n += $properties
RETURN n.id as id
```

**New:**
```cypher
MERGE (n:Node {uuid: $uuid})
SET n.name = $name,
    n.type = $type,
    n.is_type_definition = $is_type_definition,
    n.ancestors = $ancestors
SET n += $properties
RETURN n.uuid as uuid
```

### 4. Node Class (Knowledge Graph)

**Location:** `src/sophia/knowledge_graph/node.py`

**Current:**
```python
class Node(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)
```

**New:**
```python
class Node(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    type: str
    is_type_definition: bool = False
    ancestors: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)

    # Backward compat alias
    @property
    def id(self) -> str:
        return self.uuid
```

### 5. Seeder Updates

**Location:** `src/sophia/hcg_client/seeder.py`

**Responsibility:** Seed test data with logos-standard properties.

**Example Change:**
```python
# Before
hcg_client.add_node(
    node_id="table",
    node_type="location",
    properties={"name": "Table", "location_type": "surface"},
)

# After
hcg_client.add_node(
    uuid="table",
    name="Table",
    node_type="location",
    ancestors=["spatial_entity", "entity"],
    is_type_definition=False,
    properties={"location_type": "surface"},
)
```

### 6. Test Fixtures

**Location:** `tests/data/__init__.py`

Update `load_pick_and_place_scenario()` to use new Node class signature.

### 7. Cypher Test File

**Location:** `tests/data/test_data_pick_and_place.cypher`

**Current:**
```cypher
CREATE (table:Location {id: 'table', name: 'Table', type: 'surface'})
```

**New:**
```cypher
CREATE (table:Node {
    uuid: 'table',
    name: 'Table',
    type: 'location',
    is_type_definition: false,
    ancestors: ['spatial_entity', 'entity'],
    location_type: 'surface'
})
```

## Behavior Specification

### add_node() Behavior

**Preconditions:**
- Neo4j connection established
- uuid is non-empty string
- name is non-empty string
- node_type is non-empty string
- ancestors is a list (can be empty)

**Input:**
```python
{
    "uuid": str,           # Required, unique identifier
    "name": str,           # Required, human-readable
    "node_type": str,      # Required, semantic type
    "ancestors": list[str],# Required, type inheritance
    "is_type_definition": bool,  # Optional, default False
    "properties": dict | None,   # Optional, additional props
}
```

**Processing:**
1. Validate inputs via SHACL (uuid, name, type required)
2. Encode complex property values as JSON with sentinel prefix
3. Execute MERGE query with uuid as match key
4. SET all standard properties (name, type, is_type_definition, ancestors)
5. Merge additional properties
6. Return uuid on success

**Output:** `str` - The uuid of the node

**Postconditions:**
- Node exists in Neo4j with :Node label
- All standard properties set
- Additional properties merged

**Example:**
```python
# Input
client.add_node(
    uuid="red_block_001",
    name="Red Block",
    node_type="object",
    ancestors=["physical_entity", "entity"],
    is_type_definition=False,
    properties={"color": "red", "material": "plastic"}
)
# Output
"red_block_001"

# Neo4j node:
# (:Node {
#     uuid: "red_block_001",
#     name: "Red Block",
#     type: "object",
#     is_type_definition: false,
#     ancestors: ["physical_entity", "entity"],
#     color: "red",
#     material: "plastic"
# })
```

## Edge Cases & Error Handling

### Missing Required Properties

**Condition:** uuid, name, or node_type is empty/None
**Behavior:** Raise `ValueError` with descriptive message
**Example:**
```python
client.add_node(uuid="", name="Test", node_type="object", ancestors=[])
# Raises: ValueError("uuid cannot be empty")
```

### UUID Collision (MERGE behavior)

**Condition:** Node with same uuid already exists
**Behavior:** Update existing node (MERGE semantics)
**Example:**
```python
# First call creates node
client.add_node(uuid="item1", name="Item", node_type="object", ancestors=[])
# Second call updates same node
client.add_node(uuid="item1", name="Updated Item", node_type="object", ancestors=["entity"])
# Result: Single node with name="Updated Item", ancestors=["entity"]
```

### Empty Ancestors List

**Condition:** ancestors=[]
**Behavior:** Allowed (root types have no ancestors)
**Example:**
```python
client.add_node(uuid="root1", name="Root", node_type="entity", ancestors=[])
# Valid: Creates node with empty ancestors list
```

### Complex Property Encoding

**Condition:** Property value is dict or nested list
**Behavior:** JSON-encode with sentinel prefix
**Example:**
```python
client.add_node(
    uuid="x",
    name="X",
    node_type="state",
    ancestors=[],
    properties={"nested": {"a": 1}}
)
# Stored as: nested: "__LOGOS_JSON__:{\"a\": 1}"
```

## Testing Strategy

### Unit Tests

**Test: add_node creates node with all standard properties**
```python
def test_add_node_standard_properties(hcg_client, neo4j_session):
    uuid = hcg_client.add_node(
        uuid="test-001",
        name="Test Node",
        node_type="test",
        ancestors=["parent", "root"],
        is_type_definition=False,
        properties={"custom": "value"}
    )

    result = neo4j_session.run(
        "MATCH (n:Node {uuid: $uuid}) RETURN n", {"uuid": uuid}
    ).single()

    node = result["n"]
    assert node["uuid"] == "test-001"
    assert node["name"] == "Test Node"
    assert node["type"] == "test"
    assert node["is_type_definition"] == False
    assert node["ancestors"] == ["parent", "root"]
    assert node["custom"] == "value"
```

**Test: add_node rejects empty uuid**
```python
def test_add_node_rejects_empty_uuid(hcg_client):
    with pytest.raises(ValueError, match="uuid cannot be empty"):
        hcg_client.add_node(uuid="", name="Test", node_type="x", ancestors=[])
```

**Test: add_node_legacy maps old signature**
```python
def test_add_node_legacy_compatibility(hcg_client, neo4j_session):
    with pytest.warns(DeprecationWarning):
        uuid = hcg_client.add_node_legacy(
            node_id="legacy-001",
            node_type="test",
            properties={"name": "Legacy Node"}
        )

    result = neo4j_session.run(
        "MATCH (n:Node {uuid: $uuid}) RETURN n", {"uuid": uuid}
    ).single()

    node = result["n"]
    assert node["uuid"] == "legacy-001"
    assert node["name"] == "Legacy Node"
    assert node["is_type_definition"] == False
    assert node["ancestors"] == []
```

### Integration Tests

**Test: Seeder creates logos-compatible nodes**
```python
def test_seeder_creates_standard_nodes(hcg_client, neo4j_session):
    seed_pick_and_place_data(hcg_client)

    # Verify a sample node has all standard properties
    result = neo4j_session.run(
        "MATCH (n:Node {uuid: 'table'}) RETURN n"
    ).single()

    node = result["n"]
    assert "uuid" in node
    assert "name" in node
    assert "type" in node
    assert "is_type_definition" in node
    assert "ancestors" in node
```

## Files Affected

| File | Action | Description |
|------|--------|-------------|
| `src/sophia/hcg_client/client.py` | MODIFY | Update add_node signature and Cypher |
| `src/sophia/hcg_client/seeder.py` | MODIFY | Update all add_node calls |
| `src/sophia/knowledge_graph/node.py` | MODIFY | Add uuid, name, ancestors, is_type_definition |
| `tests/data/__init__.py` | MODIFY | Update Node instantiation |
| `tests/data/test_data_pick_and_place.cypher` | MODIFY | Convert to :Node with standard props |
| `tests/hcg_client/test_client.py` | MODIFY | Update tests for new signature |

## Out of Scope

- **Type registry/lookup:** Ancestors must be provided by caller; no automatic lookup from type name
- **API model changes:** logos_sophia_sdk models are OpenAPI-generated and unchanged
- **Edge properties:** Edge handling remains as-is (no ancestors on edges)
- **Migration script:** No script to migrate existing data in production Neo4j
- **CWM persistence:** Already uses HCGQueries.create_cwm_state() which is logos-compliant

## Ancestor Values Reference

For the pick-and-place scenario, use these ancestor chains:

| Type | Ancestors |
|------|-----------|
| location | `["spatial_entity", "entity"]` |
| object | `["physical_entity", "entity"]` |
| action | `["process", "entity"]` |
| goal | `["intention", "entity"]` |
| state | `["cognition"]` |
