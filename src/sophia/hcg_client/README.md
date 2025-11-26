# HCG Client - Hierarchical Cognitive Graph Management

The HCG (Hierarchical Cognitive Graph) client now reuses the canonical `logos_hcg`
package for connection management while layering on Sophia-specific SHACL validation
and helper utilities.

## Features

- **Shared LOGOS client**: Delegates connectivity/retry logic to `logos_hcg`
- **Neo4j Integration**: CRUD helpers for nodes/edges
- **SHACL Validation**: Enforces constraints on every mutation
- **Type-Safe API**: Minimal surface for CWM-A/Planner consumers
- **Extensible**: Ready for future Milvus/vector integration without copy/paste

## Architecture

```
┌────────────────────────────────────────────┐
│           Sophia HCGClient Wrapper         │
├────────────────────────────────────────────┤
│  - Extends `logos_hcg.client.HCGClient`    │
│  - Adds SHACL validation helpers           │
│  - Provides high-level graph utilities     │
└────────────────────────────────────────────┘
                   │
                   ↓
            ┌──────────────┐
            │  Neo4j (HCG) │
            └──────────────┘
```

## Installation

1. Add dependencies to your project (already included in Sophia):
   ```toml
   neo4j = ">=5.0.0"
   pyshacl = ">=0.25.0"
   tenacity = ">=8.0.0"
   ```

2. Start services with docker-compose:
   ```bash
   docker-compose -f docker-compose.hcg.dev.yml up -d
   ```

## Quick Start

```python
from sophia import HCGClient

# Initialize client
client = HCGClient(
    neo4j_uri="bolt://localhost:7687",
    neo4j_username="neo4j",
    neo4j_password="sophiadev",
)

# Add nodes (with automatic SHACL validation)
client.add_node(
    node_id="learning",
    node_type="concept",
    properties={"description": "Process of acquiring knowledge"}
)

client.add_node(
    node_id="intelligence",
    node_type="concept",
    properties={"description": "Cognitive abilities"}
)

# Add edges (with automatic SHACL validation)
client.add_edge(
    edge_id="e1",
    source_id="learning",
    target_id="intelligence",
    relation="develops",
)

# Query the graph
node = client.get_node("learning")
neighbors = client.query_neighbors("learning")
edges = client.query_edges_from("learning")

# Cleanup
client.close()
```

## API Reference

### HCGClient

Main client for managing knowledge graph.

#### Methods

**Graph Operations:**
- `add_node(node_id, node_type, properties)` - Add node with validation
- `add_edge(edge_id, source_id, target_id, relation, properties)` - Add edge with validation
- `get_node(node_id)` - Retrieve node data
- `get_edge(edge_id)` - Retrieve edge data
- `query_neighbors(node_id)` - Get all neighbors
- `query_edges_from(node_id)` - Get outgoing edges
- `delete_node(node_id)` - Delete node and its embedding

**Utility:**
- `health_check()` - Check Neo4j and Milvus health
- `clear_all()` - Clear all data (dangerous!)
- `close()` - Close all connections

### Implementation Details

- Connection pooling, retry logic, and Cypher helpers come from
  `logos_hcg.client.HCGClient`.
- Sophia adds SHACL validation plus tiny helper queries used by the API layer.
- Vector/embedding helpers will return once the shared Milvus sync module
  (`logos_hcg.sync.HCGMilvusSync`) is wired into the service.

### SHACLValidator

SHACL validation for graph mutations.

**Default Constraints:**
- All nodes must have a `type`
- All edges must have `source`, `target`, and `relation`

**Custom Shapes:**
```python
from sophia.hcg_client import SHACLValidator

custom_shapes = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <http://example.org/hcg/> .

ex:ConceptShape a sh:NodeShape ;
    sh:targetClass ex:Concept ;
    sh:property [
        sh:path ex:description ;
        sh:minCount 1 ;
        sh:datatype xsd:string ;
    ] .
"""

validator = SHACLValidator(shapes_graph=custom_shapes)
client = HCGClient(validator=validator)
```

## Examples

See `examples/hcg_client_demo.py` for a comprehensive demonstration.

Run the demo:
```bash
# Start services
docker-compose -f docker-compose.hcg.dev.yml up -d

# Run demo
poetry run python examples/hcg_client_demo.py
```

## Testing

### Unit Tests (no services required)
```bash
poetry run pytest tests/hcg_client/ -v -m "not integration"
```

### Integration Tests (requires services)
```bash
# Start services
docker-compose -f docker-compose.hcg.dev.yml up -d

# Run integration tests
poetry run pytest tests/hcg_client/test_integration.py -v

# Stop services
docker-compose -f docker-compose.hcg.dev.yml down
```

## Configuration

### Neo4j Settings
- **URI**: `bolt://localhost:7687`
- **Username**: `neo4j`
- **Password**: `sophiadev`
- **Connection Pool**: 50 connections (configurable)
- **Retry Policy**: 3 attempts with exponential backoff (2-10s)

### Milvus Settings
- **Host**: `localhost`
- **Port**: `19530`
- **Collection**: `hcg_embeddings`
- **Dimension**: 768 (configurable)
- **Index**: IVF_FLAT with L2 distance

## Integration with CWM-A/Planner

The HCG client provides the minimal API needed by CWM-A and Planner:

```python
from sophia import HCGClient, Planner, ContinuousWorkingMemoryAssociative

# Initialize HCG client
hcg = HCGClient()

# Use with Planner for goal decomposition
planner = Planner()

# Store planning results in HCG
hcg.add_node("goal_1", "goal", {"description": "Pick object"})
hcg.add_node("action_1", "action", {"name": "GRASP"})
hcg.add_edge("e1", "action_1", "goal_1", "achieves")

# Use with CWM-A for associative memory
cwm_a = ContinuousWorkingMemoryAssociative()
cwm_a.store("current_hcg_node", "goal_1")

# Query related nodes
related = hcg.query_neighbors("goal_1")
```

## Troubleshooting

### Connection Errors
```
neo4j.exceptions.ServiceUnavailable: Failed to establish connection
```
- Ensure Neo4j is running: `docker-compose -f docker-compose.hcg.dev.yml ps`
- Check Neo4j logs: `docker-compose -f docker-compose.hcg.dev.yml logs neo4j`
- Verify credentials match configuration

### Validation Errors
```
ValueError: Node validation failed: ...
```
- Check that node has required `type` field
- Verify edge has `source`, `target`, and `relation`
- Review custom SHACL shapes if using them

### Embedding Support

Vector/embedding helpers will return once the shared Milvus sync utilities are in
place. Until then, the public API focuses purely on graph mutations/queries.

## Performance Considerations

- **Batch Operations**: For bulk imports, consider using Neo4j's batch import tools
- **Connection Pooling**: Managed by `logos_hcg` and configurable via env vars
- **SHACL Validation**: Can be disabled for trusted data sources (not recommended)

## Security

- **Credentials**: Store in environment variables, not code
- **Network**: Use TLS for production deployments
- **SHACL**: Validation prevents malformed data injection
- **Connection Limits**: Pool size limits prevent resource exhaustion

## License

MIT License - see main Sophia repository
