# HCG Client - Hierarchical Cognitive Graph Management

The HCG (Hierarchical Cognitive Graph) client provides a unified interface for managing knowledge graphs with Neo4j and Milvus, featuring SHACL validation on all graph mutations.

## Features

- **Neo4j Integration**: Graph database for nodes and edges
- **Milvus Integration**: Vector database for semantic embeddings
- **SHACL Validation**: Enforces constraints on graph mutations
- **Connection Pooling**: Efficient connection management
- **Retry Logic**: Automatic retries with exponential backoff
- **Type-Safe API**: Minimal, clean API for CWM-A/Planner

## Architecture

```
┌─────────────────────────────────────────────────┐
│              HCGClient (Unified API)            │
├─────────────────┬───────────────────────────────┤
│  Neo4jAdapter   │      MilvusAdapter            │
│  - Nodes/Edges  │      - Embeddings             │
│  - Queries      │      - Similarity Search      │
└─────────────────┴───────────────────────────────┘
         │                      │
         ↓                      ↓
    ┌─────────┐           ┌──────────┐
    │ Neo4j   │           │ Milvus   │
    │ (Graph) │           │ (Vector) │
    └─────────┘           └──────────┘
```

## Installation

1. Add dependencies to your project (already included in Sophia):
   ```toml
   neo4j = ">=5.0.0"
   pymilvus = ">=2.3.0"
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
    milvus_host="localhost",
    milvus_port=19530,
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

# Add embeddings for semantic search
embedding = [0.1] * 768  # 768-dimensional vector
client.add_embedding("learning", embedding)

# Search for similar nodes
similar = client.search_similar_nodes(
    query_embedding=embedding,
    top_k=5,
    node_type_filter="concept",  # Optional: filter by type
)

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

**Vector Operations:**
- `add_embedding(node_id, embedding)` - Add/update embedding
- `search_similar_nodes(query_embedding, top_k, node_type_filter)` - Semantic search

**Utility:**
- `health_check()` - Check Neo4j and Milvus health
- `clear_all()` - Clear all data (dangerous!)
- `close()` - Close all connections

### Neo4jAdapter

Low-level Neo4j operations with connection pooling and retries.

- Automatic retry on `ServiceUnavailable` and `TransientError`
- Connection pooling (default: 50 connections)
- SHACL validation on mutations

### MilvusAdapter

Low-level Milvus operations for vector storage.

- Automatic collection creation with schema
- IVF_FLAT indexing for similarity search
- L2 distance metric

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

### Embedding Dimension Errors
```
ValueError: Embedding dimension 512 doesn't match expected dimension 768
```
- Ensure embedding vectors are 768-dimensional (default)
- Or configure MilvusAdapter with custom dimension

## Performance Considerations

- **Batch Operations**: For bulk imports, consider using Neo4j's batch import tools
- **Connection Pooling**: Adjust pool size based on workload
- **Milvus Indexing**: Larger nlist values improve accuracy but reduce speed
- **SHACL Validation**: Can be disabled for trusted data sources (not recommended)

## Security

- **Credentials**: Store in environment variables, not code
- **Network**: Use TLS for production deployments
- **SHACL**: Validation prevents malformed data injection
- **Connection Limits**: Pool size limits prevent resource exhaustion

## License

MIT License - see main Sophia repository
