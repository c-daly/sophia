# Sophia

**Non-linguistic cognitive core for Project LOGOS**

Sophia is a foundational infrastructure for building knowledge graphs and managing cognitive data structures. It provides a flexible, extensible framework for representing and storing knowledge in a graph-based format.

## Features

- **Knowledge Graph**: In-memory graph structure using NetworkX for efficient node and edge management
- **Persistent Storage**: SQLAlchemy-based database abstraction for storing knowledge graphs
- **Type-Safe Models**: Pydantic-based data models for nodes and edges
- **Configuration Management**: Flexible settings management for deployment
- **Extensible Architecture**: Clean, modular design for easy extension

## Installation

```bash
# Install the package
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
from sophia import KnowledgeGraph, Database
from sophia.knowledge_graph import Node, Edge

# Create a knowledge graph
kg = KnowledgeGraph()

# Add nodes
concept1 = Node(type="concept", properties={"name": "Learning"})
concept2 = Node(type="concept", properties={"name": "Intelligence"})

kg.add_node(concept1)
kg.add_node(concept2)

# Add edges
relation = Edge(
    source=concept1.id,
    target=concept2.id,
    relation="enables",
    properties={"strength": 0.9}
)
kg.add_edge(relation)

# Persist to database
db = Database("sqlite:///my_knowledge.db")
db.store_node(concept1.id, concept1.type, concept1.properties)
db.store_node(concept2.id, concept2.type, concept2.properties)
db.store_edge(relation.id, relation.source, relation.target, 
              relation.relation, relation.properties)
```

## Development

### Running Tests

```bash
pytest
```

### Code Quality

```bash
# Format code
black src tests

# Lint code
ruff check src tests

# Type checking
mypy src
```

## Project Structure

```
sophia/
├── src/sophia/           # Main package
│   ├── knowledge_graph/  # Knowledge graph implementation
│   ├── storage/          # Database abstraction
│   └── config/           # Configuration management
├── tests/                # Test suite
└── pyproject.toml        # Project configuration
```

## Epoch 1: Infrastructure & Knowledge Foundation

This implementation provides the foundational infrastructure for Sophia, including:

- ✅ Core knowledge graph data structures (Node, Edge, KnowledgeGraph)
- ✅ Persistent storage layer with database abstraction
- ✅ Configuration management system
- ✅ Comprehensive test coverage
- ✅ Type-safe, well-documented codebase

## License

MIT
