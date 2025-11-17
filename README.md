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

### Prerequisites

- Python >=3.11
- Poetry (for dependency management)

### Install Poetry

If you don't have Poetry installed, install it using:

```bash
curl -sSL https://install.python-poetry.org | python3 -
# or
pip install poetry
```

### Install Sophia

```bash
# Clone the repository
git clone https://github.com/c-daly/sophia.git
cd sophia

# Install dependencies (includes both runtime and development dependencies)
poetry install

# Activate the virtual environment
poetry shell
```

Alternatively, run commands without activating the shell:

```bash
poetry run python your_script.py
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
poetry run pytest
```

### Code Quality

```bash
# Format code
poetry run black src tests

# Lint code
poetry run ruff check src tests

# Type checking
poetry run mypy src
```

### Adding Dependencies

```bash
# Add a runtime dependency
poetry add package-name

# Add a development dependency
poetry add --group dev package-name

# Update dependencies
poetry update
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
