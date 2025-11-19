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

## Milestone M3: Sophia can plan simple actions ✅

**Status**: Complete (Epoch 3: Cognitive Core & Reasoning)

Sophia now has a complete cognitive architecture with:
- 🧠 **Planning**: Backward chaining for goal decomposition
- 🗺️ **World Modeling**: Knowledge graphs for representing domain knowledge
- 🎯 **Reasoning**: Causal relationships and action sequencing
- 💾 **State Management**: Tracking and updating world state

### Try the Demo

```bash
# Run the end-to-end milestone demonstration
poetry run python examples/milestone_m3_demo.py
```

This demonstrates:
1. Building a world model (knowledge graph)
2. Planning a pick-and-place task
3. Simulating execution with state updates
4. Integrating all cognitive components

See [Milestone M3 Verification](docs/MILESTONE_M3_VERIFICATION.md) for detailed documentation.

## Development

### Running Tests

```bash
poetry run pytest
```

**Test Results**: 65/65 tests passing, 98% coverage

#### Functional Tests

Sophia includes functional tests for the plan API with a pick-and-place scenario:

```bash
# Run functional tests
poetry run pytest tests/test_plan_api_pick_and_place.py -v
```

These tests validate:
- Plan generation for goal-directed tasks
- MOVE→GRASP→MOVE→RELEASE action sequences
- State management and updates
- Validation of state changes

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
│   ├── planner/          # Planning and goal decomposition
│   ├── executor/         # Action execution management
│   ├── orchestrator/     # Cognitive process coordination
│   ├── cwm_a/            # Continuous Working Memory - Associative
│   ├── cwm_g/            # Continuous Working Memory - Generative
│   └── config/           # Configuration management
├── docs/                 # Documentation and research
│   ├── research/         # Research surveys and design documents
│   └── MILESTONE_M3_VERIFICATION.md  # M3 verification document
├── examples/             # Example scripts and demonstrations
│   └── milestone_m3_demo.py  # End-to-end M3 demo
├── tests/                # Test suite
└── pyproject.toml        # Project configuration
```

## Milestones & Roadmap

### ✅ Epoch 1: Infrastructure & Knowledge Foundation

- ✅ Core knowledge graph data structures (Node, Edge, KnowledgeGraph)
- ✅ Persistent storage layer with database abstraction
- ✅ Configuration management system
- ✅ Type-safe, well-documented codebase

### ✅ Epoch 3: Cognitive Core & Reasoning (M3)

- ✅ Planning component with backward chaining
- ✅ Goal decomposition and action sequencing
- ✅ World modeling with causal relationships
- ✅ State management and updates
- ✅ Cognitive architecture integration (Orchestrator, Executor, CWM)
- ✅ Comprehensive test coverage (98%, 65 tests)

### 🚀 Future: Advanced Planning & Reasoning

- ⏳ Causal enhancement with strength annotations
- ⏳ Forward chaining for reactive planning
- ⏳ Counterfactual reasoning
- ⏳ Multi-goal planning
- ⏳ Hierarchical task decomposition

## Research & Documentation

Research documents and design notes are available in the `docs/research/` directory:

- [Causal Reasoning Methods Survey](docs/research/causal-reasoning-methods.md) - Comprehensive survey of backward/forward chaining, causal graphs, and counterfactual reasoning for HCG planning
- [Planner Applicability Notes](docs/research/planner-applicability-notes.md) - Quick reference guide for implementing causal reasoning in the Planner component
- [GNN Integration Assessment](docs/research/gnn-integration-assessment.md) - Analysis of Graph Neural Network approaches for knowledge graph enhancement, integration risks/benefits, and recommendations

## License

MIT
