#!/usr/bin/env python
"""Example demonstrating Sophia's knowledge graph and storage capabilities."""

from sophia import KnowledgeGraph, Database
from sophia.knowledge_graph import Node, Edge


def main() -> None:
    """Run example demonstrating core functionality."""
    print("=== Sophia Example: Building a Knowledge Graph ===\n")

    # Create a knowledge graph
    kg = KnowledgeGraph()
    print(f"Created empty knowledge graph: {kg.node_count()} nodes, {kg.edge_count()} edges")

    # Create nodes representing concepts
    learning = Node(type="concept", properties={"name": "Learning", "domain": "Education"})
    intelligence = Node(type="concept", properties={"name": "Intelligence", "domain": "Cognition"})
    practice = Node(type="concept", properties={"name": "Practice", "domain": "Action"})

    # Add nodes to the graph
    kg.add_node(learning)
    kg.add_node(intelligence)
    kg.add_node(practice)
    print(f"\nAdded 3 nodes: {kg.node_count()} nodes total")

    # Create relationships
    enables = Edge(
        source=learning.id,
        target=intelligence.id,
        relation="enables",
        properties={"strength": 0.9, "bidirectional": False}
    )
    
    requires = Edge(
        source=learning.id,
        target=practice.id,
        relation="requires",
        properties={"strength": 0.8, "bidirectional": True}
    )

    # Add edges to the graph
    kg.add_edge(enables)
    kg.add_edge(requires)
    print(f"Added 2 edges: {kg.edge_count()} edges total\n")

    # Query the graph
    print(f"Neighbors of '{learning.properties['name']}':")
    for neighbor in kg.get_neighbors(learning.id):
        print(f"  - {neighbor.properties['name']}")

    print(f"\nEdges from '{learning.properties['name']}':")
    for edge in kg.get_edges_from(learning.id):
        target_node = kg.get_node(edge.target)
        if target_node:
            print(f"  - {edge.relation} -> {target_node.properties['name']} (strength: {edge.properties.get('strength', 'N/A')})")

    # Persist to database
    print("\n=== Persisting to Database ===\n")
    db = Database("sqlite:///example_knowledge.db")
    
    # Store all nodes
    for node in [learning, intelligence, practice]:
        db.store_node(node.id, node.type, node.properties)
    print("Stored 3 nodes to database")

    # Store all edges
    for edge in [enables, requires]:
        db.store_edge(edge.id, edge.source, edge.target, edge.relation, edge.properties)
    print("Stored 2 edges to database")

    # Retrieve from database
    print("\nRetrieving nodes from database:")
    all_nodes = db.get_all_nodes()
    for node_data in all_nodes:
        print(f"  - {node_data['properties'].get('name', 'Unknown')} ({node_data['type']})")

    print("\nRetrieving edges from database:")
    all_edges = db.get_all_edges()
    for edge_data in all_edges:
        print(f"  - {edge_data['relation']} (strength: {edge_data['properties'].get('strength', 'N/A')})")

    print("\n=== Example Complete ===")
    print("\nCleaning up (deleting example database)...")
    import os
    if os.path.exists("example_knowledge.db"):
        os.remove("example_knowledge.db")
    print("Done!")


if __name__ == "__main__":
    main()
