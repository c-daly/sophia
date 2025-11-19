"""Example demonstrating HCG client usage with Neo4j and Milvus.

This example shows how to:
1. Connect to Neo4j and Milvus
2. Add nodes and edges with SHACL validation
3. Store embeddings in Milvus
4. Search for similar nodes
5. Query the graph

Prerequisites:
    docker-compose -f docker-compose.hcg.dev.yml up -d

Run:
    poetry run python examples/hcg_client_demo.py
"""

import random
from sophia import HCGClient


def main() -> None:
    """Demonstrate HCG client functionality."""
    
    print("=" * 60)
    print("HCG Client Demo - Sophia Knowledge Graph Management")
    print("=" * 60)
    print()
    
    # Initialize client
    print("1. Connecting to Neo4j and Milvus...")
    client = HCGClient(
        neo4j_uri="bolt://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="sophiadev",
        milvus_host="localhost",
        milvus_port=19530,
    )
    
    # Check health
    health = client.health_check()
    print(f"   Neo4j: {'✓' if health['neo4j'] else '✗'}")
    print(f"   Milvus: {'✓' if health['milvus'] else '✗'}")
    print()
    
    # Clear existing data
    print("2. Clearing existing data...")
    client.clear_all()
    print("   Done")
    print()
    
    # Add nodes
    print("3. Adding nodes to the knowledge graph...")
    nodes = [
        ("learning", "concept", {"description": "Process of acquiring knowledge"}),
        ("intelligence", "concept", {"description": "Cognitive abilities"}),
        ("reasoning", "concept", {"description": "Logical thinking"}),
        ("study", "action", {"description": "Learning activity"}),
        ("practice", "action", {"description": "Repetitive learning"}),
    ]
    
    for node_id, node_type, properties in nodes:
        client.add_node(node_id, node_type, properties)
        print(f"   Added: {node_id} ({node_type})")
    print()
    
    # Add edges
    print("4. Adding relationships...")
    edges = [
        ("e1", "study", "learning", "enables"),
        ("e2", "practice", "learning", "enables"),
        ("e3", "learning", "intelligence", "develops"),
        ("e4", "learning", "reasoning", "enhances"),
    ]
    
    for edge_id, source, target, relation in edges:
        client.add_edge(edge_id, source, target, relation)
        print(f"   Added: {source} --[{relation}]--> {target}")
    print()
    
    # Demonstrate SHACL validation
    print("5. Testing SHACL validation...")
    try:
        # This should fail - missing required type
        client.add_node("invalid", "", {})
        print("   ✗ Validation should have failed!")
    except ValueError as e:
        print(f"   ✓ Validation correctly rejected invalid node")
    print()
    
    # Query the graph
    print("6. Querying the knowledge graph...")
    
    # Get a specific node
    learning_node = client.get_node("learning")
    print(f"   Node 'learning': {learning_node['properties']['description']}")
    
    # Get neighbors
    neighbors = client.query_neighbors("learning")
    print(f"   Neighbors of 'learning': {[n['id'] for n in neighbors]}")
    
    # Get outgoing edges
    edges_from_learning = client.query_edges_from("learning")
    print(f"   Edges from 'learning': {[(e['relation'], e['target']) for e in edges_from_learning]}")
    print()
    
    # Add embeddings (simulated)
    print("7. Adding embeddings for semantic search...")
    random.seed(42)  # For reproducibility
    
    # Generate embeddings for each node
    for node_id, node_type, _ in nodes:
        # In real use, these would be from a language model
        embedding = [random.random() for _ in range(768)]
        client.add_embedding(node_id, embedding)
        print(f"   Added embedding for: {node_id}")
    print()
    
    # Search for similar nodes
    print("8. Searching for similar nodes (semantic search)...")
    
    # Get embedding for "learning" to use as query
    learning_emb = [random.random() for _ in range(768)]
    random.seed(42)  # Reset to get similar embedding
    learning_emb = [random.random() for _ in range(768)]
    
    similar_nodes = client.search_similar_nodes(
        query_embedding=learning_emb,
        top_k=3,
    )
    
    print("   Most similar nodes to 'learning':")
    for result in similar_nodes[:3]:
        node_id = result['node_id']
        distance = result['distance']
        node_data = result['node_data']
        print(f"     - {node_id} (distance: {distance:.4f})")
    print()
    
    # Filter by type
    print("9. Searching for similar 'concept' nodes only...")
    concept_similar = client.search_similar_nodes(
        query_embedding=learning_emb,
        top_k=3,
        node_type_filter="concept",
    )
    
    print("   Similar concept nodes:")
    for result in concept_similar:
        print(f"     - {result['node_id']}")
    print()
    
    # Delete a node
    print("10. Deleting a node...")
    deleted = client.delete_node("practice")
    print(f"   Deleted 'practice': {deleted}")
    
    # Verify it's gone
    practice = client.get_node("practice")
    print(f"   Verify deletion: {practice is None}")
    print()
    
    # Summary
    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print()
    print("Key Features Demonstrated:")
    print("  ✓ SHACL validation on graph mutations")
    print("  ✓ Connection pooling and retry logic")
    print("  ✓ Graph queries (nodes, edges, neighbors)")
    print("  ✓ Vector embeddings with Milvus")
    print("  ✓ Semantic similarity search")
    print("  ✓ Type-based filtering")
    print()
    
    # Cleanup
    client.close()
    print("Connection closed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure docker-compose services are running:")
        print("  docker-compose -f docker-compose.hcg.dev.yml up -d")
