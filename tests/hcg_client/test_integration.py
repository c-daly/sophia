"""Integration tests for HCG client with live Neo4j and Milvus.

These tests require docker-compose.hcg.dev.yml services to be running:
    docker-compose -f docker-compose.hcg.dev.yml up -d

Run these tests with:
    pytest tests/hcg_client/test_integration.py -v --tb=short
"""

import pytest
import time
from typing import Generator
from sophia.hcg_client import HCGClient


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def wait_for_services() -> None:
    """Wait for Neo4j and Milvus services to be ready."""
    # Give services time to start up
    time.sleep(5)


@pytest.fixture(scope="module")
def hcg_client(wait_for_services: None) -> Generator[HCGClient, None, None]:
    """Create HCG client connected to test services."""
    client = HCGClient(
        neo4j_uri="bolt://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="sophiadev",
        milvus_host="localhost",
        milvus_port=19530,
    )
    
    # Clear any existing data
    client.clear_all()
    
    yield client
    
    # Cleanup
    client.clear_all()
    client.close()


class TestHCGClientIntegration:
    """Integration tests for HCG client with live services."""

    def test_health_check(self, hcg_client: HCGClient) -> None:
        """Test that services are healthy."""
        health = hcg_client.health_check()
        assert health["neo4j"] is True, "Neo4j should be healthy"
        assert health["milvus"] is True, "Milvus should be healthy"

    def test_add_and_get_node(self, hcg_client: HCGClient) -> None:
        """Test adding and retrieving a node."""
        # Add node
        node_id = hcg_client.add_node(
            node_id="concept_1",
            node_type="concept",
            properties={"name": "Learning", "importance": 0.9},
        )
        assert node_id == "concept_1"
        
        # Get node
        node = hcg_client.get_node("concept_1")
        assert node is not None
        assert node["id"] == "concept_1"
        assert node["type"] == "concept"
        assert node["properties"]["name"] == "Learning"
        assert node["properties"]["importance"] == 0.9

    def test_add_node_validation_failure(self, hcg_client: HCGClient) -> None:
        """Test that invalid node data fails SHACL validation."""
        with pytest.raises(ValueError, match="validation failed"):
            # Missing required 'type' field
            hcg_client.add_node(
                node_id="invalid_node",
                node_type="",  # Empty type should fail validation
                properties={},
            )

    def test_add_and_get_edge(self, hcg_client: HCGClient) -> None:
        """Test adding and retrieving an edge."""
        # Add nodes first
        hcg_client.add_node("node_a", "concept", {"name": "A"})
        hcg_client.add_node("node_b", "concept", {"name": "B"})
        
        # Add edge
        edge_id = hcg_client.add_edge(
            edge_id="edge_1",
            source_id="node_a",
            target_id="node_b",
            relation="enables",
            properties={"strength": 0.8},
        )
        assert edge_id == "edge_1"
        
        # Get edge
        edge = hcg_client.get_edge("edge_1")
        assert edge is not None
        assert edge["id"] == "edge_1"
        assert edge["source"] == "node_a"
        assert edge["target"] == "node_b"
        assert edge["relation"] == "enables"
        assert edge["properties"]["strength"] == 0.8

    def test_query_neighbors(self, hcg_client: HCGClient) -> None:
        """Test querying neighbors of a node."""
        # Create a small graph
        hcg_client.add_node("center", "concept", {"name": "Center"})
        hcg_client.add_node("neighbor1", "concept", {"name": "N1"})
        hcg_client.add_node("neighbor2", "concept", {"name": "N2"})
        
        hcg_client.add_edge("e1", "center", "neighbor1", "connects")
        hcg_client.add_edge("e2", "center", "neighbor2", "connects")
        
        # Query neighbors
        neighbors = hcg_client.query_neighbors("center")
        assert len(neighbors) == 2
        neighbor_ids = {n["id"] for n in neighbors}
        assert "neighbor1" in neighbor_ids
        assert "neighbor2" in neighbor_ids

    def test_query_edges_from(self, hcg_client: HCGClient) -> None:
        """Test querying outgoing edges from a node."""
        # Create nodes and edges
        hcg_client.add_node("source", "concept", {"name": "Source"})
        hcg_client.add_node("target1", "concept", {"name": "T1"})
        hcg_client.add_node("target2", "concept", {"name": "T2"})
        
        hcg_client.add_edge("out1", "source", "target1", "leads_to")
        hcg_client.add_edge("out2", "source", "target2", "leads_to")
        
        # Query edges
        edges = hcg_client.query_edges_from("source")
        assert len(edges) == 2
        edge_ids = {e["id"] for e in edges}
        assert "out1" in edge_ids
        assert "out2" in edge_ids

    def test_delete_node(self, hcg_client: HCGClient) -> None:
        """Test deleting a node."""
        # Add node
        hcg_client.add_node("to_delete", "concept", {"name": "Delete Me"})
        
        # Verify it exists
        node = hcg_client.get_node("to_delete")
        assert node is not None
        
        # Delete it
        deleted = hcg_client.delete_node("to_delete")
        assert deleted is True
        
        # Verify it's gone
        node = hcg_client.get_node("to_delete")
        assert node is None

    def test_add_and_search_embeddings(self, hcg_client: HCGClient) -> None:
        """Test adding embeddings and searching for similar nodes."""
        # Add nodes
        hcg_client.add_node("emb_node1", "concept", {"name": "Concept 1"})
        hcg_client.add_node("emb_node2", "concept", {"name": "Concept 2"})
        hcg_client.add_node("emb_node3", "action", {"name": "Action 1"})
        
        # Add embeddings (using 768-dimensional vectors)
        import random
        random.seed(42)  # For reproducibility
        
        emb1 = [random.random() for _ in range(768)]
        emb2 = [random.random() for _ in range(768)]
        emb3 = [random.random() for _ in range(768)]
        
        hcg_client.add_embedding("emb_node1", emb1)
        hcg_client.add_embedding("emb_node2", emb2)
        hcg_client.add_embedding("emb_node3", emb3)
        
        # Give Milvus time to index
        time.sleep(2)
        
        # Search for similar nodes using emb1 as query
        results = hcg_client.search_similar_nodes(
            query_embedding=emb1,
            top_k=3,
        )
        
        assert len(results) > 0
        # First result should be emb_node1 itself (distance ~0)
        assert results[0]["node_id"] == "emb_node1"
        assert results[0]["distance"] < 0.1  # Very close to itself

    def test_search_embeddings_with_type_filter(self, hcg_client: HCGClient) -> None:
        """Test searching embeddings with node type filter."""
        # Add nodes of different types
        hcg_client.add_node("concept_node", "concept", {"name": "Concept"})
        hcg_client.add_node("action_node", "action", {"name": "Action"})
        
        import random
        random.seed(123)
        
        emb_concept = [random.random() for _ in range(768)]
        emb_action = [random.random() for _ in range(768)]
        
        hcg_client.add_embedding("concept_node", emb_concept)
        hcg_client.add_embedding("action_node", emb_action)
        
        time.sleep(2)
        
        # Search only for concepts
        results = hcg_client.search_similar_nodes(
            query_embedding=emb_concept,
            top_k=10,
            node_type_filter="concept",
        )
        
        # All results should be of type "concept"
        for result in results:
            assert result["node_type"] == "concept"

    def test_complex_graph_scenario(self, hcg_client: HCGClient) -> None:
        """Test a more complex scenario with multiple nodes and edges."""
        # Build a small knowledge graph
        nodes = [
            ("learning", "concept", {"description": "Process of acquiring knowledge"}),
            ("intelligence", "concept", {"description": "Cognitive abilities"}),
            ("study", "action", {"description": "Learning activity"}),
            ("practice", "action", {"description": "Repetitive learning"}),
        ]
        
        for node_id, node_type, props in nodes:
            hcg_client.add_node(node_id, node_type, props)
        
        edges = [
            ("e1", "study", "learning", "enables"),
            ("e2", "practice", "learning", "enables"),
            ("e3", "learning", "intelligence", "develops"),
        ]
        
        for edge_id, source, target, relation in edges:
            hcg_client.add_edge(edge_id, source, target, relation)
        
        # Query the graph
        learning_neighbors = hcg_client.query_neighbors("learning")
        assert len(learning_neighbors) >= 2  # At least study, practice, intelligence
        
        # Verify the graph structure
        edges_to_learning = [
            e for e in 
            [hcg_client.get_edge(eid) for eid in ["e1", "e2", "e3"]]
            if e and (e["source"] == "learning" or e["target"] == "learning")
        ]
        assert len(edges_to_learning) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
