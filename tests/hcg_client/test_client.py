"""Unit tests for HCG client (without live databases)."""

import pytest
from unittest.mock import Mock, patch
from sophia.hcg_client.client import HCGClient


class TestHCGClientUnit:
    """Unit tests for HCG client without live databases."""

    @patch('sophia.hcg_client.client.Neo4jAdapter')
    @patch('sophia.hcg_client.client.MilvusAdapter')
    def test_client_initialization(
        self,
        mock_milvus_cls: Mock,
        mock_neo4j_cls: Mock,
    ) -> None:
        """Test HCG client initialization."""
        mock_neo4j = Mock()
        mock_milvus = Mock()
        mock_neo4j_cls.return_value = mock_neo4j
        mock_milvus_cls.return_value = mock_milvus
        
        client = HCGClient(
            neo4j_uri="bolt://test:7687",
            milvus_host="test_host",
        )
        
        assert client is not None
        mock_neo4j_cls.assert_called_once()
        mock_milvus_cls.assert_called_once()

    @patch('sophia.hcg_client.client.Neo4jAdapter')
    @patch('sophia.hcg_client.client.MilvusAdapter')
    def test_add_node(
        self,
        mock_milvus_cls: Mock,
        mock_neo4j_cls: Mock,
    ) -> None:
        """Test adding a node through the client."""
        mock_neo4j = Mock()
        mock_milvus = Mock()
        mock_neo4j.add_node.return_value = "node1"
        mock_neo4j_cls.return_value = mock_neo4j
        mock_milvus_cls.return_value = mock_milvus
        
        client = HCGClient()
        
        result = client.add_node(
            node_id="node1",
            node_type="concept",
            properties={"name": "test"},
        )
        
        assert result == "node1"
        mock_neo4j.add_node.assert_called_once()

    @patch('sophia.hcg_client.client.Neo4jAdapter')
    @patch('sophia.hcg_client.client.MilvusAdapter')
    def test_add_edge(
        self,
        mock_milvus_cls: Mock,
        mock_neo4j_cls: Mock,
    ) -> None:
        """Test adding an edge through the client."""
        mock_neo4j = Mock()
        mock_milvus = Mock()
        mock_neo4j.add_edge.return_value = "edge1"
        mock_neo4j_cls.return_value = mock_neo4j
        mock_milvus_cls.return_value = mock_milvus
        
        client = HCGClient()
        
        result = client.add_edge(
            edge_id="edge1",
            source_id="node1",
            target_id="node2",
            relation="connects",
        )
        
        assert result == "edge1"
        mock_neo4j.add_edge.assert_called_once()

    @patch('sophia.hcg_client.client.Neo4jAdapter')
    @patch('sophia.hcg_client.client.MilvusAdapter')
    def test_get_node(
        self,
        mock_milvus_cls: Mock,
        mock_neo4j_cls: Mock,
    ) -> None:
        """Test getting a node through the client."""
        mock_neo4j = Mock()
        mock_milvus = Mock()
        node_data = {"id": "node1", "type": "concept", "properties": {}}
        mock_neo4j.get_node.return_value = node_data
        mock_neo4j_cls.return_value = mock_neo4j
        mock_milvus_cls.return_value = mock_milvus
        
        client = HCGClient()
        
        result = client.get_node("node1")
        
        assert result == node_data
        mock_neo4j.get_node.assert_called_once_with("node1")

    @patch('sophia.hcg_client.client.Neo4jAdapter')
    @patch('sophia.hcg_client.client.MilvusAdapter')
    def test_add_embedding(
        self,
        mock_milvus_cls: Mock,
        mock_neo4j_cls: Mock,
    ) -> None:
        """Test adding an embedding through the client."""
        mock_neo4j = Mock()
        mock_milvus = Mock()
        node_data = {"id": "node1", "type": "concept", "properties": {}}
        mock_neo4j.get_node.return_value = node_data
        mock_milvus.insert_embedding.return_value = "emb_node1"
        mock_neo4j_cls.return_value = mock_neo4j
        mock_milvus_cls.return_value = mock_milvus
        
        client = HCGClient()
        
        embedding = [0.1] * 768
        result = client.add_embedding(
            node_id="node1",
            embedding=embedding,
        )
        
        assert result == "emb_node1"
        mock_neo4j.get_node.assert_called_once_with("node1")
        mock_milvus.insert_embedding.assert_called_once()

    @patch('sophia.hcg_client.client.Neo4jAdapter')
    @patch('sophia.hcg_client.client.MilvusAdapter')
    def test_add_embedding_node_not_exists(
        self,
        mock_milvus_cls: Mock,
        mock_neo4j_cls: Mock,
    ) -> None:
        """Test adding embedding for non-existent node raises error."""
        mock_neo4j = Mock()
        mock_milvus = Mock()
        mock_neo4j.get_node.return_value = None
        mock_neo4j_cls.return_value = mock_neo4j
        mock_milvus_cls.return_value = mock_milvus
        
        client = HCGClient()
        
        embedding = [0.1] * 768
        
        with pytest.raises(ValueError, match="does not exist"):
            client.add_embedding(
                node_id="nonexistent",
                embedding=embedding,
            )

    @patch('sophia.hcg_client.client.Neo4jAdapter')
    @patch('sophia.hcg_client.client.MilvusAdapter')
    def test_search_similar_nodes(
        self,
        mock_milvus_cls: Mock,
        mock_neo4j_cls: Mock,
    ) -> None:
        """Test searching for similar nodes."""
        mock_neo4j = Mock()
        mock_milvus = Mock()
        
        # Mock Milvus search results
        milvus_results = [
            {"id": "emb1", "node_id": "node1", "node_type": "concept", "distance": 0.5}
        ]
        mock_milvus.search_similar.return_value = milvus_results
        
        # Mock Neo4j node data
        node_data = {"id": "node1", "type": "concept", "properties": {"name": "test"}}
        mock_neo4j.get_node.return_value = node_data
        
        mock_neo4j_cls.return_value = mock_neo4j
        mock_milvus_cls.return_value = mock_milvus
        
        client = HCGClient()
        
        query_embedding = [0.1] * 768
        results = client.search_similar_nodes(
            query_embedding=query_embedding,
            top_k=5,
        )
        
        assert len(results) == 1
        assert results[0]["node_id"] == "node1"
        assert results[0]["node_data"] == node_data
        mock_milvus.search_similar.assert_called_once()
        mock_neo4j.get_node.assert_called_once_with("node1")

    @patch('sophia.hcg_client.client.Neo4jAdapter')
    @patch('sophia.hcg_client.client.MilvusAdapter')
    def test_delete_node(
        self,
        mock_milvus_cls: Mock,
        mock_neo4j_cls: Mock,
    ) -> None:
        """Test deleting a node deletes from both Neo4j and Milvus."""
        mock_neo4j = Mock()
        mock_milvus = Mock()
        mock_neo4j.delete_node.return_value = True
        mock_milvus.delete_embedding.return_value = True
        mock_neo4j_cls.return_value = mock_neo4j
        mock_milvus_cls.return_value = mock_milvus
        
        client = HCGClient()
        
        result = client.delete_node("node1")
        
        assert result is True
        mock_neo4j.delete_node.assert_called_once_with("node1")
        mock_milvus.delete_embedding.assert_called_once_with("node1")

    @patch('sophia.hcg_client.client.Neo4jAdapter')
    @patch('sophia.hcg_client.client.MilvusAdapter')
    def test_health_check(
        self,
        mock_milvus_cls: Mock,
        mock_neo4j_cls: Mock,
    ) -> None:
        """Test health check of all components."""
        mock_neo4j = Mock()
        mock_milvus = Mock()
        mock_neo4j.health_check.return_value = True
        mock_milvus.health_check.return_value = True
        mock_neo4j_cls.return_value = mock_neo4j
        mock_milvus_cls.return_value = mock_milvus
        
        client = HCGClient()
        
        health = client.health_check()
        
        assert health["neo4j"] is True
        assert health["milvus"] is True
        mock_neo4j.health_check.assert_called_once()
        mock_milvus.health_check.assert_called_once()

    @patch('sophia.hcg_client.client.Neo4jAdapter')
    @patch('sophia.hcg_client.client.MilvusAdapter')
    def test_close(
        self,
        mock_milvus_cls: Mock,
        mock_neo4j_cls: Mock,
    ) -> None:
        """Test closing the client."""
        mock_neo4j = Mock()
        mock_milvus = Mock()
        mock_neo4j_cls.return_value = mock_neo4j
        mock_milvus_cls.return_value = mock_milvus
        
        client = HCGClient()
        client.close()
        
        mock_neo4j.close.assert_called_once()
        mock_milvus.close.assert_called_once()
