"""Unit tests for Milvus adapter (without live database)."""

import pytest
from unittest.mock import Mock, patch
from sophia.hcg_client.milvus_adapter import MilvusAdapter


class TestMilvusAdapterUnit:
    """Unit tests for Milvus adapter without live database."""

    @patch("sophia.hcg_client.milvus_adapter.connections")
    @patch("sophia.hcg_client.milvus_adapter.utility")
    @patch("sophia.hcg_client.milvus_adapter.Collection")
    def test_adapter_initialization_existing_collection(
        self,
        mock_collection_cls: Mock,
        mock_utility: Mock,
        mock_connections: Mock,
    ) -> None:
        """Test Milvus adapter initialization with existing collection."""
        mock_utility.has_collection.return_value = True
        mock_collection = Mock()
        mock_collection_cls.return_value = mock_collection

        adapter = MilvusAdapter(
            host="test_host",
            port=19530,
            collection_name="test_collection",
        )

        assert adapter is not None
        mock_connections.connect.assert_called_once()
        mock_utility.has_collection.assert_called_once()
        mock_collection_cls.assert_called()

    @patch("sophia.hcg_client.milvus_adapter.connections")
    @patch("sophia.hcg_client.milvus_adapter.utility")
    @patch("sophia.hcg_client.milvus_adapter.Collection")
    def test_adapter_initialization_new_collection(
        self,
        mock_collection_cls: Mock,
        mock_utility: Mock,
        mock_connections: Mock,
    ) -> None:
        """Test Milvus adapter initialization with new collection."""
        mock_utility.has_collection.return_value = False
        mock_collection = Mock()
        mock_collection_cls.return_value = mock_collection

        adapter = MilvusAdapter(
            host="test_host",
            port=19530,
            collection_name="new_collection",
        )

        assert adapter is not None
        mock_connections.connect.assert_called_once()
        mock_utility.has_collection.assert_called_once()
        # Should create new collection with schema
        mock_collection.create_index.assert_called_once()

    @patch("sophia.hcg_client.milvus_adapter.connections")
    @patch("sophia.hcg_client.milvus_adapter.utility")
    @patch("sophia.hcg_client.milvus_adapter.Collection")
    def test_insert_embedding(
        self,
        mock_collection_cls: Mock,
        mock_utility: Mock,
        mock_connections: Mock,
    ) -> None:
        """Test inserting an embedding."""
        mock_utility.has_collection.return_value = True
        mock_collection = Mock()
        mock_collection_cls.return_value = mock_collection

        adapter = MilvusAdapter()

        embedding = [0.1] * 768  # Default dimension
        result = adapter.insert_embedding(
            embedding_id="emb1",
            node_id="node1",
            node_type="concept",
            embedding=embedding,
        )

        assert result == "emb1"
        mock_collection.insert.assert_called_once()
        mock_collection.flush.assert_called_once()

    @patch("sophia.hcg_client.milvus_adapter.connections")
    @patch("sophia.hcg_client.milvus_adapter.utility")
    @patch("sophia.hcg_client.milvus_adapter.Collection")
    def test_insert_embedding_wrong_dimension(
        self,
        mock_collection_cls: Mock,
        mock_utility: Mock,
        mock_connections: Mock,
    ) -> None:
        """Test inserting embedding with wrong dimension raises error."""
        mock_utility.has_collection.return_value = True
        mock_collection = Mock()
        mock_collection_cls.return_value = mock_collection

        adapter = MilvusAdapter()

        embedding = [0.1] * 100  # Wrong dimension

        with pytest.raises(ValueError, match="dimension"):
            adapter.insert_embedding(
                embedding_id="emb1",
                node_id="node1",
                node_type="concept",
                embedding=embedding,
            )

    @patch("sophia.hcg_client.milvus_adapter.connections")
    @patch("sophia.hcg_client.milvus_adapter.utility")
    @patch("sophia.hcg_client.milvus_adapter.Collection")
    def test_search_similar(
        self,
        mock_collection_cls: Mock,
        mock_utility: Mock,
        mock_connections: Mock,
    ) -> None:
        """Test searching for similar embeddings."""
        mock_utility.has_collection.return_value = True
        mock_collection = Mock()

        # Mock search results
        mock_hit = Mock()
        mock_hit.id = "emb1"
        mock_hit.distance = 0.5
        mock_entity = Mock()
        mock_entity.get = lambda k: {"node_id": "node1", "node_type": "concept"}.get(k)
        mock_hit.entity = mock_entity

        mock_hits = [mock_hit]
        mock_collection.search.return_value = [mock_hits]

        mock_collection_cls.return_value = mock_collection

        adapter = MilvusAdapter()

        query_embedding = [0.1] * 768
        results = adapter.search_similar(
            query_embedding=query_embedding,
            top_k=5,
        )

        assert len(results) == 1
        assert results[0]["id"] == "emb1"
        assert results[0]["node_id"] == "node1"
        assert results[0]["distance"] == 0.5
        mock_collection.load.assert_called()
        mock_collection.search.assert_called_once()

    @patch("sophia.hcg_client.milvus_adapter.connections")
    @patch("sophia.hcg_client.milvus_adapter.utility")
    @patch("sophia.hcg_client.milvus_adapter.Collection")
    def test_close(
        self,
        mock_collection_cls: Mock,
        mock_utility: Mock,
        mock_connections: Mock,
    ) -> None:
        """Test closing the adapter."""
        mock_utility.has_collection.return_value = True
        mock_collection = Mock()
        mock_collection_cls.return_value = mock_collection

        adapter = MilvusAdapter()
        adapter.close()

        mock_collection.release.assert_called_once()
        mock_connections.disconnect.assert_called_once()
