"""Milvus adapter for HCG vector storage."""

from typing import Dict, Any, List, Optional
import logging
from pymilvus import (  # type: ignore[import-untyped]
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from pymilvus.exceptions import MilvusException  # type: ignore[import-untyped]


logger = logging.getLogger(__name__)


class MilvusAdapter:
    """Milvus adapter for HCG vector storage with connection pooling and retries.
    
    Provides vector storage and similarity search for knowledge graph embeddings.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        alias: str = "default",
        collection_name: str = "hcg_embeddings",
        dimension: int = 768,
    ) -> None:
        """Initialize Milvus adapter.
        
        Args:
            host: Milvus host
            port: Milvus port
            alias: Connection alias
            collection_name: Name of the collection for embeddings
            dimension: Dimension of embeddings
        """
        self._host = host
        self._port = port
        self._alias = alias
        self._collection_name = collection_name
        self._dimension = dimension
        self._collection: Optional[Collection] = None
        
        self._connect()
        self._initialize_collection()
        logger.info(f"Milvus adapter initialized for {host}:{port}")
    
    def _connect(self) -> None:
        """Connect to Milvus server."""
        connections.connect(
            alias=self._alias,
            host=self._host,
            port=str(self._port),
        )
        logger.info("Connected to Milvus")
    
    def _initialize_collection(self) -> None:
        """Initialize or load the collection."""
        if utility.has_collection(self._collection_name, using=self._alias):
            self._collection = Collection(
                name=self._collection_name,
                using=self._alias,
            )
            logger.info(f"Loaded existing collection: {self._collection_name}")
        else:
            self._create_collection()
    
    def _create_collection(self) -> None:
        """Create a new collection with schema."""
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=255),
            FieldSchema(name="node_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="node_type", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self._dimension),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="HCG node embeddings",
        )
        
        # Create collection
        self._collection = Collection(
            name=self._collection_name,
            schema=schema,
            using=self._alias,
        )
        
        # Create index for vector similarity search
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        self._collection.create_index(
            field_name="embedding",
            index_params=index_params,
        )
        
        logger.info(f"Created new collection: {self._collection_name}")
    
    def close(self) -> None:
        """Close the Milvus connection."""
        if self._collection:
            self._collection.release()
        connections.disconnect(alias=self._alias)
        logger.info("Milvus connection closed")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(MilvusException),
    )
    def insert_embedding(
        self,
        embedding_id: str,
        node_id: str,
        node_type: str,
        embedding: List[float],
    ) -> str:
        """Insert or update an embedding for a node.
        
        Args:
            embedding_id: Unique ID for this embedding
            node_id: Node ID in the knowledge graph
            node_type: Type of the node
            embedding: Embedding vector
            
        Returns:
            Embedding ID
            
        Raises:
            ValueError: If embedding dimension doesn't match
        """
        if len(embedding) != self._dimension:
            raise ValueError(
                f"Embedding dimension {len(embedding)} doesn't match "
                f"expected dimension {self._dimension}"
            )
        
        if not self._collection:
            raise RuntimeError("Collection not initialized")
        
        # Prepare data
        data = [
            [embedding_id],
            [node_id],
            [node_type],
            [embedding],
        ]
        
        # Insert into collection
        self._collection.insert(data)
        self._collection.flush()
        
        logger.info(f"Inserted embedding for node: {node_id}")
        return embedding_id
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(MilvusException),
    )
    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        node_type_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            node_type_filter: Optional filter by node type
            
        Returns:
            List of similar nodes with their distances
        """
        if len(query_embedding) != self._dimension:
            raise ValueError(
                f"Query embedding dimension {len(query_embedding)} doesn't match "
                f"expected dimension {self._dimension}"
            )
        
        if not self._collection:
            raise RuntimeError("Collection not initialized")
        
        # Load collection into memory for search
        self._collection.load()
        
        # Prepare search parameters
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        
        # Build expression for filtering
        expr = None
        if node_type_filter:
            expr = f'node_type == "{node_type_filter}"'
        
        # Search
        results = self._collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["node_id", "node_type"],
        )
        
        # Format results
        similar_nodes = []
        for hits in results:
            for hit in hits:
                similar_nodes.append({
                    "id": hit.id,
                    "node_id": hit.entity.get("node_id"),
                    "node_type": hit.entity.get("node_type"),
                    "distance": hit.distance,
                })
        
        return similar_nodes
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(MilvusException),
    )
    def get_embedding(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get embedding for a specific node.
        
        Args:
            node_id: Node ID
            
        Returns:
            Embedding data or None if not found
        """
        if not self._collection:
            raise RuntimeError("Collection not initialized")
        
        # Load collection
        self._collection.load()
        
        # Query by node_id
        expr = f'node_id == "{node_id}"'
        results = self._collection.query(
            expr=expr,
            output_fields=["id", "node_id", "node_type", "embedding"],
        )
        
        if not results:
            return None
        
        # Return first result
        result = results[0]
        return {
            "id": result.get("id"),
            "node_id": result.get("node_id"),
            "node_type": result.get("node_type"),
            "embedding": result.get("embedding"),
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(MilvusException),
    )
    def delete_embedding(self, node_id: str) -> bool:
        """Delete embedding for a node.
        
        Args:
            node_id: Node ID
            
        Returns:
            True if deleted, False if not found
        """
        if not self._collection:
            raise RuntimeError("Collection not initialized")
        
        # Delete by node_id
        expr = f'node_id == "{node_id}"'
        self._collection.delete(expr)
        self._collection.flush()
        
        logger.info(f"Deleted embedding for node: {node_id}")
        return True
    
    def clear_all(self) -> None:
        """Clear all embeddings from the collection."""
        if self._collection:
            self._collection.release()
            utility.drop_collection(self._collection_name, using=self._alias)
            self._initialize_collection()
        
        logger.info("Cleared all embeddings from collection")
    
    def health_check(self) -> bool:
        """Check if Milvus is healthy and reachable.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Check if we can list collections
            collections = utility.list_collections(using=self._alias)
            return isinstance(collections, list)
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
