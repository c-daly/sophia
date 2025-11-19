"""HCG Client for managing knowledge graph with Neo4j and Milvus."""

from typing import Dict, Any, List, Optional
import logging

from sophia.hcg_client.neo4j_adapter import Neo4jAdapter
from sophia.hcg_client.milvus_adapter import MilvusAdapter
from sophia.hcg_client.shacl_validator import SHACLValidator


logger = logging.getLogger(__name__)


class HCGClient:
    """Unified HCG client for managing knowledge graph with Neo4j and Milvus.

    Provides a simplified API for CWM-A/Planner to query and update HCG,
    with SHACL validation on mutations.
    """

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_username: str = "neo4j",
        neo4j_password: str = "sophiadev",
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
        validator: Optional[SHACLValidator] = None,
    ) -> None:
        """Initialize HCG client.

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            milvus_host: Milvus host
            milvus_port: Milvus port
            validator: Optional SHACL validator
        """
        self._validator = validator or SHACLValidator()
        self._neo4j = Neo4jAdapter(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            validator=self._validator,
        )
        self._milvus = MilvusAdapter(
            host=milvus_host,
            port=milvus_port,
        )
        logger.info("HCG client initialized")

    def close(self) -> None:
        """Close all connections."""
        self._neo4j.close()
        self._milvus.close()
        logger.info("HCG client connections closed")

    # Graph operations

    def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a node to the graph with SHACL validation.

        Args:
            node_id: Unique node ID
            node_type: Type of the node
            properties: Optional node properties

        Returns:
            Node ID

        Raises:
            ValueError: If validation fails
        """
        node_data = {
            "id": node_id,
            "type": node_type,
            "properties": properties or {},
        }
        return self._neo4j.add_node(node_data)

    def add_edge(
        self,
        edge_id: str,
        source_id: str,
        target_id: str,
        relation: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add an edge to the graph with SHACL validation.

        Args:
            edge_id: Unique edge ID
            source_id: Source node ID
            target_id: Target node ID
            relation: Relationship type
            properties: Optional edge properties

        Returns:
            Edge ID

        Raises:
            ValueError: If validation fails
        """
        edge_data = {
            "id": edge_id,
            "source": source_id,
            "target": target_id,
            "relation": relation,
            "properties": properties or {},
        }
        return self._neo4j.add_edge(edge_data)

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node from the graph.

        Args:
            node_id: Node ID

        Returns:
            Node data or None if not found
        """
        return self._neo4j.get_node(node_id)

    def get_edge(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """Get an edge from the graph.

        Args:
            edge_id: Edge ID

        Returns:
            Edge data or None if not found
        """
        return self._neo4j.get_edge(edge_id)

    def query_neighbors(self, node_id: str) -> List[Dict[str, Any]]:
        """Query neighbors of a node.

        Args:
            node_id: Node ID

        Returns:
            List of neighbor nodes
        """
        return self._neo4j.query_neighbors(node_id)

    def query_edges_from(self, node_id: str) -> List[Dict[str, Any]]:
        """Query outgoing edges from a node.

        Args:
            node_id: Node ID

        Returns:
            List of outgoing edges
        """
        return self._neo4j.query_edges_from(node_id)

    def delete_node(self, node_id: str) -> bool:
        """Delete a node from the graph.

        Also deletes associated embedding if present.

        Args:
            node_id: Node ID

        Returns:
            True if deleted, False if not found
        """
        # Delete from Neo4j
        deleted = self._neo4j.delete_node(node_id)

        # Delete embedding from Milvus if exists
        if deleted:
            try:
                self._milvus.delete_embedding(node_id)
            except Exception as e:
                logger.warning(f"Failed to delete embedding for {node_id}: {e}")

        return deleted

    # Vector operations

    def add_embedding(
        self,
        node_id: str,
        embedding: List[float],
    ) -> str:
        """Add or update embedding for a node.

        Args:
            node_id: Node ID
            embedding: Embedding vector

        Returns:
            Embedding ID

        Raises:
            ValueError: If node doesn't exist or embedding dimension is wrong
        """
        # Verify node exists
        node = self._neo4j.get_node(node_id)
        if not node:
            raise ValueError(f"Node {node_id} does not exist")

        # Insert embedding
        embedding_id = f"emb_{node_id}"
        return self._milvus.insert_embedding(
            embedding_id=embedding_id,
            node_id=node_id,
            node_type=node["type"],
            embedding=embedding,
        )

    def search_similar_nodes(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        node_type_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for nodes with similar embeddings.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            node_type_filter: Optional filter by node type

        Returns:
            List of similar nodes with their data and distances
        """
        # Search in Milvus
        similar = self._milvus.search_similar(
            query_embedding=query_embedding,
            top_k=top_k,
            node_type_filter=node_type_filter,
        )

        # Enrich with node data from Neo4j
        enriched_results = []
        for result in similar:
            node_id = result["node_id"]
            node = self._neo4j.get_node(node_id)
            if node:
                enriched_results.append(
                    {
                        **result,
                        "node_data": node,
                    }
                )

        return enriched_results

    # Utility methods

    def health_check(self) -> Dict[str, bool]:
        """Check health of all components.

        Returns:
            Dictionary with health status of Neo4j and Milvus
        """
        return {
            "neo4j": self._neo4j.health_check(),
            "milvus": self._milvus.health_check(),
        }

    def clear_all(self) -> None:
        """Clear all data from Neo4j and Milvus.

        WARNING: This deletes all data!
        """
        self._neo4j.clear_all()
        self._milvus.clear_all()
        logger.warning("Cleared all data from HCG")
