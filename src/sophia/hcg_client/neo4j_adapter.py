"""Neo4j adapter for HCG with connection pooling and retry logic."""

from typing import Dict, Any, List, Optional
import logging
from neo4j import GraphDatabase, Session, Driver
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from neo4j.exceptions import ServiceUnavailable, TransientError

from sophia.hcg_client.shacl_validator import SHACLValidator


logger = logging.getLogger(__name__)


class Neo4jAdapter:
    """Neo4j adapter for HCG with connection pooling, retries, and SHACL validation.
    
    Provides read/write operations for knowledge graph data with constraints.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "sophiadev",
        database: str = "neo4j",
        max_connection_pool_size: int = 50,
        validator: Optional[SHACLValidator] = None,
    ) -> None:
        """Initialize Neo4j adapter.
        
        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            database: Database name
            max_connection_pool_size: Maximum connection pool size
            validator: Optional SHACL validator for graph mutations
        """
        self._driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
            max_connection_pool_size=max_connection_pool_size,
        )
        self._database = database
        self._validator = validator or SHACLValidator()
        logger.info(f"Neo4j adapter initialized for {uri}")
    
    def close(self) -> None:
        """Close the Neo4j driver and release connections."""
        if self._driver:
            self._driver.close()
            logger.info("Neo4j driver closed")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ServiceUnavailable, TransientError)),
    )
    def add_node(self, node_data: Dict[str, Any]) -> str:
        """Add a node to the graph with SHACL validation.
        
        Args:
            node_data: Node data with 'id', 'type', and optional 'properties'
            
        Returns:
            Node ID
            
        Raises:
            ValueError: If validation fails
        """
        # Validate before mutation
        is_valid, errors = self._validator.validate_node(node_data)
        if not is_valid:
            raise ValueError(f"Node validation failed: {'; '.join(errors)}")
        
        node_id = node_data["id"]
        node_type = node_data["type"]
        properties = node_data.get("properties", {})
        
        with self._driver.session(database=self._database) as session:
            query = """
            MERGE (n:Node {id: $id})
            SET n.type = $type
            SET n += $properties
            RETURN n.id as id
            """
            result = session.run(
                query,
                id=node_id,
                type=node_type,
                properties=properties,
            )
            record = result.single()
            
        logger.info(f"Added node: {node_id}")
        return record["id"] if record else node_id
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ServiceUnavailable, TransientError)),
    )
    def add_edge(self, edge_data: Dict[str, Any]) -> str:
        """Add an edge to the graph with SHACL validation.
        
        Args:
            edge_data: Edge data with 'id', 'source', 'target', 'relation',
                      and optional 'properties'
            
        Returns:
            Edge ID
            
        Raises:
            ValueError: If validation fails
        """
        # Validate before mutation
        is_valid, errors = self._validator.validate_edge(edge_data)
        if not is_valid:
            raise ValueError(f"Edge validation failed: {'; '.join(errors)}")
        
        edge_id = edge_data["id"]
        source_id = edge_data["source"]
        target_id = edge_data["target"]
        relation = edge_data["relation"]
        properties = edge_data.get("properties", {})
        
        with self._driver.session(database=self._database) as session:
            query = """
            MATCH (source:Node {id: $source_id})
            MATCH (target:Node {id: $target_id})
            MERGE (source)-[r:RELATION {id: $edge_id}]->(target)
            SET r.relation_type = $relation
            SET r += $properties
            RETURN r.id as id
            """
            result = session.run(
                query,
                edge_id=edge_id,
                source_id=source_id,
                target_id=target_id,
                relation=relation,
                properties=properties,
            )
            record = result.single()
            
        logger.info(f"Added edge: {edge_id}")
        return record["id"] if record else edge_id
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ServiceUnavailable, TransientError)),
    )
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node from the graph.
        
        Args:
            node_id: Node ID
            
        Returns:
            Node data or None if not found
        """
        with self._driver.session(database=self._database) as session:
            query = """
            MATCH (n:Node {id: $id})
            RETURN n.id as id, n.type as type, properties(n) as props
            """
            result = session.run(query, id=node_id)
            record = result.single()
            
            if not record:
                return None
            
            # Extract properties (excluding id and type)
            props = dict(record["props"])
            props.pop("id", None)
            props.pop("type", None)
            
            return {
                "id": record["id"],
                "type": record["type"],
                "properties": props,
            }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ServiceUnavailable, TransientError)),
    )
    def get_edge(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """Get an edge from the graph.
        
        Args:
            edge_id: Edge ID
            
        Returns:
            Edge data or None if not found
        """
        with self._driver.session(database=self._database) as session:
            query = """
            MATCH (source:Node)-[r:RELATION {id: $id}]->(target:Node)
            RETURN r.id as id, source.id as source, target.id as target,
                   r.relation_type as relation, properties(r) as props
            """
            result = session.run(query, id=edge_id)
            record = result.single()
            
            if not record:
                return None
            
            # Extract properties (excluding id, relation_type)
            props = dict(record["props"])
            props.pop("id", None)
            props.pop("relation_type", None)
            
            return {
                "id": record["id"],
                "source": record["source"],
                "target": record["target"],
                "relation": record["relation"],
                "properties": props,
            }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ServiceUnavailable, TransientError)),
    )
    def query_neighbors(self, node_id: str) -> List[Dict[str, Any]]:
        """Query neighbors of a node.
        
        Args:
            node_id: Node ID
            
        Returns:
            List of neighbor nodes
        """
        with self._driver.session(database=self._database) as session:
            query = """
            MATCH (n:Node {id: $id})-[r]-(neighbor:Node)
            RETURN DISTINCT neighbor.id as id, neighbor.type as type,
                   properties(neighbor) as props
            """
            result = session.run(query, id=node_id)
            
            neighbors = []
            for record in result:
                props = dict(record["props"])
                props.pop("id", None)
                props.pop("type", None)
                
                neighbors.append({
                    "id": record["id"],
                    "type": record["type"],
                    "properties": props,
                })
            
            return neighbors
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ServiceUnavailable, TransientError)),
    )
    def query_edges_from(self, node_id: str) -> List[Dict[str, Any]]:
        """Query edges from a node.
        
        Args:
            node_id: Node ID
            
        Returns:
            List of outgoing edges
        """
        with self._driver.session(database=self._database) as session:
            query = """
            MATCH (source:Node {id: $id})-[r:RELATION]->(target:Node)
            RETURN r.id as id, source.id as source, target.id as target,
                   r.relation_type as relation, properties(r) as props
            """
            result = session.run(query, id=node_id)
            
            edges = []
            for record in result:
                props = dict(record["props"])
                props.pop("id", None)
                props.pop("relation_type", None)
                
                edges.append({
                    "id": record["id"],
                    "source": record["source"],
                    "target": record["target"],
                    "relation": record["relation"],
                    "properties": props,
                })
            
            return edges
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ServiceUnavailable, TransientError)),
    )
    def delete_node(self, node_id: str) -> bool:
        """Delete a node from the graph.
        
        Args:
            node_id: Node ID
            
        Returns:
            True if deleted, False if not found
        """
        with self._driver.session(database=self._database) as session:
            # Delete node and all its relationships
            query = """
            MATCH (n:Node {id: $id})
            DETACH DELETE n
            RETURN count(n) as deleted
            """
            result = session.run(query, id=node_id)
            record = result.single()
            
        deleted = record["deleted"] > 0 if record else False
        if deleted:
            logger.info(f"Deleted node: {node_id}")
        return deleted
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ServiceUnavailable, TransientError)),
    )
    def clear_all(self) -> None:
        """Clear all nodes and edges from the graph."""
        with self._driver.session(database=self._database) as session:
            query = "MATCH (n) DETACH DELETE n"
            session.run(query)
        
        logger.info("Cleared all nodes and edges from graph")
    
    def health_check(self) -> bool:
        """Check if Neo4j is healthy and reachable.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            with self._driver.session(database=self._database) as session:
                result = session.run("RETURN 1 as health")
                record = result.single()
                return record is not None and record["health"] == 1
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
