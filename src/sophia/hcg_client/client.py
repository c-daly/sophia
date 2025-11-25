"""Lightweight wrapper around the shared LOGOS HCG client.

This module reuses the canonical `logos_hcg` package for connection management,
retry logic, and query helpers while layering on Sophia-specific SHACL
validation and helper methods that were previously duplicated in this repo.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from logos_hcg.client import HCGClient as LogosHCGClient

from sophia.hcg_client.shacl_validator import SHACLValidator

logger = logging.getLogger(__name__)


class HCGClient(LogosHCGClient):
    """Sophia-specific helper that extends the shared LOGOS HCG client."""

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_username: str = "neo4j",
        neo4j_password: str = "sophiadev",
        neo4j_database: str = "neo4j",
        validator: Optional[SHACLValidator] = None,
    ) -> None:
        """Initialize the client and SHACL validator."""
        self._validator = validator or SHACLValidator()
        super().__init__(
            uri=neo4j_uri,
            user=neo4j_username,
            password=neo4j_password,
            database=neo4j_database,
        )

    # ------------------------------------------------------------------
    # Graph operations
    # ------------------------------------------------------------------

    def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create or update a node after SHACL validation."""
        node_data = {
            "id": node_id,
            "type": node_type,
            "properties": properties or {},
        }

        is_valid, errors = self._validator.validate_node(node_data)
        if not is_valid:
            raise ValueError(f"Node validation failed: {'; '.join(errors)}")

        query = """
        MERGE (n:Node {id: $id})
        SET n.type = $type
        SET n += $properties
        RETURN n.id as id
        """
        records = self._execute_query(
            query,
            {
                "id": node_id,
                "type": node_type,
                "properties": node_data["properties"],
            },
        )
        return str(records[0]["id"]) if records else node_id

    def add_edge(
        self,
        edge_id: str,
        source_id: str,
        target_id: str,
        relation: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create or update an edge after SHACL validation."""
        edge_data = {
            "id": edge_id,
            "source": source_id,
            "target": target_id,
            "relation": relation,
            "properties": properties or {},
        }

        is_valid, errors = self._validator.validate_edge(edge_data)
        if not is_valid:
            raise ValueError(f"Edge validation failed: {'; '.join(errors)}")

        query = """
        MATCH (source:Node {id: $source_id})
        MATCH (target:Node {id: $target_id})
        MERGE (source)-[r:RELATION {id: $edge_id}]->(target)
        SET r.relation_type = $relation
        SET r += $properties
        RETURN r.id as id
        """
        records = self._execute_query(
            query,
            {
                "edge_id": edge_id,
                "source_id": source_id,
                "target_id": target_id,
                "relation": relation,
                "properties": edge_data["properties"],
            },
        )
        return str(records[0]["id"]) if records else edge_id

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a node by ID."""
        query = """
        MATCH (n:Node {id: $id})
        RETURN n.id as id, n.type as type, properties(n) as props
        """
        records = self._execute_read(query, {"id": node_id})
        if not records:
            return None

        props = dict(records[0]["props"])
        props.pop("id", None)
        props.pop("type", None)
        return {
            "id": records[0]["id"],
            "type": records[0]["type"],
            "properties": props,
        }

    def get_edge(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """Fetch an edge by ID."""
        query = """
        MATCH (source:Node)-[r:RELATION {id: $id}]->(target:Node)
        RETURN r.id as id, source.id as source, target.id as target,
               r.relation_type as relation, properties(r) as props
        """
        records = self._execute_read(query, {"id": edge_id})
        if not records:
            return None

        props = dict(records[0]["props"])
        props.pop("id", None)
        props.pop("relation_type", None)
        return {
            "id": records[0]["id"],
            "source": records[0]["source"],
            "target": records[0]["target"],
            "relation": records[0]["relation"],
            "properties": props,
        }

    def query_neighbors(self, node_id: str) -> List[Dict[str, Any]]:
        """Return unique neighbor nodes for the provided node."""
        query = """
        MATCH (n:Node {id: $id})-[r]-(neighbor:Node)
        RETURN DISTINCT neighbor.id as id, neighbor.type as type,
               properties(neighbor) as props
        """
        records = self._execute_read(query, {"id": node_id})
        neighbors: List[Dict[str, Any]] = []
        for record in records:
            props = dict(record["props"])
            props.pop("id", None)
            props.pop("type", None)
            neighbors.append(
                {
                    "id": record["id"],
                    "type": record["type"],
                    "properties": props,
                }
            )
        return neighbors

    def query_edges_from(self, node_id: str) -> List[Dict[str, Any]]:
        """Return outgoing edges for the provided node."""
        query = """
        MATCH (source:Node {id: $id})-[r:RELATION]->(target:Node)
        RETURN r.id as id, source.id as source, target.id as target,
               r.relation_type as relation, properties(r) as props
        """
        records = self._execute_read(query, {"id": node_id})
        edges: List[Dict[str, Any]] = []
        for record in records:
            props = dict(record["props"])
            props.pop("id", None)
            props.pop("relation_type", None)
            edges.append(
                {
                    "id": record["id"],
                    "source": record["source"],
                    "target": record["target"],
                    "relation": record["relation"],
                    "properties": props,
                }
            )
        return edges

    def delete_node(self, node_id: str) -> bool:
        """Delete a node and all relationships."""
        query = """
        MATCH (n:Node {id: $id})
        DETACH DELETE n
        RETURN count(n) as deleted
        """
        records = self._execute_query(query, {"id": node_id})
        deleted = records[0]["deleted"] if records else 0
        return bool(deleted)

    def clear_all(self) -> None:
        """Remove all nodes/edges from the graph."""
        self._execute_query("MATCH (n) DETACH DELETE n")
        logger.info("Cleared all nodes and edges from Neo4j")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, bool]:
        """Return a simple component health summary."""
        neo4j_ok = False
        try:
            with self._session() as session:
                result = session.run("RETURN 1 as ok")
                record = result.single()
                neo4j_ok = bool(record and record["ok"] == 1)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Neo4j health check failed: %s", exc)

        # Milvus support has not yet migrated to the shared package.
        return {"neo4j": neo4j_ok, "milvus": False}
