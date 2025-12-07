"""Lightweight wrapper around the shared LOGOS HCG client.

This module reuses the canonical `logos_hcg` package for connection management,
retry logic, and query helpers while layering on Sophia-specific SHACL
validation and helper methods that were previously duplicated in this repo.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from contextlib import suppress
from typing import Any, Dict, List, Mapping, Optional, cast

from logos_config import get_env_value
from logos_hcg.client import HCGClient as LogosHCGClient

from sophia.hcg_client.shacl_validator import SHACLValidator

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency checked at runtime
    from pymilvus import connections as milvus_connections
except ImportError:  # pragma: no cover - fallback when pymilvus missing
    milvus_connections = None  # type: ignore[assignment]


class HCGClient(LogosHCGClient):
    """Sophia-specific helper that extends the shared LOGOS HCG client."""

    _JSON_SENTINEL = "__LOGOS_JSON__:"

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_username: str = "neo4j",
        neo4j_password: str = "neo4jtest",
        neo4j_database: str = "neo4j",
        milvus_host: Optional[str] = None,
        milvus_port: Optional[int] = None,
        validator: Optional[SHACLValidator] = None,
    ) -> None:
        """Initialize the client and SHACL validator.

        `milvus_host`/`milvus_port` are currently unused but we accept them to
        remain API-compatible with existing test fixtures and deployment
        scripts that still pass Milvus connection data alongside Neo4j creds.
        """
        self._validator = validator or SHACLValidator()
        self._milvus_host = milvus_host or get_env_value("MILVUS_HOST")
        port_value = milvus_port or get_env_value("MILVUS_PORT")
        try:
            self._milvus_port = int(port_value) if port_value else None
        except (TypeError, ValueError):
            logger.warning("Invalid MILVUS_PORT value: %s", port_value)
            self._milvus_port = None
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
        encoded_properties = self._encode_properties(
            cast(Mapping[str, Any], node_data["properties"])
        )
        records = self._execute_query(
            query,
            {
                "id": node_id,
                "type": node_type,
                "properties": encoded_properties,
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
        encoded_properties = self._encode_properties(
            cast(Mapping[str, Any], edge_data["properties"])
        )
        records = self._execute_query(
            query,
            {
                "edge_id": edge_id,
                "source_id": source_id,
                "target_id": target_id,
                "relation": relation,
                "properties": encoded_properties,
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
        props = self._decode_properties(props)
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
        props = self._decode_properties(props)
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
            props = self._decode_properties(props)
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
            props = self._decode_properties(props)
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
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_primitive(value: Any) -> bool:
        return isinstance(value, (str, int, float, bool))

    def _encode_properties(self, properties: Mapping[str, Any]) -> Dict[str, Any]:
        encoded: Dict[str, Any] = {}
        for key, value in (properties or {}).items():
            encoded[key] = self._encode_value(value)
        return encoded

    def _encode_value(self, value: Any) -> Any:
        if self._is_primitive(value):
            return value

        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            sanitized: List[Any] = []
            for item in value:
                if self._is_primitive(item):
                    sanitized.append(item)
                else:
                    return self._JSON_SENTINEL + json.dumps(value)
            return sanitized

        return self._JSON_SENTINEL + json.dumps(value)

    def _decode_properties(self, properties: Mapping[str, Any]) -> Dict[str, Any]:
        decoded: Dict[str, Any] = {}
        for key, value in properties.items():
            decoded[key] = self._decode_value(value)
        return decoded

    def _decode_value(self, value: Any) -> Any:
        if isinstance(value, str) and value.startswith(self._JSON_SENTINEL):
            json_payload = value[len(self._JSON_SENTINEL) :]
            try:
                return json.loads(json_payload)
            except json.JSONDecodeError:
                logger.warning("Failed to decode JSON payload for value: %s", value)
                return json_payload

        if isinstance(value, list):
            return [self._decode_value(item) for item in value]

        return value

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

        milvus_ok = False
        if milvus_connections is not None and self._milvus_host and self._milvus_port:
            alias = f"sophia-health-{id(self)}"
            try:
                milvus_connections.connect(
                    alias=alias,
                    host=self._milvus_host,
                    port=str(self._milvus_port),
                    timeout=2.0,
                )
                milvus_ok = True
            except Exception as exc:  # pragma: no cover - diagnostics only
                logger.warning("Milvus health check failed: %s", exc)
            finally:
                with suppress(Exception):
                    milvus_connections.disconnect(alias)
        elif self._milvus_host or self._milvus_port:
            logger.debug(
                "Skipping Milvus health check (pymilvus unavailable: %s)",
                milvus_connections is None,
            )
            milvus_ok = milvus_connections is None
        else:
            # Milvus not configured; treat as healthy for environments that
            # only require Neo4j (avoids perpetual degraded status).
            milvus_ok = True

        return {"neo4j": neo4j_ok, "milvus": milvus_ok}
