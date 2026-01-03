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
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, cast
from uuid import uuid4

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

    def _get_type_ancestors(self, node_type: str) -> List[str]:
        """Look up ancestors for a type from its type definition in Neo4j.

        The type definition's `name` field identifies what type it defines,
        while its `type` field is its parent type in the hierarchy.

        Args:
            node_type: The type name to look up (matches type definition's name)

        Returns:
            List of ancestors from the type definition, or empty list if not found
        """
        query = """
        MATCH (t:Node {name: $node_type, is_type_definition: true})
        RETURN t.ancestors as ancestors
        """
        records = self._execute_read(query, {"node_type": node_type})
        if records and records[0].get("ancestors"):
            return list(records[0]["ancestors"])
        return []

    def add_node(
        self,
        name: str,
        node_type: str,
        uuid: Optional[str] = None,
        ancestors: Optional[List[str]] = None,
        is_type_definition: bool = False,
        properties: Optional[Dict[str, Any]] = None,
        *,
        source: str = "unknown",
        derivation: str = "observed",
        confidence: Optional[float] = None,
        tags: Optional[List[str]] = None,
        links: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create or update a node with logos-standard properties and provenance.

        Args:
            name: Human-readable name (e.g., "red_block", "pick_action")
            node_type: Semantic type (e.g., "object", "action", "location")
            uuid: Unique identifier. If None, auto-generated. Provide to update existing.
            ancestors: Type inheritance chain. If None, automatically computed
                from type definition as [node_type] + type_def.ancestors
            is_type_definition: True if this node defines a type, False for instances
            properties: Additional custom properties
            source: Module/job that created this node (e.g., "planner", "jepa_runner")
            derivation: How the node was derived: "observed", "imagined", "reflected"
            confidence: Optional certainty score 0.0-1.0
            tags: Free-form labels for classification
            links: Related entity IDs (e.g., {"process_ids": [...], "plan_id": "..."})

        Returns:
            The uuid of the created/updated node

        Raises:
            ValueError: If name or node_type is empty, or validation fails
        """
        if not name:
            raise ValueError("name cannot be empty")
        if not node_type:
            raise ValueError("node_type cannot be empty")

        # Generate UUID if not provided, reject empty string
        if uuid is None:
            uuid = str(uuid4())
        elif uuid == "":
            raise ValueError("uuid cannot be empty")

        # Auto-compute ancestors if not provided
        if ancestors is None:
            if is_type_definition:
                # Type definitions should have ancestors explicitly provided
                ancestors = []
            else:
                # Instance nodes: [node_type] + type_definition.ancestors
                type_ancestors = self._get_type_ancestors(node_type)
                ancestors = [node_type] + type_ancestors

        # Generate timestamps
        now = datetime.now(timezone.utc).isoformat()

        # Build provenance properties
        provenance: Dict[str, Any] = {
            "source": source,
            "derivation": derivation,
            "created": now,
            "updated": now,
            "tags": tags or [],
            "links": links or {},
        }
        if confidence is not None:
            provenance["confidence"] = confidence

        # Merge provenance with custom properties (provenance takes precedence)
        merged_properties = {**(properties or {}), **provenance}

        node_data = {
            "uuid": uuid,
            "name": name,
            "type": node_type,
            "ancestors": ancestors,
            "is_type_definition": is_type_definition,
            "properties": merged_properties,
        }

        is_valid, errors = self._validator.validate_node(node_data)
        if not is_valid:
            raise ValueError(f"Node validation failed: {'; '.join(errors)}")

        query = """
        MERGE (n:Node {uuid: $uuid})
        SET n.name = $name,
            n.type = $type,
            n.is_type_definition = $is_type_definition,
            n.ancestors = $ancestors
        SET n += $properties
        RETURN n.uuid as uuid
        """
        encoded_properties = self._encode_properties(
            cast(Mapping[str, Any], node_data["properties"])
        )
        records = self._execute_query(
            query,
            {
                "uuid": uuid,
                "name": name,
                "type": node_type,
                "is_type_definition": is_type_definition,
                "ancestors": ancestors,
                "properties": encoded_properties,
            },
        )
        return str(records[0]["uuid"]) if records else uuid

    def update_node(
        self,
        uuid: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Update an existing node's properties and timestamp.

        Args:
            uuid: Unique identifier of the node to update
            properties: Properties to merge into the node (optional)

        Returns:
            The uuid of the updated node

        Raises:
            ValueError: If node with uuid doesn't exist
        """
        if not uuid:
            raise ValueError("uuid cannot be empty")

        # Generate updated timestamp
        now = datetime.now(timezone.utc).isoformat()

        # Build properties to update
        update_props = {**(properties or {}), "updated": now}

        query = """
        MATCH (n:Node {uuid: $uuid})
        SET n += $properties
        RETURN n.uuid as uuid
        """
        encoded_properties = self._encode_properties(
            cast(Mapping[str, Any], update_props)
        )
        records = self._execute_query(
            query,
            {
                "uuid": uuid,
                "properties": encoded_properties,
            },
        )
        if not records:
            raise ValueError(f"Node with uuid '{uuid}' not found")
        return str(records[0]["uuid"])

    def add_node_legacy(
        self,
        node_id: str,
        node_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """DEPRECATED: Use add_node() with logos-standard properties.

        This method provides backward compatibility during migration.
        Maps old signature to new:
        - node_id → uuid
        - name = properties.get("name", node_id)
        - ancestors = []
        - is_type_definition = False

        Args:
            node_id: Node identifier (maps to uuid)
            node_type: Semantic type
            properties: Additional properties

        Returns:
            The uuid of the created/updated node
        """
        import warnings

        warnings.warn(
            "add_node_legacy is deprecated, use add_node() with logos-standard properties",
            DeprecationWarning,
            stacklevel=2,
        )
        props = dict(properties) if properties else {}
        name = props.pop("name", node_id)
        return self.add_node(
            uuid=node_id,
            name=name,
            node_type=node_type,
            ancestors=[],
            is_type_definition=False,
            properties=props,
        )

    def add_edge(
        self,
        edge_id: str,
        source_uuid: str,
        target_uuid: str,
        relation: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create or update an edge after SHACL validation."""
        edge_data = {
            "id": edge_id,
            "source": source_uuid,
            "target": target_uuid,
            "relation": relation,
            "properties": properties or {},
        }

        is_valid, errors = self._validator.validate_edge(edge_data)
        if not is_valid:
            raise ValueError(f"Edge validation failed: {'; '.join(errors)}")

        query = """
        MATCH (source:Node {uuid: $source_uuid})
        MATCH (target:Node {uuid: $target_uuid})
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
                "source_uuid": source_uuid,
                "target_uuid": target_uuid,
                "relation": relation,
                "properties": encoded_properties,
            },
        )
        return str(records[0]["id"]) if records else edge_id

    def get_node(self, uuid: str) -> Optional[Dict[str, Any]]:
        """Fetch a node by uuid."""
        query = """
        MATCH (n:Node {uuid: $uuid})
        RETURN n.uuid as uuid, n.name as name, n.type as type,
               n.is_type_definition as is_type_definition,
               n.ancestors as ancestors, properties(n) as props
        """
        records = self._execute_read(query, {"uuid": uuid})
        if not records:
            return None

        props = dict(records[0]["props"])
        # Remove standard properties from props dict
        for key in ["uuid", "name", "type", "is_type_definition", "ancestors"]:
            props.pop(key, None)
        props = self._decode_properties(props)
        return {
            "uuid": records[0]["uuid"],
            "name": records[0]["name"],
            "type": records[0]["type"],
            "is_type_definition": records[0]["is_type_definition"],
            "ancestors": records[0]["ancestors"] or [],
            "properties": props,
        }

    def get_edge(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """Fetch an edge by ID."""
        query = """
        MATCH (source:Node)-[r:RELATION {id: $id}]->(target:Node)
        RETURN r.id as id, source.uuid as source, target.uuid as target,
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

    def query_neighbors(self, uuid: str) -> List[Dict[str, Any]]:
        """Return unique neighbor nodes for the provided node."""
        query = """
        MATCH (n:Node {uuid: $uuid})-[r]-(neighbor:Node)
        RETURN DISTINCT neighbor.uuid as uuid, neighbor.name as name,
               neighbor.type as type, neighbor.is_type_definition as is_type_definition,
               neighbor.ancestors as ancestors, properties(neighbor) as props
        """
        records = self._execute_read(query, {"uuid": uuid})
        neighbors: List[Dict[str, Any]] = []
        for record in records:
            props = dict(record["props"])
            for key in ["uuid", "name", "type", "is_type_definition", "ancestors"]:
                props.pop(key, None)
            props = self._decode_properties(props)
            neighbors.append(
                {
                    "uuid": record["uuid"],
                    "name": record["name"],
                    "type": record["type"],
                    "is_type_definition": record["is_type_definition"],
                    "ancestors": record["ancestors"] or [],
                    "properties": props,
                }
            )
        return neighbors

    def query_edges_from(self, uuid: str) -> List[Dict[str, Any]]:
        """Return outgoing edges for the provided node."""
        query = """
        MATCH (source:Node {uuid: $uuid})-[r:RELATION]->(target:Node)
        RETURN r.id as id, source.uuid as source, target.uuid as target,
               r.relation_type as relation, properties(r) as props
        """
        records = self._execute_read(query, {"uuid": uuid})
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

    def list_all_nodes(
        self,
        node_type: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Return all nodes in the graph, optionally filtered by type.

        Args:
            node_type: Optional filter by node type
            limit: Maximum number of nodes to return (default 1000)

        Returns:
            List of node dictionaries with uuid, name, type, ancestors, properties
        """
        if node_type:
            query = """
            MATCH (n:Node {type: $node_type})
            RETURN n.uuid as uuid, n.name as name, n.type as type,
                   n.is_type_definition as is_type_definition,
                   n.ancestors as ancestors, properties(n) as props
            LIMIT $limit
            """
            records = self._execute_read(
                query, {"node_type": node_type, "limit": limit}
            )
        else:
            query = """
            MATCH (n:Node)
            RETURN n.uuid as uuid, n.name as name, n.type as type,
                   n.is_type_definition as is_type_definition,
                   n.ancestors as ancestors, properties(n) as props
            LIMIT $limit
            """
            records = self._execute_read(query, {"limit": limit})

        nodes: List[Dict[str, Any]] = []
        for record in records:
            props = dict(record["props"])
            # Remove standard properties from props dict
            for key in ["uuid", "name", "type", "is_type_definition", "ancestors"]:
                props.pop(key, None)
            props = self._decode_properties(props)
            nodes.append(
                {
                    "uuid": record["uuid"],
                    "name": record["name"],
                    "type": record["type"],
                    "is_type_definition": record["is_type_definition"],
                    "ancestors": record["ancestors"] or [],
                    "properties": props,
                }
            )
        return nodes

    def list_all_edges(
        self,
        relation_type: Optional[str] = None,
        source_uuid: Optional[str] = None,
        target_uuid: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Return all edges in the graph with optional filters.

        Args:
            relation_type: Optional filter by relationship type
            source_uuid: Optional filter by source node uuid
            target_uuid: Optional filter by target node uuid
            limit: Maximum number of edges to return (default 1000)

        Returns:
            List of edge dictionaries with id, source, target, relation, properties
        """
        # Build query with optional filters
        conditions = []
        params: Dict[str, Any] = {"limit": limit}

        if relation_type:
            conditions.append("r.relation_type = $relation_type")
            params["relation_type"] = relation_type
        if source_uuid:
            conditions.append("source.uuid = $source_uuid")
            params["source_uuid"] = source_uuid
        if target_uuid:
            conditions.append("target.uuid = $target_uuid")
            params["target_uuid"] = target_uuid

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
        MATCH (source:Node)-[r:RELATION]->(target:Node)
        {where_clause}
        RETURN r.id as id, source.uuid as source, target.uuid as target,
               r.relation_type as relation, properties(r) as props
        LIMIT $limit
        """
        records = self._execute_read(query, params)

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

    def delete_node(self, uuid: str) -> bool:
        """Delete a node and all relationships."""
        query = """
        MATCH (n:Node {uuid: $uuid})
        DETACH DELETE n
        RETURN count(n) as deleted
        """
        records = self._execute_query(query, {"uuid": uuid})
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
