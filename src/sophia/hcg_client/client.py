"""Lightweight wrapper around the shared LOGOS HCG client.

This module reuses the canonical `logos_hcg` package for connection management,
retry logic, and query helpers while layering on Sophia-specific SHACL
validation and helper methods that were previously duplicated in this repo.

Edges are stored as *reified* edge nodes connected to source/target via
structural :FROM/:TO native Neo4j relationships.  This mirrors the foundry's
``add_edge()`` contract established in Task 1 of the cognitive-loop plan.
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
        neo4j_password: str = "logosdev",
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
        name: str,
        node_type: str,
        uuid: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        *,
        source: str = "unknown",
        derivation: str = "observed",
        confidence: Optional[float] = None,
        tags: Optional[List[str]] = None,
        links: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create or update a node with logos-standard properties and provenance.

        Type hierarchy is expressed through IS_A edge nodes, not stored as
        node properties.  Use ``add_edge(relation="IS_A")`` for hierarchy.

        Args:
            name: Human-readable name (e.g., "red_block", "pick_action")
            node_type: Semantic type (e.g., "object", "action", "location")
            uuid: Unique identifier. If None, auto-generated. Provide to update existing.
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

        # Generate UUID if not provided, reject empty string.
        # When no uuid is given, check for an existing node with the same
        # name+type to avoid creating duplicates on repeated ingestion.
        if uuid is None:
            existing = self._execute_query(
                "MATCH (n:Node {name: $name, type: $type}) RETURN n.uuid AS uuid LIMIT 1",
                {"name": name, "type": node_type},
            )
            if existing:
                uuid = existing[0]["uuid"]
            else:
                uuid = str(uuid4())
        elif uuid == "":
            raise ValueError("uuid cannot be empty")

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
            "properties": merged_properties,
        }

        is_valid, errors = self._validator.validate_node(node_data)
        if not is_valid:
            raise ValueError(f"Node validation failed: {'; '.join(errors)}")

        query = """
        MERGE (n:Node {uuid: $uuid})
        SET n.name = $name,
            n.type = $type
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
        - node_id -> uuid
        - name = properties.get("name", node_id)

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
            properties=props,
        )

    def add_edge(
        self,
        source_uuid: str,
        target_uuid: str,
        relation: str,
        edge_uuid: Optional[str] = None,
        bidirectional: bool = False,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create or update a reified edge node after SHACL validation.

        The edge is stored as a node connected to source and target::

            (source)<-[:FROM]-(edge_node)-[:TO]->(target)

        Args:
            source_uuid: Source node UUID
            target_uuid: Target node UUID
            relation: Edge type (e.g., "IS_A", "CAUSES", "LOCATED_AT")
            edge_uuid: Unique identifier for the edge node (auto-generated if None)
            bidirectional: Whether the relationship is bidirectional
            properties: Additional properties on the edge node

        Returns:
            The UUID of the created/updated edge node
        """
        from uuid import uuid4

        edge_id = edge_uuid or str(uuid4())
        edge_data = {
            "id": edge_id,
            "source": source_uuid,
            "target": target_uuid,
            "relation": relation,
            "bidirectional": bidirectional,
            "properties": properties or {},
        }

        is_valid, errors = self._validator.validate_edge(edge_data)
        if not is_valid:
            raise ValueError(f"Edge validation failed: {'; '.join(errors)}")

        now = datetime.now(timezone.utc).isoformat()

        props: Dict[str, Any] = {
            "uuid": edge_id,
            "type": "edge",
            "relation": relation,
            "source": source_uuid,
            "target": target_uuid,
            "bidirectional": bidirectional,
            "created_at": now,
            "updated_at": now,
        }
        if properties:
            encoded = self._encode_properties(cast(Mapping[str, Any], properties))
            props.update(encoded)

        # Build update-only props that preserve uuid and created_at on match.
        update_props = {
            k: v for k, v in props.items() if k not in ("uuid", "created_at")
        }
        update_props["updated_at"] = now

        # MERGE on composite key (source + target + relation) for idempotency.
        query = """
        MATCH (src:Node {uuid: $source_uuid})
        MATCH (tgt:Node {uuid: $target_uuid})
        MERGE (edge:Node {source: $source_uuid, target: $target_uuid, relation: $relation})
        ON CREATE SET edge += $props,
                      edge.name = src.name + '_' + $relation + '_' + tgt.name
        ON MATCH SET edge += $update_props
        MERGE (edge)-[:FROM]->(src)
        MERGE (edge)-[:TO]->(tgt)
        RETURN edge.uuid AS uuid
        """
        result = self._execute_query(
            query,
            {
                "source_uuid": source_uuid,
                "target_uuid": target_uuid,
                "relation": relation,
                "props": props,
                "update_props": update_props,
                "now": now,
            },
        )
        # If MERGE matched an existing edge, return its UUID instead
        if result and result[0].get("uuid"):
            return str(result[0]["uuid"])
        return edge_id

    def get_node(self, uuid: str) -> Optional[Dict[str, Any]]:
        """Fetch a content node by uuid.

        Returns None for edge nodes (those with a ``relation`` property).
        """
        query = """
        MATCH (n:Node {uuid: $uuid})
        WHERE n.relation IS NULL
        RETURN n.uuid as uuid, n.name as name, n.type as type,
               properties(n) as props
        """
        records = self._execute_read(query, {"uuid": uuid})
        if not records:
            return None

        props = dict(records[0]["props"])
        # Remove standard properties from props dict
        for key in ["uuid", "name", "type"]:
            props.pop(key, None)
        props = self._decode_properties(props)
        return {
            "uuid": records[0]["uuid"],
            "name": records[0]["name"],
            "type": records[0]["type"],
            "properties": props,
        }

    def find_node_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Find a content node by exact name match.

        Returns the first matching node or None.
        """
        query = """
        MATCH (n:Node {name: $name})
        WHERE n.relation IS NULL
        RETURN n.uuid as uuid, n.name as name, n.type as type,
               properties(n) as props
        LIMIT 1
        """
        records = self._execute_read(query, {"name": name})
        if not records:
            return None

        props = dict(records[0]["props"])
        for key in ["uuid", "name", "type"]:
            props.pop(key, None)
        props = self._decode_properties(props)
        return {
            "uuid": records[0]["uuid"],
            "name": records[0]["name"],
            "type": records[0]["type"],
            "properties": props,
        }

    def get_nodes_batch(self, uuids: list[str]) -> list[dict]:
        """Fetch multiple content nodes by uuid in a single query.

        Returns a list of node dicts with keys: uuid, name, type, properties.
        Edge nodes (those with a ``relation`` property) are excluded.
        Returns an empty list when *uuids* is empty.
        """
        if not uuids:
            return []

        query = """
        MATCH (n:Node) WHERE n.uuid IN $uuids AND n.relation IS NULL
        RETURN n.uuid as uuid, n.name as name, n.type as type,
               properties(n) as props
        """
        records = self._execute_read(query, {"uuids": uuids})

        results: list[dict] = []
        for record in records:
            props = dict(record["props"])
            for key in ["uuid", "name", "type"]:
                props.pop(key, None)
            props = self._decode_properties(props)
            results.append(
                {
                    "uuid": record["uuid"],
                    "name": record["name"],
                    "type": record["type"],
                    "properties": props,
                }
            )
        return results

    def find_nodes_by_names(self, names: list[str]) -> dict[str, dict]:
        """Find multiple content nodes by exact name match in a single query.

        Returns a dict mapping each name to its first matching node dict.
        Edge nodes (those with a ``relation`` property) are excluded.
        Returns an empty dict when *names* is empty.
        """
        if not names:
            return {}

        query = """
        MATCH (n:Node) WHERE n.name IN $names AND n.relation IS NULL
        RETURN n.uuid as uuid, n.name as name, n.type as type,
               properties(n) as props
        """
        records = self._execute_read(query, {"names": names})

        results: dict[str, dict] = {}
        for record in records:
            name = record["name"]
            if name in results:
                continue  # keep first match per name
            props = dict(record["props"])
            for key in ["uuid", "name", "type"]:
                props.pop(key, None)
            props = self._decode_properties(props)
            results[name] = {
                "uuid": record["uuid"],
                "name": name,
                "type": record["type"],
                "properties": props,
            }
        return results

    def get_edge(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a reified edge node by UUID.

        Queries an edge node (has ``relation`` property) and resolves its
        source/target via :FROM/:TO structural relationships.
        """
        query = """
        MATCH (edge:Node {uuid: $uuid})
        WHERE edge.relation IS NOT NULL
        OPTIONAL MATCH (edge)-[:FROM]->(src:Node)
        OPTIONAL MATCH (edge)-[:TO]->(tgt:Node)
        RETURN edge.uuid as id,
               edge.source as source,
               edge.target as target,
               edge.relation as relation,
               edge.bidirectional as bidirectional,
               properties(edge) as props
        """
        records = self._execute_read(query, {"uuid": edge_id})
        if not records:
            return None

        props = dict(records[0]["props"])
        # Remove structural properties from the extra-props dict
        for key in [
            "uuid",
            "name",
            "type",
            "relation",
            "source",
            "target",
            "bidirectional",
            "created_at",
            "updated_at",
        ]:
            props.pop(key, None)
        props = self._decode_properties(props)
        return {
            "id": records[0]["id"],
            "source": records[0]["source"],
            "target": records[0]["target"],
            "relation": records[0]["relation"],
            "bidirectional": bool(records[0].get("bidirectional", False)),
            "properties": props,
        }

    def query_neighbors(self, uuid: str) -> List[Dict[str, Any]]:
        """Return unique content-node neighbors connected via edge nodes.

        Traverses outgoing edges (node is source) and incoming edges
        (node is target) through reified edge nodes.
        """
        query = """
        MATCH (n:Node {uuid: $uuid})
        // Find neighbors via outgoing edges (n is source)
        OPTIONAL MATCH (n)<-[:FROM]-(:Node)-[:TO]->(out_neighbor:Node)
        WHERE out_neighbor.relation IS NULL
        // Find neighbors via incoming edges (n is target)
        OPTIONAL MATCH (n)<-[:TO]-(:Node)-[:FROM]->(in_neighbor:Node)
        WHERE in_neighbor.relation IS NULL
        // Collect and return unique neighbors
        WITH collect(DISTINCT out_neighbor) + collect(DISTINCT in_neighbor) as all_neighbors
        UNWIND all_neighbors as neighbor
        WHERE neighbor IS NOT NULL AND neighbor.uuid <> $uuid
        RETURN DISTINCT neighbor.uuid as uuid, neighbor.name as name,
               neighbor.type as type, properties(neighbor) as props
        """
        records = self._execute_read(query, {"uuid": uuid})
        neighbors: List[Dict[str, Any]] = []
        for record in records:
            props = dict(record["props"])
            for key in ["uuid", "name", "type"]:
                props.pop(key, None)
            props = self._decode_properties(props)
            neighbors.append(
                {
                    "uuid": record["uuid"],
                    "name": record["name"],
                    "type": record["type"],
                    "properties": props,
                }
            )
        return neighbors

    def query_edges_from(self, uuid: str) -> List[Dict[str, Any]]:
        """Return outgoing reified edges where the node is the source."""
        query = """
        MATCH (edge:Node)-[:FROM]->(src:Node {uuid: $uuid})
        WHERE edge.relation IS NOT NULL
        OPTIONAL MATCH (edge)-[:TO]->(tgt:Node)
        RETURN edge.uuid as id,
               edge.source as source,
               edge.target as target,
               edge.relation as relation,
               edge.bidirectional as bidirectional,
               properties(edge) as props
        """
        records = self._execute_read(query, {"uuid": uuid})
        edges: List[Dict[str, Any]] = []
        for record in records:
            props = dict(record["props"])
            for key in [
                "uuid",
                "name",
                "type",
                "relation",
                "source",
                "target",
                "bidirectional",
                "created_at",
                "updated_at",
            ]:
                props.pop(key, None)
            props = self._decode_properties(props)
            edges.append(
                {
                    "id": record["id"],
                    "source": record["source"],
                    "target": record["target"],
                    "relation": record["relation"],
                    "bidirectional": bool(record.get("bidirectional", False)),
                    "properties": props,
                }
            )
        return edges

    def list_all_nodes(
        self,
        node_type: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Return all content nodes in the graph, optionally filtered by type.

        Edge nodes (those with a ``relation`` property) are excluded.

        Args:
            node_type: Optional filter by node type
            limit: Maximum number of nodes to return (default 1000)

        Returns:
            List of node dictionaries with uuid, name, type, properties
        """
        if node_type:
            query = """
            MATCH (n:Node {type: $node_type})
            WHERE n.relation IS NULL
            RETURN n.uuid as uuid, n.name as name, n.type as type,
                   properties(n) as props
            LIMIT $limit
            """
            records = self._execute_read(
                query, {"node_type": node_type, "limit": limit}
            )
        else:
            query = """
            MATCH (n:Node)
            WHERE n.relation IS NULL
            RETURN n.uuid as uuid, n.name as name, n.type as type,
                   properties(n) as props
            LIMIT $limit
            """
            records = self._execute_read(query, {"limit": limit})

        nodes: List[Dict[str, Any]] = []
        for record in records:
            props = dict(record["props"])
            # Remove standard properties from props dict
            for key in ["uuid", "name", "type"]:
                props.pop(key, None)
            props = self._decode_properties(props)
            nodes.append(
                {
                    "uuid": record["uuid"],
                    "name": record["name"],
                    "type": record["type"],
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
        """Return all reified edge nodes in the graph with optional filters.

        Args:
            relation_type: Optional filter by relation type
            source_uuid: Optional filter by source node uuid
            target_uuid: Optional filter by target node uuid
            limit: Maximum number of edges to return (default 1000)

        Returns:
            List of edge dictionaries with id, source, target, relation, properties
        """
        # Build query with optional filters
        conditions = ["edge.relation IS NOT NULL"]
        params: Dict[str, Any] = {"limit": limit}

        if relation_type:
            conditions.append("edge.relation = $relation_type")
            params["relation_type"] = relation_type
        if source_uuid:
            conditions.append("edge.source = $source_uuid")
            params["source_uuid"] = source_uuid
        if target_uuid:
            conditions.append("edge.target = $target_uuid")
            params["target_uuid"] = target_uuid

        where_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
        MATCH (edge:Node)
        {where_clause}
        RETURN edge.uuid as id,
               edge.source as source,
               edge.target as target,
               edge.relation as relation,
               edge.bidirectional as bidirectional,
               properties(edge) as props
        LIMIT $limit
        """
        records = self._execute_read(query, params)

        edges: List[Dict[str, Any]] = []
        for record in records:
            props = dict(record["props"])
            for key in [
                "uuid",
                "name",
                "type",
                "relation",
                "source",
                "target",
                "bidirectional",
                "created_at",
                "updated_at",
            ]:
                props.pop(key, None)
            props = self._decode_properties(props)
            edges.append(
                {
                    "id": record["id"],
                    "source": record["source"],
                    "target": record["target"],
                    "relation": record["relation"],
                    "bidirectional": bool(record.get("bidirectional", False)),
                    "properties": props,
                }
            )
        return edges

    def get_subgraph(self, node_uuids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Return nodes and connecting edge nodes for the given UUIDs.

        Args:
            node_uuids: List of content-node UUIDs to include

        Returns:
            Dict with ``"nodes"`` and ``"edges"`` lists
        """
        if not node_uuids:
            return {"nodes": [], "edges": []}

        # Fetch the content nodes
        nodes_query = """
        MATCH (n:Node)
        WHERE n.uuid IN $uuids AND n.relation IS NULL
        RETURN n.uuid as uuid, n.name as name, n.type as type,
               properties(n) as props
        """
        node_records = self._execute_read(nodes_query, {"uuids": node_uuids})

        nodes: List[Dict[str, Any]] = []
        for record in node_records:
            props = dict(record["props"])
            for key in ["uuid", "name", "type"]:
                props.pop(key, None)
            props = self._decode_properties(props)
            nodes.append(
                {
                    "uuid": record["uuid"],
                    "name": record["name"],
                    "type": record["type"],
                    "properties": props,
                }
            )

        # Fetch edge nodes where both source and target are in the set
        edges_query = """
        MATCH (edge:Node)
        WHERE edge.relation IS NOT NULL
          AND edge.source IN $uuids
          AND edge.target IN $uuids
        RETURN edge.uuid as id,
               edge.source as source,
               edge.target as target,
               edge.relation as relation,
               edge.bidirectional as bidirectional,
               properties(edge) as props
        """
        edge_records = self._execute_read(edges_query, {"uuids": node_uuids})

        edges: List[Dict[str, Any]] = []
        for record in edge_records:
            props = dict(record["props"])
            for key in [
                "uuid",
                "name",
                "type",
                "relation",
                "source",
                "target",
                "bidirectional",
                "created_at",
                "updated_at",
            ]:
                props.pop(key, None)
            props = self._decode_properties(props)
            edges.append(
                {
                    "id": record["id"],
                    "source": record["source"],
                    "target": record["target"],
                    "relation": record["relation"],
                    "bidirectional": bool(record.get("bidirectional", False)),
                    "properties": props,
                }
            )

        return {"nodes": nodes, "edges": edges}

    def delete_node(self, uuid: str) -> bool:
        """Delete a content node, its edge nodes, and all relationships."""
        # First delete any edge nodes that reference this node
        cleanup_query = """
        MATCH (edge:Node)
        WHERE edge.relation IS NOT NULL
          AND (edge.source = $uuid OR edge.target = $uuid)
        DETACH DELETE edge
        """
        self._execute_query(cleanup_query, {"uuid": uuid})

        # Then delete the content node itself
        query = """
        MATCH (n:Node {uuid: $uuid})
        DETACH DELETE n
        RETURN count(n) as deleted
        """
        records = self._execute_query(query, {"uuid": uuid})
        deleted = records[0]["deleted"] if records else 0
        return bool(deleted)

    def delete_edge(self, edge_uuid: str) -> bool:
        """Delete a reified edge by its uuid.

        Edges are stored as ``:Node`` records (see :meth:`add_edge`), so
        removing an edge is just :meth:`delete_node` on the edge's uuid. This
        wrapper names the intent at the call site -- e.g. dropping a stale
        ``IS_A`` edge when a member is retyped (#149 review).
        """
        return self.delete_node(edge_uuid)

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
