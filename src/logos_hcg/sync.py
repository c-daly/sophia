"""
HCG-Milvus Synchronization Utility

This module provides bidirectional synchronization between Neo4j graph nodes
and Milvus vector embeddings, implementing Section 4.2 of the LOGOS specification.

Key features:
- Upsert embeddings to Milvus with metadata tracking
- Sync metadata back to Neo4j nodes
- Batch operations for efficiency
- Delete synchronization (cascade deletes)
- Health checks for consistency verification

See Project LOGOS spec: Section 4.2 (Vector Integration)
"""

import logging
from datetime import datetime, timezone
from typing import Any, Literal, cast
from uuid import UUID

from pymilvus import Collection, connections, utility

logger = logging.getLogger(__name__)

# Collection name mapping for HCG node types
COLLECTION_NAMES = {
    "Entity": "hcg_entity_embeddings",
    "Concept": "hcg_concept_embeddings",
    "State": "hcg_state_embeddings",
    "Process": "hcg_process_embeddings",
}

NodeType = Literal["Entity", "Concept", "State", "Process"]


class MilvusSyncError(Exception):
    """Raised when synchronization with Milvus fails."""

    pass


class HCGMilvusSync:
    """
    Synchronization manager for HCG-Milvus bidirectional sync.

    Handles:
    - Upserting embeddings to Milvus
    - Tracking metadata in Neo4j
    - Batch operations
    - Delete synchronization
    - Consistency verification

    Example:
        sync = HCGMilvusSync(milvus_host="localhost", milvus_port="19530")
        sync.upsert_embedding(
            node_type="Entity",
            uuid="entity-uuid",
            embedding=[0.1, 0.2, ...],
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
    """

    def __init__(
        self,
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        alias: str = "default",
    ):
        """
        Initialize sync manager with Milvus connection.

        Args:
            milvus_host: Milvus server host
            milvus_port: Milvus server port
            alias: Connection alias for Milvus
        """
        self.host = milvus_host
        self.port = milvus_port
        self.alias = alias
        self._connected = False
        self._collections: dict[str, Collection] = {}

    def connect(self) -> None:
        """
        Establish connection to Milvus and load collections.

        Raises:
            MilvusSyncError: If connection fails
        """
        try:
            connections.connect(
                alias=self.alias,
                host=self.host,
                port=self.port,
            )
            self._connected = True
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")

            # Load collections into memory
            for node_type, collection_name in COLLECTION_NAMES.items():
                if utility.has_collection(collection_name, using=self.alias):
                    collection = Collection(name=collection_name, using=self.alias)
                    collection.load()
                    self._collections[node_type] = collection
                    logger.debug(f"Loaded collection: {collection_name}")
                else:
                    logger.warning(f"Collection not found: {collection_name}")

        except Exception as e:
            raise MilvusSyncError(f"Failed to connect to Milvus: {e}") from e

    def disconnect(self) -> None:
        """Disconnect from Milvus."""
        if self._connected:
            connections.disconnect(self.alias)
            self._connected = False
            self._collections.clear()
            logger.info("Disconnected from Milvus")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

    def _get_collection(self, node_type: NodeType) -> Collection:
        """
        Get Milvus collection for a node type.

        Args:
            node_type: Type of HCG node

        Returns:
            Milvus Collection object

        Raises:
            MilvusSyncError: If collection not available
        """
        if not self._connected:
            raise MilvusSyncError("Not connected to Milvus. Call connect() first.")

        if node_type not in self._collections:
            raise MilvusSyncError(
                f"Collection for {node_type} not loaded. "
                f"Run init_milvus_collections.py to create collections."
            )

        return self._collections[node_type]

    def upsert_embedding(
        self,
        node_type: NodeType,
        uuid: str | UUID,
        embedding: list[float],
        model: str,
    ) -> dict[str, Any]:
        """
        Upsert an embedding to Milvus with metadata.

        This creates or updates the embedding in Milvus and returns metadata
        that should be stored in the Neo4j node.

        Args:
            node_type: Type of HCG node (Entity, Concept, State, Process)
            uuid: Node UUID
            embedding: Vector embedding (dimensionality must match collection schema)
            model: Model name used to generate the embedding

        Returns:
            Dictionary with embedding metadata:
            {
                "embedding_id": str,
                "embedding_model": str,
                "last_sync": datetime
            }

        Raises:
            MilvusSyncError: If upsert fails
        """
        uuid_str = str(uuid)
        collection = self._get_collection(node_type)

        try:
            # Prepare data for insertion
            # Milvus uses upsert based on primary key (uuid)
            timestamp = int(datetime.now(timezone.utc).timestamp())
            data = [
                [uuid_str],  # uuid
                [embedding],  # embedding
                [model],  # embedding_model
                [timestamp],  # last_sync
            ]

            # Upsert to Milvus (replaces if exists)
            collection.insert(data)
            collection.flush()

            logger.info(f"Upserted embedding for {node_type} {uuid_str}")

            return {
                "embedding_id": uuid_str,
                "embedding_model": model,
                "last_sync": datetime.now(timezone.utc),
            }

        except Exception as e:
            raise MilvusSyncError(
                f"Failed to upsert embedding for {node_type} {uuid_str}: {e}"
            ) from e

    def batch_upsert_embeddings(
        self,
        node_type: NodeType,
        embeddings: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Batch upsert embeddings to Milvus.

        Args:
            node_type: Type of HCG node
            embeddings: List of dicts with keys: uuid, embedding, model

        Returns:
            List of embedding metadata dicts (one per input)

        Raises:
            MilvusSyncError: If batch upsert fails
        """
        if not embeddings:
            return []

        collection = self._get_collection(node_type)

        try:
            # Prepare batch data
            timestamp = int(datetime.now(timezone.utc).timestamp())
            uuids = [str(e["uuid"]) for e in embeddings]
            vectors = [e["embedding"] for e in embeddings]
            models = [e["model"] for e in embeddings]
            timestamps = [timestamp] * len(embeddings)

            data = [uuids, vectors, models, timestamps]

            # Batch insert
            collection.insert(data)
            collection.flush()

            logger.info(f"Batch upserted {len(embeddings)} embeddings for {node_type}")

            # Return metadata for each embedding
            sync_time = datetime.now(timezone.utc)
            return [
                {
                    "embedding_id": uuid,
                    "embedding_model": model,
                    "last_sync": sync_time,
                }
                for uuid, model in zip(uuids, models, strict=True)
            ]

        except Exception as e:
            raise MilvusSyncError(
                f"Failed to batch upsert embeddings for {node_type}: {e}"
            ) from e

    def delete_embedding(
        self,
        node_type: NodeType,
        uuid: str | UUID,
    ) -> bool:
        """
        Delete an embedding from Milvus.

        This should be called when a node is deleted from Neo4j to maintain sync.

        Args:
            node_type: Type of HCG node
            uuid: Node UUID

        Returns:
            True if deletion was successful

        Raises:
            MilvusSyncError: If deletion fails
        """
        uuid_str = str(uuid)
        collection = self._get_collection(node_type)

        try:
            # Delete by primary key
            expr = f'uuid == "{uuid_str}"'
            collection.delete(expr)
            collection.flush()

            logger.info(f"Deleted embedding for {node_type} {uuid_str}")
            return True

        except Exception as e:
            raise MilvusSyncError(
                f"Failed to delete embedding for {node_type} {uuid_str}: {e}"
            ) from e

    def batch_delete_embeddings(
        self,
        node_type: NodeType,
        uuids: list[str | UUID],
    ) -> int:
        """
        Batch delete embeddings from Milvus.

        Args:
            node_type: Type of HCG node
            uuids: List of node UUIDs to delete

        Returns:
            Number of embeddings deleted

        Raises:
            MilvusSyncError: If batch deletion fails
        """
        if not uuids:
            return 0

        collection = self._get_collection(node_type)
        uuid_strs = [str(u) for u in uuids]

        try:
            # Build delete expression (OR of all UUIDs)
            uuid_list = ", ".join([f'"{u}"' for u in uuid_strs])
            expr = f"uuid in [{uuid_list}]"
            collection.delete(expr)
            collection.flush()

            logger.info(f"Batch deleted {len(uuids)} embeddings for {node_type}")
            return len(uuids)

        except Exception as e:
            raise MilvusSyncError(
                f"Failed to batch delete embeddings for {node_type}: {e}"
            ) from e

    def get_embedding(
        self,
        node_type: NodeType,
        uuid: str | UUID,
    ) -> dict[str, Any] | None:
        """
        Retrieve an embedding and its metadata from Milvus.

        Args:
            node_type: Type of HCG node
            uuid: Node UUID

        Returns:
            Dictionary with embedding data or None if not found:
            {
                "uuid": str,
                "embedding": list[float],
                "embedding_model": str,
                "last_sync": int (unix timestamp)
            }

        Raises:
            MilvusSyncError: If query fails
        """
        uuid_str = str(uuid)
        collection = self._get_collection(node_type)

        try:
            # Query by primary key
            expr = f'uuid == "{uuid_str}"'
            results = collection.query(
                expr=expr,
                output_fields=["uuid", "embedding", "embedding_model", "last_sync"],
            )

            if not results:
                return None

            return cast(dict[str, Any], results[0])

        except Exception as e:
            raise MilvusSyncError(
                f"Failed to get embedding for {node_type} {uuid_str}: {e}"
            ) from e

    def verify_sync(
        self,
        node_type: NodeType,
        neo4j_uuids: set[str],
    ) -> dict[str, Any]:
        """
        Verify synchronization consistency between Neo4j and Milvus.

        Compares the set of UUIDs in Neo4j with those in Milvus to detect:
        - Orphaned embeddings (in Milvus but not in Neo4j)
        - Missing embeddings (in Neo4j but not in Milvus)

        Args:
            node_type: Type of HCG node
            neo4j_uuids: Set of UUIDs from Neo4j for this node type

        Returns:
            Dictionary with consistency report:
            {
                "in_sync": bool,
                "neo4j_count": int,
                "milvus_count": int,
                "orphaned_embeddings": list[str],  # In Milvus but not Neo4j
                "missing_embeddings": list[str],   # In Neo4j but not Milvus
            }

        Raises:
            MilvusSyncError: If verification fails
        """
        collection = self._get_collection(node_type)

        try:
            # Get all UUIDs from Milvus
            results = collection.query(
                expr="uuid != ''",  # Match all
                output_fields=["uuid"],
            )
            milvus_uuids = {r["uuid"] for r in results}

            # Find discrepancies
            orphaned = milvus_uuids - neo4j_uuids
            missing = neo4j_uuids - milvus_uuids

            report = {
                "in_sync": len(orphaned) == 0 and len(missing) == 0,
                "neo4j_count": len(neo4j_uuids),
                "milvus_count": len(milvus_uuids),
                "orphaned_embeddings": sorted(orphaned),
                "missing_embeddings": sorted(missing),
            }

            if not report["in_sync"]:
                logger.warning(
                    f"Sync inconsistency detected for {node_type}: "
                    f"{len(orphaned)} orphaned, {len(missing)} missing"
                )

            return report

        except Exception as e:
            raise MilvusSyncError(f"Failed to verify sync for {node_type}: {e}") from e

    def health_check(self) -> dict[str, Any]:
        """
        Perform health check on Milvus collections.

        Returns:
            Dictionary with health status:
            {
                "connected": bool,
                "collections": {
                    "Entity": {"exists": bool, "loaded": bool, "count": int},
                    ...
                }
            }
        """
        status: dict[str, Any] = {
            "connected": self._connected,
            "collections": {},
        }

        if not self._connected:
            return status

        for node_type, collection_name in COLLECTION_NAMES.items():
            collection_status = {
                "exists": False,
                "loaded": False,
                "count": 0,
            }

            try:
                if utility.has_collection(collection_name, using=self.alias):
                    collection_status["exists"] = True
                    collection = Collection(name=collection_name, using=self.alias)

                    # Check if loaded
                    try:
                        collection_status["count"] = collection.num_entities
                        collection_status["loaded"] = True
                    except Exception:
                        # Collection exists but not loaded
                        collection_status["loaded"] = False

            except Exception as e:
                logger.error(f"Error checking collection {collection_name}: {e}")

            status["collections"][node_type] = collection_status

        return status
