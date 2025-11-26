"""Media ingestion service for handling uploads and Neo4j persistence."""

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from fastapi import UploadFile

from sophia.storage.media_storage import MediaStorageService
from sophia.models.media_models import (
    MediaType,
    MediaIngestResponse,
    MediaMetadata,
    MediaSampleResponse,
    MediaSamplesListResponse,
    MediaSampleQuery,
)
from sophia.hcg_client import HCGClient


logger = logging.getLogger(__name__)


class MediaIngestionService:
    """Service for ingesting media and storing metadata in Neo4j."""

    def __init__(
        self,
        hcg_client: HCGClient,
        storage_service: MediaStorageService,
        jepa_runner: Optional[Any] = None,
    ):
        """Initialize media ingestion service.

        Args:
            hcg_client: HCG client for Neo4j operations
            storage_service: Media storage service for file operations
            jepa_runner: Optional JEPA runner for perception processing
        """
        self.hcg_client = hcg_client
        self.storage = storage_service
        self.jepa_runner = jepa_runner

    async def ingest_media(
        self,
        file: UploadFile,
        media_type: MediaType,
        question: Optional[str] = None,
    ) -> MediaIngestResponse:
        """Ingest uploaded media file.

        Args:
            file: Uploaded file
            media_type: Type of media
            question: Optional perception question associated with the upload

        Returns:
            Media ingestion response with sample metadata
        """
        # Store file to disk
        sample_id, file_path, file_size = await self.storage.store_file(
            file, media_type
        )

        # Extract metadata
        metadata = self.storage.extract_metadata(file_path, media_type)

        # Compute file hash for deduplication
        file_hash = self.storage.compute_file_hash(file_path)

        # Get current timestamp
        timestamp = datetime.now(timezone.utc)

        # Create MediaSample node in Neo4j
        neo4j_node_id = self._create_media_sample_node(
            sample_id=sample_id,
            media_type=media_type,
            file_path=file_path,
            file_size=file_size,
            file_hash=file_hash,
            timestamp=timestamp,
            metadata=metadata,
            question=question,
        )

        logger.info(
            f"Successfully ingested {media_type.value} sample {sample_id} "
            f"(Neo4j: {neo4j_node_id})"
        )

        # Notify JEPA runner if available
        if self.jepa_runner:
            try:
                await self._notify_jepa_runner(
                    sample_id=sample_id,
                    media_type=media_type,
                    file_path=file_path,
                    metadata=metadata,
                    question=question,
                )
                logger.info(f"Notified JEPA runner about sample {sample_id}")
            except Exception as e:
                logger.warning(f"Failed to notify JEPA runner: {e}")

        return MediaIngestResponse(
            sample_id=sample_id,
            media_type=media_type,
            file_path=file_path,
            file_size=file_size,
            timestamp=timestamp,
            metadata=metadata.model_dump(),
            neo4j_node_id=neo4j_node_id,
        )

    def _create_media_sample_node(
        self,
        sample_id: str,
        media_type: MediaType,
        file_path: str,
        file_size: int,
        file_hash: str,
        timestamp: datetime,
        metadata: MediaMetadata,
        question: Optional[str] = None,
    ) -> str:
        """Create MediaSample node in Neo4j.

        Args:
            sample_id: Sample identifier
            media_type: Type of media
            file_path: Path to stored file
            file_size: File size in bytes
            file_hash: SHA256 hash of file
            timestamp: Upload timestamp
            metadata: Extracted metadata
            question: Optional perception question

        Returns:
            Neo4j node ID
        """
        properties = {
            "sample_id": sample_id,
            "media_type": media_type.value,
            "file_path": file_path,
            "file_size": file_size,
            "file_hash": file_hash,
            "timestamp": timestamp.isoformat(),
            "ingested_at": timestamp.isoformat(),
        }

        # Add optional question
        if question:
            properties["question"] = question

        # Add extracted metadata as flat properties
        metadata_dict = metadata.model_dump(exclude_none=True)
        for key, value in metadata_dict.items():
            properties[f"metadata_{key}"] = value

        # Create node using HCG client
        self.hcg_client.add_node(
            node_id=sample_id,
            node_type="media_sample",
            properties=properties,
        )

        logger.debug(f"Created MediaSample node {sample_id} in Neo4j")

        return sample_id

    def get_media_sample(self, sample_id: str) -> Optional[MediaSampleResponse]:
        """Retrieve media sample metadata from Neo4j.

        Args:
            sample_id: Sample identifier

        Returns:
            Media sample response or None if not found
        """
        try:
            node_data = self.hcg_client.get_node(sample_id)
            if not node_data:
                return None

            # Node data might be flat or nested - handle both
            properties = node_data.get("properties", node_data)

            # Reconstruct metadata from flat properties
            metadata_dict = {}
            for key, value in properties.items():
                if key.startswith("metadata_"):
                    field_name = key.replace("metadata_", "")
                    metadata_dict[field_name] = value
            metadata = MediaMetadata(**metadata_dict)

            # Count simulations using this sample
            simulation_count = self._count_simulations_for_sample(sample_id)

            return MediaSampleResponse(
                sample_id=properties.get("sample_id", sample_id),
                media_type=MediaType(properties["media_type"]),
                file_path=properties["file_path"],
                file_size=properties["file_size"],
                file_hash=properties["file_hash"],
                timestamp=datetime.fromisoformat(properties["timestamp"]),
                metadata=metadata,
                neo4j_node_id=properties.get("id", sample_id),
                simulation_count=simulation_count,
            )

        except Exception as e:
            logger.error(f"Failed to retrieve media sample {sample_id}: {e}")
            return None

    def list_media_samples(self, query: MediaSampleQuery) -> MediaSamplesListResponse:
        """List media samples with optional filtering.

        Args:
            query: Query parameters

        Returns:
            List of media samples
        """
        # Build Neo4j query
        cypher_conditions = ["n.type = 'media_sample'"]
        parameters: Dict[str, Any] = {}

        if query.media_type:
            cypher_conditions.append("n.media_type = $media_type")
            parameters["media_type"] = query.media_type.value

        if query.after_timestamp:
            cypher_conditions.append("n.timestamp > $after_timestamp")
            parameters["after_timestamp"] = query.after_timestamp.isoformat()

        where_clause = " AND ".join(cypher_conditions)

        # Query for total count
        count_query = f"""
        MATCH (n)
        WHERE {where_clause}
        RETURN count(n) as total
        """

        # Query for paginated results
        list_query = f"""
        MATCH (n)
        WHERE {where_clause}
        RETURN n
        ORDER BY n.timestamp DESC
        SKIP $offset
        LIMIT $limit
        """

        parameters["offset"] = query.offset
        parameters["limit"] = query.limit

        try:
            # Get total count
            with self.hcg_client.driver.session() as session:  # type: ignore[attr-defined]
                result = session.run(count_query, parameters)
                count_record = result.single()
                total = int(count_record["total"]) if count_record else 0

                # Get samples
                result = session.run(list_query, parameters)
                samples = []

                for record in result:
                    node_dict = record["n"]
                    # Handle both Neo4j node objects and dicts
                    if hasattr(node_dict, "__getitem__"):
                        properties = dict(node_dict)
                    else:
                        properties = node_dict

                    # Reconstruct metadata
                    metadata_dict = {}
                    for key, value in properties.items():
                        if key.startswith("metadata_"):
                            field_name = key.replace("metadata_", "")
                            metadata_dict[field_name] = value
                    metadata = MediaMetadata(**metadata_dict)

                    samples.append(
                        MediaSampleResponse(
                            sample_id=properties["sample_id"],
                            media_type=MediaType(properties["media_type"]),
                            file_path=properties["file_path"],
                            file_size=properties["file_size"],
                            file_hash=properties["file_hash"],
                            timestamp=datetime.fromisoformat(properties["timestamp"]),
                            metadata=metadata,
                            neo4j_node_id=properties["sample_id"],
                            simulation_count=0,  # Optimize: compute only if needed
                        )
                    )

            return MediaSamplesListResponse(
                samples=samples,
                total=total,
                limit=query.limit,
                offset=query.offset,
            )

        except Exception as e:
            logger.error(f"Failed to list media samples: {e}")
            return MediaSamplesListResponse(
                samples=[],
                total=0,
                limit=query.limit,
                offset=query.offset,
            )

    def _count_simulations_for_sample(self, sample_id: str) -> int:
        """Count simulations that used this media sample.

        Args:
            sample_id: Sample identifier

        Returns:
            Number of simulations
        """
        query = """
        MATCH (m {sample_id: $sample_id})-[:USED_IN]->(s)
        WHERE s.type = 'simulation'
        RETURN count(s) as count
        """

        try:
            with self.hcg_client.driver.session() as session:  # type: ignore[attr-defined]
                result = session.run(query, {"sample_id": sample_id})
                record = result.single()
                return record["count"] if record else 0
        except Exception as e:
            logger.debug(f"Failed to count simulations for {sample_id}: {e}")
            return 0

    def link_sample_to_simulation(self, sample_id: str, simulation_id: str) -> bool:
        """Link a media sample to a simulation that used it.

        Args:
            sample_id: Sample identifier
            simulation_id: Simulation identifier

        Returns:
            True if linked successfully
        """
        try:
            self.hcg_client.add_edge(
                edge_id=f"e_{sample_id}_{simulation_id}",
                source_id=sample_id,
                target_id=simulation_id,
                relation="used_in",
            )
            logger.debug(f"Linked sample {sample_id} to simulation {simulation_id}")
            return True
        except Exception as e:
            logger.error(
                f"Failed to link sample {sample_id} to simulation {simulation_id}: {e}"
            )
            return False

    async def _notify_jepa_runner(
        self,
        sample_id: str,
        media_type: MediaType,
        file_path: str,
        metadata: MediaMetadata,
        question: Optional[str] = None,
    ) -> None:
        """Notify JEPA runner about new media sample for perception processing.

        This hook processes the media sample through the JEPA runner to generate
        embeddings representing physical understanding. The embeddings are then
        stored in Milvus for later use in simulations.

        Args:
            sample_id: Sample identifier
            media_type: Type of media
            file_path: Path to stored file
            metadata: Extracted metadata
            question: Optional perception question
        """
        logger.info(
            f"[JEPA Hook] Processing sample {sample_id} ({media_type.value}) "
            f"at {file_path}"
        )
        if question:
            logger.info(f"[JEPA Hook] Perception question: {question}")

        if not self.jepa_runner:
            logger.warning("[JEPA Hook] JEPA runner not available, skipping processing")
            return

        try:
            # Process media through JEPA runner
            result = await self.jepa_runner.process_media_sample(
                sample_id=sample_id,
                file_path=file_path,
                media_type=media_type.value,
                metadata=metadata.model_dump(exclude_none=True),
                question=question,
            )

            # Store embeddings in Milvus
            await self._store_jepa_embeddings(
                sample_id=sample_id,
                embeddings=result["embeddings"],
                metadata={
                    "media_type": media_type.value,
                    "model_version": result["model_version"],
                    "confidence": result["confidence"],
                    "question": question,
                },
            )

            logger.info(
                f"[JEPA Hook] Successfully processed sample {sample_id}, "
                f"stored {len(result['embeddings'])} embedding sets in Milvus"
            )

        except Exception as e:
            logger.error(f"[JEPA Hook] Failed to process sample {sample_id}: {e}")
            # Don't raise - media ingestion should succeed even if JEPA processing fails

    async def _store_jepa_embeddings(
        self,
        sample_id: str,
        embeddings: Dict[str, list],
        metadata: Dict[str, Any],
    ) -> None:
        """Store JEPA-generated embeddings in Milvus.

        Args:
            sample_id: Media sample identifier
            embeddings: Dictionary of embedding vectors (visual, physics, etc.)
            metadata: Additional metadata to store with embeddings
        """
        if not hasattr(self.hcg_client, "_milvus"):
            logger.warning(
                f"[JEPA Hook] Milvus not available, skipping embedding storage for {sample_id}"
            )
            return

        try:
            # Store each embedding type (visual, physics, etc.)
            for embedding_type, embedding_vector in embeddings.items():
                embedding_id = f"{sample_id}_{embedding_type}"

                # Insert embedding into Milvus using private adapter
                self.hcg_client._milvus.insert_embedding(
                    embedding_id=embedding_id,
                    node_id=sample_id,
                    node_type=f"media_embedding_{embedding_type}",
                    embedding=embedding_vector,
                )

                logger.info(
                    f"[JEPA Hook] Stored {embedding_type} embedding {embedding_id} in Milvus"
                )

            # Create edge in Neo4j linking media sample to its embeddings
            for embedding_type in embeddings.keys():
                embedding_id = f"{sample_id}_{embedding_type}"
                try:
                    self.hcg_client.add_edge(
                        edge_id=f"e_{sample_id}_{embedding_id}",
                        source_id=sample_id,
                        target_id=embedding_id,
                        relation="has_embedding",
                        properties={"embedding_type": embedding_type, **metadata},
                    )
                except Exception as e:
                    logger.warning(f"Failed to create Neo4j edge for embedding: {e}")

        except Exception as e:
            logger.error(
                f"[JEPA Hook] Failed to store embeddings for {sample_id} in Milvus: {e}"
            )
            # Don't raise - media ingestion should succeed even if embedding storage fails
