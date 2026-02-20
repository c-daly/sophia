"""Processes Hermes proposals -- Sophia's cognitive intake pathway.

Sophia receives structured proposals from Hermes (entities, embeddings,
text metadata) and decides what to ingest into the graph. She also searches
for relevant existing context to return.

Sophia operates on embeddings, not text. Text properties exist on nodes
for Hermes's benefit when context is returned.
"""

import logging
import uuid as uuid_mod
from typing import Any

logger = logging.getLogger(__name__)

ENTITY_MATCH_THRESHOLD = 0.5

# Milvus collection types to search for context.
SEARCHABLE_COLLECTIONS = ("Entity", "Concept", "State", "Process")

# Map proposed node types to the Milvus collection used for dedup/storage.
_NODE_TYPE_TO_COLLECTION = {
    "entity": "Entity",
    "concept": "Concept",
    "state": "State",
    "process": "Process",
}


def _collection_for(node_type: str) -> str:
    """Return the Milvus collection name for a given semantic node type."""
    return _NODE_TYPE_TO_COLLECTION.get(node_type.lower(), "Entity")


class ProposalProcessor:
    """Processes proposals from Hermes into graph knowledge."""

    def __init__(self, hcg_client: Any, milvus_sync: Any) -> None:
        self._hcg = hcg_client
        self._milvus = milvus_sync

    def process(self, proposal: dict) -> dict:
        """Process a proposal: search for context, decide what to ingest."""
        stored_ids: list[str] = []
        relevant_context: list[dict] = []
        # Track entity name -> node uuid for edge resolution.
        name_to_uuid: dict[str, str] = {}

        # 1. Search for relevant existing context using document embedding
        doc_emb = proposal.get("document_embedding")
        if doc_emb and doc_emb.get("embedding"):
            for collection in SEARCHABLE_COLLECTIONS:
                try:
                    matches = self._milvus.search_similar(
                        node_type=collection,
                        query_embedding=doc_emb["embedding"],
                        top_k=5,
                    )
                    for match in matches:
                        node = self._hcg.get_node(match["uuid"])
                        if node:
                            relevant_context.append(
                                {
                                    "node_uuid": match["uuid"],
                                    "name": node.get("name", ""),
                                    "type": node.get("type", ""),
                                    "properties": node.get("properties", {}),
                                    "score": match["score"],
                                }
                            )
                            if node.get("name"):
                                name_to_uuid[node["name"]] = match["uuid"]
                except Exception as e:
                    logger.debug(f"Search in {collection} failed: {e}")

            relevant_context.sort(key=lambda x: x.get("score", float("inf")))
            relevant_context = relevant_context[:10]

        # 2. Ingest proposed nodes
        for proposed in proposal.get("proposed_nodes", []):
            name = proposed.get("name", "").strip()
            if not name:
                continue

            node_type = proposed.get("type", "unknown")
            embedding = proposed.get("embedding")
            model = proposed.get("model", "unknown")
            collection = _collection_for(node_type)

            # 2a. Search for existing node with similar embedding
            if embedding:
                try:
                    existing = self._milvus.search_similar(
                        node_type=collection,
                        query_embedding=embedding,
                        top_k=1,
                    )
                    if existing and existing[0]["score"] < ENTITY_MATCH_THRESHOLD:
                        existing_node = self._hcg.get_node(existing[0]["uuid"])
                        if existing_node:
                            logger.info(
                                f"Entity '{name}' matches existing node "
                                f"'{existing_node.get('name')}' "
                                f"(L2={existing[0]['score']:.3f}), skipping creation"
                            )
                            name_to_uuid[name] = existing[0]["uuid"]
                            if not any(
                                c["node_uuid"] == existing[0]["uuid"]
                                for c in relevant_context
                            ):
                                relevant_context.append(
                                    {
                                        "node_uuid": existing[0]["uuid"],
                                        "name": existing_node.get("name", ""),
                                        "type": existing_node.get("type", ""),
                                        "properties": existing_node.get(
                                            "properties", {}
                                        ),
                                        "score": existing[0]["score"],
                                    }
                                )
                            continue
                except Exception as e:
                    logger.debug(f"Entity match search failed for '{name}': {e}")

            # 2b. No match -- create the node
            try:
                node_uuid = self._hcg.add_node(
                    name=name,
                    node_type=node_type,
                    source=proposal.get("source_service", "hermes"),
                    derivation="observed",
                    confidence=proposal.get("confidence", 0.7),
                    properties={
                        "raw_text": proposal.get("raw_text", ""),
                        **proposed.get("properties", {}),
                    },
                )
                stored_ids.append(node_uuid)
                name_to_uuid[name] = node_uuid
            except Exception as e:
                logger.error(f"Failed to create node '{name}': {e}")
                continue

            # 2c. Connect to type definition via IS_A edge
            type_def_uuid = f"type_{node_type}"
            try:
                self._hcg.add_edge(
                    source_uuid=node_uuid,
                    target_uuid=type_def_uuid,
                    relation="IS_A",
                )
            except Exception as e:
                logger.debug(
                    "Could not create IS_A edge to type '%s': %s",
                    node_type,
                    e,
                )

            # 2d. Store embedding in Milvus
            if embedding:
                try:
                    self._milvus.upsert_embedding(
                        node_type=collection,
                        uuid=node_uuid,
                        embedding=embedding,
                        model=model,
                    )
                except Exception as e:
                    logger.warning(f"Embedding storage failed for '{name}': {e}")

        # 3. Ingest proposed edges
        stored_edge_ids: list[str] = []
        for edge in proposal.get("proposed_edges") or []:
            src_name = edge.get("source_name", "")
            tgt_name = edge.get("target_name", "")
            src_uuid = name_to_uuid.get(src_name)
            tgt_uuid = name_to_uuid.get(tgt_name)

            if not src_uuid or not tgt_uuid:
                logger.debug(
                    "Skipping edge %s -> %s: missing node UUID " "(available: %s)",
                    src_name,
                    tgt_name,
                    list(name_to_uuid.keys()),
                )
                continue

            relation = edge.get("relation", "RELATED_TO")
            bidirectional = edge.get("bidirectional", False)
            properties = dict(edge.get("properties", {}))
            properties.setdefault("confidence", edge.get("confidence", 0.5))

            try:
                edge_uuid = self._hcg.add_edge(
                    source_uuid=src_uuid,
                    target_uuid=tgt_uuid,
                    relation=relation,
                    bidirectional=bidirectional,
                    properties=properties,
                )
                stored_edge_ids.append(edge_uuid)
            except Exception as e:
                logger.error(
                    "Failed to create edge %s -[%s]-> %s: %s",
                    src_name,
                    relation,
                    tgt_name,
                    e,
                )
                continue

            # Store edge embedding in Milvus
            embedding = edge.get("embedding")
            model = edge.get("model", "unknown")
            if embedding:
                try:
                    self._milvus.upsert_embedding(
                        node_type="Edge",
                        uuid=edge_uuid,
                        embedding=embedding,
                        model=model,
                    )
                except Exception as e:
                    logger.warning(
                        "Edge embedding storage failed for %s: %s",
                        edge_uuid,
                        e,
                    )

        # 4. Create experiment_run node if pipeline metadata is present
        experiment_run_id: str | None = None
        pipeline = (proposal.get("metadata") or {}).get("pipeline")
        if pipeline and (stored_ids or stored_edge_ids):
            experiment_run_id = self._create_experiment_run(
                proposal=proposal,
                pipeline=pipeline,
                stored_node_ids=stored_ids,
                stored_edge_ids=stored_edge_ids,
            )

        return {
            "stored_node_ids": stored_ids,
            "stored_edge_ids": stored_edge_ids,
            "relevant_context": relevant_context,
            "experiment_run_id": experiment_run_id,
        }

    def _create_experiment_run(
        self,
        proposal: dict,
        pipeline: dict,
        stored_node_ids: list[str],
        stored_edge_ids: list[str],
    ) -> str | None:
        """Create an experiment_run node and PRODUCED edges to outputs.

        Returns the experiment_run uuid or None on failure.
        """
        run_uuid = str(uuid_mod.uuid4())
        metadata = proposal.get("metadata") or {}
        tags = list(metadata.get("experiment_tags", []))

        try:
            self._hcg.add_node(
                name=f"run_{run_uuid[:8]}",
                node_type="experiment_run",
                uuid=run_uuid,
                source="sophia",
                derivation="observed",
                confidence=1.0,
                properties={
                    "proposal_id": proposal.get("proposal_id", ""),
                    "correlation_id": proposal.get("correlation_id", ""),
                    "ner_provider": pipeline.get("ner_provider", ""),
                    "embedding_provider": pipeline.get("embedding_provider", ""),
                    "ner_duration_ms": pipeline.get("ner_duration_ms", 0),
                    "relation_duration_ms": pipeline.get("relation_duration_ms", 0),
                    "embedding_duration_ms": pipeline.get("embedding_duration_ms", 0),
                    "total_duration_ms": pipeline.get("total_duration_ms", 0),
                    "entity_count": pipeline.get("entity_count", 0),
                    "edge_count": pipeline.get("edge_count", 0),
                    "experiment_tags": tags,
                },
            )
        except Exception as e:
            logger.warning("Failed to create experiment_run node: %s", e)
            return None

        # Link to produced nodes
        for node_uuid in stored_node_ids:
            try:
                self._hcg.add_edge(
                    source_uuid=run_uuid,
                    target_uuid=node_uuid,
                    relation="PRODUCED",
                )
            except Exception as e:
                logger.debug("Failed to link experiment_run to node %s: %s", node_uuid, e)

        # Link to produced edges
        for edge_uuid in stored_edge_ids:
            try:
                self._hcg.add_edge(
                    source_uuid=run_uuid,
                    target_uuid=edge_uuid,
                    relation="PRODUCED",
                )
            except Exception as e:
                logger.debug("Failed to link experiment_run to edge %s: %s", edge_uuid, e)

        logger.info(
            "Created experiment_run %s: %d nodes, %d edges, ner=%s, emb=%s",
            run_uuid[:8],
            len(stored_node_ids),
            len(stored_edge_ids),
            pipeline.get("ner_provider"),
            pipeline.get("embedding_provider"),
        )
        return run_uuid
