"""Processes Hermes proposals -- Sophia's cognitive intake pathway.

Sophia receives structured proposals from Hermes (entities, embeddings,
text metadata) and decides what to ingest into the graph. She also searches
for relevant existing context to return.

Sophia operates on embeddings, not text. Text properties exist on nodes
for Hermes's benefit when context is returned.
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)

ENTITY_MATCH_THRESHOLD = 0.5


class ProposalProcessor:
    """Processes proposals from Hermes into graph knowledge."""

    def __init__(self, hcg_client: Any, milvus_sync: Any) -> None:
        self._hcg = hcg_client
        self._milvus = milvus_sync

    def process(self, proposal: dict) -> dict:
        """Process a proposal: search for context, decide what to ingest."""
        stored_ids: list[str] = []
        relevant_context: list[dict] = []

        # 1. Search for relevant existing context using document embedding
        doc_emb = proposal.get("document_embedding")
        if doc_emb and doc_emb.get("embedding"):
            for node_type in ["Entity", "Concept", "State", "Process", "Edge"]:
                try:
                    matches = self._milvus.search_similar(
                        node_type=node_type,
                        query_embedding=doc_emb["embedding"],
                        top_k=5,
                    )
                    for match in matches:
                        node = self._hcg.get_node(match["uuid"])
                        if node:
                            relevant_context.append({
                                "node_uuid": match["uuid"],
                                "name": node.get("name", ""),
                                "type": node.get("type", ""),
                                "properties": node.get("properties", {}),
                                "score": match["score"],
                            })
                except Exception as e:
                    logger.debug(f"Search in {node_type} failed: {e}")

            relevant_context.sort(key=lambda x: x.get("score", float("inf")))
            relevant_context = relevant_context[:10]

        # 2. Ingest proposed nodes
        for proposed in proposal.get("proposed_nodes", []):
            name = proposed.get("name", "").strip()
            if not name:
                continue

            node_type = proposed.get("type", "unknown")
            embedding = proposed.get("embedding")
            embedding_id = proposed.get("embedding_id")
            model = proposed.get("model", "unknown")

            # 2a. Search for existing node with similar embedding
            if embedding:
                try:
                    existing = self._milvus.search_similar(
                        node_type="Entity",
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
                            if not any(c["node_uuid"] == existing[0]["uuid"]
                                       for c in relevant_context):
                                relevant_context.append({
                                    "node_uuid": existing[0]["uuid"],
                                    "name": existing_node.get("name", ""),
                                    "type": existing_node.get("type", ""),
                                    "properties": existing_node.get("properties", {}),
                                    "score": existing[0]["score"],
                                })
                            continue
                except Exception as e:
                    logger.debug(f"Entity match search failed for '{name}': {e}")

            # 2b. No match -- create the node
            try:
                node_uuid = self._hcg.add_node(
                    name=name,
                    node_type=node_type,
                    properties={
                        "source": proposal.get("source_service", "hermes"),
                        "derivation": "observed",
                        "confidence": proposal.get("confidence", 0.7),
                        "raw_text": proposal.get("raw_text", ""),
                        **proposed.get("properties", {}),
                    },
                )
                stored_ids.append(node_uuid)
            except Exception as e:
                logger.error(f"Failed to create node '{name}': {e}")
                continue

            # 2c. Store embedding in Milvus
            if embedding and embedding_id:
                try:
                    self._milvus.upsert_embedding(
                        node_type="Entity",
                        uuid=node_uuid,
                        embedding=embedding,
                        model=model,
                    )
                except Exception as e:
                    logger.warning(f"Embedding storage failed for '{name}': {e}")

        return {
            "stored_node_ids": stored_ids,
            "relevant_context": relevant_context,
        }
