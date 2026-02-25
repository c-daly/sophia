"""Processes Hermes proposals -- Sophia's cognitive intake pathway.

Sophia receives structured proposals from Hermes (entities, embeddings,
text metadata) and decides what to ingest into the graph. She also searches
for relevant existing context to return.

Sophia operates on embeddings, not text. Text properties exist on nodes
for Hermes's benefit when context is returned.
"""

import logging
from typing import Any, Literal, Tuple

from sophia.ingestion.type_classifier import TypeClassifier

logger = logging.getLogger(__name__)

try:
    from logos_observability import get_tracer

    tracer = get_tracer("sophia.proposal_processor")
except ImportError:
    from contextlib import nullcontext

    class _NoopTracer:
        def start_as_current_span(self, name: str, **kw: Any) -> nullcontext:  # type: ignore[type-arg]
            return nullcontext()

    tracer: Any = _NoopTracer()  # type: ignore[no-redef]

ENTITY_MATCH_THRESHOLD = 0.5

# Node/collection type literal shared with logos_hcg.
NodeType = Literal["Entity", "Concept", "State", "Process", "Edge"]

# All Milvus collections used by proposal processing.
ALL_MILVUS_COLLECTIONS: Tuple[NodeType, ...] = (
    "Entity",
    "Concept",
    "State",
    "Process",
    "Edge",
)

# Milvus collection types to search for context.
SEARCHABLE_COLLECTIONS = ("Entity", "Concept", "State", "Process")

# Map proposed node types to the Milvus collection used for dedup/storage.
_NODE_TYPE_TO_COLLECTION = {
    # General knowledge types — classifier assigns these
    "entity": "Entity",
    "concept": "Concept",
    "location": "Entity",
    "object": "Entity",
    "state": "Entity",
    "process": "Entity",
    "agent": "Entity",
    # Reserved internal types — only Sophia subsystems assign these
    "reserved_state": "State",
    "reserved_process": "Process",
    "reserved_agent": "Process",
    "reserved_action": "Process",
    "reserved_goal": "Process",
    "reserved_plan": "Process",
    "reserved_simulation": "Process",
    "reserved_execution": "Process",
    "reserved_media_sample": "Entity",
}

# Keys that must not be overwritten by untrusted proposal properties.
_RESERVED_EDGE_KEYS = frozenset(
    {
        "uuid",
        "source",
        "target",
        "relation",
        "type",
        "bidirectional",
        "created_at",
        "updated_at",
    }
)


def _collection_for(node_type: str) -> str:
    """Return the Milvus collection name for a given semantic node type."""
    return _NODE_TYPE_TO_COLLECTION.get(node_type.lower(), "Entity")


class ProposalProcessor:
    """Processes proposals from Hermes into graph knowledge."""

    def __init__(
        self,
        hcg_client: Any,
        milvus_sync: Any,
    ) -> None:
        self._hcg = hcg_client
        self._milvus = milvus_sync
        self._classifier = TypeClassifier(milvus=milvus_sync, hcg=hcg_client)

    def process(self, proposal: dict) -> dict:
        """Process a proposal: search for context, decide what to ingest."""
        stored_ids: list[str] = []
        relevant_context: list[dict] = []
        # Track entity name -> node uuid for edge resolution.
        name_to_uuid: dict[str, str] = {}

        with tracer.start_as_current_span(
            "proposal_processor.process",
            attributes={"proposal.id": proposal.get("proposal_id", "")},
        ):
            # 1. Search for relevant existing context using document embedding
            doc_emb = proposal.get("document_embedding")
            if doc_emb and doc_emb.get("embedding"):
                with tracer.start_as_current_span("proposal_processor.context_search"):
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
                                        if node["name"] in name_to_uuid:
                                            logger.warning(
                                                "Name collision in context: '%s' already mapped "
                                                "to %s, overwriting with %s",
                                                node["name"],
                                                name_to_uuid[node["name"]],
                                                match["uuid"],
                                            )
                                        name_to_uuid[node["name"]] = match["uuid"]
                        except Exception as e:
                            logger.debug(f"Search in {collection} failed: {e}")

                    relevant_context.sort(key=lambda x: x.get("score", float("inf")))
                    relevant_context = relevant_context[:10]

            # 2. Ingest proposed nodes
            with tracer.start_as_current_span("proposal_processor.ingest_nodes"):
                for proposed in proposal.get("proposed_nodes", []):
                    name = proposed.get("name", "").strip()
                    if not name:
                        continue

                    embedding = proposed.get("embedding")
                    model = proposed.get("model", "unknown")

                    # Classify using embedding-space centroids (Hermes type hint ignored)
                    if embedding:
                        classification = self._classifier.classify(embedding)
                        node_type = classification.type_name
                    else:
                        classification = None
                        node_type = proposed.get("type", "entity")

                    collection = _collection_for(node_type)

                    # 2a. Search for existing node with similar embedding
                    if embedding:
                        try:
                            existing = self._milvus.search_similar(
                                node_type=collection,
                                query_embedding=embedding,
                                top_k=1,
                            )
                            if (
                                existing
                                and existing[0]["score"] < ENTITY_MATCH_THRESHOLD
                            ):
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
                            logger.debug(
                                f"Entity match search failed for '{name}': {e}"
                            )

                    # 2b. No match -- create the node
                    try:
                        node_props = {
                            "raw_text": proposal.get("raw_text", ""),
                            **proposed.get("properties", {}),
                        }
                        if classification:
                            node_props["type_confidence"] = classification.confidence
                            node_props["needs_reclassification"] = (
                                classification.needs_reclassification
                            )

                        node_uuid = self._hcg.add_node(
                            name=name,
                            node_type=node_type,
                            source=proposal.get("source_service", "hermes"),
                            derivation="observed",
                            confidence=proposal.get("confidence", 0.7),
                            properties=node_props,
                        )
                        stored_ids.append(node_uuid)
                        name_to_uuid[name] = node_uuid
                    except Exception as e:
                        logger.error(f"Failed to create node '{name}': {e}")
                        continue

                    # 2c. Connect to type definition via IS_A edge.
                    # Ensure the type-definition node exists (MERGE is idempotent).
                    type_def_uuid = f"type_{node_type}"
                    try:
                        self._hcg.add_node(
                            uuid=type_def_uuid,
                            name=node_type,
                            node_type="type_definition",
                            source="sophia",
                            derivation="observed",
                        )
                        self._hcg.add_edge(
                            source_uuid=node_uuid,
                            target_uuid=type_def_uuid,
                            relation="IS_A",
                        )
                    except Exception as e:
                        logger.warning(
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
                            logger.warning(
                                f"Embedding storage failed for '{name}': {e}"
                            )

                    # 2e. Incrementally update the type centroid
                    if classification and embedding:
                        try:
                            type_node = self._hcg.get_node(classification.type_uuid)
                            props = (
                                type_node.get("properties", {})
                                if type_node
                                and isinstance(type_node.get("properties"), dict)
                                else {}
                            )
                            member_count = props.get("member_count", 0)
                            current_centroid = props.get("centroid")

                            if (
                                isinstance(member_count, int)
                                and isinstance(current_centroid, list)
                                and current_centroid
                            ):
                                self._classifier.update_centroid_for_assignment(
                                    type_uuid=classification.type_uuid,
                                    new_embedding=embedding,
                                    current_centroid=current_centroid,
                                    member_count=member_count,
                                    model=model,
                                )
                            elif not current_centroid:
                                # First node of this type — initialize centroid
                                self._milvus.update_centroid(
                                    type_uuid=classification.type_uuid,
                                    centroid=embedding,
                                    model=model,
                                )
                        except Exception as e:
                            logger.debug(
                                "Centroid update skipped for type '%s': %s",
                                node_type,
                                e,
                            )

            # 3. Ingest proposed edges
            with tracer.start_as_current_span("proposal_processor.ingest_edges"):
                stored_edge_ids: list[str] = []
                for edge in proposal.get("proposed_edges") or []:
                    src_name = edge.get("source_name", "")
                    tgt_name = edge.get("target_name", "")
                    src_uuid = name_to_uuid.get(src_name)
                    tgt_uuid = name_to_uuid.get(tgt_name)

                    # Fallback: look up unresolved names in Neo4j
                    if not src_uuid:
                        try:
                            found = self._hcg.find_node_by_name(src_name)
                            if found:
                                src_uuid = found.get("uuid")
                        except Exception as e:
                            logger.debug(
                                "Neo4j fallback lookup failed for '%s': %s", src_name, e
                            )
                    if not tgt_uuid:
                        try:
                            found = self._hcg.find_node_by_name(tgt_name)
                            if found:
                                tgt_uuid = found.get("uuid")
                        except Exception as e:
                            logger.debug(
                                "Neo4j fallback lookup failed for '%s': %s", tgt_name, e
                            )

                    if not src_uuid or not tgt_uuid:
                        logger.debug(
                            "Skipping edge %s -> %s: missing node UUID (available: %s)",
                            src_name,
                            tgt_name,
                            list(name_to_uuid.keys()),
                        )
                        continue

                    relation = edge.get("relation", "RELATED_TO")
                    bidirectional = edge.get("bidirectional", False)
                    properties = dict(edge.get("properties") or {})
                    properties["confidence"] = edge.get("confidence", 0.5)
                    # Strip reserved keys so untrusted input cannot overwrite them
                    for key in _RESERVED_EDGE_KEYS:
                        properties.pop(key, None)

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

            return {
                "stored_node_ids": stored_ids,
                "stored_edge_ids": stored_edge_ids,
                "relevant_context": relevant_context,
            }
