"""Processes Hermes proposals -- Sophia's cognitive intake pathway.

Sophia receives structured proposals from Hermes (entities, embeddings,
text metadata) and decides what to ingest into the graph. She also searches
for relevant existing context to return.

Sophia operates on embeddings, not text. Text properties exist on nodes
for Hermes's benefit when context is returned.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal, Tuple

from sophia.ingestion.type_classifier import TypeClassifier

logger = logging.getLogger(__name__)


class EmbeddingPersistenceError(RuntimeError):
    """Raised when flushing pending embeddings to Milvus fails.

    Ingestion must not silently swallow a failed embedding write: an empty
    ``hcg_*_embeddings`` collection starves the type classifier and emergent
    type discovery. This error surfaces the failure to the caller so a
    partially-ingested batch is reported instead of being reported as success.
    """

    def __init__(
        self,
        message: str,
        *,
        failures: "dict[str, str] | None" = None,
    ) -> None:
        super().__init__(message)
        # Map of collection_type -> stringified underlying error.
        self.failures: dict[str, str] = failures or {}


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


def _squared_l2_distance(a: list[float], b: list[float]) -> float:
    """Squared Euclidean (L2) distance between two vectors.

    Matches Milvus' L2 metric (which returns squared distance), so distances
    computed here are directly comparable to ``ENTITY_MATCH_THRESHOLD`` and to
    ``search_similar`` scores. Lower is more similar.
    """
    return sum((x - y) ** 2 for x, y in zip(a, b, strict=True))


class ProposalProcessor:
    """Processes proposals from Hermes into graph knowledge."""

    def __init__(
        self,
        hcg_client: Any,
        milvus_sync: Any,
        event_bus: Any | None = None,
        redis_client: Any | None = None,
    ) -> None:
        self._hcg = hcg_client
        self._milvus = milvus_sync
        self._classifier = TypeClassifier(milvus=milvus_sync, hcg=hcg_client)
        self._event_bus = event_bus
        self._redis = redis_client
        self._seen_type_uuids: set[str] = set()

    def _publish_batch_event(
        self,
        stored_node_ids: list[str],
        stored_edge_ids: list[str],
        new_types: list[dict[str, str]],
        updated_types: list[dict[str, str]],
        affected_node_uuids: list[str],
    ) -> None:
        """Publish a batch event summarizing the proposal processing."""
        if self._event_bus is None:
            return
        try:
            self._event_bus.publish(
                "logos:sophia:proposal_processed",
                {
                    "event_type": "proposal_processed",
                    "source": "sophia",
                    "payload": {
                        "new_types": new_types,
                        "updated_types": updated_types,
                        "stored_node_ids": stored_node_ids,
                        "stored_edge_ids": stored_edge_ids,
                        "affected_node_uuids": affected_node_uuids,
                    },
                },
            )
        except Exception:
            logger.exception("Failed to publish proposal_processed event")

    def _write_type_snapshot(self) -> None:
        """Write full type list to Redis for Hermes initial sync."""
        if self._redis is None:
            return
        try:
            records = self._hcg.get_all_type_definitions()
            snapshot: dict[str, dict[str, Any]] = {}
            for record in records:
                name = record.get("name", "")
                if not name:
                    continue
                props = record.get("properties")
                if isinstance(props, dict):
                    member_count = props.get("member_count", 0)
                else:
                    member_count = 0
                snapshot[name] = {
                    "uuid": record.get("uuid", ""),
                    "member_count": member_count,
                }
            self._redis.set("logos:ontology:types", json.dumps(snapshot))
        except Exception:
            logger.exception("Failed to write type snapshot to Redis")

    def process(self, proposal: dict) -> dict:
        """Process a proposal: search for context, decide what to ingest."""
        stored_ids: list[str] = []
        relevant_context: list[dict] = []
        # Track entity name -> node uuid for edge resolution.
        name_to_uuid: dict[str, str] = {}
        # Track types and affected nodes for batch event.
        new_types: list[dict[str, str]] = []
        updated_types: list[dict[str, str]] = []
        affected_node_uuids: list[str] = []
        pending_embeddings: dict[str, list[dict]] = {}

        with tracer.start_as_current_span(
            "proposal_processor.process",
            attributes={"proposal.id": proposal.get("proposal_id", "")},
        ):
            # 1. Search for relevant existing context using document embedding
            doc_emb = proposal.get("document_embedding")
            if doc_emb and doc_emb.get("embedding"):
                with tracer.start_as_current_span("proposal_processor.context_search"):

                    def _search_collection(
                        coll: str,
                    ) -> tuple[str, list[dict[str, Any]]]:
                        try:
                            return coll, self._milvus.search_similar(
                                node_type=coll,
                                query_embedding=doc_emb["embedding"],
                                top_k=5,
                            )
                        except Exception as e:
                            logger.debug(f"Search in {coll} failed: {e}")
                            return coll, []

                    all_matches = []
                    with ThreadPoolExecutor(
                        max_workers=len(SEARCHABLE_COLLECTIONS)
                    ) as executor:
                        futures = {
                            executor.submit(_search_collection, c): c
                            for c in SEARCHABLE_COLLECTIONS
                        }
                        for future in as_completed(futures):
                            _coll, matches = future.result()
                            all_matches.extend(matches)

                    match_uuids = [m["uuid"] for m in all_matches]
                    nodes_by_uuid = {}
                    if match_uuids:
                        batch_nodes = self._hcg.get_nodes_batch(match_uuids)
                        nodes_by_uuid = {n["uuid"]: n for n in batch_nodes}

                    for match in all_matches:
                        node = nodes_by_uuid.get(match["uuid"])
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

                    relevant_context.sort(key=lambda x: x.get("score", float("inf")))
                    relevant_context = relevant_context[:10]

            # 2. Ingest proposed nodes
            # Collect centroid updates to flush after the node loop.
            centroid_updates: dict[str, list[tuple[list[float], str]]] = {}

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

                    # 2a-pre. In-process dedup against siblings created earlier
                    # in THIS ingest. The Milvus index (2a) is only flushed after
                    # the node loop (see pending_embeddings, 2d), so the search
                    # below cannot see a node created moments ago in the same
                    # batch -- the same entity proposed twice would mint two
                    # nodes (#148). Identity is embedding-based, never name-based:
                    # compare this mention's embedding (squared L2) against the
                    # pending embeddings of nodes already created this ingest and
                    # reuse the closest sibling below ENTITY_MATCH_THRESHOLD. A
                    # node with NO embedding carries no meaning signal, so it is
                    # deliberately not deduped here -- the downstream resolver
                    # handles that later.
                    if embedding:
                        best_uuid: str | None = None
                        best_dist = ENTITY_MATCH_THRESHOLD
                        # Scope to THIS node's collection only. The persisted
                        # Milvus dedup (2a) searches node_type=collection, so
                        # cross-collection merges must not happen here either --
                        # batch membership must not change whether two nodes
                        # collapse into one (#151 review).
                        for pending_node in pending_embeddings.get(collection, []):
                            dist = _squared_l2_distance(
                                embedding, pending_node["embedding"]
                            )
                            if dist < best_dist:
                                best_dist = dist
                                best_uuid = pending_node["uuid"]
                        if best_uuid is not None:
                            logger.info(
                                "Entity '%s' matches a sibling created earlier in "
                                "this ingest (%s, L2=%.3f); reusing, skipping "
                                "creation",
                                name,
                                best_uuid,
                                best_dist,
                            )
                            name_to_uuid[name] = best_uuid
                            continue

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

                        # Preserve Hermes' initial NER type pick as provenance / a
                        # weak prior for emergence (#505); the authoritative `type`
                        # is the centroid-classified one above.
                        node_props["hermes_type_hint"] = proposed.get("type")

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
                        affected_node_uuids.append(node_uuid)
                    except Exception as e:
                        logger.error(f"Failed to create node '{name}': {e}")
                        continue

                    # 2c. Connect to type definition via IS_A edge.
                    # Ensure the type-definition node exists (MERGE is idempotent).
                    type_def_uuid = f"type_{node_type}"
                    _is_new_type = type_def_uuid not in self._seen_type_uuids
                    self._seen_type_uuids.add(type_def_uuid)
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

                    # Track type as new or updated for the batch event.
                    type_entry = {"uuid": type_def_uuid, "name": node_type}
                    if _is_new_type and not any(
                        t["uuid"] == type_def_uuid for t in new_types
                    ):
                        new_types.append(type_entry)
                    elif not _is_new_type and not any(
                        t["uuid"] == type_def_uuid for t in updated_types
                    ):
                        updated_types.append(type_entry)

                    # 2d. Collect embedding for batch upsert
                    if embedding:
                        pending_embeddings.setdefault(collection, []).append(
                            {
                                "uuid": node_uuid,
                                "embedding": embedding,
                                "model": model,
                            }
                        )

                    # 2e. Collect centroid update (deferred to after node loop)
                    if classification and embedding:
                        centroid_updates.setdefault(
                            classification.type_uuid, []
                        ).append((embedding, model))

            # 2f. Flush deferred centroid updates
            with tracer.start_as_current_span("proposal_processor.centroid_updates"):
                for type_uuid, assignments in centroid_updates.items():
                    try:
                        type_node = self._hcg.get_node(type_uuid)
                        props = (
                            type_node.get("properties", {})
                            if type_node
                            and isinstance(type_node.get("properties"), dict)
                            else {}
                        )
                        member_count = props.get("member_count", 0)
                        current_centroid = props.get("centroid")

                        for embedding_val, model_val in assignments:
                            if (
                                isinstance(member_count, int)
                                and isinstance(current_centroid, list)
                                and current_centroid
                            ):
                                current_centroid = (
                                    self._classifier.update_centroid_for_assignment(
                                        type_uuid=type_uuid,
                                        new_embedding=embedding_val,
                                        current_centroid=current_centroid,
                                        member_count=member_count,
                                        model=model_val,
                                    )
                                )
                                member_count += 1
                            elif not current_centroid:
                                self._milvus.update_centroid(
                                    type_uuid=type_uuid,
                                    centroid=embedding_val,
                                    model=model_val,
                                )
                                current_centroid = embedding_val
                                member_count = 1

                        self._hcg.update_node(
                            type_uuid,
                            {
                                "member_count": member_count,
                                "centroid": current_centroid,
                            },
                        )
                    except Exception as e:
                        logger.debug(
                            "Centroid update skipped for type %s: %s", type_uuid, e
                        )

            # 3. Ingest proposed edges
            with tracer.start_as_current_span("proposal_processor.ingest_edges"):
                stored_edge_ids: list[str] = []
                dropped_count = 0
                proposed_edges = proposal.get("proposed_edges") or []

                # Pre-resolve unresolved edge names in batch
                unresolved_names: set[str] = set()
                for edge in proposed_edges:
                    src_name = edge.get("source_name", "")
                    tgt_name = edge.get("target_name", "")
                    if src_name and src_name not in name_to_uuid:
                        unresolved_names.add(src_name)
                    if tgt_name and tgt_name not in name_to_uuid:
                        unresolved_names.add(tgt_name)

                if unresolved_names:
                    try:
                        resolved = self._hcg.find_nodes_by_names(list(unresolved_names))
                        for name, node_data in resolved.items():
                            if node_data and node_data.get("uuid"):
                                name_to_uuid[name] = node_data["uuid"]
                    except Exception as e:
                        logger.debug("Batch name resolution failed: %s", e)

                for edge in proposed_edges:
                    src_name = edge.get("source_name", "")
                    tgt_name = edge.get("target_name", "")
                    src_uuid = name_to_uuid.get(src_name)
                    tgt_uuid = name_to_uuid.get(tgt_name)

                    if not src_uuid or not tgt_uuid:
                        dropped_count += 1
                        # Do not log entity names: source/target names are derived
                        # from arbitrary ingested documents (potential PII). Key the
                        # diagnostic on the proposal id, the relation type, and which
                        # endpoint failed to resolve.
                        logger.debug(
                            "Dropping edge in proposal %s: relation=%s "
                            "unresolved src=%s tgt=%s",
                            proposal.get("proposal_id", ""),
                            edge.get("relation", "RELATED_TO"),
                            not src_uuid,
                            not tgt_uuid,
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

                    # Collect edge embedding for batch upsert
                    embedding = edge.get("embedding")
                    model = edge.get("model", "unknown")
                    if embedding:
                        pending_embeddings.setdefault("Edge", []).append(
                            {
                                "uuid": edge_uuid,
                                "embedding": embedding,
                                "model": model,
                            }
                        )

                # Only emit at INFO when something dropped (the signal we care
                # about); the all-resolved case is already covered by the
                # per-proposal summary logged in the API layer.
                if dropped_count:
                    # received = created + dropped_unresolved + errored (every
                    # proposed edge takes exactly one of those three paths), so
                    # surface the errored term too — otherwise the counts don't
                    # reconcile when an edge resolves but add_edge raises.
                    errored_count = (
                        len(proposed_edges) - dropped_count - len(stored_edge_ids)
                    )
                    logger.info(
                        "Edge ingestion for proposal %s: received=%d created=%d "
                        "dropped_unresolved=%d errored=%d",
                        proposal.get("proposal_id", ""),
                        len(proposed_edges),
                        len(stored_edge_ids),
                        dropped_count,
                        errored_count,
                    )

            # Flush all pending embeddings in batch.
            #
            # A failed Milvus write must NOT be silently swallowed: when it is,
            # ingestion reports success while the hcg_*_embeddings collections
            # stay empty, starving the type classifier and emergent type
            # discovery. We attempt every collection (so one bad collection
            # does not mask the others), collect any failures, and raise so the
            # caller sees the error instead of a false success.
            embedding_failures: dict[str, str] = {}
            for collection_type, batch in pending_embeddings.items():
                if batch:
                    try:
                        self._milvus.batch_upsert_embeddings(
                            node_type=collection_type, embeddings=batch
                        )
                    except Exception as e:
                        logger.error(
                            "Batch embedding upsert failed for %s: %s",
                            collection_type,
                            e,
                        )
                        embedding_failures[collection_type] = str(e)

            if embedding_failures:
                # The Neo4j nodes/edges (steps 2-3) already committed, but their
                # embeddings did not land in Milvus. Orphaned nodes are invisible
                # to the dedup search (search_similar), so a retry would re-create
                # duplicates -- and no batch event is emitted for writes that
                # half-landed. Data is wipeable down to the type baseline, so
                # best-effort roll the partial graph writes back; a retry then
                # re-creates cleanly. (Centroid drift is left as-is: regenerable.)
                rolled_back = 0
                for uuid in stored_ids:
                    try:
                        # delete_node also removes edge-nodes touching this node.
                        self._hcg.delete_node(uuid)
                        rolled_back += 1
                    except Exception:
                        logger.exception(
                            "Rollback: failed to delete node %s after "
                            "embedding-persistence failure",
                            uuid,
                        )
                # Edges are reified as Node entities. Deleting stored_ids drops the
                # edges touching them, but an edge added this batch between two
                # PRE-EXISTING nodes must be deleted by its own uuid too, or it
                # survives the rollback (gemini asked to roll back stored_edge_ids).
                for edge_uuid in stored_edge_ids:
                    try:
                        self._hcg.delete_node(edge_uuid)
                        rolled_back += 1
                    except Exception:
                        logger.exception(
                            "Rollback: failed to delete edge %s after "
                            "embedding-persistence failure",
                            edge_uuid,
                        )
                logger.critical(
                    "Embedding persistence failed for %s; rolled back %d/%d graph "
                    "node(s)+edge(s) so the batch is cleanly retryable.",
                    sorted(embedding_failures),
                    rolled_back,
                    len(stored_ids) + len(stored_edge_ids),
                )
                raise EmbeddingPersistenceError(
                    "Failed to persist embeddings to Milvus for "
                    f"{sorted(embedding_failures)}; ingestion did not complete and "
                    "the partially-written graph nodes were rolled back.",
                    failures=embedding_failures,
                )

            # Write type snapshot BEFORE publishing event so subscribers
            # see up-to-date data when they react to the event.
            if new_types or updated_types:
                self._write_type_snapshot()

            # Publish batch event summarising what changed.
            if stored_ids or stored_edge_ids or new_types or updated_types:
                self._publish_batch_event(
                    stored_node_ids=stored_ids,
                    stored_edge_ids=stored_edge_ids,
                    new_types=new_types,
                    updated_types=updated_types,
                    affected_node_uuids=affected_node_uuids,
                )

            return {
                "stored_node_ids": stored_ids,
                "stored_edge_ids": stored_edge_ids,
                "relevant_context": relevant_context,
            }
