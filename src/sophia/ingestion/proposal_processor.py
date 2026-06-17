"""Processes Hermes proposals -- Sophia's cognitive intake pathway.

Sophia receives structured proposals from Hermes (entities, embeddings,
text metadata) and decides what to ingest into the graph. She also searches
for relevant existing context to return.

Sophia operates on embeddings, not text. Text properties exist on nodes
for Hermes's benefit when context is returned.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal, Tuple

from sophia.maintenance import placement
from sophia.maintenance.type_snapshot import publish_type_snapshot

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

# Naming-driven typing (#505, DESIGN s3/s5): a content node's `node_type` is its
# REALM (entity/concept/process) -- an infrastructure label for Milvus collection
# routing, NOT a membership assertion. Membership is the instance->type IS_A edge,
# parked at the realm root at ingest and refined by drainage.
#
# Realm triage collapses the hermes NER ontology type (`ONTOLOGY_TYPES`) to one of
# the 3 realms, LLM-free (the NER already ran). It is a tunable heuristic over the
# existing NER output, not a hard truth.
_ONTOLOGY_TYPE_TO_REALM = {
    # entity realm
    "entity": "entity",
    "location": "entity",
    "object": "entity",
    "agent": "entity",
    "workspace": "entity",
    "zone": "entity",
    # process realm
    "process": "process",
    "action": "process",
    # concept realm
    "concept": "concept",
    "state": "concept",
    "data": "concept",
    "goal": "concept",
    "plan": "concept",
    "capability": "concept",
}

# Single content embedding collection, keyed by uuid: every content node type shares
# ONE collection (Chris: "one collection is how it's supposed to be"). The realm
# `node_type` does not select a collection. The separate TypeCentroid and
# media/visual (CLIP/JEPA) collections are unaffected.
_CONTENT_COLLECTION = "Entity"

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


def _realm_for(ontology_type: str) -> str:
    """Collapse a hermes NER ontology type to one of the 3 realms.

    LLM-free triage over the existing NER output (#505, DESIGN s5). An unknown,
    missing, or non-string ontology type (a malformed payload) defaults to the
    ``entity`` realm rather than raising.
    """
    return _ONTOLOGY_TYPE_TO_REALM.get(
        str(ontology_type or "").strip().lower(), "entity"
    )


def _collection_for(node_type: str) -> str:
    """Return the single Milvus content collection for any content node type.

    Content embeds into one collection keyed by uuid; ``node_type`` (the realm) is
    not a collection selector (#505).
    """
    return _CONTENT_COLLECTION


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
        self._event_bus = event_bus
        self._redis = redis_client
        # Publish the current type snapshot immediately: ingest stopped
        # minting types (#505), so the per-batch `new_types` gate below never
        # fires in production and a freshly reseeded stack would leave
        # logos:ontology:types absent — hermes's TypeRegistry then boots
        # empty (sophia#195). Fail-soft inside _write_type_snapshot.
        self._write_type_snapshot()

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
        """Write full type list to Redis for Hermes initial sync.

        Delegates to the shared positional snapshot writer so ingest, the
        scheduler reconcile loop, and the emergence event all produce an
        identical snapshot of the real (incoming-IS_A) type layer.
        """
        publish_type_snapshot(self._hcg, self._redis)

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
            # Resolve the seeded realm roots (entity/concept/process) BY NAME from
            # the type-definition catalog so each new instance can be parked under
            # its realm via an IS_A edge (#505). Mirrors the maintenance tier's
            # name->uuid resolution (emergence_handler); built once per batch.
            # Fail-soft: if the catalog query fails (e.g. Neo4j transient error),
            # proceed with an empty map -- nodes are created but left unparked
            # (the maintenance reconcile loop re-parks them later).
            try:
                uuid_by_name = {
                    n["name"].strip().lower(): n["uuid"]
                    for n in self._hcg.get_all_type_definitions()
                    if n.get("name") and n.get("uuid")
                }
            except Exception:
                logger.exception(
                    "ingest: failed to fetch realm-root catalog; nodes will be "
                    "created unparked and re-parked by the reconcile loop"
                )
                uuid_by_name = {}

            with tracer.start_as_current_span("proposal_processor.ingest_nodes"):
                for proposed in proposal.get("proposed_nodes", []):
                    name = proposed.get("name", "").strip()
                    if not name:
                        continue

                    embedding = proposed.get("embedding")
                    model = proposed.get("model", "unknown")

                    # Realm triage (#505, DESIGN s5): collapse the hermes NER
                    # ontology type to one of the 3 realms, LLM-free (the NER
                    # already ran). `node_type` is the realm -- an infra label for
                    # Milvus collection routing, NOT a membership assertion. Fine
                    # typing happens later in drainage, which reparents the IS_A edge.
                    node_type = _realm_for(proposed.get("type"))

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
                        # Preserve Hermes' initial NER type pick as provenance / a
                        # weak prior for emergence (#505). The authoritative type is
                        # read by walking the instance->type IS_A edge, never stored.
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

                    # 2c. Park the new instance under its realm root via a single
                    # upward IS_A edge -- membership IS the edge now (#505, DESIGN
                    # s3/s5). The realm root is resolved BY NAME from the type-def
                    # catalog (never a `type_<name>` slug). Drainage later reparents
                    # this edge to the fine type. Every membership write goes through
                    # `placement` (the consolidation invariant), carrying placed_by.
                    realm_root_uuid = uuid_by_name.get(node_type)
                    if realm_root_uuid is not None:
                        placement.attach(
                            node_uuid,
                            realm_root_uuid,
                            hcg=self._hcg,
                            children_of={},
                            placed_by="root_fallback",
                        )
                    else:
                        logger.warning(
                            "ingest: realm %r has no catalog uuid; node %s left "
                            "unparked",
                            node_type,
                            node_uuid,
                        )

                    # 2d. Collect embedding for batch upsert
                    if embedding:
                        pending_embeddings.setdefault(collection, []).append(
                            {
                                "uuid": node_uuid,
                                "embedding": embedding,
                                "model": model,
                            }
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
