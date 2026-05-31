"""The 'type_emergence' maintenance handler (#505).

Dispatched by MaintenanceScheduler as handlers['type_emergence'](type_uuid=...).
Dependencies (load_members / name_fn / mint_fn / candidates_fn) are injected so
the orchestration is unit-testable without Neo4j / Milvus / Hermes. Emergence
always mints NEW types from the residue; it never attaches to existing ones.
"""

from __future__ import annotations

import logging
import uuid as uuid_lib

from sophia.maintenance.config import MaintenanceConfig
from sophia.maintenance.emergence_clustering import find_emergent_clusters
from sophia.maintenance.emergence_types import Member
from sophia.maintenance.structural_signature import build_signature

logger = logging.getLogger(__name__)

ONTOLOGY_CHANGED_CHANNEL = "ontology.type_created"


def _type_name(type_uuid: str) -> str:
    """Convention: type-definition uuids are 'type_<name>'."""
    return type_uuid[len("type_"):] if type_uuid.startswith("type_") else type_uuid


def current_categories(hcg) -> list[str]:
    """Existing type-definition labels, excluding `entity` and reserved_* types."""
    out: list[str] = []
    for node in hcg.list_all_nodes(node_type="type_definition"):
        name = node.get("name")
        if not name or name == "entity" or name.startswith("reserved_"):
            continue
        out.append(name)
    return out


def load_type_members(hcg, milvus, type_uuid: str) -> list[Member]:
    """Load all members of a type as Member objects (embedding + structural signature).

    Embeddings come from Milvus; the structural signature is built from the node's
    outgoing reified edges (relation + resolved neighbor type). Nodes without an
    embedding are skipped (they can't be clustered).
    """
    from sophia.ingestion.proposal_processor import _collection_for

    type_name = _type_name(type_uuid)
    members: list[Member] = []
    for row in hcg.list_all_nodes(node_type=type_name):
        uuid = row["uuid"]
        # Embeddings live in a Milvus collection keyed by the canonical NodeType
        # ('Entity'/'Concept'/...), not the semantic type string.
        emb = milvus.get_embedding(
            node_type=_collection_for(row.get("type") or type_name), uuid=uuid
        )
        if not emb or not emb.get("embedding"):
            continue
        edges = hcg.query_edges_from(uuid)
        target_uuids = [e["target"] for e in edges if e.get("target")]
        target_nodes = {
            n["uuid"]: n for n in (hcg.get_nodes_batch(target_uuids) if target_uuids else [])
        }
        neighbors = [
            {
                "relation": e.get("relation"),
                "neighbor_name": target_nodes.get(e.get("target"), {}).get("name"),
                "neighbor_type": target_nodes.get(e.get("target"), {}).get("type"),
            }
            for e in edges
        ]
        props = row.get("properties", {}) or {}
        members.append(
            Member(
                uuid=uuid,
                name=row.get("name", uuid),
                embedding=emb["embedding"],
                signature=build_signature(neighbors),
                current_type=row.get("type", type_name),
                hermes_type_hint=props.get("hermes_type_hint"),
                neighbors=neighbors,
                model=emb.get("model"),
            )
        )
    return members


class EmergenceHandler:
    def __init__(
        self,
        *,
        config: MaintenanceConfig,
        hcg,
        milvus,
        event_bus,
        hermes_url: str,
        token: str,
        load_members,
        name_fn,
        mint_fn,
        candidates_fn,
    ):
        self._config = config
        self._hcg = hcg
        self._milvus = milvus
        self._event_bus = event_bus
        self._hermes_url = hermes_url
        self._token = token
        self._load_members = load_members
        self._name_fn = name_fn
        self._mint_fn = mint_fn
        self._candidates_fn = candidates_fn

    def run(self, type_uuid: str) -> None:
        members = self._load_members(type_uuid)
        clusters = find_emergent_clusters(
            members,
            min_cluster_size=self._config.min_cluster_size,
            variance_threshold=self._config.variance_threshold,
            min_cohesion_improvement=self._config.min_cohesion_improvement,
        )
        if not clusters:
            logger.info("emergence: no qualifying clusters in %s", type_uuid)
            return

        candidates = self._candidates_fn()
        for cluster in clusters:
            name = self._name_fn(cluster, candidates, self._hermes_url, self._token)
            if name is None or name.confidence < self._config.hermes_confidence_floor:
                logger.info("emergence: skip cluster (no/low-confidence name)")
                continue
            cluster_id = uuid_lib.uuid4().hex[:8]
            new_type_uuid = self._mint_fn(
                cluster,
                name,
                hcg=self._hcg,
                milvus=self._milvus,
                source_cluster_id=cluster_id,
            )
            if self._event_bus is not None:
                self._event_bus.publish(
                    ONTOLOGY_CHANGED_CHANNEL,
                    {
                        "type_uuid": new_type_uuid,
                        "name": name.label,
                        "ancestors": ["root"],
                    },
                )
            logger.info(
                "emergence: minted %s from %d members", name.label, cluster.size
            )


def build_emergence_handler(*, config, hcg, milvus, event_bus, hermes_url, token):
    """Return the callable registered as handlers['type_emergence']."""
    from sophia.maintenance.hermes_naming import name_cluster
    from sophia.maintenance.type_minting import mint_type

    handler = EmergenceHandler(
        config=config,
        hcg=hcg,
        milvus=milvus,
        event_bus=event_bus,
        hermes_url=hermes_url,
        token=token,
        load_members=lambda u: load_type_members(hcg, milvus, u),
        name_fn=lambda c, cand, url, tok: name_cluster(
            c, candidates=cand, hermes_url=url, token=tok
        ),
        mint_fn=mint_type,
        candidates_fn=lambda: current_categories(hcg),
    )

    def _run(type_uuid: str) -> None:
        handler.run(type_uuid=type_uuid)

    return _run
