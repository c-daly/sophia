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

logger = logging.getLogger(__name__)

ONTOLOGY_CHANGED_CHANNEL = "ontology.type_created"


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
