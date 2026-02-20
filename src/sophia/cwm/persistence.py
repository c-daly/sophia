"""CWM Persistence Service.

Handles persisting Causal World Model (CWM) states to Neo4j via logos_hcg.

All CWM states are stored as ``type = "state"`` nodes with tags:
- ``"cwm"`` — marks the node as a CWM state
- ``"subsystem:abstract"``  (CWM-A) — entities, relations, causal rules
- ``"subsystem:grounded"``  (CWM-G) — JEPA outputs, sensor predictions, physics
- ``"subsystem:emotional"`` (CWM-E) — persona state, sentiment, reflections
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Literal

from logos_hcg.queries import HCGQueries

# Import the canonical CWMState envelope from cwm_a
from sophia.cwm_a.state_service import CWMState

logger = logging.getLogger(__name__)

CWMType = Literal["state"]

# Map SDK model_type to subsystem tag
MODEL_TYPE_TO_SUBSYSTEM = {
    "CWM_A": "subsystem:abstract",
    "CWM_G": "subsystem:grounded",
    "CWM_E": "subsystem:emotional",
}

# Reverse mapping (subsystem tag → SDK model_type)
SUBSYSTEM_TO_MODEL_TYPE = {v: k for k, v in MODEL_TYPE_TO_SUBSYSTEM.items()}


class CWMPersistence:
    """Service for persisting CWM states to Neo4j."""

    def __init__(self, neo4j_driver: Any, database: str = "neo4j") -> None:
        """Initialize with Neo4j driver.

        Args:
            neo4j_driver: Neo4j driver instance from HCG client
            database: Neo4j database name
        """
        self._driver = neo4j_driver
        self._database = database

    def persist(self, state: CWMState) -> str:
        """Persist a CWMState to Neo4j.

        Args:
            state: CWMState envelope to persist (provenance is in data)

        Returns:
            The state_id of the persisted state
        """
        # Determine subsystem tag from model_type
        subsystem_tag = MODEL_TYPE_TO_SUBSYSTEM.get(state.model_type)

        # Extract provenance from data (simplified CWMState has provenance in data)
        data = state.data or {}
        source = data.get("source", "unknown")
        derivation = data.get("derivation", "observed")
        confidence = data.get("confidence")
        tags = list(data.get("tags", []))
        links = data.get("links", {})

        # Ensure CWM marker and subsystem tags are present
        if "cwm" not in tags:
            tags.append("cwm")
        if subsystem_tag and subsystem_tag not in tags:
            tags.append(subsystem_tag)

        # Serialize data payload (full data including provenance)
        data_json = json.dumps(data) if data else "{}"

        # Serialize links
        links_json = json.dumps(links) if links else "{}"

        query = HCGQueries.create_cwm_state()
        params = {
            "uuid": state.state_id,
            "name": f"cwm_{state.timestamp.strftime('%Y%m%d_%H%M%S')}",
            "type": "state",
            "timestamp": state.timestamp.isoformat(),
            "source": source,
            "confidence": confidence if confidence is not None else 1.0,
            "status": derivation,  # status field stores derivation value
            "payload": data_json,
            "links": links_json,
            "tags": tags,
            "embedding_id": None,
            "embedding_type": None,
        }

        with self._driver.session(database=self._database) as session:
            result = session.run(query, params)
            record = result.single()
            if not record:
                raise RuntimeError(f"Failed to persist CWM state: {state.state_id}")

            subsystem = subsystem_tag or "unknown"
            logger.info(f"Persisted CWM state: {subsystem} id={state.state_id}")

        return state.state_id

    def find_states(
        self,
        subsystems: list[str] | None = None,
        after_timestamp: datetime | None = None,
        limit: int = 20,
    ) -> list[CWMState]:
        """Find CWM states with optional filters.

        Args:
            subsystems: List of subsystem tags to filter by
                       (e.g. ["subsystem:abstract", "subsystem:emotional"]).
                       If None, returns all CWM states.
            after_timestamp: Only return states after this time
            limit: Max results to return

        Returns:
            List of CWMState envelopes ordered by timestamp desc
        """
        subsystem_tags = subsystems or []

        query = HCGQueries.find_cwm_states()
        params = {
            "subsystem_tags": subsystem_tags,
            "after_timestamp": after_timestamp.isoformat() if after_timestamp else None,
            "limit": limit,
        }

        states = []
        with self._driver.session(database=self._database) as session:
            result = session.run(query, params)
            for record in result:
                node = record["s"]
                cwm_state = self._node_to_cwm_state(node)
                if cwm_state:
                    states.append(cwm_state)

        return states

    def _node_to_cwm_state(self, node: dict) -> CWMState | None:
        """Convert a Neo4j node to a CWMState envelope.

        Args:
            node: Neo4j node properties

        Returns:
            CWMState envelope (simplified: provenance in data) or None if conversion fails
        """
        try:
            # Determine model_type from subsystem tag
            tags = node.get("tags", [])
            model_type = "CWM_A"  # default
            for tag in tags:
                if tag in SUBSYSTEM_TO_MODEL_TYPE:
                    model_type = SUBSYSTEM_TO_MODEL_TYPE[tag]
                    break

            # Parse payload
            payload_str = node.get("payload", "{}")
            if isinstance(payload_str, str):
                payload_dict = json.loads(payload_str)
            else:
                payload_dict = payload_str or {}

            # Parse links
            links_str = node.get("links", "{}")
            if isinstance(links_str, str):
                links_dict = json.loads(links_str)
            else:
                links_dict = links_str or {}

            # Parse timestamp - ensure it's always a datetime
            raw_timestamp = node.get("timestamp")
            if raw_timestamp is None:
                timestamp = datetime.now(timezone.utc)
            elif hasattr(raw_timestamp, "to_native"):
                timestamp = raw_timestamp.to_native()
            elif isinstance(raw_timestamp, str):
                timestamp = datetime.fromisoformat(raw_timestamp)
            else:
                timestamp = raw_timestamp

            # Build data dict with provenance (simplified CWMState)
            # Start with ALL payload data (preserves entry, etc.)
            data = payload_dict.copy() if payload_dict else {}

            # Overlay provenance fields (these override payload if present)
            data.update(
                {
                    "source": node.get("source", "unknown"),
                    "derivation": node.get("status", "observed"),
                    "confidence": node.get("confidence"),
                    "created": node.get("created", timestamp.isoformat()),
                    "updated": node.get("updated", timestamp.isoformat()),
                    "tags": node.get("tags", []),
                    "links": links_dict,
                }
            )

            # Build CWMState (thin envelope)
            return CWMState(
                state_id=node["uuid"],
                model_type=model_type,
                timestamp=timestamp,  # type: ignore[arg-type]
                data=data,
            )
        except Exception as e:
            logger.warning(f"Failed to convert node to CWMState: {e}")
            return None
