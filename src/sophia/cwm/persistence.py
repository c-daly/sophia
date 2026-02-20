"""CWM Persistence Service.

Handles persisting Causal World Model (CWM) states to Neo4j via logos_hcg.

CWM Types:
- cwm_a: Abstract reasoning - entities, relations, causal rules
- cwm_g: Grounded - JEPA outputs, sensor predictions, physics
- cwm_e: Emotional - persona state, sentiment, reflections
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

CWMType = Literal["cwm_a", "cwm_g", "cwm_e"]

# Map SDK model_type to internal type
MODEL_TYPE_MAP = {
    "CWM_A": "cwm_a",
    "CWM_G": "cwm_g",
    "CWM_E": "cwm_e",
}

# Reverse mapping
TYPE_MODEL_MAP = {v: k for k, v in MODEL_TYPE_MAP.items()}


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
        # Map model_type to internal type
        cwm_type = MODEL_TYPE_MAP.get(state.model_type, state.model_type.lower())

        # Extract provenance from data (simplified CWMState has provenance in data)
        data = state.data or {}
        source = data.get("source", "unknown")
        derivation = data.get("derivation", "observed")
        confidence = data.get("confidence")
        tags = data.get("tags", [])
        links = data.get("links", {})

        # Serialize data payload (full data including provenance)
        data_json = json.dumps(data) if data else "{}"

        # Serialize links
        links_json = json.dumps(links) if links else "{}"

        query = HCGQueries.create_cwm_state()
        params = {
            "uuid": state.state_id,
            "name": f"{cwm_type}_{state.timestamp.strftime('%Y%m%d_%H%M%S')}",
            "type": cwm_type,
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

            logger.info(f"Persisted CWM state: {cwm_type} id={state.state_id}")

        return state.state_id

    def find_states(
        self,
        types: list[CWMType] | None = None,
        after_timestamp: datetime | None = None,
        limit: int = 20,
    ) -> list[CWMState]:
        """Find CWM states with optional filters.

        Args:
            types: List of CWM types to include (default: all)
            after_timestamp: Only return states after this time
            limit: Max results to return

        Returns:
            List of CWMState envelopes ordered by timestamp desc
        """
        if types is None:
            types = ["cwm_a", "cwm_g", "cwm_e"]

        query = HCGQueries.find_cwm_states()
        params = {
            "types": types,
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
            cwm_type = node.get("type", "cwm_a")
            model_type = TYPE_MODEL_MAP.get(cwm_type, "CWM_A")

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
