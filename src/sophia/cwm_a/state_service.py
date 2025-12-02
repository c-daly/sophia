"""CWM-A State Service for normalized state emission.

This service wraps entity/relationship updates in CWMState envelopes,
providing consistent state emissions that comply with the unified
CWM state contract defined in PHASE2_SPEC.
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class EntityDiff(BaseModel):
    """Represents a change to an entity."""

    entity_id: str
    entity_type: str
    operation: str = Field(description="create, update, or delete")
    before: Optional[Dict[str, Any]] = None
    after: Optional[Dict[str, Any]] = None
    changed_properties: Optional[List[str]] = None


class RelationDiff(BaseModel):
    """Represents a change to a relationship."""

    source_id: str
    target_id: str
    relation_type: str
    operation: str = Field(description="create or delete")
    properties: Optional[Dict[str, Any]] = None


class ValidationResult(BaseModel):
    """SHACL validation result."""

    passed: bool
    violations: List[str] = Field(default_factory=list)
    validator_version: str = "shacl-v1"


class CWMAGraphPayload(BaseModel):
    """Payload for CWM-A state emissions.

    Contains normalized entity/relationship diffs and validation status.
    """

    entities: List[EntityDiff] = Field(default_factory=list)
    relations: List[RelationDiff] = Field(default_factory=list)
    violations: List[str] = Field(default_factory=list)
    validation: ValidationResult = Field(
        default_factory=lambda: ValidationResult(passed=True)
    )


class CWMStateLinks(BaseModel):
    """Links to related HCG entities."""

    process_ids: Optional[List[str]] = None
    plan_id: Optional[str] = None
    entity_ids: Optional[List[str]] = None
    media_sample_id: Optional[str] = None
    persona_entry_id: Optional[str] = None
    talos_run_id: Optional[str] = None


class CWMState(BaseModel):
    """Unified CWM state envelope.

    All CWM emissions (CWM-A, CWM-G, CWM-E) share this envelope format
    for consistent consumption by clients, logs, and Neo4j.
    """

    state_id: str = Field(description="Globally unique identifier (cwm_<model>_<uuid>)")
    model_type: str = Field(description="CWM_A, CWM_G, or CWM_E")
    source: str = Field(description="Subsystem that emitted the record")
    timestamp: datetime
    confidence: float = Field(ge=0.0, le=1.0, description="Certainty score")
    status: str = Field(description="observed, imagined, or reflected")
    links: CWMStateLinks
    tags: List[str] = Field(default_factory=list)
    data: CWMAGraphPayload


class CWMAStateService:
    """Service for emitting CWM-A state records.

    Tracks entity/relationship changes and emits properly formatted
    CWMState envelopes for each state update.
    """

    def __init__(self, source: str = "cwm_a_service") -> None:
        """Initialize the state service.

        Args:
            source: Identifier for the emitting subsystem
        """
        self._source = source
        self._state_history: List[CWMState] = []
        self._current_snapshot: Dict[str, Dict[str, Any]] = {}
        logger.info(f"CWM-A State Service initialized with source: {source}")

    def _generate_state_id(self) -> str:
        """Generate a unique state ID."""
        return f"cwm_a_{uuid.uuid4().hex[:12]}"

    def _compute_entity_diff(
        self,
        entity_id: str,
        entity_type: str,
        new_properties: Dict[str, Any],
    ) -> EntityDiff:
        """Compute the diff between current and new entity state.

        Args:
            entity_id: Entity identifier
            entity_type: Type of entity
            new_properties: New property values

        Returns:
            EntityDiff describing the change
        """
        snapshot_key = f"entity:{entity_id}"
        before = self._current_snapshot.get(snapshot_key)

        if before is None:
            # New entity
            operation = "create"
            changed_properties = list(new_properties.keys())
        else:
            # Update existing entity
            operation = "update"
            changed_properties = [
                k
                for k in set(list(before.keys()) + list(new_properties.keys()))
                if before.get(k) != new_properties.get(k)
            ]

        return EntityDiff(
            entity_id=entity_id,
            entity_type=entity_type,
            operation=operation,
            before=before,
            after=new_properties,
            changed_properties=changed_properties,
        )

    def _update_snapshot(
        self,
        entity_diffs: List[EntityDiff],
        relation_diffs: List[RelationDiff],
    ) -> None:
        """Update the internal snapshot with applied changes.

        Args:
            entity_diffs: Entity changes applied
            relation_diffs: Relationship changes applied
        """
        for entity_diff in entity_diffs:
            key = f"entity:{entity_diff.entity_id}"
            if entity_diff.operation == "delete":
                self._current_snapshot.pop(key, None)
            else:
                self._current_snapshot[key] = entity_diff.after or {}

        for rel_diff in relation_diffs:
            key = f"rel:{rel_diff.source_id}:{rel_diff.relation_type}:{rel_diff.target_id}"
            if rel_diff.operation == "delete":
                self._current_snapshot.pop(key, None)
            else:
                self._current_snapshot[key] = rel_diff.properties or {}

    def emit_state_update(
        self,
        entity_diffs: List[EntityDiff],
        relation_diffs: Optional[List[RelationDiff]] = None,
        validation: Optional[ValidationResult] = None,
        confidence: float = 1.0,
        status: str = "observed",
        tags: Optional[List[str]] = None,
        links: Optional[CWMStateLinks] = None,
    ) -> CWMState:
        """Emit a CWM-A state update.

        Args:
            entity_diffs: List of entity changes
            relation_diffs: List of relationship changes
            validation: SHACL validation result
            confidence: Confidence score (0.0-1.0)
            status: State status (observed, imagined, reflected)
            tags: Optional tags for filtering
            links: Links to related HCG entities

        Returns:
            CWMState envelope containing the update
        """
        relation_diffs = relation_diffs or []
        validation = validation or ValidationResult(passed=True)
        tags = tags or []
        links = links or CWMStateLinks()

        # Collect entity IDs for links
        entity_ids = [d.entity_id for d in entity_diffs]
        if links.entity_ids:
            entity_ids.extend(links.entity_ids)
        links.entity_ids = list(set(entity_ids))

        # Build the payload
        payload = CWMAGraphPayload(
            entities=entity_diffs,
            relations=relation_diffs,
            violations=validation.violations,
            validation=validation,
        )

        # Create the state envelope
        state = CWMState(
            state_id=self._generate_state_id(),
            model_type="CWM_A",
            source=self._source,
            timestamp=datetime.now(timezone.utc),
            confidence=confidence,
            status=status,
            links=links,
            tags=tags,
            data=payload,
        )

        # Update internal snapshot
        self._update_snapshot(entity_diffs, relation_diffs)

        # Store in history
        self._state_history.append(state)

        logger.info(
            f"Emitted CWM-A state {state.state_id}: "
            f"{len(entity_diffs)} entities, {len(relation_diffs)} relations, "
            f"validation={'passed' if validation.passed else 'failed'}"
        )

        return state

    def emit_entity_update(
        self,
        entity_id: str,
        entity_type: str,
        properties: Dict[str, Any],
        validation_passed: bool = True,
        validation_violations: Optional[List[str]] = None,
        source: Optional[str] = None,
        confidence: float = 1.0,
        tags: Optional[List[str]] = None,
    ) -> CWMState:
        """Convenience method to emit a single entity update.

        Args:
            entity_id: Entity identifier
            entity_type: Type of entity
            properties: New property values
            validation_passed: Whether SHACL validation passed
            validation_violations: List of validation violations
            source: Override source identifier
            confidence: Confidence score
            tags: Optional tags

        Returns:
            CWMState envelope
        """
        # Compute diff
        diff = self._compute_entity_diff(entity_id, entity_type, properties)

        validation = ValidationResult(
            passed=validation_passed,
            violations=validation_violations or [],
        )

        # Temporarily override source if provided
        original_source = self._source
        if source:
            self._source = source

        try:
            return self.emit_state_update(
                entity_diffs=[diff],
                validation=validation,
                confidence=confidence,
                tags=tags or [f"entity_type:{entity_type}"],
            )
        finally:
            self._source = original_source

    def emit_entity_deletion(
        self,
        entity_id: str,
        entity_type: str,
        source: Optional[str] = None,
    ) -> CWMState:
        """Emit a state update for entity deletion.

        Args:
            entity_id: Entity identifier
            entity_type: Type of entity
            source: Override source identifier

        Returns:
            CWMState envelope
        """
        snapshot_key = f"entity:{entity_id}"
        before = self._current_snapshot.get(snapshot_key, {})

        diff = EntityDiff(
            entity_id=entity_id,
            entity_type=entity_type,
            operation="delete",
            before=before,
            after=None,
        )

        original_source = self._source
        if source:
            self._source = source

        try:
            return self.emit_state_update(
                entity_diffs=[diff],
                tags=[f"entity_type:{entity_type}", "operation:delete"],
            )
        finally:
            self._source = original_source

    def emit_relation_update(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        operation: str = "create",
        properties: Optional[Dict[str, Any]] = None,
    ) -> CWMState:
        """Emit a state update for a relationship change.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relation_type: Type of relationship
            operation: create or delete
            properties: Relationship properties

        Returns:
            CWMState envelope
        """
        diff = RelationDiff(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            operation=operation,
            properties=properties,
        )

        return self.emit_state_update(
            entity_diffs=[],
            relation_diffs=[diff],
            tags=[f"relation_type:{relation_type}", f"operation:{operation}"],
            links=CWMStateLinks(entity_ids=[source_id, target_id]),
        )

    def get_state_history(self, limit: int = 100) -> List[CWMState]:
        """Get recent state history.

        Args:
            limit: Maximum number of states to return

        Returns:
            List of recent CWMState records
        """
        return self._state_history[-limit:]

    def get_latest_state(self) -> Optional[CWMState]:
        """Get the most recent state record.

        Returns:
            Most recent CWMState or None if no history
        """
        if self._state_history:
            return self._state_history[-1]
        return None

    def clear_history(self) -> None:
        """Clear state history (preserves current snapshot)."""
        self._state_history.clear()
        logger.info("CWM-A state history cleared")

    def get_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Get the current entity/relation snapshot.

        Returns:
            Dictionary of current entity/relation states
        """
        return self._current_snapshot.copy()
