"""Tests for CWM-A State Service."""

import pytest
from datetime import datetime, timezone

from sophia.cwm_a import (
    CWMAStateService,
    CWMState,
    CWMAGraphPayload,
    CWMStateLinks,
    EntityDiff,
    RelationDiff,
    ValidationResult,
)


class TestEntityDiff:
    """Tests for EntityDiff model."""

    def test_create_entity_diff(self):
        """Test creating an entity diff for a new entity."""
        diff = EntityDiff(
            entity_id="red_block",
            entity_type="block",
            operation="create",
            before=None,
            after={"location": "table", "color": "red"},
            changed_properties=["location", "color"],
        )
        
        assert diff.entity_id == "red_block"
        assert diff.entity_type == "block"
        assert diff.operation == "create"
        assert diff.before is None
        assert diff.after == {"location": "table", "color": "red"}
        assert diff.changed_properties == ["location", "color"]

    def test_update_entity_diff(self):
        """Test creating an entity diff for an update."""
        diff = EntityDiff(
            entity_id="red_block",
            entity_type="block",
            operation="update",
            before={"location": "table", "color": "red"},
            after={"location": "bin", "color": "red"},
            changed_properties=["location"],
        )
        
        assert diff.operation == "update"
        assert diff.before["location"] == "table"
        assert diff.after["location"] == "bin"
        assert "location" in diff.changed_properties

    def test_delete_entity_diff(self):
        """Test creating an entity diff for deletion."""
        diff = EntityDiff(
            entity_id="red_block",
            entity_type="block",
            operation="delete",
            before={"location": "bin", "color": "red"},
            after=None,
        )
        
        assert diff.operation == "delete"
        assert diff.after is None


class TestRelationDiff:
    """Tests for RelationDiff model."""

    def test_create_relation_diff(self):
        """Test creating a relation diff."""
        diff = RelationDiff(
            source_id="red_block",
            target_id="bin",
            relation_type="located_at",
            operation="create",
            properties={"since": "2024-01-01"},
        )
        
        assert diff.source_id == "red_block"
        assert diff.target_id == "bin"
        assert diff.relation_type == "located_at"
        assert diff.operation == "create"

    def test_delete_relation_diff(self):
        """Test creating a relation deletion diff."""
        diff = RelationDiff(
            source_id="red_block",
            target_id="table",
            relation_type="located_at",
            operation="delete",
        )
        
        assert diff.operation == "delete"


class TestValidationResult:
    """Tests for ValidationResult model."""

    def test_passed_validation(self):
        """Test passed validation result."""
        result = ValidationResult(passed=True)
        
        assert result.passed is True
        assert result.violations == []
        assert result.validator_version == "shacl-v1"

    def test_failed_validation(self):
        """Test failed validation result."""
        result = ValidationResult(
            passed=False,
            violations=["Missing required property: uuid", "Invalid type for location"],
        )
        
        assert result.passed is False
        assert len(result.violations) == 2


class TestCWMAStateService:
    """Tests for CWMAStateService."""

    def test_initialization(self):
        """Test service initialization."""
        service = CWMAStateService(source="test_service")
        
        assert service._source == "test_service"
        assert service._state_history == []
        assert service._current_snapshot == {}

    def test_generate_state_id(self):
        """Test state ID generation."""
        service = CWMAStateService()
        
        state_id = service._generate_state_id()
        
        assert state_id.startswith("cwm_a_")
        assert len(state_id) == 18  # "cwm_a_" + 12 hex chars

    def test_emit_state_update_basic(self):
        """Test basic state update emission."""
        service = CWMAStateService(source="test")
        
        diff = EntityDiff(
            entity_id="red_block",
            entity_type="block",
            operation="create",
            after={"location": "table"},
        )
        
        state = service.emit_state_update(entity_diffs=[diff])
        
        assert state.model_type == "CWM_A"
        assert state.source == "test"
        assert state.status == "observed"
        assert state.confidence == 1.0
        assert len(state.data.entities) == 1
        assert state.data.entities[0].entity_id == "red_block"

    def test_emit_state_update_with_validation(self):
        """Test state update with validation result."""
        service = CWMAStateService()
        
        validation = ValidationResult(
            passed=False,
            violations=["Invalid property"],
        )
        
        state = service.emit_state_update(
            entity_diffs=[],
            validation=validation,
        )
        
        assert state.data.validation.passed is False
        assert "Invalid property" in state.data.violations

    def test_emit_state_update_with_relations(self):
        """Test state update with relation diffs."""
        service = CWMAStateService()
        
        relation_diff = RelationDiff(
            source_id="red_block",
            target_id="bin",
            relation_type="located_at",
            operation="create",
        )
        
        state = service.emit_state_update(
            entity_diffs=[],
            relation_diffs=[relation_diff],
        )
        
        assert len(state.data.relations) == 1
        assert state.data.relations[0].source_id == "red_block"

    def test_emit_entity_update_convenience(self):
        """Test the convenience method for single entity updates."""
        service = CWMAStateService()
        
        state = service.emit_entity_update(
            entity_id="gripper",
            entity_type="actuator",
            properties={"position": "home", "holding": None},
        )
        
        assert state.model_type == "CWM_A"
        assert len(state.data.entities) == 1
        assert state.data.entities[0].entity_id == "gripper"
        assert state.data.entities[0].operation == "create"

    def test_emit_entity_update_tracks_changes(self):
        """Test that entity updates track changes correctly."""
        service = CWMAStateService()
        
        # First update - create
        state1 = service.emit_entity_update(
            entity_id="block",
            entity_type="object",
            properties={"x": 0, "y": 0},
        )
        assert state1.data.entities[0].operation == "create"
        
        # Second update - update
        state2 = service.emit_entity_update(
            entity_id="block",
            entity_type="object",
            properties={"x": 10, "y": 0},
        )
        assert state2.data.entities[0].operation == "update"
        assert state2.data.entities[0].before == {"x": 0, "y": 0}
        assert state2.data.entities[0].after == {"x": 10, "y": 0}

    def test_emit_entity_deletion(self):
        """Test entity deletion emission."""
        service = CWMAStateService()
        
        # First create the entity
        service.emit_entity_update(
            entity_id="temp_block",
            entity_type="block",
            properties={"x": 5},
        )
        
        # Then delete it
        state = service.emit_entity_deletion(
            entity_id="temp_block",
            entity_type="block",
        )
        
        assert state.data.entities[0].operation == "delete"
        assert state.data.entities[0].before == {"x": 5}
        assert state.data.entities[0].after is None

    def test_emit_relation_update(self):
        """Test relation update emission."""
        service = CWMAStateService()
        
        state = service.emit_relation_update(
            source_id="block_a",
            target_id="block_b",
            relation_type="stacked_on",
            operation="create",
        )
        
        assert len(state.data.relations) == 1
        assert state.data.relations[0].relation_type == "stacked_on"
        assert "block_a" in state.links.entity_ids
        assert "block_b" in state.links.entity_ids

    def test_state_history(self):
        """Test state history retrieval."""
        service = CWMAStateService()
        
        # Emit multiple states
        for i in range(5):
            service.emit_entity_update(
                entity_id=f"entity_{i}",
                entity_type="test",
                properties={"index": i},
            )
        
        history = service.get_state_history(limit=3)
        
        assert len(history) == 3
        # Should be the last 3
        assert history[-1].data.entities[0].entity_id == "entity_4"

    def test_get_latest_state(self):
        """Test getting the latest state."""
        service = CWMAStateService()
        
        # No history
        assert service.get_latest_state() is None
        
        # Add a state
        service.emit_entity_update(
            entity_id="latest",
            entity_type="test",
            properties={"value": 42},
        )
        
        latest = service.get_latest_state()
        assert latest is not None
        assert latest.data.entities[0].entity_id == "latest"

    def test_clear_history(self):
        """Test clearing state history."""
        service = CWMAStateService()
        
        service.emit_entity_update("e1", "test", {"x": 1})
        service.emit_entity_update("e2", "test", {"x": 2})
        
        assert len(service.get_state_history()) == 2
        
        service.clear_history()
        
        assert len(service.get_state_history()) == 0
        # Snapshot should still exist
        assert len(service.get_snapshot()) > 0

    def test_get_snapshot(self):
        """Test getting the current snapshot."""
        service = CWMAStateService()
        
        service.emit_entity_update("block", "object", {"x": 1, "y": 2})
        service.emit_relation_update("block", "table", "on", "create")
        
        snapshot = service.get_snapshot()
        
        assert "entity:block" in snapshot
        assert "rel:block:on:table" in snapshot

    def test_tags_and_links(self):
        """Test custom tags and links."""
        service = CWMAStateService()
        
        links = CWMStateLinks(
            plan_id="plan_123",
            process_ids=["proc_1", "proc_2"],
        )
        
        state = service.emit_state_update(
            entity_diffs=[],
            tags=["custom:tag", "source:executor"],
            links=links,
        )
        
        assert "custom:tag" in state.tags
        assert state.links.plan_id == "plan_123"
        assert "proc_1" in state.links.process_ids

    def test_confidence_and_status(self):
        """Test custom confidence and status."""
        service = CWMAStateService()
        
        state = service.emit_state_update(
            entity_diffs=[],
            confidence=0.75,
            status="imagined",
        )
        
        assert state.confidence == 0.75
        assert state.status == "imagined"

    def test_cwm_state_envelope_format(self):
        """Test that the CWMState envelope follows the spec."""
        service = CWMAStateService(source="sophia_api")
        
        diff = EntityDiff(
            entity_id="test_entity",
            entity_type="entity",
            operation="create",
            after={"prop": "value"},
        )
        
        state = service.emit_state_update(entity_diffs=[diff])
        
        # Verify all required fields per PHASE2_SPEC
        assert state.state_id.startswith("cwm_a_")
        assert state.model_type == "CWM_A"
        assert state.source == "sophia_api"
        assert isinstance(state.timestamp, datetime)
        assert 0.0 <= state.confidence <= 1.0
        assert state.status in ["observed", "imagined", "reflected"]
        assert isinstance(state.links, CWMStateLinks)
        assert isinstance(state.tags, list)
        assert isinstance(state.data, CWMAGraphPayload)


class TestCWMStateModels:
    """Tests for CWMState and related models."""

    def test_cwm_state_serialization(self):
        """Test CWMState model serialization."""
        state = CWMState(
            state_id="cwm_a_test123",
            model_type="CWM_A",
            source="test",
            timestamp=datetime.now(timezone.utc),
            confidence=0.95,
            status="observed",
            links=CWMStateLinks(entity_ids=["e1", "e2"]),
            tags=["test"],
            data=CWMAGraphPayload(
                entities=[
                    EntityDiff(
                        entity_id="e1",
                        entity_type="test",
                        operation="create",
                        after={"x": 1},
                    )
                ],
            ),
        )
        
        # Should be serializable
        data = state.model_dump()
        
        assert data["state_id"] == "cwm_a_test123"
        assert data["model_type"] == "CWM_A"
        assert len(data["data"]["entities"]) == 1

    def test_cwma_graph_payload_defaults(self):
        """Test CWMAGraphPayload default values."""
        payload = CWMAGraphPayload()
        
        assert payload.entities == []
        assert payload.relations == []
        assert payload.violations == []
        assert payload.validation.passed is True

    def test_cwm_state_links_optional(self):
        """Test CWMStateLinks optional fields."""
        links = CWMStateLinks()
        
        assert links.process_ids is None
        assert links.plan_id is None
        assert links.entity_ids is None
        
        links_with_values = CWMStateLinks(
            entity_ids=["e1"],
            plan_id="p1",
        )
        
        assert links_with_values.entity_ids == ["e1"]
        assert links_with_values.plan_id == "p1"
