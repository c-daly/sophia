"""Unit tests for feedback models."""

import pytest
from pydantic import ValidationError

from sophia.feedback.models import FeedbackPayload, StateDiff, StepResult


class TestStepResult:
    """Tests for StepResult model."""

    def test_success_step(self) -> None:
        """Test creating a successful step result."""
        result = StepResult(
            step_index=0,
            action="MOVE",
            outcome="success",
            duration_ms=150,
        )
        assert result.step_index == 0
        assert result.action == "MOVE"
        assert result.outcome == "success"
        assert result.error is None
        assert result.duration_ms == 150

    def test_failure_step(self) -> None:
        """Test creating a failed step result."""
        result = StepResult(
            step_index=1,
            action="GRASP",
            outcome="failure",
            error="Gripper fault",
        )
        assert result.outcome == "failure"
        assert result.error == "Gripper fault"

    def test_skipped_step(self) -> None:
        """Test creating a skipped step result."""
        result = StepResult(
            step_index=2,
            action="RELEASE",
            outcome="skipped",
        )
        assert result.outcome == "skipped"


class TestStateDiff:
    """Tests for StateDiff model."""

    def test_empty_diff(self) -> None:
        """Test creating empty state diff."""
        diff = StateDiff()
        assert diff.added_nodes == []
        assert diff.removed_nodes == []
        assert diff.modified_nodes == []

    def test_diff_with_changes(self) -> None:
        """Test creating state diff with changes."""
        diff = StateDiff(
            added_nodes=["node-1", "node-2"],
            modified_nodes=["node-3"],
        )
        assert len(diff.added_nodes) == 2
        assert len(diff.modified_nodes) == 1


class TestFeedbackPayload:
    """Tests for FeedbackPayload model."""

    def test_requires_correlation_key(self) -> None:
        """Test that at least one correlation key is required."""
        with pytest.raises(ValidationError) as exc_info:
            FeedbackPayload(
                feedback_type="plan",
                outcome="created",
                reason="test",
            )
        assert "correlation_id, plan_id, or execution_id" in str(exc_info.value)

    def test_accepts_correlation_id_only(self) -> None:
        """Test payload with only correlation_id."""
        payload = FeedbackPayload(
            correlation_id="req-abc",
            feedback_type="observation",
            outcome="accepted",
            reason="Accepted observation",
        )
        assert payload.correlation_id == "req-abc"
        assert payload.plan_id is None
        assert payload.execution_id is None

    def test_accepts_plan_id_only(self) -> None:
        """Test payload with only plan_id."""
        payload = FeedbackPayload(
            plan_id="plan-123",
            feedback_type="plan",
            outcome="created",
            reason="Created plan",
        )
        assert payload.plan_id == "plan-123"

    def test_accepts_execution_id_only(self) -> None:
        """Test payload with only execution_id."""
        payload = FeedbackPayload(
            execution_id="exec-001",
            feedback_type="execution",
            outcome="success",
            reason="Execution complete",
        )
        assert payload.execution_id == "exec-001"

    def test_accepts_multiple_correlation_keys(self) -> None:
        """Test payload with multiple correlation keys."""
        payload = FeedbackPayload(
            correlation_id="req-abc",
            plan_id="plan-123",
            execution_id="exec-001",
            feedback_type="execution",
            outcome="success",
            reason="All keys present",
        )
        assert payload.correlation_id == "req-abc"
        assert payload.plan_id == "plan-123"
        assert payload.execution_id == "exec-001"

    def test_with_step_results(self) -> None:
        """Test payload with step results."""
        payload = FeedbackPayload(
            plan_id="plan-123",
            feedback_type="execution",
            outcome="success",
            reason="Executed",
            step_results=[
                StepResult(step_index=0, action="MOVE", outcome="success"),
                StepResult(step_index=1, action="GRASP", outcome="success"),
            ],
        )
        assert len(payload.step_results) == 2

    def test_with_state_diff(self) -> None:
        """Test payload with state diff."""
        payload = FeedbackPayload(
            correlation_id="req-abc",
            feedback_type="observation",
            outcome="accepted",
            reason="Added nodes",
            state_diff=StateDiff(added_nodes=["node-1"]),
            node_ids_created=["node-1"],
        )
        assert payload.state_diff.added_nodes == ["node-1"]
        assert payload.node_ids_created == ["node-1"]

    def test_timestamp_auto_generated(self) -> None:
        """Test that timestamp is auto-generated."""
        payload = FeedbackPayload(
            plan_id="plan-123",
            feedback_type="plan",
            outcome="created",
            reason="test",
        )
        assert payload.timestamp is not None

    def test_source_service_default(self) -> None:
        """Test default source_service value."""
        payload = FeedbackPayload(
            plan_id="plan-123",
            feedback_type="plan",
            outcome="created",
            reason="test",
        )
        assert payload.source_service == "sophia"

    def test_feedback_type_literals(self) -> None:
        """Test valid feedback_type values."""
        for ftype in ["observation", "plan", "execution", "validation"]:
            payload = FeedbackPayload(
                plan_id="plan-123",
                feedback_type=ftype,
                outcome="created",
                reason="test",
            )
            assert payload.feedback_type == ftype

    def test_outcome_literals(self) -> None:
        """Test valid outcome values."""
        for outcome in [
            "accepted",
            "rejected",
            "created",
            "success",
            "failure",
            "partial",
        ]:
            payload = FeedbackPayload(
                plan_id="plan-123",
                feedback_type="plan",
                outcome=outcome,
                reason="test",
            )
            assert payload.outcome == outcome
