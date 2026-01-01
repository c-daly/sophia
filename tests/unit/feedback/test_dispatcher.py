"""Unit tests for FeedbackDispatcher."""

from unittest.mock import MagicMock, patch

import pytest

from sophia.feedback.dispatcher import FeedbackDispatcher
from sophia.feedback.models import FeedbackPayload


class TestFeedbackDispatcher:
    """Tests for FeedbackDispatcher."""

    @pytest.fixture
    def mock_queue(self) -> MagicMock:
        """Create a mock queue."""
        queue = MagicMock()
        queue.enqueue.return_value = "fb-123"
        return queue

    @pytest.fixture
    def valid_payload(self) -> FeedbackPayload:
        """Create a valid test payload."""
        return FeedbackPayload(
            plan_id="plan-123",
            feedback_type="plan",
            outcome="created",
            reason="Test plan",
        )

    def test_emit_when_enabled(
        self, mock_queue: MagicMock, valid_payload: FeedbackPayload
    ) -> None:
        """Test emit when dispatcher is enabled."""
        dispatcher = FeedbackDispatcher(mock_queue, enabled=True)

        result = dispatcher.emit(valid_payload)

        assert result == "fb-123"
        mock_queue.enqueue.assert_called_once_with(valid_payload)

    def test_emit_when_disabled(
        self, mock_queue: MagicMock, valid_payload: FeedbackPayload
    ) -> None:
        """Test emit when dispatcher is disabled."""
        dispatcher = FeedbackDispatcher(mock_queue, enabled=False)

        result = dispatcher.emit(valid_payload)

        assert result is None
        mock_queue.enqueue.assert_not_called()

    def test_emit_when_queue_none(self, valid_payload: FeedbackPayload) -> None:
        """Test emit when queue is None."""
        dispatcher = FeedbackDispatcher(None, enabled=True)

        result = dispatcher.emit(valid_payload)

        assert result is None

    def test_emit_handles_redis_error(
        self, mock_queue: MagicMock, valid_payload: FeedbackPayload
    ) -> None:
        """Test emit handles Redis connection error gracefully."""
        from redis.exceptions import ConnectionError as RedisConnectionError

        mock_queue.enqueue.side_effect = RedisConnectionError("Connection refused")
        dispatcher = FeedbackDispatcher(mock_queue, enabled=True)

        result = dispatcher.emit(valid_payload)

        assert result is None

    def test_enabled_property_with_none_queue(self) -> None:
        """Test that enabled is False when queue is None."""
        dispatcher = FeedbackDispatcher(None, enabled=True)
        assert dispatcher.enabled is False

    def test_enabled_property_with_queue(self, mock_queue: MagicMock) -> None:
        """Test that enabled reflects constructor value when queue exists."""
        dispatcher = FeedbackDispatcher(mock_queue, enabled=True)
        assert dispatcher.enabled is True

        dispatcher2 = FeedbackDispatcher(mock_queue, enabled=False)
        assert dispatcher2.enabled is False
