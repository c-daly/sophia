"""Interface for emitting feedback from API endpoints."""

import logging

from redis.exceptions import ConnectionError as RedisConnectionError

from sophia.feedback.models import FeedbackPayload
from sophia.feedback.queue import FeedbackQueue

logger = logging.getLogger(__name__)


class FeedbackDispatcher:
    """Interface for emitting feedback from API endpoints."""

    def __init__(self, queue: FeedbackQueue | None, enabled: bool = True):
        """Initialize dispatcher.

        Args:
            queue: The feedback queue (None if Redis unavailable)
            enabled: Whether feedback emission is enabled
        """
        self.queue = queue
        self.enabled = enabled and queue is not None

    def emit(self, payload: FeedbackPayload) -> str | None:
        """Emit feedback to queue.

        Args:
            payload: The feedback payload to emit

        Returns:
            Message ID if queued, None if disabled or failed
        """
        if not self.enabled:
            logger.debug(f"Feedback disabled, skipping: {payload.feedback_type}")
            return None

        if self.queue is None:
            logger.debug("No queue available, skipping feedback")
            return None

        try:
            message_id = self.queue.enqueue(payload)
            logger.info(
                f"Feedback queued: {message_id} "
                f"type={payload.feedback_type} outcome={payload.outcome}"
            )
            return message_id
        except RedisConnectionError as e:
            logger.error(f"Failed to queue feedback: {e}")
            return None
