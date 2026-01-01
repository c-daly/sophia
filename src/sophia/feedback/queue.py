"""Redis-backed queue for feedback messages."""

import json
import logging
from datetime import datetime

import redis

from sophia.feedback.models import FeedbackPayload

logger = logging.getLogger(__name__)


class FeedbackQueue:
    """Redis-backed queue for feedback messages."""

    QUEUE_KEY = "sophia:feedback:pending"
    FAILED_KEY = "sophia:feedback:failed"
    MAX_RETRIES = 5

    def __init__(self, redis_url: str):
        """Initialize queue with Redis connection.

        Args:
            redis_url: Redis connection URL (e.g., redis://localhost:6379/0)
        """
        self.redis = redis.from_url(redis_url)

    def enqueue(self, payload: FeedbackPayload) -> str:
        """Add feedback to queue.

        Args:
            payload: The feedback payload to queue

        Returns:
            Message ID for tracking
        """
        message_id = f"fb-{datetime.utcnow().timestamp()}"
        message = {
            "id": message_id,
            "payload": payload.model_dump(mode="json"),
            "attempts": 0,
            "created_at": datetime.utcnow().isoformat(),
        }
        self.redis.lpush(self.QUEUE_KEY, json.dumps(message))
        return message_id

    def dequeue(self, timeout: int = 5) -> dict | None:
        """Block-pop next message from queue.

        Args:
            timeout: Seconds to wait for message

        Returns:
            Message dict or None on timeout
        """
        result = self.redis.brpop(self.QUEUE_KEY, timeout=timeout)
        if result:
            data: dict = json.loads(result[1])
            return data
        return None

    def requeue_with_backoff(self, message: dict) -> None:
        """Put message back with incremented attempt count.

        Args:
            message: The message to requeue
        """
        message["attempts"] += 1
        message["next_attempt_after"] = datetime.utcnow().timestamp() + (
            2 ** message["attempts"]
        )
        self.redis.lpush(self.QUEUE_KEY, json.dumps(message))

    def move_to_failed(self, message: dict, error: str) -> None:
        """Move message to failed queue after max retries.

        Args:
            message: The failed message
            error: Final error description
        """
        message["failed_at"] = datetime.utcnow().isoformat()
        message["final_error"] = error
        self.redis.lpush(self.FAILED_KEY, json.dumps(message))

    def pending_count(self) -> int:
        """Get number of messages waiting in queue."""
        count: int = self.redis.llen(self.QUEUE_KEY)
        return count

    def failed_count(self) -> int:
        """Get number of failed messages."""
        count: int = self.redis.llen(self.FAILED_KEY)
        return count
