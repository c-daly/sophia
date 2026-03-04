"""Redis-backed job queue for maintenance tasks."""

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

PRIORITY_KEYS = {
    "high": "sophia:maintenance:queue:high",
    "normal": "sophia:maintenance:queue:normal",
    "low": "sophia:maintenance:queue:low",
}

# Ordered from highest to lowest priority for brpop
PRIORITY_ORDER = [
    PRIORITY_KEYS["high"],
    PRIORITY_KEYS["normal"],
    PRIORITY_KEYS["low"],
]


@dataclass
class MaintenanceJob:
    """A maintenance job to be processed."""

    id: str
    job_type: str
    priority: str
    params: dict
    created_at: str
    attempts: int = 0


class MaintenanceQueue:
    """Redis-backed queue for maintenance jobs.

    Uses separate Redis lists per priority level. brpop across the lists
    in priority order ensures high-priority jobs are dequeued first.
    """

    QUEUE_KEY_HIGH = PRIORITY_KEYS["high"]
    QUEUE_KEY_NORMAL = PRIORITY_KEYS["normal"]
    QUEUE_KEY_LOW = PRIORITY_KEYS["low"]
    FAILED_KEY = "sophia:maintenance:failed"
    MAX_RETRIES = 3

    def __init__(self, redis_client: Any) -> None:
        """Initialize queue with a Redis client.

        Args:
            redis_client: Redis client instance.
        """
        self.redis = redis_client

    def _key_for_priority(self, priority: str) -> str:
        """Return the Redis list key for the given priority.

        Args:
            priority: Job priority ("low", "normal", "high").

        Returns:
            Redis key string.
        """
        return PRIORITY_KEYS.get(priority, PRIORITY_KEYS["normal"])

    def enqueue(
        self,
        job_type: str,
        priority: str = "normal",
        params: dict | None = None,
    ) -> str:
        """Add a maintenance job to the queue.

        Args:
            job_type: Type of maintenance task (e.g. "consolidation", "pruning").
            priority: Job priority ("low", "normal", "high").
            params: Optional parameters for the job.

        Returns:
            Job ID for tracking.
        """
        job_id = f"maint-{uuid.uuid4().hex[:12]}"
        job_data = {
            "id": job_id,
            "job_type": job_type,
            "priority": priority,
            "params": params if params is not None else {},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "attempts": 0,
        }
        key = self._key_for_priority(priority)
        self.redis.lpush(key, json.dumps(job_data))
        logger.info(
            "Enqueued maintenance job %s (type=%s, priority=%s)",
            job_id,
            job_type,
            priority,
        )
        return job_id

    def dequeue(self, timeout: int = 1) -> MaintenanceJob | None:
        """Block-pop the next job from the queue, respecting priority order.

        High-priority jobs are dequeued before normal, which are dequeued
        before low. This is achieved by passing the keys to brpop in order;
        brpop returns from the first non-empty list.

        Args:
            timeout: Seconds to wait for a job.

        Returns:
            MaintenanceJob or None on timeout.
        """
        result = self.redis.brpop(PRIORITY_ORDER, timeout=timeout)
        if result:
            data: dict = json.loads(result[1])
            return MaintenanceJob(**data)
        return None

    def requeue(self, job: MaintenanceJob) -> None:
        """Put a job back on the queue with incremented attempt count.

        If the job has reached MAX_RETRIES, it is moved to the failed queue
        instead. The job is re-enqueued to its original priority key.

        Args:
            job: The job to requeue.
        """
        job.attempts += 1
        if job.attempts >= self.MAX_RETRIES:
            self.move_to_failed(job)
            return
        key = self._key_for_priority(job.priority)
        self.redis.lpush(key, json.dumps(asdict(job)))
        logger.info("Requeued job %s (attempt %d)", job.id, job.attempts)

    def move_to_failed(self, job: MaintenanceJob) -> None:
        """Move a job to the failed queue.

        Args:
            job: The job that has permanently failed.
        """
        data = asdict(job)
        data["failed_at"] = datetime.now(timezone.utc).isoformat()
        self.redis.lpush(self.FAILED_KEY, json.dumps(data))
        logger.warning(
            "Job %s moved to failed queue after %d attempts", job.id, job.attempts
        )

    def pending_count(self) -> int:
        """Get number of jobs waiting across all priority queues."""
        total = 0
        for key in PRIORITY_ORDER:
            total += self.redis.llen(key)
        return total
