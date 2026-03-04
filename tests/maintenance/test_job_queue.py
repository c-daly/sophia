"""Tests for MaintenanceQueue Redis-backed job queue."""

import json
from unittest.mock import MagicMock


from sophia.maintenance.job_queue import MaintenanceJob, MaintenanceQueue


class TestMaintenanceQueue:
    """Tests for MaintenanceQueue."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_redis = MagicMock()
        self.queue = MaintenanceQueue(self.mock_redis)

    def test_enqueue_creates_job_with_metadata(self) -> None:
        """Enqueue should create a job with UUID, type, priority, and timestamp."""
        job_id = self.queue.enqueue(
            "consolidation", priority="high", params={"threshold": 0.8}
        )

        assert isinstance(job_id, str)
        assert job_id.startswith("maint-")

        # Verify lpush was called with correct queue key
        self.mock_redis.lpush.assert_called_once()
        call_args = self.mock_redis.lpush.call_args
        assert call_args[0][0] == "sophia:maintenance:pending"

        # Parse the pushed JSON and verify structure
        pushed_data = json.loads(call_args[0][1])
        assert pushed_data["id"] == job_id
        assert pushed_data["job_type"] == "consolidation"
        assert pushed_data["priority"] == "high"
        assert pushed_data["params"] == {"threshold": 0.8}
        assert pushed_data["attempts"] == 0
        assert "created_at" in pushed_data

    def test_enqueue_defaults(self) -> None:
        """Enqueue with defaults should use normal priority and empty params."""
        self.queue.enqueue("pruning")

        pushed_data = json.loads(self.mock_redis.lpush.call_args[0][1])
        assert pushed_data["priority"] == "normal"
        assert pushed_data["params"] == {}

    def test_dequeue_returns_job(self) -> None:
        """Dequeue should return a MaintenanceJob when data is available."""
        job_data = {
            "id": "maint-abc123",
            "job_type": "consolidation",
            "priority": "normal",
            "params": {},
            "created_at": "2026-01-01T00:00:00",
            "attempts": 0,
        }
        self.mock_redis.brpop.return_value = (
            "sophia:maintenance:pending",
            json.dumps(job_data),
        )

        job = self.queue.dequeue(timeout=1)

        assert job is not None
        assert isinstance(job, MaintenanceJob)
        assert job.id == "maint-abc123"
        assert job.job_type == "consolidation"
        assert job.priority == "normal"
        assert job.attempts == 0
        self.mock_redis.brpop.assert_called_once_with(
            "sophia:maintenance:pending", timeout=1
        )

    def test_dequeue_returns_none_on_timeout(self) -> None:
        """Dequeue should return None when brpop times out."""
        self.mock_redis.brpop.return_value = None

        job = self.queue.dequeue(timeout=1)

        assert job is None

    def test_pending_count(self) -> None:
        """pending_count should return the queue length."""
        self.mock_redis.llen.return_value = 5

        assert self.queue.pending_count() == 5
        self.mock_redis.llen.assert_called_once_with("sophia:maintenance:pending")

    def test_requeue_increments_attempts(self) -> None:
        """Requeue should increment attempts and push back to pending queue."""
        job = MaintenanceJob(
            id="maint-abc123",
            job_type="consolidation",
            priority="normal",
            params={},
            created_at="2026-01-01T00:00:00",
            attempts=1,
        )

        self.queue.requeue(job)

        assert job.attempts == 2
        self.mock_redis.lpush.assert_called_once()
        call_args = self.mock_redis.lpush.call_args
        assert call_args[0][0] == "sophia:maintenance:pending"

    def test_requeue_moves_to_failed_after_max_retries(self) -> None:
        """Requeue should move job to failed queue when max retries exceeded."""
        job = MaintenanceJob(
            id="maint-abc123",
            job_type="consolidation",
            priority="normal",
            params={},
            created_at="2026-01-01T00:00:00",
            attempts=3,  # Already at MAX_RETRIES
        )

        self.queue.requeue(job)

        # Should push to failed key, not pending
        call_args = self.mock_redis.lpush.call_args
        assert call_args[0][0] == "sophia:maintenance:failed"
        pushed_data = json.loads(call_args[0][1])
        assert "failed_at" in pushed_data
