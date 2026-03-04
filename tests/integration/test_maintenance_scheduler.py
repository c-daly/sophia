"""Integration test for MaintenanceScheduler with real Redis.

Requires: Redis running on localhost:6379
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
import redis as redis_lib

from sophia.maintenance.config import MaintenanceConfig
from sophia.maintenance.job_queue import MaintenanceQueue
from sophia.maintenance.scheduler import MaintenanceScheduler

REDIS_AVAILABLE = False
try:
    _r = redis_lib.from_url("redis://localhost:6379/0")
    _r.ping()
    REDIS_AVAILABLE = True
    _r.close()
except Exception:
    pass

pytestmark = pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")


class TestMaintenanceSchedulerIntegration:
    """Integration tests for MaintenanceScheduler against real Redis."""

    def setup_method(self) -> None:
        self.redis = redis_lib.from_url("redis://localhost:6379/0")
        # Clean up test keys before each test
        self.redis.delete("sophia:maintenance:pending")
        self.redis.delete("sophia:maintenance:failed")

    def teardown_method(self) -> None:
        self.redis.delete("sophia:maintenance:pending")
        self.redis.delete("sophia:maintenance:failed")
        self.redis.close()

    def _make_mock_event_bus(self) -> MagicMock:
        """Create a mock EventBus that records subscriptions."""
        bus = MagicMock()
        bus.subscribe = MagicMock()
        bus.listen = MagicMock()
        bus.stop = MagicMock()
        bus.close = MagicMock()
        return bus

    def test_proposal_processed_enqueues_relationship_discovery(self) -> None:
        """_on_proposal_processed enqueues relationship_discovery job."""
        queue = MaintenanceQueue(self.redis)
        event_bus = self._make_mock_event_bus()
        handler_called: list[dict] = []

        def mock_handler(**kwargs: object) -> None:
            handler_called.append(kwargs)

        config = MaintenanceConfig(
            periodic_enabled=False,
            event_driven_enabled=False,
            threshold_enabled=False,
        )

        scheduler = MaintenanceScheduler(
            queue=queue,
            event_bus=event_bus,
            config=config,
            handlers={"relationship_discovery": mock_handler},
        )

        # Simulate the event that the EventBus would deliver
        scheduler._on_proposal_processed(
            {
                "event_type": "proposal_processed",
                "source": "sophia",
                "payload": {
                    "affected_node_uuids": ["n1", "n2"],
                    "new_types": [],
                    "updated_types": [],
                },
            }
        )

        # Verify job was enqueued into real Redis
        assert queue.pending_count() == 1

        # Dequeue and verify contents
        job = queue.dequeue(timeout=1)
        assert job is not None
        assert job.job_type == "relationship_discovery"
        assert job.params == {"node_uuids": ["n1", "n2"]}

    def test_proposal_processed_enqueues_type_emergence(self) -> None:
        """_on_proposal_processed enqueues type_emergence for updated_types."""
        queue = MaintenanceQueue(self.redis)
        event_bus = self._make_mock_event_bus()

        config = MaintenanceConfig(
            periodic_enabled=False,
            event_driven_enabled=False,
            threshold_enabled=False,
        )

        scheduler = MaintenanceScheduler(
            queue=queue,
            event_bus=event_bus,
            config=config,
            handlers={
                "relationship_discovery": lambda **kw: None,
                "type_emergence": lambda **kw: None,
            },
        )

        scheduler._on_proposal_processed(
            {
                "event_type": "proposal_processed",
                "source": "sophia",
                "payload": {
                    "affected_node_uuids": ["n1"],
                    "new_types": [],
                    "updated_types": ["Vehicle", "Tool"],
                },
            }
        )

        # 1 relationship_discovery + 2 type_emergence = 3 jobs
        assert queue.pending_count() == 3

    def test_threshold_crossed_enqueues_high_priority(self) -> None:
        """_on_threshold_crossed enqueues with high priority."""
        queue = MaintenanceQueue(self.redis)
        event_bus = self._make_mock_event_bus()

        config = MaintenanceConfig(
            periodic_enabled=False,
            post_ingestion_enabled=False,
            threshold_enabled=False,
        )

        scheduler = MaintenanceScheduler(
            queue=queue,
            event_bus=event_bus,
            config=config,
            handlers={"consolidation": lambda **kw: None},
        )

        scheduler._on_threshold_crossed(
            {
                "event_type": "threshold_crossed",
                "payload": {
                    "job_type": "consolidation",
                    "params": {"type_name": "Animal"},
                },
            }
        )

        assert queue.pending_count() == 1
        job = queue.dequeue(timeout=1)
        assert job is not None
        assert job.priority == "high"
        assert job.params == {"type_name": "Animal"}

    @pytest.mark.asyncio
    async def test_dispatch_loop_processes_jobs(self) -> None:
        """Dispatch loop dequeues and processes jobs from real Redis."""
        queue = MaintenanceQueue(self.redis)
        event_bus = self._make_mock_event_bus()
        results: list[dict] = []

        def mock_handler(**kwargs: object) -> None:
            results.append(kwargs)

        config = MaintenanceConfig(
            periodic_enabled=False,
            event_driven_enabled=False,
            post_ingestion_enabled=False,
        )

        scheduler = MaintenanceScheduler(
            queue=queue,
            event_bus=event_bus,
            config=config,
            handlers={"type_emergence": mock_handler},
        )

        # Enqueue a job directly into Redis
        queue.enqueue("type_emergence", params={"type_uuid": "t1"})

        # Start the dispatch loop briefly
        scheduler._running = True
        scheduler._semaphore = asyncio.Semaphore(2)
        task = asyncio.create_task(scheduler._dispatch_loop())
        await asyncio.sleep(2)
        scheduler._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert len(results) == 1
        assert results[0] == {"type_uuid": "t1"}

    @pytest.mark.asyncio
    async def test_dispatch_loop_handles_unknown_job_type(self) -> None:
        """Dispatch loop skips jobs with no registered handler."""
        queue = MaintenanceQueue(self.redis)
        event_bus = self._make_mock_event_bus()

        config = MaintenanceConfig(
            periodic_enabled=False,
            event_driven_enabled=False,
            post_ingestion_enabled=False,
        )

        scheduler = MaintenanceScheduler(
            queue=queue,
            event_bus=event_bus,
            config=config,
            handlers={},  # No handlers registered
        )

        queue.enqueue("unknown_job", params={"x": 1})

        scheduler._running = True
        scheduler._semaphore = asyncio.Semaphore(2)
        task = asyncio.create_task(scheduler._dispatch_loop())
        await asyncio.sleep(2)
        scheduler._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Job was consumed (dequeued) but not re-enqueued
        assert queue.pending_count() == 0

    def test_requeue_and_fail_after_max_retries(self) -> None:
        """Jobs that fail repeatedly end up in the failed queue."""
        queue = MaintenanceQueue(self.redis)

        queue.enqueue("test_job", params={"key": "val"})
        job = queue.dequeue(timeout=1)
        assert job is not None

        # Requeue up to MAX_RETRIES
        for _ in range(MaintenanceQueue.MAX_RETRIES):
            queue.requeue(job)

        # After MAX_RETRIES requeues, job should be in failed queue
        failed_count: int = self.redis.llen("sophia:maintenance:failed")
        assert failed_count == 1

        # Pending queue should have the intermediate requeues minus the final one
        # (MAX_RETRIES - 1 requeues went to pending, last one went to failed)
        pending_count = queue.pending_count()
        assert pending_count == MaintenanceQueue.MAX_RETRIES - 1
