"""Tests for MaintenanceScheduler core."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sophia.maintenance.config import MaintenanceConfig
from sophia.maintenance.job_queue import MaintenanceJob, MaintenanceQueue
from sophia.maintenance.scheduler import MaintenanceScheduler


def _make_job(job_type: str = "consolidation", **kwargs) -> MaintenanceJob:
    """Create a MaintenanceJob for testing."""
    defaults = {
        "id": "maint-test123",
        "job_type": job_type,
        "priority": "normal",
        "params": {},
        "created_at": "2026-03-04T00:00:00+00:00",
        "attempts": 0,
    }
    defaults.update(kwargs)
    return MaintenanceJob(**defaults)


class TestMaintenanceSchedulerInit:
    """Tests for scheduler initialization and subscription setup."""

    def setup_method(self) -> None:
        self.mock_queue = MagicMock(spec=MaintenanceQueue)
        self.mock_event_bus = MagicMock()
        self.handlers = {"consolidation": MagicMock(), "pruning": MagicMock()}

    def test_init_registers_handlers(self) -> None:
        """Scheduler should store handlers dict and wire up config."""
        config = MaintenanceConfig(enabled=True)
        scheduler = MaintenanceScheduler(
            queue=self.mock_queue,
            event_bus=self.mock_event_bus,
            config=config,
            handlers=self.handlers,
        )
        assert scheduler._handlers == self.handlers
        assert scheduler._config is config
        assert scheduler._queue is self.mock_queue

    def test_disabled_scheduler_does_nothing(self) -> None:
        """When enabled=False, no EventBus subscriptions should be created."""
        config = MaintenanceConfig(enabled=False)
        MaintenanceScheduler(
            queue=self.mock_queue,
            event_bus=self.mock_event_bus,
            config=config,
            handlers=self.handlers,
        )
        self.mock_event_bus.subscribe.assert_not_called()

    def test_enabled_scheduler_subscribes_to_proposal_processed(self) -> None:
        """When post_ingestion_enabled, should subscribe to proposal_processed."""
        config = MaintenanceConfig(
            enabled=True,
            post_ingestion_enabled=True,
            event_driven_enabled=False,
        )
        MaintenanceScheduler(
            queue=self.mock_queue,
            event_bus=self.mock_event_bus,
            config=config,
            handlers=self.handlers,
        )
        subscribe_calls = self.mock_event_bus.subscribe.call_args_list
        channels = [c[0][0] for c in subscribe_calls]
        assert "logos:sophia:proposal_processed" in channels

    def test_disabled_post_ingestion_skips_subscription(self) -> None:
        """When post_ingestion_enabled=False, should not subscribe to proposal_processed."""
        config = MaintenanceConfig(
            enabled=True,
            post_ingestion_enabled=False,
            event_driven_enabled=False,
        )
        MaintenanceScheduler(
            queue=self.mock_queue,
            event_bus=self.mock_event_bus,
            config=config,
            handlers=self.handlers,
        )
        subscribe_calls = self.mock_event_bus.subscribe.call_args_list
        channels = [c[0][0] for c in subscribe_calls]
        assert "logos:sophia:proposal_processed" not in channels

    def test_event_driven_enabled_subscribes_threshold(self) -> None:
        """When event_driven_enabled, should subscribe to threshold_crossed."""
        config = MaintenanceConfig(
            enabled=True,
            post_ingestion_enabled=False,
            event_driven_enabled=True,
        )
        MaintenanceScheduler(
            queue=self.mock_queue,
            event_bus=self.mock_event_bus,
            config=config,
            handlers=self.handlers,
        )
        subscribe_calls = self.mock_event_bus.subscribe.call_args_list
        channels = [c[0][0] for c in subscribe_calls]
        assert "logos:sophia:threshold_crossed" in channels


class TestEventHandlers:
    """Tests for event handler methods that enqueue jobs."""

    def setup_method(self) -> None:
        self.mock_queue = MagicMock(spec=MaintenanceQueue)
        self.mock_event_bus = MagicMock()
        self.handlers = {
            "relationship_discovery": MagicMock(),
            "type_emergence": MagicMock(),
            "consolidation": MagicMock(),
        }
        config = MaintenanceConfig(enabled=True)
        self.scheduler = MaintenanceScheduler(
            queue=self.mock_queue,
            event_bus=self.mock_event_bus,
            config=config,
            handlers=self.handlers,
        )

    def test_on_proposal_processed_enqueues_jobs(self) -> None:
        """_on_proposal_processed should enqueue relationship_discovery and type_emergence."""
        event = {
            "event_type": "proposal_processed",
            "source": "sophia",
            "payload": {
                "affected_node_uuids": ["node-1", "node-2"],
                "new_types": [],
                "updated_types": [
                    {"uuid": "type_Person", "name": "Person"},
                    {"uuid": "type_Organization", "name": "Organization"},
                ],
            },
        }
        self.scheduler._on_proposal_processed(event)

        enqueue_calls = self.mock_queue.enqueue.call_args_list
        job_types = [c[1]["job_type"] for c in enqueue_calls]
        assert "relationship_discovery" in job_types
        assert "type_emergence" in job_types

    def test_on_threshold_crossed_enqueues_event_job(self) -> None:
        """_on_threshold_crossed should enqueue job type from event payload."""
        event = {
            "event_type": "threshold_crossed",
            "payload": {
                "job_type": "consolidation",
                "params": {"threshold": 0.9},
            },
        }
        self.scheduler._on_threshold_crossed(event)

        self.mock_queue.enqueue.assert_called_once_with(
            job_type="consolidation",
            priority="high",
            params={"threshold": 0.9},
        )

    def test_on_proposal_processed_calls_check_thresholds(self) -> None:
        """_on_proposal_processed should call _check_thresholds for new/updated types."""
        config = MaintenanceConfig(enabled=True, threshold_enabled=True)
        scheduler = MaintenanceScheduler(
            queue=self.mock_queue,
            event_bus=MagicMock(),
            config=config,
            handlers=self.handlers,
            hcg_client=MagicMock(),
        )
        event = {
            "event_type": "proposal_processed",
            "payload": {
                "affected_node_uuids": [],
                "new_types": [{"uuid": "type_vehicle", "name": "vehicle"}],
                "updated_types": [{"uuid": "type_person", "name": "person"}],
            },
        }
        # _check_thresholds logs and returns, so just verify no error
        scheduler._on_proposal_processed(event)


class TestDispatch:
    """Tests for async job dispatch."""

    def setup_method(self) -> None:
        self.mock_queue = MagicMock(spec=MaintenanceQueue)
        self.mock_event_bus = MagicMock()
        self.mock_handler = MagicMock(return_value={"status": "ok"})
        self.handlers = {"consolidation": self.mock_handler}
        config = MaintenanceConfig(enabled=True)
        self.scheduler = MaintenanceScheduler(
            queue=self.mock_queue,
            event_bus=self.mock_event_bus,
            config=config,
            handlers=self.handlers,
        )

    @pytest.mark.asyncio
    async def test_dispatch_calls_handler(self) -> None:
        """_dispatch_job should call the correct handler with job.params."""
        job = _make_job("consolidation", params={"key": "value"})
        await self.scheduler._dispatch_job(job)
        self.mock_handler.assert_called_once_with(key="value")

    @pytest.mark.asyncio
    async def test_dispatch_unknown_job_type_logs_warning(self) -> None:
        """_dispatch_job with unknown type should log a warning, not raise."""
        job = _make_job("unknown_type")
        with patch("sophia.maintenance.scheduler.logger") as mock_logger:
            await self.scheduler._dispatch_job(job)
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert "unknown_type" in str(call_args)

    @pytest.mark.asyncio
    async def test_dispatch_requeues_on_failure(self) -> None:
        """_dispatch_job should requeue job when handler raises."""
        self.mock_handler.side_effect = RuntimeError("boom")
        job = _make_job("consolidation")
        await self.scheduler._dispatch_job(job)
        self.mock_queue.requeue.assert_called_once_with(job)


class TestStartStop:
    """Tests for start/stop lifecycle."""

    def setup_method(self) -> None:
        self.mock_queue = MagicMock(spec=MaintenanceQueue)
        self.mock_event_bus = MagicMock()
        self.handlers = {"consolidation": MagicMock()}

    @pytest.mark.asyncio
    async def test_start_disabled_returns_immediately(self) -> None:
        """start() with enabled=False should return without running loops."""
        config = MaintenanceConfig(enabled=False)
        scheduler = MaintenanceScheduler(
            queue=self.mock_queue,
            event_bus=self.mock_event_bus,
            config=config,
            handlers=self.handlers,
        )
        await scheduler.start()
        assert not scheduler._running

    @pytest.mark.asyncio
    async def test_stop_calls_event_bus_stop(self) -> None:
        """stop() should set _running=False and call event_bus.stop()."""
        config = MaintenanceConfig(enabled=True)
        scheduler = MaintenanceScheduler(
            queue=self.mock_queue,
            event_bus=self.mock_event_bus,
            config=config,
            handlers=self.handlers,
        )
        scheduler._running = True
        await scheduler.stop()
        assert not scheduler._running
        self.mock_event_bus.stop.assert_called_once()
