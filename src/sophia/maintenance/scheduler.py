"""Core maintenance scheduler that ties triggers to the job queue."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from sophia.maintenance.config import MaintenanceConfig
from sophia.maintenance.job_queue import MaintenanceJob, MaintenanceQueue

logger = logging.getLogger(__name__)


class MaintenanceScheduler:
    """Scheduler that listens for events and dispatches maintenance jobs.

    Supports four trigger sources:
    - Post-ingestion: reacts to proposal_processed events
    - Event-driven: reacts to threshold_crossed events
    - Periodic: runs full scans on a configurable interval
    - Threshold: (configured via event-driven channel)

    Each trigger enqueues jobs into the MaintenanceQueue. The dispatch loop
    dequeues and runs them through registered handler callables.
    """

    def __init__(
        self,
        queue: MaintenanceQueue,
        event_bus: Any,
        config: MaintenanceConfig,
        handlers: dict[str, Callable],
        hcg_client: Any | None = None,
    ) -> None:
        """Initialize the scheduler.

        Args:
            queue: Redis-backed maintenance job queue.
            event_bus: EventBus instance for pub/sub subscriptions.
            config: Maintenance scheduler configuration.
            handlers: Mapping of job_type -> callable handler.
            hcg_client: Optional HCG client for graph operations.
        """
        self.queue = queue
        self.event_bus = event_bus
        self.config = config
        self.handlers = handlers
        self.hcg_client = hcg_client
        self._running = False
        self._semaphore: asyncio.Semaphore | None = None

        if config.enabled:
            self._setup_subscriptions()

    def _setup_subscriptions(self) -> None:
        """Subscribe to EventBus channels based on config toggles."""
        if self.config.post_ingestion_enabled:
            self.event_bus.subscribe(
                "logos:sophia:proposal_processed",
                self._on_proposal_processed,
            )
            logger.info("Subscribed to logos:sophia:proposal_processed")

        if self.config.event_driven_enabled:
            self.event_bus.subscribe(
                "logos:sophia:threshold_crossed",
                self._on_threshold_crossed,
            )
            logger.info("Subscribed to logos:sophia:threshold_crossed")

    def _on_proposal_processed(self, event: dict) -> None:
        """Handle proposal_processed events by enqueuing discovery jobs.

        Enqueues relationship_discovery for affected nodes and
        type_emergence for updated types.

        Args:
            event: Event payload with affected_node_ids and updated_types.
        """
        affected_nodes = event.get("affected_node_ids", [])
        updated_types = event.get("updated_types", [])

        if affected_nodes:
            self.queue.enqueue(
                job_type="relationship_discovery",
                priority="normal",
                params={"node_ids": affected_nodes},
            )

        if updated_types:
            self.queue.enqueue(
                job_type="type_emergence",
                priority="normal",
                params={"types": updated_types},
            )

    def _on_threshold_crossed(self, event: dict) -> None:
        """Handle threshold_crossed events by enqueuing the specified job.

        Args:
            event: Event payload with job_type and optional params.
        """
        job_type = event.get("job_type", "")
        params = event.get("params", {})

        if not job_type:
            logger.warning("threshold_crossed event missing job_type: %s", event)
            return

        self.queue.enqueue(
            job_type=job_type,
            priority="normal",
            params=params,
        )

    async def start(self) -> None:
        """Start the scheduler dispatch and periodic loops.

        If the scheduler is disabled via config, returns immediately.
        """
        if not self.config.enabled:
            logger.info("Maintenance scheduler disabled, not starting")
            return

        self._running = True
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_jobs)
        logger.info("Maintenance scheduler started")

        tasks = [self._dispatch_loop()]
        if self.config.periodic_enabled:
            tasks.append(self._periodic_loop())

        await asyncio.gather(*tasks)

    def stop(self) -> None:
        """Signal the scheduler to stop."""
        self._running = False
        logger.info("Maintenance scheduler stopping")

    async def _dispatch_loop(self) -> None:
        """Continuously dequeue and dispatch jobs."""
        while self._running:
            job = await asyncio.to_thread(self.queue.dequeue, timeout=1)
            if job is None:
                continue
            assert self._semaphore is not None
            async with self._semaphore:
                await self._dispatch_job(job)

    async def _periodic_loop(self) -> None:
        """Periodically enqueue full scan jobs."""
        while self._running:
            await asyncio.sleep(self.config.periodic_interval_seconds)
            if not self._running:
                break
            self.queue.enqueue(
                job_type="full_scan",
                priority="low",
                params={},
            )
            logger.info("Enqueued periodic full_scan job")

    async def _dispatch_job(self, job: MaintenanceJob) -> None:
        """Dispatch a single job to the appropriate handler.

        Args:
            job: The maintenance job to dispatch.
        """
        handler = self.handlers.get(job.job_type)
        if handler is None:
            logger.warning(
                "No handler for job type %s (job %s), skipping",
                job.job_type,
                job.id,
            )
            return

        try:
            await asyncio.to_thread(handler, job)
            logger.info("Completed job %s (type=%s)", job.id, job.job_type)
        except Exception:
            logger.exception("Job %s failed, requeuing", job.id)
            self.queue.requeue(job)
