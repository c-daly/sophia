"""Core maintenance scheduler that ties triggers to the job queue."""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
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
    - Threshold: checks member counts against configured thresholds

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
        self._queue = queue
        self._event_bus = event_bus
        self._config = config
        self._handlers = handlers
        self._hcg = hcg_client
        self._running = False
        self._semaphore: asyncio.Semaphore | None = None
        self._listener_thread: threading.Thread | None = None
        self._job_tasks: set[asyncio.Task] = set()

        if config.enabled:
            self._setup_subscriptions()

    def _setup_subscriptions(self) -> None:
        """Subscribe to EventBus channels based on config toggles."""
        if self._config.post_ingestion_enabled:
            self._event_bus.subscribe(
                "logos:sophia:proposal_processed",
                self._on_proposal_processed,
            )
            logger.info("Subscribed to logos:sophia:proposal_processed")

        if self._config.event_driven_enabled:
            self._event_bus.subscribe(
                "logos:sophia:threshold_crossed",
                self._on_threshold_crossed,
            )
            logger.info("Subscribed to logos:sophia:threshold_crossed")

    def _on_proposal_processed(self, event: dict) -> None:
        """Handle proposal_processed events by enqueuing discovery jobs.

        Enqueues relationship_discovery for affected nodes and
        type_emergence for updated types.

        Note: Called from the EventBus listener thread. Must remain synchronous.
        Exceptions are caught to prevent killing the listener thread.

        Args:
            event: Event payload with affected_node_uuids and type UUIDs.
        """
        try:
            payload = event.get("payload", {})
            affected_nodes = payload.get("affected_node_uuids", [])
            new_types = payload.get("new_types", [])
            updated_types = payload.get("updated_types", [])

            if affected_nodes and "relationship_discovery" in self._handlers:
                self._queue.enqueue(
                    job_type="relationship_discovery",
                    priority="normal",
                    params={"node_uuids": affected_nodes},
                )

            all_types = new_types + updated_types
            if all_types and "type_emergence" in self._handlers:
                for type_entry in all_types:
                    self._queue.enqueue(
                        job_type="type_emergence",
                        priority="normal",
                        params={"type_uuid": type_entry["uuid"]},
                    )

            # Threshold check for new/updated types
            all_type_uuids = [t["uuid"] for t in new_types + updated_types]
            if self._config.threshold_enabled and all_type_uuids:
                self._check_thresholds(all_type_uuids)
        except Exception:
            logger.exception("Error handling proposal_processed event")

    def _on_threshold_crossed(self, event: dict) -> None:
        """Handle threshold_crossed events by enqueuing the specified job.

        Note: Called from the EventBus listener thread. Must remain synchronous.
        Exceptions are caught to prevent killing the listener thread.

        Args:
            event: Event payload with job_type and optional params.
        """
        try:
            payload = event.get("payload", {})
            job_type = payload.get("job_type", "")
            params = payload.get("params", {})

            if not job_type:
                logger.warning("threshold_crossed event missing job_type: %s", event)
                return

            if job_type in self._handlers:
                self._queue.enqueue(
                    job_type=job_type,
                    priority="high",
                    params=params,
                )
            else:
                logger.warning(
                    "threshold_crossed: no handler for job_type %s", job_type
                )
        except Exception:
            logger.exception("Error handling threshold_crossed event")

    def _check_thresholds(self, type_uuids: list[str]) -> None:
        """Check if any types cross the member count threshold."""
        if self._hcg is None:
            return
        # TODO: Implement when HCGClient exposes member count queries
        logger.debug(
            "Threshold checking not yet implemented, skipping %d types",
            len(type_uuids),
        )

    async def start(self) -> None:
        """Start the scheduler: listener thread + dispatch loop + periodic timer.

        If the scheduler is disabled via config, returns immediately.
        """
        if not self._config.enabled:
            logger.info("Maintenance scheduler disabled, not starting")
            return

        self._running = True
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent_jobs)

        # Start EventBus listener in background thread
        self._listener_thread = threading.Thread(
            target=self._event_bus.listen, daemon=True, name="maintenance-listener"
        )
        self._listener_thread.start()
        logger.info("Maintenance scheduler started")

        # Run dispatch loop and periodic timer concurrently
        tasks = [asyncio.create_task(self._dispatch_loop())]
        if self._config.periodic_enabled:
            tasks.append(asyncio.create_task(self._periodic_loop()))
        if self._config.rollup_enabled:
            tasks.append(asyncio.create_task(self._rollup_loop()))

        await asyncio.gather(*tasks)

    async def stop(self) -> None:
        """Signal the scheduler to stop and drain in-flight jobs."""
        self._running = False
        if self._event_bus is not None:
            self._event_bus.stop()
        if self._listener_thread is not None:
            await asyncio.to_thread(self._listener_thread.join, 5)
        # Drain in-flight job tasks before closing Redis
        if self._job_tasks:
            logger.info(
                "Waiting for %d in-flight jobs to complete", len(self._job_tasks)
            )
            await asyncio.gather(*self._job_tasks, return_exceptions=True)
        logger.info("Maintenance scheduler stopped")

    async def _dispatch_loop(self) -> None:
        """Continuously dequeue and dispatch jobs."""
        while self._running:
            try:
                job = await asyncio.to_thread(self._queue.dequeue, 1)
                if job is None:
                    continue
                assert self._semaphore is not None
                task = asyncio.create_task(self._run_job_with_semaphore(job))
                self._job_tasks.add(task)
                task.add_done_callback(self._job_tasks.discard)
            except Exception:
                if self._running:
                    logger.exception("Error in dispatch loop")
                    await asyncio.sleep(1)
        # Drain: process any remaining jobs after stop signal
        while True:
            try:
                job = self._queue.dequeue(timeout=0)
                if job is None:
                    break
                assert self._semaphore is not None
                task = asyncio.create_task(self._run_job_with_semaphore(job))
                self._job_tasks.add(task)
                task.add_done_callback(self._job_tasks.discard)
            except Exception:
                logger.exception("Error draining dispatch queue")
                break

    async def _run_job_with_semaphore(self, job: MaintenanceJob) -> None:
        """Run a job bounded by the concurrency semaphore."""
        assert self._semaphore is not None
        async with self._semaphore:
            await self._dispatch_job(job)

    async def _periodic_loop(self) -> None:
        """Periodically enqueue full scan jobs."""
        interval = self._config.periodic_interval_seconds
        first_run = True
        while self._running:
            if first_run:
                first_run = False
            else:
                await asyncio.sleep(interval)
                if not self._running:
                    break
            try:
                logger.info("Periodic maintenance scan triggered")
                # Periodic scans only trigger type_emergence. Relationship discovery
                # requires per-node embeddings and is too expensive for full-graph
                # sweeps; it runs only via post-ingestion triggers on affected nodes.
                if "type_emergence" in self._handlers and self._hcg is not None:
                    try:
                        all_types = await asyncio.to_thread(
                            self._hcg.get_all_type_definitions
                        )
                    except Exception:
                        logger.exception("Failed to fetch types for periodic scan")
                        all_types = []
                    for td in all_types:
                        type_uuid = td.get("uuid", "")
                        if type_uuid:
                            self._queue.enqueue(
                                job_type="type_emergence",
                                priority="low",
                                params={"type_uuid": type_uuid},
                            )
            except Exception:
                if self._running:
                    logger.exception("Error in periodic loop")
                    await asyncio.sleep(1)

    async def _rollup_loop(self) -> None:
        """Periodically troll the type layer and enqueue a type_rollup pass (#160).

        Decoupled from the per-type emergence cadence: the rollup is idempotent,
        so a plain timer is enough -- if there are new types to organise it does,
        otherwise it is a cheap no-op. Unlike _periodic_loop it sleeps BEFORE the
        first pass by design: there is nothing to roll up until ingestion and
        emergence have populated the type layer, so an immediate fire would be a
        guaranteed no-op (greptile #161).
        """
        interval = self._config.rollup_interval_seconds
        while self._running:
            await asyncio.sleep(interval)
            if not self._running:
                break
            try:
                if "type_rollup" in self._handlers:
                    logger.info("Type-rollup scan triggered")
                    self._queue.enqueue(
                        job_type="type_rollup", priority="low", params={}
                    )
            except Exception:
                if self._running:
                    logger.exception("Error in rollup loop")
                    await asyncio.sleep(1)

    async def _dispatch_job(self, job: MaintenanceJob) -> None:
        """Dispatch a single job to the appropriate handler.

        Args:
            job: The maintenance job to dispatch.
        """
        handler = self._handlers.get(job.job_type)
        if handler is None:
            logger.warning(
                "No handler for job type %s (job %s), skipping",
                job.job_type,
                job.id,
            )
            return

        logger.info("Dispatching maintenance job %s: %s", job.id, job.job_type)
        try:
            if inspect.iscoroutinefunction(handler):
                await handler(**job.params)
            else:
                await asyncio.to_thread(handler, **job.params)
            logger.info("Maintenance job %s completed", job.id)
        except Exception:
            logger.exception("Maintenance job %s failed", job.id)
            self._queue.requeue(job)
