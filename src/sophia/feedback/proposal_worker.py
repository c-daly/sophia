"""Background worker that processes proposals from the Redis queue."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from sophia.feedback.proposal_queue import ProposalQueue

if TYPE_CHECKING:
    from sophia.ingestion.proposal_processor import ProposalProcessor

logger = logging.getLogger(__name__)


class ProposalWorker:
    """Background worker that dequeues proposals and runs ProposalProcessor."""

    def __init__(
        self,
        queue: ProposalQueue,
        processor: ProposalProcessor,
        context_ttl: int = 3600,
        error_backoff: float = 1.0,
    ):
        self.queue = queue
        self.processor = processor
        self.context_ttl = context_ttl
        # Seconds to wait after a dequeue error before retrying, so a Redis
        # outage throttles the loop instead of spinning hot.
        self._error_backoff = error_backoff
        self._running = False

    async def start(self) -> None:
        self._running = True
        logger.info("Proposal worker started")

        while self._running:
            await self._process_one()

    def stop(self) -> None:
        self._running = False
        logger.info("Proposal worker stopping")

    async def _process_one(self) -> None:
        try:
            message = await asyncio.to_thread(self.queue.dequeue, timeout=1)
        except Exception as e:
            # Redis unavailable (e.g. a transient outage): log, back off, and
            # keep the loop alive so the worker resumes when Redis recovers.
            # The dequeue is a blocking brpop; previously this call sat outside
            # any guard, so a dropped connection killed the worker task and
            # silently stopped the cache pipeline until a sophia restart.
            logger.warning(
                "Proposal dequeue failed (Redis unavailable?); backing off %.1fs: %s",
                self._error_backoff,
                e,
                exc_info=True,
            )
            # Don't burn the full backoff if we're shutting down mid-outage.
            if self._running:
                await asyncio.sleep(self._error_backoff)
            return
        if not message:
            return

        proposal = message.get("payload", {})
        conversation_id = message.get("conversation_id")

        try:
            result = await asyncio.to_thread(self.processor.process, proposal)
            relevant_context = result.get("relevant_context", [])

            if conversation_id and relevant_context:
                await asyncio.to_thread(
                    self.queue.store_context,
                    conversation_id,
                    relevant_context,
                    self.context_ttl,
                )

            logger.info(
                "Processed proposal %s: %d nodes, %d edges, %d context items",
                message.get("id", "?"),
                len(result.get("stored_node_ids", [])),
                len(result.get("stored_edge_ids", [])),
                len(relevant_context),
            )
        except Exception as e:
            logger.error("Proposal processing failed for %s: %s", message.get("id"), e)
