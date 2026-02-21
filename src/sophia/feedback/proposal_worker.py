"""Background worker that processes proposals from the Redis queue."""

import asyncio
import logging
from typing import Any

from sophia.feedback.proposal_queue import ProposalQueue

logger = logging.getLogger(__name__)


class ProposalWorker:
    """Background worker that dequeues proposals and runs ProposalProcessor."""

    def __init__(
        self,
        queue: ProposalQueue,
        processor: Any,
        context_ttl: int = 3600,
    ):
        self.queue = queue
        self.processor = processor
        self.context_ttl = context_ttl
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
        message = await asyncio.to_thread(self.queue.dequeue, timeout=1)
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
