"""Background worker that sends feedback to Hermes."""

import asyncio
import logging
from datetime import datetime

import httpx

from sophia.feedback.queue import FeedbackQueue

logger = logging.getLogger(__name__)


class FeedbackWorker:
    """Background worker that sends feedback to Hermes."""

    def __init__(
        self,
        queue: FeedbackQueue,
        hermes_url: str,
        timeout: float = 10.0,
    ):
        """Initialize worker.

        Args:
            queue: The feedback queue to process
            hermes_url: Base URL for Hermes service
            timeout: HTTP timeout for requests
        """
        self.queue = queue
        self.hermes_url = hermes_url.rstrip("/")
        self.timeout = timeout
        self._running = False

    async def start(self) -> None:
        """Start processing loop."""
        self._running = True
        logger.info("Feedback worker started")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            while self._running:
                await self._process_one(client)

    def stop(self) -> None:
        """Signal worker to stop."""
        self._running = False
        logger.info("Feedback worker stopping")

    async def _process_one(self, client: httpx.AsyncClient) -> None:
        """Process single message from queue."""
        # Run blocking Redis call in thread to avoid blocking event loop
        message = await asyncio.to_thread(self.queue.dequeue, timeout=1)
        if not message:
            return

        # Check backoff
        next_attempt = message.get("next_attempt_after", 0)
        now = datetime.utcnow().timestamp()
        if now < next_attempt:
            # Not ready yet, put it back without incrementing attempts
            self.queue.requeue(message)
            # Sleep until ready (or at least a bit) to avoid busy-loop
            sleep_time = min(next_attempt - now, 1.0)
            await asyncio.sleep(sleep_time)
            return

        # Attempt send
        error = ""
        try:
            response = await client.post(
                f"{self.hermes_url}/feedback",
                json=message["payload"],
            )

            if response.status_code == 201:
                logger.info(f"Feedback {message['id']} sent successfully")
                return

            error = f"HTTP {response.status_code}: {response.text[:100]}"
            logger.warning(f"Feedback {message['id']} rejected: {error}")

        except httpx.RequestError as e:
            error = str(e)
            logger.warning(f"Feedback {message['id']} failed: {error}")

        # Handle retry or fail
        if message["attempts"] >= FeedbackQueue.MAX_RETRIES:
            self.queue.move_to_failed(message, error)
            logger.error(
                f"Feedback {message['id']} moved to failed after "
                f"{message['attempts']} attempts"
            )
        else:
            self.queue.requeue_with_backoff(message)
            logger.info(
                f"Feedback {message['id']} requeued (attempt {message['attempts'] + 1})"
            )
