"""Tests for ProposalWorker resilience to Redis outages.

Regression: ``dequeue()`` does a blocking ``brpop``; when Redis drops the
connection it raises. Previously that call sat OUTSIDE any try/except, so the
exception propagated through ``start()``'s loop and killed the worker task
permanently -- the background cache pipeline silently stopped until a sophia
restart, which left every chat turn on the slow synchronous path. The worker
must instead log, back off, and keep looping so it resumes when Redis recovers.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from sophia.feedback.proposal_worker import ProposalWorker


class TestProposalWorkerResilience:
    async def test_process_one_survives_dequeue_error(self, monkeypatch):
        """A Redis error from dequeue must not propagate out of _process_one."""
        queue = MagicMock()
        queue.dequeue.side_effect = ConnectionError("redis down")
        worker = ProposalWorker(queue=queue, processor=MagicMock())

        slept: list[float] = []

        async def fake_sleep(delay):
            slept.append(delay)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        # Must not raise -- previously this killed the worker task.
        await worker._process_one()

        assert slept, "worker should back off after a dequeue error"

    async def test_worker_resumes_after_dequeue_error(self, monkeypatch):
        """After a transient dequeue error, the next message is processed."""
        queue = MagicMock()
        queue.dequeue.side_effect = [
            ConnectionError("redis down"),
            {"id": "m1", "payload": {"x": 1}, "conversation_id": "c1"},
        ]
        processor = MagicMock()
        processor.process.return_value = {
            "relevant_context": [{"node_uuid": "u"}],
            "stored_node_ids": [],
            "stored_edge_ids": [],
        }
        worker = ProposalWorker(queue=queue, processor=processor)
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

        await worker._process_one()  # errors, backs off, returns
        await worker._process_one()  # recovers, processes the message

        processor.process.assert_called_once_with({"x": 1})
        queue.store_context.assert_called_once()
