"""Unit tests for ProposalQueue and ProposalWorker."""

import json
from unittest.mock import MagicMock, patch

import pytest

from sophia.feedback.proposal_queue import ProposalQueue
from sophia.feedback.proposal_worker import ProposalWorker


class TestProposalQueue:
    """Tests for ProposalQueue."""

    @pytest.fixture
    def mock_redis(self) -> MagicMock:
        """Create a mock Redis client."""
        return MagicMock()

    @pytest.fixture
    def queue(self, mock_redis: MagicMock) -> ProposalQueue:
        """Create a ProposalQueue with mocked Redis."""
        with patch("sophia.feedback.proposal_queue.redis") as mock_redis_mod:
            mock_redis_mod.from_url.return_value = mock_redis
            q = ProposalQueue("redis://localhost:6379/0")
        return q

    def test_enqueue_dequeue(self, queue: ProposalQueue, mock_redis: MagicMock) -> None:
        """Test enqueue calls lpush, dequeue via brpop returns parsed message."""
        proposal = {"proposal_id": "p-1", "proposed_nodes": [{"name": "Alice"}]}

        # Enqueue
        message_id = queue.enqueue(proposal, conversation_id="conv-1")
        assert message_id.startswith("pq-")
        mock_redis.lpush.assert_called_once()

        # Verify the JSON that was pushed
        call_args = mock_redis.lpush.call_args
        assert call_args[0][0] == ProposalQueue.QUEUE_KEY
        pushed_json = call_args[0][1]
        pushed = json.loads(pushed_json)
        assert pushed["id"] == message_id
        assert pushed["payload"] == proposal
        assert pushed["conversation_id"] == "conv-1"
        assert pushed["attempts"] == 0
        assert "created_at" in pushed

        # Dequeue -- simulate brpop returning (key, json_bytes)
        mock_redis.brpop.return_value = (
            ProposalQueue.QUEUE_KEY,
            pushed_json.encode(),
        )
        result = queue.dequeue(timeout=5)
        assert result is not None
        assert result["id"] == message_id
        assert result["payload"] == proposal
        mock_redis.brpop.assert_called_once_with(ProposalQueue.QUEUE_KEY, timeout=5)

    def test_dequeue_returns_none_on_timeout(
        self, queue: ProposalQueue, mock_redis: MagicMock
    ) -> None:
        """Test dequeue returns None when brpop times out."""
        mock_redis.brpop.return_value = None
        result = queue.dequeue(timeout=1)
        assert result is None

    def test_store_and_get_context(
        self, queue: ProposalQueue, mock_redis: MagicMock
    ) -> None:
        """Test store_context calls setex, get_context returns parsed JSON."""
        conversation_id = "conv-42"
        context = [
            {"node_uuid": "n-1", "name": "Alice", "type": "Entity", "score": 0.1},
            {"node_uuid": "n-2", "name": "Bob", "type": "Entity", "score": 0.2},
        ]

        # Store
        queue.store_context(conversation_id, context, ttl=1800)
        expected_key = f"{ProposalQueue.CONTEXT_PREFIX}{conversation_id}"
        mock_redis.setex.assert_called_once_with(
            expected_key, 1800, json.dumps(context)
        )

        # Get
        mock_redis.get.return_value = json.dumps(context).encode()
        result = queue.get_context(conversation_id)
        assert result == context
        mock_redis.get.assert_called_once_with(expected_key)

    def test_get_context_returns_empty_list_when_missing(
        self, queue: ProposalQueue, mock_redis: MagicMock
    ) -> None:
        """Test get_context returns [] when key does not exist."""
        mock_redis.get.return_value = None
        result = queue.get_context("nonexistent")
        assert result == []

    def test_pending_count(self, queue: ProposalQueue, mock_redis: MagicMock) -> None:
        """Test pending_count delegates to llen."""
        mock_redis.llen.return_value = 7
        assert queue.pending_count() == 7
        mock_redis.llen.assert_called_once_with(ProposalQueue.QUEUE_KEY)

    def test_enqueue_without_conversation_id(
        self, queue: ProposalQueue, mock_redis: MagicMock
    ) -> None:
        """Test enqueue works without a conversation_id."""
        message_id = queue.enqueue({"proposal_id": "p-2"})
        assert message_id.startswith("pq-")
        pushed = json.loads(mock_redis.lpush.call_args[0][1])
        assert pushed["conversation_id"] is None


class TestProposalWorker:
    """Tests for ProposalWorker."""

    @pytest.fixture
    def mock_queue(self) -> MagicMock:
        """Create a mock ProposalQueue."""
        return MagicMock(spec=ProposalQueue)

    @pytest.fixture
    def mock_processor(self) -> MagicMock:
        """Create a mock ProposalProcessor."""
        processor = MagicMock()
        processor.process.return_value = {
            "stored_node_ids": ["n-1"],
            "stored_edge_ids": [],
            "relevant_context": [
                {"node_uuid": "n-1", "name": "Alice", "type": "Entity", "score": 0.1},
            ],
        }
        return processor

    @pytest.fixture
    def worker(
        self, mock_queue: MagicMock, mock_processor: MagicMock
    ) -> ProposalWorker:
        """Create a ProposalWorker with mocked dependencies."""
        return ProposalWorker(mock_queue, mock_processor, context_ttl=600)

    @pytest.mark.asyncio
    async def test_process_one_with_message(
        self,
        worker: ProposalWorker,
        mock_queue: MagicMock,
        mock_processor: MagicMock,
    ) -> None:
        """Test _process_one dequeues, processes, and stores context."""
        message = {
            "id": "prop-abc123",
            "payload": {"proposal_id": "p-1"},
            "conversation_id": "conv-1",
            "attempts": 0,
        }
        mock_queue.dequeue.return_value = message

        await worker._process_one()

        mock_queue.dequeue.assert_called_once_with(timeout=1)
        mock_processor.process.assert_called_once_with({"proposal_id": "p-1"})
        # asyncio.to_thread may pass ttl as a keyword arg depending on
        # Python internals and mock spec binding, so check both forms.
        mock_queue.store_context.assert_called_once()
        call_args = mock_queue.store_context.call_args
        assert call_args[0][0] == "conv-1"
        assert call_args[0][1] == [
            {"node_uuid": "n-1", "name": "Alice", "type": "Entity", "score": 0.1}
        ]
        # ttl may appear as positional arg[2] or keyword "ttl"
        if len(call_args[0]) > 2:
            assert call_args[0][2] == 600
        else:
            assert call_args[1].get("ttl") == 600

    @pytest.mark.asyncio
    async def test_process_one_no_message(
        self,
        worker: ProposalWorker,
        mock_queue: MagicMock,
        mock_processor: MagicMock,
    ) -> None:
        """Test _process_one returns early when no message available."""
        mock_queue.dequeue.return_value = None

        await worker._process_one()

        mock_processor.process.assert_not_called()
        mock_queue.store_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_one_no_conversation_id(
        self,
        worker: ProposalWorker,
        mock_queue: MagicMock,
        mock_processor: MagicMock,
    ) -> None:
        """Test _process_one skips context storage when no conversation_id."""
        message = {
            "id": "prop-xyz",
            "payload": {"proposal_id": "p-2"},
            "conversation_id": None,
            "attempts": 0,
        }
        mock_queue.dequeue.return_value = message

        await worker._process_one()

        mock_processor.process.assert_called_once()
        mock_queue.store_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_stop(
        self,
        worker: ProposalWorker,
        mock_queue: MagicMock,
    ) -> None:
        """Test start/stop lifecycle."""
        call_count = 0

        def dequeue_side_effect(timeout=1):
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                worker.stop()
            return None

        mock_queue.dequeue.side_effect = dequeue_side_effect

        await worker.start()

        assert not worker._running
        assert call_count >= 3

    @pytest.mark.asyncio
    async def test_process_one_handles_processor_error(
        self,
        worker: ProposalWorker,
        mock_queue: MagicMock,
        mock_processor: MagicMock,
    ) -> None:
        """Test _process_one handles processor exceptions gracefully."""
        message = {
            "id": "prop-err",
            "payload": {"proposal_id": "p-fail"},
            "conversation_id": "conv-1",
            "attempts": 0,
        }
        mock_queue.dequeue.return_value = message
        mock_processor.process.side_effect = RuntimeError("Graph unavailable")

        # Should not raise
        await worker._process_one()

        mock_queue.store_context.assert_not_called()
