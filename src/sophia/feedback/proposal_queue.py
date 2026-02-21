"""Redis-backed queue for async proposal processing."""

import json
import logging
from datetime import datetime, timezone

import redis

logger = logging.getLogger(__name__)


class ProposalQueue:
    """Redis-backed queue for Hermes proposals."""

    QUEUE_KEY = "sophia:proposals:pending"
    CONTEXT_PREFIX = "sophia:context:"

    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)

    def enqueue(self, proposal: dict, conversation_id: str | None = None) -> str:
        message_id = f"pq-{datetime.now(timezone.utc).timestamp()}"
        message = {
            "id": message_id,
            "payload": proposal,
            "conversation_id": conversation_id,
            "attempts": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.redis.lpush(self.QUEUE_KEY, json.dumps(message))
        return message_id

    def dequeue(self, timeout: int = 5) -> dict | None:
        result = self.redis.brpop(self.QUEUE_KEY, timeout=timeout)
        if result:
            return json.loads(result[1])
        return None

    def store_context(
        self, conversation_id: str, context: list[dict], ttl: int = 3600
    ) -> None:
        key = f"{self.CONTEXT_PREFIX}{conversation_id}"
        self.redis.setex(key, ttl, json.dumps(context))

    def get_context(self, conversation_id: str) -> list[dict]:
        key = f"{self.CONTEXT_PREFIX}{conversation_id}"
        data = self.redis.get(key)
        if data:
            return json.loads(data)
        return []

    def pending_count(self) -> int:
        return self.redis.llen(self.QUEUE_KEY)
