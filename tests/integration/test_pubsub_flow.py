"""Integration test: sophia publishes events that hermes can subscribe to.

Requires: Redis running on localhost:6379
"""

from __future__ import annotations

import json
import threading
import time

import pytest
import redis

from logos_config import RedisConfig
from logos_events import EventBus

REDIS_AVAILABLE = False
try:
    r = redis.from_url("redis://localhost:6379/0")
    r.ping()
    REDIS_AVAILABLE = True
except Exception:
    pass

pytestmark = pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")


class TestPubSubFlow:
    """Test the full pub/sub event flow."""

    def test_proposal_processed_event_received(self):
        """Published proposal_processed event is received by subscriber."""
        config = RedisConfig()
        received: list[dict] = []

        # Subscriber
        sub_bus = EventBus(config)
        sub_bus.subscribe(
            "logos:sophia:proposal_processed", lambda e: received.append(e)
        )
        listener = threading.Thread(target=sub_bus.listen, daemon=True)
        listener.start()
        time.sleep(0.2)

        # Publisher
        pub_bus = EventBus(config)
        pub_bus.publish(
            "logos:sophia:proposal_processed",
            {
                "event_type": "proposal_processed",
                "source": "sophia",
                "payload": {
                    "new_types": ["vehicle"],
                    "updated_types": ["person"],
                    "stored_node_ids": ["n1"],
                    "stored_edge_ids": ["e1"],
                    "affected_node_uuids": ["n1"],
                },
            },
        )
        pub_bus.close()

        time.sleep(0.3)
        sub_bus.stop()

        assert len(received) == 1
        assert received[0]["event_type"] == "proposal_processed"
        assert received[0]["payload"]["new_types"] == ["vehicle"]

    def test_type_snapshot_written_and_readable(self):
        """Type snapshot written to Redis is readable."""
        config = RedisConfig()
        r = redis.from_url(config.url)

        snapshot = {
            "person": {"uuid": "t1", "member_count": 10},
            "location": {"uuid": "t2", "member_count": 5},
        }
        r.set("logos:ontology:types", json.dumps(snapshot))

        raw = r.get("logos:ontology:types")
        loaded = json.loads(raw)
        assert loaded == snapshot

        # Cleanup
        r.delete("logos:ontology:types")
        r.close()
