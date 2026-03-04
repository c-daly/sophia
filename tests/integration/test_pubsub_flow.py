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
_probe = None
try:
    _probe_config = RedisConfig()
    _probe = redis.from_url(_probe_config.url)
    _probe.ping()
    REDIS_AVAILABLE = True
except Exception:
    pass
finally:
    if _probe is not None:
        try:
            _probe.close()
        except Exception:
            pass

pytestmark = pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")


class TestPubSubFlow:
    """Test the full pub/sub event flow."""

    def test_proposal_processed_event_received(self):
        """Published proposal_processed event is received by subscriber."""
        config = RedisConfig()
        received: list[dict] = []
        received_event = threading.Event()

        def on_event(e: dict) -> None:
            received.append(e)
            received_event.set()

        # Subscriber
        sub_bus = EventBus(config)
        sub_bus.subscribe("logos:sophia:proposal_processed", on_event)
        listener = threading.Thread(target=sub_bus.listen, daemon=True)
        listener.start()
        time.sleep(0.5)  # allow subscription to register

        try:
            # Publisher
            pub_bus = EventBus(config)
            try:
                pub_bus.publish(
                    "logos:sophia:proposal_processed",
                    {
                        "event_type": "proposal_processed",
                        "source": "sophia",
                        "payload": {
                            "new_type_uuids": ["type_vehicle"],
                            "updated_type_uuids": ["type_person"],
                            "stored_node_ids": ["n1"],
                            "stored_edge_ids": ["e1"],
                            "affected_node_uuids": ["n1"],
                        },
                    },
                )
            finally:
                pub_bus.close()

            assert received_event.wait(timeout=2.0), "Subscriber did not receive event"

            assert len(received) == 1
            assert received[0]["event_type"] == "proposal_processed"
            assert received[0]["payload"]["new_type_uuids"] == ["type_vehicle"]
        finally:
            sub_bus.stop()
            sub_bus.close()

    def test_write_type_snapshot_writes_correct_format(self):
        """_write_type_snapshot writes correct JSON to Redis key."""
        from unittest.mock import MagicMock

        from sophia.ingestion.proposal_processor import ProposalProcessor

        config = RedisConfig()
        r = redis.from_url(config.url)

        # Clean up before test
        r.delete("logos:ontology:types")

        try:
            mock_hcg = MagicMock()
            mock_hcg.get_all_type_definitions.return_value = [
                {
                    "name": "person",
                    "uuid": "type_person",
                    "properties": {"member_count": 10},
                },
                {
                    "name": "location",
                    "uuid": "type_location",
                    "properties": {"member_count": 5},
                },
            ]

            processor = ProposalProcessor(
                hcg_client=mock_hcg,
                milvus_sync=MagicMock(),
                event_bus=None,
                redis_client=r,
            )
            processor._write_type_snapshot()

            raw = r.get("logos:ontology:types")
            assert raw is not None
            snapshot = json.loads(raw)
            assert "person" in snapshot
            assert snapshot["person"]["uuid"] == "type_person"
            assert snapshot["person"]["member_count"] == 10
            assert "location" in snapshot
        finally:
            r.delete("logos:ontology:types")
            r.close()
