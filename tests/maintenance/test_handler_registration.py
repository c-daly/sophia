"""Task 11: build_emergence_handler factory produces the scheduler callable (#505)."""

from __future__ import annotations

import inspect

from sophia.maintenance.config import MaintenanceConfig
from sophia.maintenance.emergence_handler import build_emergence_handler


def test_build_emergence_handler_callable_with_type_uuid():
    handler = build_emergence_handler(
        config=MaintenanceConfig(),
        hcg=object(),
        milvus=object(),
        event_bus=None,
        hermes_url="http://h",
        token="t",
    )
    assert callable(handler)
    # The scheduler dispatches handlers['type_emergence'](type_uuid=...).
    assert "type_uuid" in inspect.signature(handler).parameters
