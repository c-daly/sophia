"""Tests for sophia.maintenance.config.MaintenanceConfig."""

from __future__ import annotations

import os
from unittest import mock

from sophia.maintenance.config import MaintenanceConfig


class TestMaintenanceConfig:
    def test_default_values(self):
        config = MaintenanceConfig()
        assert config.enabled is True
        assert config.post_ingestion_enabled is True
        assert config.periodic_enabled is True
        assert config.periodic_interval_seconds == 3600
        assert config.event_driven_enabled is True
        assert config.threshold_enabled is True
        assert config.type_member_count_threshold == 100
        assert config.max_concurrent_jobs == 2
        # Rollup (revision) tier is deferred for naming-driven-typing B1 and
        # gated OFF by default until it is converted onto placement.py.
        assert config.rollup_enabled is False

    def test_env_override(self):
        with mock.patch.dict(
            os.environ,
            {
                "SOPHIA_MAINTENANCE_ENABLED": "false",
                "SOPHIA_MAINTENANCE_PERIODIC_INTERVAL_SECONDS": "600",
                "SOPHIA_MAINTENANCE_MAX_CONCURRENT_JOBS": "4",
            },
        ):
            config = MaintenanceConfig()
            assert config.enabled is False
            assert config.periodic_interval_seconds == 600
            assert config.max_concurrent_jobs == 4

    def test_individual_triggers_toggleable(self):
        with mock.patch.dict(
            os.environ,
            {
                "SOPHIA_MAINTENANCE_POST_INGESTION_ENABLED": "false",
                "SOPHIA_MAINTENANCE_PERIODIC_ENABLED": "false",
                "SOPHIA_MAINTENANCE_EVENT_DRIVEN_ENABLED": "false",
                "SOPHIA_MAINTENANCE_THRESHOLD_ENABLED": "false",
            },
        ):
            config = MaintenanceConfig()
            assert config.post_ingestion_enabled is False
            assert config.periodic_enabled is False
            assert config.event_driven_enabled is False
            assert config.threshold_enabled is False
