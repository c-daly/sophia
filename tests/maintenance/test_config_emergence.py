"""Tests for the #505 emergence tunables on MaintenanceConfig."""

from __future__ import annotations

import os
from unittest import mock

from sophia.maintenance.config import MaintenanceConfig


class TestEmergenceTunables:
    def test_defaults(self):
        cfg = MaintenanceConfig()
        assert cfg.variance_threshold > 0
        assert cfg.min_cluster_size >= 2
        assert cfg.max_cluster_size >= cfg.min_cluster_size
        assert 0.0 < cfg.min_cohesion_improvement <= 1.0
        assert 0.0 <= cfg.hermes_confidence_floor <= 1.0

    def test_env_override(self):
        with mock.patch.dict(
            os.environ, {"SOPHIA_MAINTENANCE_MIN_CLUSTER_SIZE": "5"}
        ):
            cfg = MaintenanceConfig()
            assert cfg.min_cluster_size == 5
