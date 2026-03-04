"""Configuration for the maintenance scheduler."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MaintenanceConfig(BaseSettings):
    """Configuration for async KG maintenance scheduling."""

    model_config = SettingsConfigDict(env_prefix="SOPHIA_MAINTENANCE_")

    enabled: bool = Field(
        default=True, description="Master switch for maintenance scheduler"
    )
    post_ingestion_enabled: bool = Field(
        default=True, description="Queue checks after proposal processing"
    )
    periodic_enabled: bool = Field(default=True, description="Run periodic graph scans")
    periodic_interval_seconds: int = Field(
        default=3600, ge=1, description="Interval between periodic scans"
    )
    event_driven_enabled: bool = Field(
        default=True, description="React to specific EventBus channels"
    )
    threshold_enabled: bool = Field(
        default=True, description="Trigger jobs on metric thresholds"
    )
    type_member_count_threshold: int = Field(
        default=100, description="Member count to trigger ontology evolution"
    )
    max_concurrent_jobs: int = Field(
        default=2, ge=1, description="Max simultaneous maintenance jobs"
    )
