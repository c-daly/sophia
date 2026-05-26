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

    # --- Ontology evolution / emergence (#505) tunables ---
    variance_threshold: float = Field(
        default=0.6,
        gt=0,
        description="Mean squared distance from centroid above which a type is a "
        "junk-drawer candidate (cheap pre-filter for emergence).",
    )
    min_cluster_size: int = Field(
        default=3, ge=2, description="Smallest cluster that may be minted into a type."
    )
    max_cluster_size: int = Field(
        default=50,
        ge=2,
        description="Largest membership sent verbatim to Hermes name_cluster; "
        "larger clusters are sampled.",
    )
    min_cohesion_improvement: float = Field(
        default=0.15,
        gt=0,
        le=1.0,
        description="Minimum fractional variance reduction a split must achieve "
        "to be accepted.",
    )
    hermes_confidence_floor: float = Field(
        default=0.5,
        ge=0,
        le=1.0,
        description="Discard Hermes cluster names below this confidence.",
    )
