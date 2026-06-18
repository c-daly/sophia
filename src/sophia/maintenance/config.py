"""Configuration for the maintenance scheduler."""

from __future__ import annotations

from typing import Literal

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
    hermes_confidence_floor: float = Field(
        default=0.5,
        ge=0,
        le=1.0,
        description="Discard Hermes cluster names below this confidence.",
    )
    rollup_enabled: bool = Field(
        # The rollup is now fully edge-based (#209):
        #   (a) realm roots are resolved by uuid positionally via _root_uuid_by_name
        #       (landed in commit 5e8825b; no `type_<name>` slugs remain), and
        #   (b) membership is derived from the IS_A edge target in _type_uuid_map,
        #       not from the `type_uuid` / `type` property (which doesn't exist on
        #       a reseeded graph).
        # Both blockers are gone; the rollup can run on positional/reseeded graphs.
        default=True,
        description="Run the periodic type-level rollup that groups the flat "
        "layer of emergent types into super-types (#160). Enabled after "
        "edge-based realm-root resolution and IS_A-edge read-path (#209).",
    )
    rollup_interval_seconds: int = Field(
        default=600,
        description="How often the type_rollup pass trolls the type layer.",
    )
    rollup_min_cluster_size: int = Field(
        default=2, ge=2, description="Min types in a leaf group during rollup."
    )
    rollup_min_supercluster_size: int = Field(
        default=2, ge=2, description="Min child groups needed to mint a super-type."
    )
    rollup_tier2_selection: Literal["silhouette", "threshold"] = Field(
        default="silhouette",
        description="Tier-2 super-cluster selector. 'silhouette' = legacy "
        "global-partition (silhouette-argmax over agglomeration); collapses to "
        "one diffuse blob on type-centroids and finds nothing. 'threshold' = "
        "cosine-threshold connected components (neighborhood frame, sophia #220).",
    )
    rollup_sim_threshold: float = Field(
        default=0.70,
        ge=0,
        le=1.0,
        description="Cosine-similarity threshold for the 'threshold' tier-2 "
        "selector: type-centroids at/above it group into one super-type.",
    )
    type_match_threshold: float = Field(
        default=0.9,
        ge=0,
        le=1.0,
        description="Cosine similarity above which an emergent cluster is "
        "reconciled into an existing type (its members are retyped to that type) "
        "instead of minting a new duplicate type (#504 match-before-mint).",
    )
