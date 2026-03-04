"""Configuration for feedback emission system."""

from pydantic import Field
from pydantic_settings import BaseSettings

from logos_config import RedisConfig


class FeedbackConfig(BaseSettings):
    """Configuration for feedback emission."""

    enabled: bool = Field(
        default=True,
        description="Enable/disable feedback emission",
    )
    redis: RedisConfig = Field(
        default_factory=RedisConfig,
        description="Redis configuration",
    )
    hermes_url: str = Field(
        default="http://localhost:18000",
        description="Hermes base URL",
    )
    worker_timeout: float = Field(
        default=10.0,
        description="HTTP timeout for Hermes requests",
    )

    model_config = {"env_prefix": "SOPHIA_FEEDBACK_"}
