"""Configuration for feedback emission system."""

from pydantic import Field
from pydantic_settings import BaseSettings


class FeedbackConfig(BaseSettings):
    """Configuration for feedback emission."""

    enabled: bool = Field(
        default=True,
        description="Enable/disable feedback emission",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
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
