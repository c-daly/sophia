"""Configuration settings for Sophia."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Settings(BaseModel):
    """Global configuration settings for Sophia.

    Attributes:
        db_url: Database connection URL
        data_dir: Directory for storing data files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """

    model_config = ConfigDict(frozen=False)

    db_url: str = Field(
        default="sqlite:///sophia.db", description="Database connection URL"
    )
    data_dir: Path = Field(
        default=Path("./data"), description="Directory for storing data files"
    )
    log_level: str = Field(default="INFO", description="Logging level")

    def ensure_data_dir(self) -> None:
        """Ensure data directory exists."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
