"""Tests for Settings configuration."""

import pytest
from pathlib import Path

from sophia.config.settings import Settings


pytestmark = pytest.mark.unit


def test_settings_defaults() -> None:
    """Test default settings values."""
    settings = Settings()

    assert settings.db_url == "sqlite:///sophia.db"
    assert settings.data_dir == Path("./data")
    assert settings.log_level == "INFO"


def test_settings_custom_values() -> None:
    """Test creating settings with custom values."""
    settings = Settings(
        db_url="postgresql://localhost/sophia",
        data_dir=Path("/custom/data"),
        log_level="DEBUG",
    )

    assert settings.db_url == "postgresql://localhost/sophia"
    assert settings.data_dir == Path("/custom/data")
    assert settings.log_level == "DEBUG"


def test_ensure_data_dir(temp_dir: Path) -> None:
    """Test ensuring data directory exists."""
    data_path = temp_dir / "data"
    settings = Settings(data_dir=data_path)

    assert not data_path.exists()
    settings.ensure_data_dir()
    assert data_path.exists()
    assert data_path.is_dir()
