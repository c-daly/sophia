"""Pytest configuration and fixtures."""

from collections.abc import Iterator
from pathlib import Path
import tempfile

import pytest


@pytest.fixture
def temp_db_path() -> Iterator[str]:
    """Create a temporary database path for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        yield tmp_file.name


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
