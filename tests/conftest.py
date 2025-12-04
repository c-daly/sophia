"""Pytest configuration and fixtures."""

import os
from collections.abc import Iterator
from pathlib import Path
import tempfile

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (no external services)")
    config.addinivalue_line(
        "markers", "integration: Integration tests (requires Neo4j/Milvus)"
    )
    config.addinivalue_line("markers", "e2e: End-to-end tests (full stack)")
    config.addinivalue_line("markers", "slow: Slow tests (skipped by default)")


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up consistent environment variables for all tests.

    Tests assume infrastructure (Neo4j, Milvus) is already running and seeded.
    Use scripts/run_integration.sh for automated setup, or manually:
      docker compose -f docker-compose.test.yml up -d
      python scripts/seed_test_data.py

    Environment variables can be set by:
      - scripts/run_integration.sh (sets NEO4J_URI, etc.)
      - docker-compose.test.sophia.yml
      - Manual export before running pytest
    """
    # Set defaults only if not already set (allows override by scripts)
    # Note: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD should be set by CI or test runner
    # Do not set defaults here - let the neo4j_uri fixtures handle defaults
    os.environ.setdefault("SOPHIA_API_TOKEN", "test-token-12345")

    yield


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
