"""Tests for main sophia module."""

import pytest
import sophia


pytestmark = pytest.mark.unit


def test_version() -> None:
    """Test that version is defined."""
    assert hasattr(sophia, "__version__")
    assert isinstance(sophia.__version__, str)


def test_exports() -> None:
    """Test that main classes are exported."""
    assert hasattr(sophia, "KnowledgeGraph")
    assert hasattr(sophia, "Database")
    assert hasattr(sophia, "Orchestrator")
    assert hasattr(sophia, "ContinuousWorkingMemoryAssociative")
    assert hasattr(sophia, "ContinuousWorkingMemoryGenerative")
    assert hasattr(sophia, "Planner")
    assert hasattr(sophia, "Executor")
    assert hasattr(sophia, "HCGClient")
