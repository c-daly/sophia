"""Tests for main sophia module."""

import sophia


def test_version() -> None:
    """Test that version is defined."""
    assert hasattr(sophia, "__version__")
    assert isinstance(sophia.__version__, str)


def test_exports() -> None:
    """Test that main classes are exported."""
    assert hasattr(sophia, "KnowledgeGraph")
    assert hasattr(sophia, "Database")
