"""Tests for the CWM-A (Continuous Working Memory - Associative) module."""

import pytest
from sophia.cwm_a import ContinuousWorkingMemoryAssociative


pytestmark = pytest.mark.unit


def test_cwm_a_creation() -> None:
    """Test creating a CWM-A instance."""
    cwm_a = ContinuousWorkingMemoryAssociative()
    assert cwm_a is not None
    assert cwm_a.size() == 0


def test_cwm_a_store_retrieve() -> None:
    """Test storing and retrieving items from CWM-A."""
    cwm_a = ContinuousWorkingMemoryAssociative()

    # Store an item
    cwm_a.store("key1", "value1")
    assert cwm_a.size() == 1

    # Retrieve the item
    result = cwm_a.retrieve("key1")
    assert result == "value1"


def test_cwm_a_retrieve_nonexistent() -> None:
    """Test retrieving a non-existent key returns None."""
    cwm_a = ContinuousWorkingMemoryAssociative()
    result = cwm_a.retrieve("nonexistent")
    assert result is None


def test_cwm_a_clear() -> None:
    """Test clearing the memory."""
    cwm_a = ContinuousWorkingMemoryAssociative()

    # Store some items
    cwm_a.store("key1", "value1")
    cwm_a.store("key2", "value2")
    assert cwm_a.size() == 2

    # Clear the memory
    cwm_a.clear()
    assert cwm_a.size() == 0
    assert cwm_a.retrieve("key1") is None


def test_cwm_a_overwrite() -> None:
    """Test overwriting an existing key."""
    cwm_a = ContinuousWorkingMemoryAssociative()

    cwm_a.store("key1", "value1")
    assert cwm_a.retrieve("key1") == "value1"

    cwm_a.store("key1", "value2")
    assert cwm_a.retrieve("key1") == "value2"
    assert cwm_a.size() == 1
