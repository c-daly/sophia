"""Tests for the CWM-G (Continuous Working Memory - Generative) module."""

from sophia.cwm_g import ContinuousWorkingMemoryGenerative


def test_cwm_g_creation() -> None:
    """Test creating a CWM-G instance."""
    cwm_g = ContinuousWorkingMemoryGenerative()
    assert cwm_g is not None
    assert cwm_g.size() == 0


def test_cwm_g_add_items() -> None:
    """Test adding items to CWM-G."""
    cwm_g = ContinuousWorkingMemoryGenerative()

    # Add items
    cwm_g.add("item1")
    assert cwm_g.size() == 1

    cwm_g.add("item2")
    assert cwm_g.size() == 2


def test_cwm_g_get_buffer() -> None:
    """Test getting the buffer from CWM-G."""
    cwm_g = ContinuousWorkingMemoryGenerative()

    cwm_g.add("item1")
    cwm_g.add("item2")

    buffer = cwm_g.get_buffer()
    assert buffer == ["item1", "item2"]

    # Ensure it's a copy
    buffer.append("item3")
    assert cwm_g.size() == 2


def test_cwm_g_clear() -> None:
    """Test clearing the buffer."""
    cwm_g = ContinuousWorkingMemoryGenerative()

    cwm_g.add("item1")
    cwm_g.add("item2")
    assert cwm_g.size() == 2

    cwm_g.clear()
    assert cwm_g.size() == 0
    assert cwm_g.get_buffer() == []


def test_cwm_g_multiple_types() -> None:
    """Test storing different types of items."""
    cwm_g = ContinuousWorkingMemoryGenerative()

    cwm_g.add("string")
    cwm_g.add(42)
    cwm_g.add({"key": "value"})

    buffer = cwm_g.get_buffer()
    assert buffer == ["string", 42, {"key": "value"}]
