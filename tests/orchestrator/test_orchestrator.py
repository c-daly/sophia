"""Tests for the Orchestrator module."""

from sophia.orchestrator import Orchestrator


def test_orchestrator_creation() -> None:
    """Test creating an orchestrator instance."""
    orchestrator = Orchestrator()
    assert orchestrator is not None
    assert not orchestrator.is_running()


def test_orchestrator_start_stop() -> None:
    """Test starting and stopping the orchestrator."""
    orchestrator = Orchestrator()

    # Initially not running
    assert not orchestrator.is_running()

    # Start the orchestrator
    orchestrator.start()
    assert orchestrator.is_running()

    # Stop the orchestrator
    orchestrator.stop()
    assert not orchestrator.is_running()


def test_orchestrator_multiple_starts() -> None:
    """Test that multiple starts don't cause issues."""
    orchestrator = Orchestrator()

    orchestrator.start()
    assert orchestrator.is_running()

    orchestrator.start()
    assert orchestrator.is_running()
