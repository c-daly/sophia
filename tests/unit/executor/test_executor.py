"""Tests for the Executor module."""

import pytest
from sophia.executor import Executor


pytestmark = pytest.mark.unit


def test_executor_creation() -> None:
    """Test creating an executor instance."""
    executor = Executor()
    assert executor is not None
    assert not executor.is_executing()
    assert not executor.has_queued_actions()


def test_executor_queue_action() -> None:
    """Test queuing actions for execution."""
    executor = Executor()

    action1 = {"name": "action1", "type": "test"}
    executor.queue_action(action1)

    assert executor.has_queued_actions()
    queue = executor.get_queue()
    assert len(queue) == 1
    assert queue[0] == action1


def test_executor_get_queue() -> None:
    """Test getting the execution queue."""
    executor = Executor()

    action1 = {"name": "action1"}
    action2 = {"name": "action2"}

    executor.queue_action(action1)
    executor.queue_action(action2)

    queue = executor.get_queue()
    assert len(queue) == 2
    assert queue[0] == action1
    assert queue[1] == action2

    # Ensure it's a copy
    queue.append({"name": "action3"})
    assert len(executor.get_queue()) == 2


def test_executor_clear_queue() -> None:
    """Test clearing the execution queue."""
    executor = Executor()

    executor.queue_action({"name": "action1"})
    executor.queue_action({"name": "action2"})
    assert executor.has_queued_actions()

    executor.clear_queue()
    assert not executor.has_queued_actions()
    assert len(executor.get_queue()) == 0


def test_executor_has_queued_actions() -> None:
    """Test checking if executor has queued actions."""
    executor = Executor()

    assert not executor.has_queued_actions()

    executor.queue_action({"name": "action1"})
    assert executor.has_queued_actions()

    executor.clear_queue()
    assert not executor.has_queued_actions()


def test_executor_is_executing() -> None:
    """Test checking if executor is executing."""
    executor = Executor()
    assert not executor.is_executing()
