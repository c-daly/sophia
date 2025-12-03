"""Tests for the Planner module."""

import pytest
from sophia.planner import Planner


pytestmark = pytest.mark.unit


def test_planner_creation() -> None:
    """Test creating a planner instance."""
    planner = Planner()
    assert planner is not None
    assert not planner.has_goals()


def test_planner_add_goal() -> None:
    """Test adding goals to the planner."""
    planner = Planner()

    goal1 = {"name": "goal1", "description": "Test goal"}
    planner.add_goal(goal1)

    assert planner.has_goals()
    goals = planner.get_goals()
    assert len(goals) == 1
    assert goals[0] == goal1


def test_planner_get_goals() -> None:
    """Test getting goals from the planner."""
    planner = Planner()

    goal1 = {"name": "goal1"}
    goal2 = {"name": "goal2"}

    planner.add_goal(goal1)
    planner.add_goal(goal2)

    goals = planner.get_goals()
    assert len(goals) == 2
    assert goals[0] == goal1
    assert goals[1] == goal2

    # Ensure it's a copy
    goals.append({"name": "goal3"})
    assert len(planner.get_goals()) == 2


def test_planner_clear_goals() -> None:
    """Test clearing goals from the planner."""
    planner = Planner()

    planner.add_goal({"name": "goal1"})
    planner.add_goal({"name": "goal2"})
    assert planner.has_goals()

    planner.clear_goals()
    assert not planner.has_goals()
    assert len(planner.get_goals()) == 0


def test_planner_has_goals() -> None:
    """Test checking if planner has goals."""
    planner = Planner()

    assert not planner.has_goals()

    planner.add_goal({"name": "goal1"})
    assert planner.has_goals()

    planner.clear_goals()
    assert not planner.has_goals()
