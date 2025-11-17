"""Planning component for cognitive task planning."""

from typing import List, Dict, Any


class Planner:
    """Planner manages cognitive task planning and goal decomposition.

    The Planner is responsible for breaking down high-level goals into
    actionable steps and managing the planning process.
    """

    def __init__(self) -> None:
        """Initialize the planner."""
        self._goals: List[Dict[str, Any]] = []

    def add_goal(self, goal: Dict[str, Any]) -> None:
        """Add a goal to the planning queue.

        Args:
            goal: A dictionary representing the goal
        """
        self._goals.append(goal)

    def get_goals(self) -> List[Dict[str, Any]]:
        """Get all current goals.

        Returns:
            List of goals
        """
        return self._goals.copy()

    def clear_goals(self) -> None:
        """Clear all goals from the planner."""
        self._goals.clear()

    def has_goals(self) -> bool:
        """Check if there are any active goals.

        Returns:
            True if there are goals, False otherwise
        """
        return len(self._goals) > 0
