"""Execution component for carrying out planned actions."""

from typing import List, Dict, Any


class Executor:
    """Executor carries out planned actions and manages execution state.

    The Executor is responsible for executing planned actions and
    managing the execution lifecycle.
    """

    def __init__(self) -> None:
        """Initialize the executor."""
        self._execution_queue: List[Dict[str, Any]] = []
        self._is_executing: bool = False

    def queue_action(self, action: Dict[str, Any]) -> None:
        """Queue an action for execution.

        Args:
            action: A dictionary representing the action to execute
        """
        self._execution_queue.append(action)

    def get_queue(self) -> List[Dict[str, Any]]:
        """Get the current execution queue.

        Returns:
            Copy of the execution queue
        """
        return self._execution_queue.copy()

    def clear_queue(self) -> None:
        """Clear all actions from the execution queue."""
        self._execution_queue.clear()

    def is_executing(self) -> bool:
        """Check if the executor is currently executing.

        Returns:
            True if executing, False otherwise
        """
        return self._is_executing

    def has_queued_actions(self) -> bool:
        """Check if there are queued actions.

        Returns:
            True if there are queued actions, False otherwise
        """
        return len(self._execution_queue) > 0
