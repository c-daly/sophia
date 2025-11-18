"""Planning component for cognitive task planning."""

from typing import List, Dict, Any, Optional

from sophia.knowledge_graph.graph import KnowledgeGraph


class Planner:
    """Planner manages cognitive task planning and goal decomposition.

    The Planner is responsible for breaking down high-level goals into
    actionable steps and managing the planning process.
    """

    def __init__(self, knowledge_graph: Optional[KnowledgeGraph] = None) -> None:
        """Initialize the planner.

        Args:
            knowledge_graph: Optional knowledge graph for planning
        """
        self._goals: List[Dict[str, Any]] = []
        self._kg = knowledge_graph or KnowledgeGraph()
        self._current_state: Dict[str, Any] = {}

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

    def plan(self, goal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create a plan to achieve a goal using backward chaining.

        Args:
            goal: Goal specification with description and target_state

        Returns:
            Ordered list of action steps to achieve the goal
        """
        plan_steps: List[Dict[str, Any]] = []

        # Find goal node in knowledge graph
        target_state = goal.get("target_state", "")

        # Backward chain from goal to find required actions
        # Find the action that achieves the goal state
        for node_id in self._kg._nodes.keys():
            node = self._kg.get_node(node_id)
            if node and node.type == "action":
                # Check if this action achieves the goal
                edges = self._kg.get_edges_from(node_id)
                for edge in edges:
                    if (
                        edge.relation == "achieves"
                        and edge.properties.get("state") == target_state
                    ):
                        # Found the final action, now trace back prerequisites
                        plan_steps = self._trace_prerequisites(node_id)
                        break
                if plan_steps:
                    break

        return plan_steps

    def _trace_prerequisites(self, action_id: str) -> List[Dict[str, Any]]:
        """Trace prerequisites for an action using backward chaining.

        Args:
            action_id: ID of the action to trace

        Returns:
            Ordered list of actions leading to the target action
        """
        plan: List[Dict[str, Any]] = []
        visited = set()

        def trace(current_id: str) -> None:
            if current_id in visited:
                return
            visited.add(current_id)

            # Find actions that enable this action
            prerequisites = []
            for node_id in self._kg._nodes.keys():
                edges = self._kg.get_edges_from(node_id)
                for edge in edges:
                    if edge.target == current_id and edge.relation == "enables":
                        prerequisites.append(node_id)

            # Recursively trace prerequisites
            for prereq_id in prerequisites:
                trace(prereq_id)

            # Add current action to plan
            node = self._kg.get_node(current_id)
            if node:
                plan.append(
                    {
                        "id": current_id,
                        "name": node.properties.get("name", current_id),
                        "type": node.type,
                        "action_type": node.properties.get("action_type", node.type),
                        "target": node.properties.get("target", ""),
                    }
                )

        trace(action_id)
        return plan

    def get_state(self) -> Dict[str, Any]:
        """Get the current state tracked by the planner.

        Returns:
            Current state dictionary
        """
        return self._current_state.copy()

    def update_state(self, updates: Dict[str, Any]) -> None:
        """Update the current state.

        Args:
            updates: Dictionary of state updates to apply
        """
        self._current_state.update(updates)
