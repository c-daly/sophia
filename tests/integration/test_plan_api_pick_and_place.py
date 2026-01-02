"""Functional test for Sophia plan API with pick-and-place scenario.

This test validates the planning and state management capabilities
for a pick-and-place task as specified in issue requirements.
"""

import pytest
from typing import Dict, Any, List

from sophia.planner import Planner
from tests.data import load_pick_and_place_scenario, get_initial_state


pytestmark = pytest.mark.integration


class TestPlanAPIPickAndPlace:
    """Functional tests for Sophia plan API against pick-and-place scenario."""

    @pytest.fixture
    def planner_with_scenario(self) -> Planner:
        """Create a planner with the pick-and-place scenario loaded."""
        kg = load_pick_and_place_scenario()
        planner = Planner(knowledge_graph=kg)
        # Initialize with starting state
        planner.update_state(get_initial_state())
        return planner

    def test_plan_api_returns_move_grasp_move_release_sequence(
        self, planner_with_scenario: Planner
    ) -> None:
        """Test that plan API returns correct MOVE→GRASP→MOVE→RELEASE sequence.

        Validates:
        - Plan is generated for 'red block in bin' goal
        - Plan follows MOVE→GRASP→MOVE→RELEASE pattern
        - All steps reference HCG nodes
        """
        # Define goal
        goal = {"description": "red block in bin", "target_state": "red_block_in_bin"}

        # Call plan API
        plan = planner_with_scenario.plan(goal)

        # Assert plan is not empty
        assert len(plan) > 0, "Plan should contain action steps"

        # Assert plan has exactly 4 steps (MOVE→GRASP→MOVE→RELEASE)
        assert len(plan) == 4, f"Expected 4 steps, got {len(plan)}"

        # Verify action sequence
        expected_sequence = ["MOVE", "GRASP", "MOVE", "RELEASE"]
        actual_sequence = [step["action_type"] for step in plan]

        assert (
            actual_sequence == expected_sequence
        ), f"Expected {expected_sequence}, got {actual_sequence}"

        # Verify each step references HCG nodes
        for step in plan:
            assert "id" in step, "Each step should have an ID (HCG node reference)"
            assert "name" in step, "Each step should have a name"
            assert "action_type" in step, "Each step should have an action type"
            assert "target" in step, "Each step should have a target"

        # Verify specific actions
        assert plan[0]["id"] == "move_to_red_block"
        assert plan[1]["id"] == "grasp_red_block"
        assert plan[2]["id"] == "move_to_bin"
        assert plan[3]["id"] == "release_red_block"

    def test_state_api_reflects_initial_state(
        self, planner_with_scenario: Planner
    ) -> None:
        """Test that state API returns the correct initial state."""
        state = planner_with_scenario.get_state()

        # Verify initial state
        assert "red_block" in state
        assert state["red_block"]["location"] == "table"
        assert state["red_block"]["grasped"] is False

        assert "blue_block" in state
        assert state["blue_block"]["location"] == "table"
        assert state["blue_block"]["grasped"] is False

        assert "gripper" in state
        assert state["gripper"]["position"] == "home"
        assert state["gripper"]["holding"] is None

    def test_apply_plan_updates_state(self, planner_with_scenario: Planner) -> None:
        """Test that applying plan (or simulation) updates state correctly.

        Simulates Talos shim by manually applying action effects.
        """
        # Get initial state
        initial_state = planner_with_scenario.get_state()
        assert initial_state["red_block"]["location"] == "table"
        assert initial_state["gripper"]["holding"] is None

        # Generate plan
        goal = {"description": "red block in bin", "target_state": "red_block_in_bin"}
        plan = planner_with_scenario.plan(goal)

        # Simulate applying each step (Talos shim simulation)
        state_updates = self._simulate_plan_execution(plan, initial_state)

        # Apply state updates
        planner_with_scenario.update_state(state_updates)

        # Verify state changes
        final_state = planner_with_scenario.get_state()
        assert final_state["red_block"]["location"] == "bin"
        assert final_state["red_block"]["grasped"] is False
        assert final_state["gripper"]["holding"] is None
        assert final_state["gripper"]["position"] == "bin"

    def test_invalid_state_update_validation(
        self, planner_with_scenario: Planner
    ) -> None:
        """Test that invalid state updates fail validation (SHACL-like).

        This simulates SHACL validation by checking constraints.
        """
        # Try to create invalid state (block in two places)
        with pytest.raises(ValueError, match="validation"):
            invalid_update = {
                "red_block": {"location": "table", "grasped": True},
                "gripper": {"holding": "red_block"},
                # Constraint: if gripper holds red_block, red_block can't be on table
            }
            self._validate_state_update(invalid_update)

    def test_plan_references_real_hcg_nodes(
        self, planner_with_scenario: Planner
    ) -> None:
        """Test that plan steps reference actual nodes in the HCG."""
        goal = {"description": "red block in bin", "target_state": "red_block_in_bin"}
        plan = planner_with_scenario.plan(goal)

        # Get all node IDs from the knowledge graph
        kg = planner_with_scenario._kg
        all_node_ids = set(kg._nodes.keys())

        # Verify each plan step references a real HCG node
        for step in plan:
            assert (
                step["id"] in all_node_ids
            ), f"Plan step {step['id']} should reference a real HCG node"

    def test_plan_goal_not_achievable_returns_empty(self) -> None:
        """Test that planning for unachievable goal returns empty plan."""
        # Create minimal scenario without proper action chain
        planner = Planner()
        goal = {"description": "impossible goal", "target_state": "impossible_state"}

        plan = planner.plan(goal)

        # Should return empty plan when goal is not achievable
        assert len(plan) == 0, "Unachievable goal should return empty plan"

    def _simulate_plan_execution(
        self, plan: List[Dict[str, Any]], initial_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate execution of a plan and return state updates.

        This simulates what Talos shim would do.

        Args:
            plan: List of action steps
            initial_state: Current state

        Returns:
            Dictionary of state updates
        """
        state = initial_state.copy()

        for step in plan:
            action_type = step["action_type"]
            target = step["target"]

            if action_type == "MOVE":
                # Update gripper position
                if "red_block" in target:
                    state["gripper"]["position"] = "red_block"
                elif "bin" in target:
                    state["gripper"]["position"] = "bin"

            elif action_type == "GRASP":
                # Grasp object
                if "red_block" in target:
                    state["red_block"]["grasped"] = True
                    state["gripper"]["holding"] = "red_block"

            elif action_type == "RELEASE":
                # Release object at current position
                if "red_block" in target:
                    state["red_block"]["grasped"] = False
                    state["red_block"]["location"] = state["gripper"]["position"]
                    state["gripper"]["holding"] = None

        return state

    def _validate_state_update(self, state_update: Dict[str, Any]) -> None:
        """Validate state update against constraints (SHACL-like).

        Args:
            state_update: Proposed state update

        Raises:
            ValueError: If state update violates constraints
        """
        # Basic SHACL-like validation
        # Constraint 1: If gripper holds an object, that object must be grasped
        if (
            "gripper" in state_update
            and state_update["gripper"].get("holding") is not None
        ):
            held_object = state_update["gripper"]["holding"]
            if held_object in state_update:
                if not state_update[held_object].get("grasped", False):
                    raise ValueError(
                        f"State validation failed: gripper holds {held_object} but object not grasped"
                    )

        # Constraint 2: If object is grasped, it can't have a fixed location
        for obj_id, obj_state in state_update.items():
            if obj_id in ["red_block", "blue_block"]:
                if obj_state.get("grasped") and obj_state.get("location") in [
                    "table",
                    "bin",
                ]:
                    raise ValueError(
                        f"State validation failed: {obj_id} is grasped but has fixed location"
                    )
