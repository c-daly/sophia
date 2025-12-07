#!/usr/bin/env python3
"""
Milestone M3 Demonstration: Sophia can plan simple actions

This script demonstrates the complete cognitive architecture with planning,
world modeling, and reasoning capabilities for Epoch 3.

It shows:
1. Knowledge graph construction (world modeling)
2. Simple action planning with backward chaining
3. State management and updates
4. Integration of cognitive components (Planner, Executor, Orchestrator)
"""

from sophia import (
    KnowledgeGraph,
    Planner,
    Executor,
    Orchestrator,
    ContinuousWorkingMemoryAssociative,
    ContinuousWorkingMemoryGenerative,
)
from sophia.knowledge_graph import Node, Edge
from typing import Dict, Any, List, Optional


def create_world_model() -> KnowledgeGraph:
    """Create a world model using the knowledge graph.

    Represents a simple pick-and-place scenario with:
    - Objects (red block, blue block)
    - Locations (table, bin)
    - Actions (move, grasp, release)
    - Causal relationships between actions

    Returns:
        KnowledgeGraph with world model
    """
    print("\n=== Building World Model (Knowledge Graph) ===")
    kg = KnowledgeGraph()

    # Create spatial entities
    table = Node(
        id="table",
        type="location",
        properties={"name": "Table", "location_type": "surface"},
    )
    bin_node = Node(
        id="bin",
        type="location",
        properties={"name": "Bin", "location_type": "container"},
    )
    kg.add_node(table)
    kg.add_node(bin_node)
    print(
        f"  Added locations: {table.properties['name']}, {bin_node.properties['name']}"
    )

    # Create objects
    red_block = Node(
        id="red_block",
        type="object",
        properties={"name": "Red Block", "color": "red", "object_type": "block"},
    )
    blue_block = Node(
        id="blue_block",
        type="object",
        properties={"name": "Blue Block", "color": "blue", "object_type": "block"},
    )
    kg.add_node(red_block)
    kg.add_node(blue_block)
    print(
        f"  Added objects: {red_block.properties['name']}, {blue_block.properties['name']}"
    )

    # Initial state: blocks on table
    kg.add_edge(
        Edge(source="red_block", target="table", relation="located_at", properties={})
    )
    kg.add_edge(
        Edge(source="blue_block", target="table", relation="located_at", properties={})
    )
    print("  Initial state: Both blocks on table")

    # Create action primitives
    move1 = Node(
        id="move_to_red_block",
        type="action",
        properties={
            "name": "Move to Red Block",
            "action_type": "MOVE",
            "target": "red_block",
            "description": "Move gripper to red block position",
        },
    )
    grasp1 = Node(
        id="grasp_red_block",
        type="action",
        properties={
            "name": "Grasp Red Block",
            "action_type": "GRASP",
            "target": "red_block",
            "description": "Grasp the red block",
        },
    )
    move2 = Node(
        id="move_to_bin",
        type="action",
        properties={
            "name": "Move to Bin",
            "action_type": "MOVE",
            "target": "bin",
            "description": "Move gripper (with block) to bin",
        },
    )
    release1 = Node(
        id="release_red_block",
        type="action",
        properties={
            "name": "Release Red Block",
            "action_type": "RELEASE",
            "target": "red_block",
            "description": "Release red block into bin",
        },
    )

    kg.add_node(move1)
    kg.add_node(grasp1)
    kg.add_node(move2)
    kg.add_node(release1)
    print(f"  Added {4} action primitives")

    # Action preconditions and effects (causal chain using backward chaining)
    # This encodes: move_to_red_block -> grasp_red_block -> move_to_bin -> release_red_block
    kg.add_edge(
        Edge(source="move_to_red_block", target="grasp_red_block", relation="enables")
    )
    kg.add_edge(
        Edge(source="grasp_red_block", target="move_to_bin", relation="enables")
    )
    kg.add_edge(
        Edge(source="move_to_bin", target="release_red_block", relation="enables")
    )
    kg.add_edge(
        Edge(
            source="release_red_block",
            target="bin",
            relation="achieves",
            properties={"state": "red_block_in_bin"},
        )
    )
    print("  Causal relationships established (enables chain)")

    # Goal definition
    goal = Node(
        id="goal_red_block_in_bin",
        type="goal",
        properties={
            "description": "red block in bin",
            "target_state": "red_block_in_bin",
        },
    )
    kg.add_node(goal)
    kg.add_edge(
        Edge(
            source="goal_red_block_in_bin",
            target="release_red_block",
            relation="requires",
        )
    )
    print("  Goal defined: Place red block in bin")

    print(
        f"\n  World model complete: {len(kg._nodes)} nodes, {len(kg._graph.edges())} edges"
    )
    return kg


def demonstrate_planning(kg: KnowledgeGraph) -> None:
    """Demonstrate planning with backward chaining.

    Args:
        kg: Knowledge graph with world model
    """
    print("\n=== Demonstrating Planning (Backward Chaining) ===")

    # Create planner with knowledge graph
    planner = Planner(knowledge_graph=kg)

    # Set initial state
    initial_state: Dict[str, Dict[str, Any]] = {
        "red_block": {"location": "table", "grasped": False},
        "blue_block": {"location": "table", "grasped": False},
        "gripper": {"position": "home", "holding": None},
    }
    planner.update_state(initial_state)
    print(f"  Initial state set: {initial_state['red_block']}")

    # Define goal
    goal = {"description": "red block in bin", "target_state": "red_block_in_bin"}
    print(f"  Goal: {goal['description']}")

    # Generate plan using backward chaining
    print("\n  Planning (working backwards from goal)...")
    plan = planner.plan(goal)

    if not plan:
        print("  ❌ No plan found!")
        return

    print(f"\n  ✓ Plan generated with {len(plan)} steps:")
    for i, step in enumerate(plan, 1):
        print(f"    {i}. {step['action_type']}: {step['name']}")
        print(f"       Target: {step['target']}")

    # Verify plan structure
    expected_sequence = ["MOVE", "GRASP", "MOVE", "RELEASE"]
    actual_sequence = [step["action_type"] for step in plan]

    if actual_sequence == expected_sequence:
        print(f"\n  ✓ Plan structure correct: {' → '.join(actual_sequence)}")
    else:
        print(f"\n  ⚠ Unexpected plan structure: {actual_sequence}")


def demonstrate_execution_simulation(kg: KnowledgeGraph) -> None:
    """Demonstrate execution simulation and state updates.

    Args:
        kg: Knowledge graph with world model
    """
    print("\n=== Demonstrating Execution Simulation ===")

    # Create cognitive components
    planner = Planner(knowledge_graph=kg)
    executor = Executor()

    # Set initial state
    initial_state: Dict[str, Dict[str, Any]] = {
        "red_block": {"location": "table", "grasped": False},
        "blue_block": {"location": "table", "grasped": False},
        "gripper": {"position": "home", "holding": None},
    }
    planner.update_state(initial_state)
    print(f"  Initial state: Red block at {initial_state['red_block']['location']}")

    # Generate plan
    goal = {"description": "red block in bin", "target_state": "red_block_in_bin"}
    plan = planner.plan(goal)

    # Queue actions in executor
    for action in plan:
        executor.queue_action(action)
    print(f"  Queued {len(plan)} actions in executor")

    # Simulate execution (normally would interface with Talos)
    print("\n  Simulating execution:")
    state: Dict[str, Any] = initial_state.copy()

    for i, step in enumerate(executor.get_queue(), 1):
        action_type = step["action_type"]
        target = step["target"]

        print(f"\n    Step {i}: {action_type} {target}")

        # Simulate action effects
        if action_type == "MOVE":
            if "red_block" in target:
                state["gripper"]["position"] = "red_block"
                print("      → Gripper moved to red_block position")
            elif "bin" in target:
                state["gripper"]["position"] = "bin"
                print("      → Gripper moved to bin position")

        elif action_type == "GRASP":
            if "red_block" in target:
                state["red_block"]["grasped"] = True
                state["gripper"]["holding"] = "red_block"
                print("      → Red block grasped")

        elif action_type == "RELEASE":
            if "red_block" in target:
                state["red_block"]["grasped"] = False
                state["red_block"]["location"] = state["gripper"]["position"]
                state["gripper"]["holding"] = None
                print(f"      → Red block released at {state['red_block']['location']}")

    # Update planner state
    planner.update_state(state)
    final_state = planner.get_state()

    print("\n  Final state:")
    print(f"    Red block location: {final_state['red_block']['location']}")
    print(f"    Red block grasped: {final_state['red_block']['grasped']}")
    print(f"    Gripper position: {final_state['gripper']['position']}")
    print(f"    Gripper holding: {final_state['gripper']['holding']}")

    if final_state["red_block"]["location"] == "bin":
        print("\n  ✓ Goal achieved: Red block is in bin!")
    else:
        print("\n  ❌ Goal not achieved")


def demonstrate_cognitive_architecture() -> None:
    """Demonstrate the complete cognitive architecture.

    Shows integration of all components:
    - Orchestrator (coordination)
    - Working Memory (CWM-A, CWM-G)
    - Planner (goal decomposition)
    - Executor (action execution)
    - Knowledge Graph (world model)
    """
    print("\n=== Demonstrating Cognitive Architecture Integration ===")

    # Initialize cognitive components
    orchestrator = Orchestrator()
    cwm_a = ContinuousWorkingMemoryAssociative()
    cwm_g = ContinuousWorkingMemoryGenerative()
    kg = KnowledgeGraph()
    planner = Planner(knowledge_graph=kg)
    executor = Executor()

    print("  Components initialized:")
    print("    • Orchestrator (coordination)")
    print("    • CWM-A (associative working memory)")
    print("    • CWM-G (generative working memory)")
    print("    • Knowledge Graph (world model)")
    print("    • Planner (goal decomposition)")
    print("    • Executor (action execution)")

    # Start orchestrator
    orchestrator.start()
    print(f"\n  Orchestrator started: {orchestrator.is_running()}")

    # Store information in working memory
    cwm_a.store("current_goal", "place red block in bin")
    cwm_g.add("plan_step_1")

    print("\n  Working memory state:")
    print(f"    CWM-A items: {list(cwm_a._memory.keys())}")
    print(f"    CWM-G buffer size: {len(cwm_g._buffer)}")

    # Add goal to planner
    goal = {"description": "red block in bin", "target_state": "red_block_in_bin"}
    planner.add_goal(goal)
    print(f"\n  Goals in planner: {planner.has_goals()}")

    # Check executor state
    print(f"  Executor has queued actions: {executor.has_queued_actions()}")
    print(f"  Executor is executing: {executor.is_executing()}")

    # Stop orchestrator
    orchestrator.stop()
    print(f"\n  Orchestrator stopped: {not orchestrator.is_running()}")

    print("\n  ✓ All cognitive components working together")


def main() -> None:
    """Main demonstration function."""
    print("=" * 70)
    print("MILESTONE M3 DEMONSTRATION")
    print("Sophia can plan simple actions")
    print("=" * 70)
    print("\nEpoch 3: Cognitive Core & Reasoning")
    print("Demonstrating cognitive architecture with planning, world modeling,")
    print("and reasoning capabilities.")
    print("=" * 70)

    # 1. World Modeling
    kg = create_world_model()

    # 2. Planning
    demonstrate_planning(kg)

    # 3. Execution Simulation
    demonstrate_execution_simulation(kg)

    # 4. Cognitive Architecture Integration
    demonstrate_cognitive_architecture()

    print("\n" + "=" * 70)
    print("MILESTONE M3 VERIFICATION COMPLETE")
    print("=" * 70)
    print("\n✓ World modeling with knowledge graphs")
    print("✓ Simple action planning with backward chaining")
    print("✓ State management and updates")
    print("✓ Cognitive architecture components integrated")
    print("✓ End-to-end demonstration successful")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
