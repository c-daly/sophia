"""Test data loading utilities."""

from typing import Dict, Any
from sophia.knowledge_graph import KnowledgeGraph, Node, Edge


def load_pick_and_place_scenario() -> KnowledgeGraph:
    """Load the pick-and-place test scenario into a knowledge graph.

    This replicates the structure from test_data_pick_and_place.cypher
    in a Python-based knowledge graph.

    Returns:
        KnowledgeGraph populated with pick-and-place scenario
    """
    kg = KnowledgeGraph()

    # Create spatial entities (locations)
    table = Node(
        id="table",
        type="location",
        properties={"name": "Table", "location_type": "surface"},
    )
    bin_node = Node(
        id="bin", type="location", properties={"name": "Bin", "location_type": "container"}
    )

    kg.add_node(table)
    kg.add_node(bin_node)

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

    # Initial state: blocks on table
    kg.add_edge(
        Edge(source="red_block", target="table", relation="located_at", properties={})
    )
    kg.add_edge(
        Edge(source="blue_block", target="table", relation="located_at", properties={})
    )

    # Create action primitives
    move1 = Node(
        id="move_to_red_block",
        type="action",
        properties={
            "name": "Move to Red Block",
            "action_type": "MOVE",
            "target": "red_block",
        },
    )
    grasp1 = Node(
        id="grasp_red_block",
        type="action",
        properties={
            "name": "Grasp Red Block",
            "action_type": "GRASP",
            "target": "red_block",
        },
    )
    move2 = Node(
        id="move_to_bin",
        type="action",
        properties={"name": "Move to Bin", "action_type": "MOVE", "target": "bin"},
    )
    release1 = Node(
        id="release_red_block",
        type="action",
        properties={
            "name": "Release Red Block",
            "action_type": "RELEASE",
            "target": "red_block",
        },
    )

    kg.add_node(move1)
    kg.add_node(grasp1)
    kg.add_node(move2)
    kg.add_node(release1)

    # Action preconditions and effects (causal chain)
    kg.add_edge(Edge(source="move_to_red_block", target="grasp_red_block", relation="enables"))
    kg.add_edge(Edge(source="grasp_red_block", target="move_to_bin", relation="enables"))
    kg.add_edge(Edge(source="move_to_bin", target="release_red_block", relation="enables"))
    kg.add_edge(
        Edge(
            source="release_red_block",
            target="bin",
            relation="achieves",
            properties={"state": "red_block_in_bin"},
        )
    )

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
        Edge(source="goal_red_block_in_bin", target="release_red_block", relation="requires")
    )

    return kg


def get_initial_state() -> Dict[str, Any]:
    """Get the initial state for the pick-and-place scenario.

    Returns:
        Dictionary representing initial state
    """
    return {
        "red_block": {"location": "table", "grasped": False},
        "blue_block": {"location": "table", "grasped": False},
        "gripper": {"position": "home", "holding": None},
    }
