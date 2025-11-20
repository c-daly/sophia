"""Utilities for seeding HCG with test data."""

import logging

from sophia.hcg_client.client import HCGClient


logger = logging.getLogger(__name__)


def seed_pick_and_place_data(hcg_client: HCGClient) -> None:
    """Seed Neo4j with pick-and-place scenario data.

    This replicates the structure from test_data_pick_and_place.cypher.

    Args:
        hcg_client: HCG client instance
    """
    logger.info("Seeding pick-and-place data into Neo4j...")

    # Create spatial entities (locations)
    hcg_client.add_node(
        node_id="table",
        node_type="location",
        properties={"name": "Table", "location_type": "surface"},
    )
    hcg_client.add_node(
        node_id="bin",
        node_type="location",
        properties={"name": "Bin", "location_type": "container"},
    )

    # Create objects
    hcg_client.add_node(
        node_id="red_block",
        node_type="object",
        properties={"name": "Red Block", "color": "red", "object_type": "block"},
    )
    hcg_client.add_node(
        node_id="blue_block",
        node_type="object",
        properties={"name": "Blue Block", "color": "blue", "object_type": "block"},
    )

    # Initial state: blocks on table
    hcg_client.add_edge(
        edge_id="e_red_block_on_table",
        source_id="red_block",
        target_id="table",
        relation="located_at",
    )
    hcg_client.add_edge(
        edge_id="e_blue_block_on_table",
        source_id="blue_block",
        target_id="table",
        relation="located_at",
    )

    # Create action primitives
    hcg_client.add_node(
        node_id="move_to_red_block",
        node_type="action",
        properties={
            "name": "Move to Red Block",
            "action_type": "MOVE",
            "target": "red_block",
        },
    )
    hcg_client.add_node(
        node_id="grasp_red_block",
        node_type="action",
        properties={
            "name": "Grasp Red Block",
            "action_type": "GRASP",
            "target": "red_block",
        },
    )
    hcg_client.add_node(
        node_id="move_to_bin",
        node_type="action",
        properties={"name": "Move to Bin", "action_type": "MOVE", "target": "bin"},
    )
    hcg_client.add_node(
        node_id="release_red_block",
        node_type="action",
        properties={
            "name": "Release Red Block",
            "action_type": "RELEASE",
            "target": "red_block",
        },
    )

    # Action preconditions and effects (causal chain)
    hcg_client.add_edge(
        edge_id="e_move_enables_grasp",
        source_id="move_to_red_block",
        target_id="grasp_red_block",
        relation="enables",
    )
    hcg_client.add_edge(
        edge_id="e_grasp_enables_move",
        source_id="grasp_red_block",
        target_id="move_to_bin",
        relation="enables",
    )
    hcg_client.add_edge(
        edge_id="e_move_enables_release",
        source_id="move_to_bin",
        target_id="release_red_block",
        relation="enables",
    )
    hcg_client.add_edge(
        edge_id="e_release_achieves_bin",
        source_id="release_red_block",
        target_id="bin",
        relation="achieves",
        properties={"state": "red_block_in_bin"},
    )

    # Goal definition
    hcg_client.add_node(
        node_id="goal_red_block_in_bin",
        node_type="goal",
        properties={
            "description": "red block in bin",
            "target_state": "red_block_in_bin",
        },
    )
    hcg_client.add_edge(
        edge_id="e_goal_requires_release",
        source_id="goal_red_block_in_bin",
        target_id="release_red_block",
        relation="requires",
    )

    # Create initial state node
    hcg_client.add_node(
        node_id="current_state",
        node_type="state",
        properties={
            "red_block": {"location": "table", "grasped": False},
            "blue_block": {"location": "table", "grasped": False},
            "gripper": {"position": "home", "holding": None},
        },
    )

    logger.info("Pick-and-place data seeded successfully")
