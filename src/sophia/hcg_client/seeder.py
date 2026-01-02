"""Utilities for seeding HCG with test data."""

import logging
from typing import List

from sophia.hcg_client.client import HCGClient


logger = logging.getLogger(__name__)

# Ancestor chains for all node types used in sophia
# Based on logos core ontology hierarchy
ANCESTORS = {
    # Core pick-and-place types
    "location": ["spatial_entity", "entity"],
    "object": ["physical_entity", "entity"],
    "action": ["process", "entity"],
    "goal": ["intention", "entity"],
    "state": ["cognition"],
    # API and planning types
    "plan": ["process", "entity"],
    "imagined_state": ["cognition"],
    "imagined_process": ["process", "entity"],
    "simulation": ["abstraction", "entity"],
    # Execution container types (parallel to simulation for observed)
    "execution": ["abstraction", "entity"],
    "process": ["entity"],
    # Hermes ingestion types
    "hermes_proposal": ["intention", "entity"],
    "proposed_plan_step": ["process", "entity"],
    "proposed_imagined_state": ["cognition"],
    "proposed_tool_call": ["process", "entity"],
    # Media types
    "media_sample": ["data", "entity"],
}

# Types required by sophia API endpoints (must exist for auto-compute ancestors)
REQUIRED_TYPES = [
    "simulation",
    "imagined_process",
    "imagined_state",
    "execution",
    "process",
]


def seed_type_definitions(hcg_client: HCGClient) -> None:
    """Seed Neo4j with type definitions from ANCESTORS.

    Creates type definition nodes for all types in ANCESTORS dict.
    These are required for the HCG client's auto-compute ancestors feature.

    Args:
        hcg_client: HCG client instance
    """
    logger.info("Seeding type definitions into Neo4j...")

    for type_name, ancestors in ANCESTORS.items():
        # Determine the parent type (first ancestor)
        parent_type = ancestors[0] if ancestors else "entity"

        hcg_client.add_node(
            uuid=f"type_{type_name}",
            name=type_name,
            node_type=parent_type,
            ancestors=ancestors,
            is_type_definition=True,
            source="bootstrap",
            derivation="observed",
        )

    logger.info(f"Seeded {len(ANCESTORS)} type definitions")


def verify_required_types(hcg_client: HCGClient) -> List[str]:
    """Verify that all required type definitions exist in Neo4j.

    Args:
        hcg_client: HCG client instance

    Returns:
        List of missing type names (empty if all exist)
    """
    missing = []
    for type_name in REQUIRED_TYPES:
        ancestors = hcg_client._get_type_ancestors(type_name)
        if not ancestors:
            missing.append(type_name)
    return missing


def seed_pick_and_place_data(hcg_client: HCGClient) -> None:
    """Seed Neo4j with pick-and-place scenario data.

    This replicates the structure from test_data_pick_and_place.cypher.

    Args:
        hcg_client: HCG client instance
    """
    logger.info("Seeding pick-and-place data into Neo4j...")

    # Create spatial entities (locations)
    hcg_client.add_node(
        uuid="table",
        name="Table",
        node_type="location",
        ancestors=ANCESTORS["location"],
        is_type_definition=False,
        properties={"location_type": "surface"},
        source="bootstrap",
        derivation="observed",
    )
    hcg_client.add_node(
        uuid="bin",
        name="Bin",
        node_type="location",
        ancestors=ANCESTORS["location"],
        is_type_definition=False,
        properties={"location_type": "container"},
        source="bootstrap",
        derivation="observed",
    )

    # Create objects
    hcg_client.add_node(
        uuid="red_block",
        name="Red Block",
        node_type="object",
        ancestors=ANCESTORS["object"],
        is_type_definition=False,
        properties={"color": "red", "object_type": "block"},
        source="bootstrap",
        derivation="observed",
    )
    hcg_client.add_node(
        uuid="blue_block",
        name="Blue Block",
        node_type="object",
        ancestors=ANCESTORS["object"],
        is_type_definition=False,
        properties={"color": "blue", "object_type": "block"},
        source="bootstrap",
        derivation="observed",
    )

    # Initial state: blocks on table
    hcg_client.add_edge(
        edge_id="e_red_block_on_table",
        source_uuid="red_block",
        target_uuid="table",
        relation="located_at",
    )
    hcg_client.add_edge(
        edge_id="e_blue_block_on_table",
        source_uuid="blue_block",
        target_uuid="table",
        relation="located_at",
    )

    # Create action primitives
    hcg_client.add_node(
        uuid="move_to_red_block",
        name="Move to Red Block",
        node_type="action",
        ancestors=ANCESTORS["action"],
        is_type_definition=False,
        properties={"action_type": "MOVE", "target": "red_block"},
        source="bootstrap",
        derivation="observed",
    )
    hcg_client.add_node(
        uuid="grasp_red_block",
        name="Grasp Red Block",
        node_type="action",
        ancestors=ANCESTORS["action"],
        is_type_definition=False,
        properties={"action_type": "GRASP", "target": "red_block"},
        source="bootstrap",
        derivation="observed",
    )
    hcg_client.add_node(
        uuid="move_to_bin",
        name="Move to Bin",
        node_type="action",
        ancestors=ANCESTORS["action"],
        is_type_definition=False,
        properties={"action_type": "MOVE", "target": "bin"},
        source="bootstrap",
        derivation="observed",
    )
    hcg_client.add_node(
        uuid="release_red_block",
        name="Release Red Block",
        node_type="action",
        ancestors=ANCESTORS["action"],
        is_type_definition=False,
        properties={"action_type": "RELEASE", "target": "red_block"},
        source="bootstrap",
        derivation="observed",
    )

    # Action preconditions and effects (causal chain)
    hcg_client.add_edge(
        edge_id="e_move_enables_grasp",
        source_uuid="move_to_red_block",
        target_uuid="grasp_red_block",
        relation="enables",
    )
    hcg_client.add_edge(
        edge_id="e_grasp_enables_move",
        source_uuid="grasp_red_block",
        target_uuid="move_to_bin",
        relation="enables",
    )
    hcg_client.add_edge(
        edge_id="e_move_enables_release",
        source_uuid="move_to_bin",
        target_uuid="release_red_block",
        relation="enables",
    )
    hcg_client.add_edge(
        edge_id="e_release_achieves_bin",
        source_uuid="release_red_block",
        target_uuid="bin",
        relation="achieves",
        properties={"state": "red_block_in_bin"},
    )

    # Goal definition
    hcg_client.add_node(
        uuid="goal_red_block_in_bin",
        name="Goal: Red Block in Bin",
        node_type="goal",
        ancestors=ANCESTORS["goal"],
        is_type_definition=False,
        properties={
            "description": "red block in bin",
            "target_state": "red_block_in_bin",
        },
        source="bootstrap",
        derivation="observed",
    )
    hcg_client.add_edge(
        edge_id="e_goal_requires_release",
        source_uuid="goal_red_block_in_bin",
        target_uuid="release_red_block",
        relation="requires",
    )

    # Create initial state node
    hcg_client.add_node(
        uuid="current_state",
        name="Current State",
        node_type="state",
        ancestors=ANCESTORS["state"],
        is_type_definition=False,
        properties={
            "red_block": {"location": "table", "grasped": False},
            "blue_block": {"location": "table", "grasped": False},
            "gripper": {"position": "home", "holding": None},
        },
        source="bootstrap",
        derivation="observed",
    )

    logger.info("Pick-and-place data seeded successfully")
