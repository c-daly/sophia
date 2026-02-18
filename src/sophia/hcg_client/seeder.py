"""Utilities for seeding HCG with test data.

Type hierarchy is expressed through IS_A edge nodes in the reified-edge
model, not stored as node properties.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List
from sophia.hcg_client.client import HCGClient


logger = logging.getLogger(__name__)

# Type hierarchy expressed as parent -> children mapping.
# Each entry maps a type name to its parent type.
# These relationships will be created as IS_A edge nodes.
TYPE_HIERARCHY: Dict[str, str] = {
    # Core pick-and-place types
    "location": "spatial_entity",
    "spatial_entity": "entity",
    "object": "physical_entity",
    "physical_entity": "entity",
    "action": "process",
    "goal": "intention",
    "intention": "entity",
    "state": "cognition",
    # API and planning types
    "plan": "process",
    "imagined_state": "cognition",
    "imagined_process": "process",
    "simulation": "abstraction",
    "abstraction": "entity",
    # Execution container types (parallel to simulation for observed)
    "execution": "abstraction",
    "process": "entity",
    # Hermes ingestion types
    "hermes_proposal": "intention",
    "proposed_plan_step": "process",
    "proposed_imagined_state": "cognition",
    "proposed_tool_call": "process",
    # Media types
    "media_sample": "data",
    "data": "entity",
    # CWM types
    "cwm_a": "cwm",
    "cwm_g": "cwm",
    "cwm_e": "cwm",
    "cwm": "cognition",
    "cognition": "entity",
}

# Types required by sophia API endpoints
REQUIRED_TYPES = [
    "simulation",
    "imagined_process",
    "imagined_state",
    "execution",
    "process",
]


def seed_type_definitions(hcg_client: HCGClient) -> None:
    """Seed Neo4j with type definition nodes and IS_A edge hierarchy.

    Creates a type node for each type and connects them via IS_A edges
    to express the type hierarchy.

    Args:
        hcg_client: HCG client instance
    """
    logger.info("Seeding type definitions into Neo4j...")

    # First pass: create all type nodes
    created_types = set()
    all_types = set(TYPE_HIERARCHY.keys()) | set(TYPE_HIERARCHY.values())

    for type_name in all_types:
        if type_name not in created_types:
            hcg_client.add_node(
                uuid=f"type_{type_name}",
                name=type_name,
                node_type="type",
                source="bootstrap",
                derivation="observed",
            )
            created_types.add(type_name)

    # Second pass: create IS_A edges for the hierarchy
    for child_type, parent_type in TYPE_HIERARCHY.items():
        hcg_client.add_edge(
            edge_uuid=f"edge_is_a_{child_type}_{parent_type}",
            source_uuid=f"type_{child_type}",
            target_uuid=f"type_{parent_type}",
            relation="IS_A",
        )

    logger.info(
        f"Seeded {len(created_types)} type nodes with {len(TYPE_HIERARCHY)} IS_A edges"
    )


def verify_required_types(hcg_client: HCGClient) -> List[str]:
    """Verify that all required type definition nodes exist in Neo4j.

    Args:
        hcg_client: HCG client instance

    Returns:
        List of missing type names (empty if all exist)
    """
    missing = []
    for type_name in REQUIRED_TYPES:
        node = hcg_client.get_node(f"type_{type_name}")
        if not node:
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
        properties={"location_type": "surface"},
        source="bootstrap",
        derivation="observed",
    )
    hcg_client.add_node(
        uuid="bin",
        name="Bin",
        node_type="location",
        properties={"location_type": "container"},
        source="bootstrap",
        derivation="observed",
    )

    # Create objects
    hcg_client.add_node(
        uuid="red_block",
        name="Red Block",
        node_type="object",
        properties={"color": "red", "object_type": "block"},
        source="bootstrap",
        derivation="observed",
    )
    hcg_client.add_node(
        uuid="blue_block",
        name="Blue Block",
        node_type="object",
        properties={"color": "blue", "object_type": "block"},
        source="bootstrap",
        derivation="observed",
    )

    # Initial state: blocks on table
    hcg_client.add_edge(
        edge_uuid="e_red_block_on_table",
        source_uuid="red_block",
        target_uuid="table",
        relation="LOCATED_AT",
    )
    hcg_client.add_edge(
        edge_uuid="e_blue_block_on_table",
        source_uuid="blue_block",
        target_uuid="table",
        relation="LOCATED_AT",
    )

    # Create action primitives
    hcg_client.add_node(
        uuid="move_to_red_block",
        name="Move to Red Block",
        node_type="action",
        properties={"action_type": "MOVE", "target": "red_block"},
        source="bootstrap",
        derivation="observed",
    )
    hcg_client.add_node(
        uuid="grasp_red_block",
        name="Grasp Red Block",
        node_type="action",
        properties={"action_type": "GRASP", "target": "red_block"},
        source="bootstrap",
        derivation="observed",
    )
    hcg_client.add_node(
        uuid="move_to_bin",
        name="Move to Bin",
        node_type="action",
        properties={"action_type": "MOVE", "target": "bin"},
        source="bootstrap",
        derivation="observed",
    )
    hcg_client.add_node(
        uuid="release_red_block",
        name="Release Red Block",
        node_type="action",
        properties={"action_type": "RELEASE", "target": "red_block"},
        source="bootstrap",
        derivation="observed",
    )

    # Action preconditions and effects (causal chain)
    hcg_client.add_edge(
        edge_uuid="e_move_enables_grasp",
        source_uuid="move_to_red_block",
        target_uuid="grasp_red_block",
        relation="ENABLES",
    )
    hcg_client.add_edge(
        edge_uuid="e_grasp_enables_move",
        source_uuid="grasp_red_block",
        target_uuid="move_to_bin",
        relation="ENABLES",
    )
    hcg_client.add_edge(
        edge_uuid="e_move_enables_release",
        source_uuid="move_to_bin",
        target_uuid="release_red_block",
        relation="ENABLES",
    )
    hcg_client.add_edge(
        edge_uuid="e_release_achieves_bin",
        source_uuid="release_red_block",
        target_uuid="bin",
        relation="ACHIEVES",
        properties={"state": "red_block_in_bin"},
    )

    # Goal definition
    hcg_client.add_node(
        uuid="goal_red_block_in_bin",
        name="Goal: Red Block in Bin",
        node_type="goal",
        properties={
            "description": "red block in bin",
            "target_state": "red_block_in_bin",
        },
        source="bootstrap",
        derivation="observed",
    )
    hcg_client.add_edge(
        edge_uuid="e_goal_requires_release",
        source_uuid="goal_red_block_in_bin",
        target_uuid="release_red_block",
        relation="REQUIRES",
    )

    # Create initial state node
    hcg_client.add_node(
        uuid="current_state",
        name="Current State",
        node_type="state",
        properties={
            "red_block": {"location": "table", "grasped": False},
            "blue_block": {"location": "table", "grasped": False},
            "gripper": {"position": "home", "holding": None},
        },
        source="bootstrap",
        derivation="observed",
    )

    logger.info("Pick-and-place data seeded successfully")


def seed_plan_data(hcg_client: HCGClient) -> None:
    """Seed plan nodes into Neo4j.

    Creates sample plans that reference the pick-and-place scenario.

    Args:
        hcg_client: HCG client instance
    """
    logger.info("Seeding plan data into Neo4j...")

    # Plan 1: completed plan for red block
    hcg_client.add_node(
        uuid="plan_red_block_to_bin",
        name="Plan: Red Block to Bin",
        node_type="plan",
        properties={
            "goal_id": "goal_red_block_in_bin",
            "status": "completed",
            "steps": [
                {
                    "id": "move_to_red_block",
                    "name": "Move to Red Block",
                    "action_type": "MOVE",
                },
                {
                    "id": "grasp_red_block",
                    "name": "Grasp Red Block",
                    "action_type": "GRASP",
                },
                {"id": "move_to_bin", "name": "Move to Bin", "action_type": "MOVE"},
                {
                    "id": "release_red_block",
                    "name": "Release Red Block",
                    "action_type": "RELEASE",
                },
            ],
        },
        source="planner",
        derivation="observed",
    )
    hcg_client.add_edge(
        edge_uuid="e_plan_red_for_goal",
        source_uuid="plan_red_block_to_bin",
        target_uuid="goal_red_block_in_bin",
        relation="ACHIEVES",
    )

    # Plan 2: pending plan for blue block
    hcg_client.add_node(
        uuid="plan_blue_block_to_bin",
        name="Plan: Blue Block to Bin",
        node_type="plan",
        properties={
            "status": "pending",
            "steps": [
                {
                    "id": "move_to_blue_block",
                    "name": "Move to Blue Block",
                    "action_type": "MOVE",
                },
                {
                    "id": "grasp_blue_block",
                    "name": "Grasp Blue Block",
                    "action_type": "GRASP",
                },
                {"id": "move_to_bin", "name": "Move to Bin", "action_type": "MOVE"},
                {
                    "id": "release_blue_block",
                    "name": "Release Blue Block",
                    "action_type": "RELEASE",
                },
            ],
        },
        source="planner",
        derivation="imagined",
    )

    logger.info("Plan data seeded successfully")


def seed_persona_entries(hcg_client: HCGClient) -> None:
    """Seed persona diary entries as CWM-E states via CWMPersistence.

    Args:
        hcg_client: HCG client instance (driver used for CWMPersistence)
    """
    from sophia.cwm.persistence import CWMPersistence
    from sophia.cwm_a.state_service import CWMState

    logger.info("Seeding persona diary entries into Neo4j...")

    persistence = CWMPersistence(
        neo4j_driver=hcg_client.driver,
        database=hcg_client.database,
    )

    now = datetime.now(timezone.utc)

    entries = [
        {
            "entry_id": "persona_seed_001",
            "entry_type": "observation",
            "content": "I notice the workspace has two blocks on the table. The red block "
            "and blue block are both resting on the surface. This is a familiar "
            "starting configuration for pick-and-place tasks.",
            "summary": "Observed blocks on table",
            "trigger": "workspace_scan",
            "sentiment": "neutral",
            "confidence": 0.9,
            "emotion_tags": ["curiosity", "readiness"],
            "related_process_ids": [],
            "related_goal_ids": [],
            "metadata": {},
        },
        {
            "entry_id": "persona_seed_002",
            "entry_type": "reflection",
            "content": "Successfully completed the red block sorting task. The plan "
            "executed all four steps without errors. I am becoming more confident "
            "in my ability to handle pick-and-place operations.",
            "summary": "Reflected on successful red block task",
            "trigger": "plan_completed",
            "sentiment": "positive",
            "confidence": 0.85,
            "emotion_tags": ["satisfaction", "confidence"],
            "related_process_ids": ["plan_red_block_to_bin"],
            "related_goal_ids": ["goal_red_block_in_bin"],
            "metadata": {"plan_duration_ms": 1250},
        },
        {
            "entry_id": "persona_seed_003",
            "entry_type": "decision",
            "content": "I have decided to prioritize the blue block next. The bin has "
            "space and the blue block is in a reachable position. Moving it "
            "will clear the table for future operations.",
            "summary": "Decided to sort blue block next",
            "trigger": "goal_evaluation",
            "sentiment": "positive",
            "confidence": 0.75,
            "emotion_tags": ["determination", "planning"],
            "related_process_ids": ["plan_blue_block_to_bin"],
            "related_goal_ids": [],
            "metadata": {},
        },
        {
            "entry_id": "persona_seed_004",
            "entry_type": "belief",
            "content": "I believe the current workspace layout is efficient for "
            "sequential sorting. Objects are well-separated and the bin is "
            "positioned within reach. No obstacles detected.",
            "summary": "Belief about workspace efficiency",
            "trigger": "environment_assessment",
            "sentiment": "positive",
            "confidence": 0.8,
            "emotion_tags": ["confidence", "awareness"],
            "related_process_ids": [],
            "related_goal_ids": [],
            "metadata": {"workspace_score": 0.92},
        },
    ]

    for i, entry_data in enumerate(entries):
        # Stagger timestamps so ordering works
        entry_timestamp = now - timedelta(minutes=len(entries) - i)

        state = CWMState(
            state_id=f"cwm_e_{entry_data['entry_id']}",
            model_type="CWM_E",
            timestamp=entry_timestamp,
            data={
                "entry": entry_data,
                "source": "persona_api",
                "derivation": "observed",
                "confidence": entry_data["confidence"],
                "tags": [f"entry_type:{entry_data['entry_type']}"],
                "links": {},
            },
        )

        persistence.persist(state)

    logger.info(f"Seeded {len(entries)} persona diary entries")
