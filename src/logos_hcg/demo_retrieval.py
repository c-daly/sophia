"""
HCG Retrieval Demonstration Script

This script demonstrates retrieving entities from the HCG and validates M1 acceptance criteria:
- Query entities by UUID
- Query entities by name
- Traverse IS_A relationships
- Traverse HAS_STATE relationships

Usage:
    python -m logos_hcg.demo_retrieval --uri bolt://localhost:7687 --user neo4j --password logosdev

See Project LOGOS spec: Section 4.1 (Core Ontology and Data Model)
Reference: docs/PHASE1_VERIFY.md - M1 checklist
"""

import argparse
import logging
import os
import sys

from logos_hcg.client import HCGClient, HCGConnectionError

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_entity(entity):
    """Print entity details."""
    print(f"  UUID: {entity.uuid}")
    print(f"  Name: {entity.name}")
    if entity.description:
        print(f"  Description: {entity.description}")
    if hasattr(entity, "created_at") and entity.created_at:
        print(f"  Created: {entity.created_at}")


def print_concept(concept):
    """Print concept details."""
    print(f"  UUID: {concept.uuid}")
    print(f"  Name: {concept.name}")
    if concept.description:
        print(f"  Description: {concept.description}")


def print_state(state):
    """Print state details."""
    print(f"  UUID: {state.uuid}")
    if hasattr(state, "name") and state.name:
        print(f"  Name: {state.name}")
    print(f"  Timestamp: {state.timestamp}")


def demo_basic_retrieval(client: HCGClient):
    """
    Demonstrate basic entity retrieval by UUID and name.

    M1 Criteria:
    - Can query entities by UUID
    - Can query entities by name
    """
    print_section("1. Basic Entity Retrieval")

    # Retrieve by UUID
    print("\n1.1 Retrieve RobotArm by UUID:")
    entity = client.find_entity_by_uuid("entity-robot-arm-01")
    if entity:
        print("  ✓ Found entity:")
        print_entity(entity)
    else:
        print("  ✗ Entity not found (run load_hcg.py first)")
        return False

    # Retrieve by name
    print("\n1.2 Retrieve RobotArm by name:")
    entities = client.find_entities_by_name("RobotArm01")
    if entities:
        print(f"  ✓ Found {len(entities)} entity(ies):")
        for e in entities:
            print_entity(e)
    else:
        print("  ✗ No entities found")
        return False

    return True


def demo_relationship_traversal(client: HCGClient):
    """
    Demonstrate relationship traversal.

    M1 Criteria:
    - Can traverse IS_A relationships to find entity types
    - Can traverse HAS_STATE relationships to find current states
    """
    print_section("2. Relationship Traversal")

    # Get entity type via IS_A
    print("\n2.1 Get entity type (IS_A relationship):")
    entity_uuid = "entity-robot-arm-01"
    concept = client.get_entity_type(entity_uuid)
    if concept:
        print(f"  ✓ Entity '{entity_uuid}' is a:")
        print_concept(concept)
    else:
        print("  ✗ No type relationship found")
        return False

    # Get entity states via HAS_STATE
    print("\n2.2 Get entity states (HAS_STATE relationship):")
    states = client.get_entity_states(entity_uuid)
    if states:
        print(f"  ✓ Found {len(states)} state(s):")
        for i, state in enumerate(states, 1):
            print(f"\n  State {i}:")
            print_state(state)
    else:
        print("  ✗ No states found")
        return False

    # Get current state
    print("\n2.3 Get current/most recent state:")
    current_state = client.get_entity_current_state(entity_uuid)
    if current_state:
        print("  ✓ Current state:")
        print_state(current_state)
    else:
        print("  ✗ No current state found")

    return True


def demo_counts_and_statistics(client: HCGClient):
    """
    Demonstrate node counts and statistics.

    M1 Criteria:
    - Can query and aggregate node counts
    """
    print_section("3. Node Counts and Statistics")

    counts = client.count_nodes_by_type()
    print("\n  Node counts in HCG:")
    print(f"    Entities:  {counts.get('entity_count', 0)}")
    print(f"    Concepts:  {counts.get('concept_count', 0)}")
    print(f"    States:    {counts.get('state_count', 0)}")
    print(f"    Processes: {counts.get('process_count', 0)}")

    return True


def demo_all_entities(client: HCGClient):
    """
    Demonstrate listing all entities.

    M1 Criteria:
    - Can list all entities with pagination
    """
    print_section("4. List All Entities")

    entities = client.find_all_entities(limit=10)
    if entities:
        print(f"\n  ✓ Found {len(entities)} entity(ies) (showing first 10):")
        for i, entity in enumerate(entities, 1):
            print(f"\n  Entity {i}:")
            print_entity(entity)
    else:
        print("  ℹ No entities found in HCG")

    return True


def demo_all_concepts(client: HCGClient):
    """
    Demonstrate listing all concepts.

    M1 Criteria:
    - Can list all concepts
    """
    print_section("5. List All Concepts")

    concepts = client.find_all_concepts()
    if concepts:
        print(f"\n  ✓ Found {len(concepts)} concept(s):")
        for i, concept in enumerate(concepts, 1):
            print(f"\n  Concept {i}:")
            print_concept(concept)
    else:
        print("  ℹ No concepts found in HCG")

    return True


def main():
    """Main entry point for the retrieval demo."""
    parser = argparse.ArgumentParser(
        description="Demonstrate HCG entity retrieval and relationship traversal"
    )
    parser.add_argument(
        "--uri",
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j connection URI (default: bolt://localhost:7687)",
    )
    parser.add_argument(
        "--user",
        default=os.getenv("NEO4J_USER", "neo4j"),
        help="Neo4j username (default: neo4j)",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("NEO4J_PASSWORD", "logosdev"),
        help="Neo4j password (default: logosdev)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  LOGOS HCG Retrieval Demonstration")
    print("  M1 Milestone: HCG can store and retrieve entities")
    print("=" * 70)
    print(f"\nConnecting to Neo4j at {args.uri}...")

    try:
        with HCGClient(args.uri, args.user, args.password) as client:
            print("✓ Connected successfully\n")

            # Run all demos
            success = True
            success = demo_basic_retrieval(client) and success
            success = demo_relationship_traversal(client) and success
            success = demo_counts_and_statistics(client) and success
            success = demo_all_entities(client) and success
            success = demo_all_concepts(client) and success

            # Summary
            print("\n" + "=" * 70)
            if success:
                print("  ✅ M1 DEMONSTRATION COMPLETE")
                print("=" * 70)
                print("\n  All acceptance criteria verified:")
                print("    ✓ Can query entities by UUID")
                print("    ✓ Can query entities by name")
                print("    ✓ Can traverse IS_A relationships")
                print("    ✓ Can traverse HAS_STATE relationships")
                print("    ✓ Can retrieve node counts and statistics")
                print("\n  Reference: docs/PHASE1_VERIFY.md - M1 checklist")
                print()
                sys.exit(0)
            else:
                print("  ⚠ DEMONSTRATION INCOMPLETE")
                print("=" * 70)
                print("\n  Some entities or relationships are missing.")
                print("  Run the loader first:")
                print("    python -m logos_hcg.load_hcg")
                print()
                sys.exit(1)

    except HCGConnectionError as e:
        logger.error(f"✗ Connection failed: {e}")
        logger.error("\nTroubleshooting:")
        logger.error("  1. Ensure Neo4j is running:")
        logger.error("     docker compose -f infra/docker-compose.hcg.dev.yml up -d")
        logger.error("  2. Wait for Neo4j to be ready (15-30 seconds)")
        logger.error("  3. Verify connection details")
        sys.exit(1)
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
