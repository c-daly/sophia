"""
HCG Data Loader - Loads ontology and seed entities into Neo4j.

This script provides a deterministic way to bootstrap the HCG with:
1. Core ontology (constraints, indexes, concepts)
2. Seed entities (RobotArm, Manipulator, etc.)

Usage:
    python -m logos_hcg.load_hcg --uri bolt://localhost:7687 --user neo4j --password logosdev

    Or with environment variables:
    NEO4J_URI=bolt://localhost:7687 python -m logos_hcg.load_hcg

See Project LOGOS spec: Section 4.1 (Core Ontology and Data Model)
Reference: docs/PHASE1_VERIFY.md - M1 checklist
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HCGLoader:
    """Loads HCG ontology and seed data into Neo4j."""

    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize loader with Neo4j connection.

        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            user: Neo4j username
            password: Neo4j password
        """
        self.uri = uri
        self.user = user
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            logger.info(f"✓ Connected to Neo4j at {uri}")
        except ServiceUnavailable as e:
            logger.error(f"✗ Failed to connect to Neo4j at {uri}: {e}")
            raise

    def close(self):
        """Close the driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("✓ Closed Neo4j connection")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def load_cypher_file(self, file_path: Path, description: str) -> bool:
        """
        Load a Cypher script file into Neo4j.

        Args:
            file_path: Path to the .cypher file
            description: Human-readable description of what's being loaded

        Returns:
            True if successful, False otherwise
        """
        if not file_path.exists():
            logger.error(f"✗ File not found: {file_path}")
            return False

        logger.info(f"Loading {description} from {file_path.name}...")

        try:
            with open(file_path) as f:
                script_content = f.read()

            # Parse Cypher statements more carefully
            # Remove single-line comments
            lines = []
            for line in script_content.split("\n"):
                # Remove inline comments
                if "//" in line:
                    line = line[: line.index("//")]
                line = line.strip()
                if line:
                    lines.append(line)

            # Join lines and split by semicolons
            cleaned_script = " ".join(lines)
            statements = [s.strip() for s in cleaned_script.split(";") if s.strip()]

            with self.driver.session() as session:
                for i, statement in enumerate(statements):
                    # Skip empty statements
                    if not statement:
                        continue

                    try:
                        result = session.run(statement)
                        # Consume result to ensure execution
                        summary = result.consume()

                        # Log meaningful operations
                        counters = summary.counters
                        if counters.constraints_added > 0:
                            logger.info(
                                f"  ✓ Added {counters.constraints_added} constraint(s)"
                            )
                        if counters.indexes_added > 0:
                            logger.info(f"  ✓ Added {counters.indexes_added} index(es)")
                        if counters.nodes_created > 0:
                            logger.info(f"  ✓ Created {counters.nodes_created} node(s)")
                        if counters.relationships_created > 0:
                            logger.info(
                                f"  ✓ Created {counters.relationships_created} relationship(s)"
                            )

                    except Neo4jError as e:
                        # Some errors are expected (e.g., constraint already exists)
                        if (
                            "already exists" in str(e).lower()
                            or "equivalent" in str(e).lower()
                        ):
                            logger.debug("  (skipped - already exists)")
                        else:
                            logger.warning(f"  Warning on statement {i+1}: {e}")
                            # Continue with other statements

            logger.info(f"✓ Loaded {description}")
            return True

        except Exception as e:
            logger.error(f"✗ Failed to load {description}: {e}")
            return False

    def verify_constraints(self) -> dict:
        """
        Verify that required constraints exist.

        Returns:
            Dictionary with constraint counts
        """
        logger.info("Verifying constraints...")

        with self.driver.session() as session:
            result = session.run("SHOW CONSTRAINTS")
            constraints = [record["name"] for record in result]

        required_constraints = [
            "logos_entity_uuid",
            "logos_concept_uuid",
            "logos_state_uuid",
            "logos_process_uuid",
            "logos_concept_name",
        ]

        found = {}
        for constraint in required_constraints:
            exists = any(constraint in c for c in constraints)
            found[constraint] = exists
            status = "✓" if exists else "✗"
            logger.info(f"  {status} {constraint}: {'found' if exists else 'MISSING'}")

        return found

    def verify_indexes(self) -> dict:
        """
        Verify that required indexes exist.

        Returns:
            Dictionary with index counts
        """
        logger.info("Verifying indexes...")

        with self.driver.session() as session:
            result = session.run("SHOW INDEXES")
            indexes = [record["name"] for record in result]

        required_indexes = [
            "logos_entity_name",
            "logos_state_timestamp",
            "logos_process_timestamp",
        ]

        found = {}
        for index in required_indexes:
            exists = any(index in i for i in indexes)
            found[index] = exists
            status = "✓" if exists else "✗"
            logger.info(f"  {status} {index}: {'found' if exists else 'MISSING'}")

        return found

    def verify_concepts(self) -> int:
        """
        Verify that concepts are loaded.

        Returns:
            Number of concepts found
        """
        logger.info("Verifying concepts...")

        with self.driver.session() as session:
            result = session.run("MATCH (c:Concept) RETURN count(c) AS count")
            record = result.single()
            count = record["count"] if record else 0

        logger.info(f"  ✓ Found {count} concept(s)")
        return int(count)

    def verify_entities(self) -> int:
        """
        Verify that seed entities are loaded.

        Returns:
            Number of entities found
        """
        logger.info("Verifying seed entities...")

        with self.driver.session() as session:
            result = session.run("MATCH (e:Entity) RETURN count(e) AS count")
            record = result.single()
            count = record["count"] if record else 0

        logger.info(f"  ✓ Found {count} entity(ies)")
        return int(count)

    def create_seed_entities(self) -> bool:
        """
        Create canonical seed entities (RobotArm, Manipulator, etc.).

        These are the minimal entities required for M1 verification.

        Returns:
            True if successful, False otherwise
        """
        logger.info("Creating seed entities...")

        seed_queries = [
            # Create Manipulator concept
            """
            MERGE (c:Concept {uuid: 'concept-manipulator'})
            ON CREATE SET
                c.name = 'Manipulator',
                c.description = 'A robotic manipulator or arm capable of movement and grasping',
                c.created_at = datetime()
            """,
            # Create RobotArm entity
            """
            MERGE (e:Entity {uuid: 'entity-robot-arm-01'})
            ON CREATE SET
                e.name = 'RobotArm01',
                e.description = 'Canonical six-axis robotic manipulator for M1 verification',
                e.created_at = datetime()
            """,
            # Create IS_A relationship
            """
            MATCH (e:Entity {uuid: 'entity-robot-arm-01'})
            MATCH (c:Concept {uuid: 'concept-manipulator'})
            MERGE (e)-[:IS_A]->(c)
            """,
            # Create initial state for RobotArm
            """
            MERGE (s:State {uuid: 'state-robot-arm-01-initial'})
            ON CREATE SET
                s.name = 'RobotArm01-Idle',
                s.description = 'Robot arm in idle state',
                s.timestamp = datetime()
            """,
            # Link state to entity
            """
            MATCH (e:Entity {uuid: 'entity-robot-arm-01'})
            MATCH (s:State {uuid: 'state-robot-arm-01-initial'})
            MERGE (e)-[:HAS_STATE]->(s)
            """,
        ]

        try:
            with self.driver.session() as session:
                for query in seed_queries:
                    result = session.run(query)
                    result.consume()

            logger.info("✓ Seed entities created")
            return True

        except Neo4jError as e:
            logger.error(f"✗ Failed to create seed entities: {e}")
            return False

    def load_all(self, repo_root: Path) -> bool:
        """
        Load all ontology and seed data.

        Args:
            repo_root: Root path of the repository

        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 60)
        logger.info("HCG Data Loader - Loading Ontology and Seed Data")
        logger.info("=" * 60)

        # Step 1: Load core ontology
        ontology_path = repo_root / "ontology" / "core_ontology.cypher"
        if not self.load_cypher_file(ontology_path, "core ontology"):
            return False

        # Step 2: Verify ontology loaded correctly
        logger.info("")
        constraints = self.verify_constraints()
        self.verify_indexes()
        concepts = self.verify_concepts()

        # Check if basic requirements are met
        if not all(constraints.values()):
            logger.warning(
                "⚠ Some constraints are missing - they may have been created previously"
            )

        # Step 3: Create seed entities
        logger.info("")
        if not self.create_seed_entities():
            return False

        # Step 4: Load test data (optional - for full M1 demo)
        logger.info("")
        test_data_path = repo_root / "ontology" / "test_data_pick_and_place.cypher"
        if test_data_path.exists():
            self.load_cypher_file(test_data_path, "pick-and-place test data")
        else:
            logger.info("ℹ Pick-and-place test data not found (optional)")

        # Step 5: Final verification
        logger.info("")
        entities = self.verify_entities()

        logger.info("")
        logger.info("=" * 60)
        logger.info("✓ HCG Loading Complete")
        logger.info("=" * 60)
        logger.info(f"  Concepts: {concepts}")
        logger.info(f"  Entities: {entities}")
        logger.info("")
        logger.info("Access Neo4j Browser at: http://localhost:7474")
        logger.info(f"  Username: {self.user}")
        logger.info(f"  Password: {'*' * len(self.user)}")
        logger.info("")
        logger.info("Try running queries like:")
        logger.info("  MATCH (e:Entity)-[:IS_A]->(c:Concept) RETURN e.name, c.name;")
        logger.info("  MATCH (e:Entity)-[:HAS_STATE]->(s:State) RETURN e.name, s.name;")
        logger.info("")

        return True


def main():
    """Main entry point for the HCG loader."""
    parser = argparse.ArgumentParser(
        description="Load LOGOS HCG ontology and seed data into Neo4j"
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
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).parent.parent,
        help="Repository root path (default: auto-detected)",
    )

    args = parser.parse_args()

    try:
        with HCGLoader(args.uri, args.user, args.password) as loader:
            success = loader.load_all(args.repo_root)
            sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
