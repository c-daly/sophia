#!/usr/bin/env python3
"""Seed test data into Neo4j for integration/e2e tests.

This script should be run AFTER docker services are up and ready.
It clears existing data and seeds the pick-and-place scenario.

Usage:
    python scripts/seed_test_data.py

Or via the test runner:
    scripts/run_integration.sh  # handles docker + seeding + tests
"""

import os
import sys
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sophia.hcg_client import HCGClient
from sophia.hcg_client.seeder import seed_pick_and_place_data


def wait_for_neo4j(uri: str, user: str, password: str, max_retries: int = 30) -> bool:
    """Wait for Neo4j to be ready."""
    logger.info(f"Waiting for Neo4j at {uri}...")
    
    for attempt in range(max_retries):
        try:
            client = HCGClient(
                neo4j_uri=uri,
                neo4j_username=user,
                neo4j_password=password,
            )
            # Try a simple query
            client._driver.verify_connectivity()
            client.close()
            logger.info("Neo4j is ready!")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                if attempt % 5 == 0:
                    logger.info(f"Attempt {attempt + 1}/{max_retries}: Neo4j not ready yet...")
                time.sleep(2)
            else:
                logger.error(f"Failed to connect to Neo4j after {max_retries} attempts: {e}")
                return False
    return False


def seed_data(uri: str, user: str, password: str) -> bool:
    """Clear and seed test data."""
    logger.info("Seeding test data...")
    
    try:
        client = HCGClient(
            neo4j_uri=uri,
            neo4j_username=user,
            neo4j_password=password,
        )
        
        # Clear existing data
        logger.info("Clearing existing data...")
        client.clear_all()
        
        # Seed pick-and-place scenario
        logger.info("Seeding pick-and-place scenario...")
        seed_pick_and_place_data(client)
        
        # Verify
        nodes = client.get_all_nodes()
        logger.info(f"Seeded {len(nodes)} nodes")
        
        client.close()
        logger.info("Test data seeded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to seed data: {e}")
        return False


def main():
    # Configuration from env vars (matching docker-compose.test.yml)
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "neo4jtest")
    
    logger.info("=== Sophia Test Data Seeder ===")
    logger.info(f"Neo4j URI: {neo4j_uri}")
    
    if not wait_for_neo4j(neo4j_uri, neo4j_user, neo4j_password):
        logger.error("Cannot proceed without Neo4j connection")
        sys.exit(1)
    
    if not seed_data(neo4j_uri, neo4j_user, neo4j_password):
        logger.error("Failed to seed test data")
        sys.exit(1)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
