"""
End-to-End Test Fixtures for Sophia

Provides fixtures for testing complete Sophia workflows with real
Neo4j and Milvus services. The test stack must be running:

    ./scripts/start_services.sh

These tests validate the full Sophia service including:
- API endpoints with real database connectivity
- SHACL validation in the data path
- CWM state emission
- JEPA simulation workflows
"""

import os
from datetime import datetime, timezone
import pytest
import httpx

# E2E tests require the stack to be running
pytestmark = pytest.mark.e2e

# Service configuration from environment
# Sophia runs in container on port 38001
SOPHIA_PORT = os.getenv("SOPHIA_PORT", "38001")
SOPHIA_URL = os.getenv("SOPHIA_URL", f"http://localhost:{SOPHIA_PORT}")

# Infrastructure ports (Sophia uses 37xxx/39xxx offset)
NEO4J_HTTP_PORT = os.getenv("NEO4J_HTTP_PORT", "37474")
NEO4J_BOLT_PORT = os.getenv("NEO4J_BOLT_PORT", "37687")
MILVUS_PORT = os.getenv("MILVUS_PORT", "39530")
MILVUS_METRICS_PORT = os.getenv("MILVUS_METRICS_PORT", "39091")

# Neo4j connection config
NEO4J_URI = os.getenv("NEO4J_URI", f"bolt://localhost:{NEO4J_BOLT_PORT}")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4jtest")

# Milvus connection config
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")

# Test auth token (for endpoints requiring authentication)
# Must match SOPHIA_API_TOKEN in docker-compose.test.yml
TEST_AUTH_TOKEN = os.getenv("SOPHIA_API_TOKEN", "test-token-12345")


@pytest.fixture(scope="session")
def sophia_url() -> str:
    """Base URL for Sophia API."""
    return SOPHIA_URL


@pytest.fixture(scope="session")
def auth_headers() -> dict:
    """Authentication headers for protected endpoints."""
    return {"Authorization": f"Bearer {TEST_AUTH_TOKEN}"}


@pytest.fixture(scope="session")
def neo4j_config() -> dict:
    """Neo4j connection configuration."""
    return {
        "uri": NEO4J_URI,
        "user": NEO4J_USER,
        "password": NEO4J_PASSWORD,
    }


@pytest.fixture(scope="session")
def milvus_config() -> dict:
    """Milvus connection configuration."""
    return {
        "host": MILVUS_HOST,
        "port": int(MILVUS_PORT),
    }


@pytest.fixture(scope="session")
def infrastructure_ports() -> dict:
    """Port configuration for infrastructure services."""
    return {
        "neo4j_http": int(NEO4J_HTTP_PORT),
        "neo4j_bolt": int(NEO4J_BOLT_PORT),
        "milvus_grpc": int(MILVUS_PORT),
        "milvus_health": int(MILVUS_METRICS_PORT),
    }


def check_neo4j_health(ports: dict) -> bool:
    """Check if Neo4j is healthy via HTTP API."""
    try:
        # Neo4j browser endpoint returns JSON about the instance
        resp = httpx.get(
            f"http://localhost:{ports['neo4j_http']}/",
            timeout=5,
        )
        return resp.status_code == 200
    except Exception:
        return False


def check_milvus_health(ports: dict) -> bool:
    """Check if Milvus is healthy via metrics endpoint."""
    try:
        resp = httpx.get(
            f"http://localhost:{ports['milvus_health']}/healthz",
            timeout=5,
        )
        return resp.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def verify_infrastructure(infrastructure_ports: dict) -> None:
    """
    Verify that infrastructure services are running before tests.

    This fixture runs automatically and will skip all e2e tests
    if the required services are not available.
    """
    neo4j_ok = check_neo4j_health(infrastructure_ports)
    milvus_ok = check_milvus_health(infrastructure_ports)

    if not neo4j_ok and not milvus_ok:
        pytest.skip(
            "E2E infrastructure not running. Start with: ./scripts/start_services.sh"
        )
    elif not neo4j_ok:
        pytest.skip(f"Neo4j not available on port {infrastructure_ports['neo4j_http']}")
    elif not milvus_ok:
        pytest.skip(
            f"Milvus not available on port {infrastructure_ports['milvus_health']}"
        )


@pytest.fixture
def unique_id() -> str:
    """Generate a unique ID for test isolation."""
    import uuid

    return f"e2e_test_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def test_timestamp() -> str:
    """ISO timestamp for test data."""
    return datetime.now(timezone.utc).isoformat()
