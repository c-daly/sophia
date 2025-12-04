"""
Integration Test Fixtures for Sophia

Provides fixtures for testing Sophia API endpoints with real
Neo4j and Milvus services. The test stack must be running:

    ./scripts/test_integration.sh up

These tests make HTTP requests to the running Sophia service.
"""

import os

import pytest
import httpx

# Integration tests require the stack to be running
pytestmark = pytest.mark.integration

# Service configuration from environment
# Sophia runs on port 38001 in test stack
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

# Test auth token
TEST_AUTH_TOKEN = os.getenv("SOPHIA_API_TOKEN", "test-token-12345")


@pytest.fixture(scope="module")
def sophia_url() -> str:
    """Base URL for Sophia API."""
    return SOPHIA_URL


@pytest.fixture(scope="module")
def api_token() -> str:
    """API authentication token."""
    return TEST_AUTH_TOKEN


@pytest.fixture(scope="module")
def auth_headers(api_token: str) -> dict:
    """Authentication headers for protected endpoints."""
    return {"Authorization": f"Bearer {api_token}"}


@pytest.fixture(scope="module")
def neo4j_uri() -> str:
    """Neo4j connection URI."""
    return NEO4J_URI


@pytest.fixture(scope="module")
def neo4j_username() -> str:
    """Neo4j username."""
    return NEO4J_USER


@pytest.fixture(scope="module")
def neo4j_password() -> str:
    """Neo4j password."""
    return NEO4J_PASSWORD


@pytest.fixture(scope="module")
def milvus_host() -> str:
    """Milvus host."""
    return MILVUS_HOST


@pytest.fixture(scope="module")
def milvus_port() -> int:
    """Milvus port."""
    return int(MILVUS_PORT)


@pytest.fixture(scope="module")
def infrastructure_ports() -> dict:
    """Port configuration for infrastructure services."""
    return {
        "neo4j_http": int(NEO4J_HTTP_PORT),
        "neo4j_bolt": int(NEO4J_BOLT_PORT),
        "milvus_grpc": int(MILVUS_PORT),
        "milvus_health": int(MILVUS_METRICS_PORT),
        "sophia": int(SOPHIA_PORT),
    }


def check_sophia_health(url: str) -> bool:
    """Check if Sophia is healthy."""
    try:
        resp = httpx.get(f"{url}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def check_neo4j_health(ports: dict) -> bool:
    """Check if Neo4j is healthy via HTTP API."""
    try:
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


@pytest.fixture(scope="module")
def verify_sophia(sophia_url: str) -> None:
    """Verify Sophia service is running before tests."""
    if not check_sophia_health(sophia_url):
        pytest.skip(
            f"Sophia not available at {sophia_url}. "
            "Start with: ./scripts/test_integration.sh up"
        )


@pytest.fixture(scope="module")
def verify_infrastructure(infrastructure_ports: dict) -> None:
    """
    Verify that infrastructure services are running before tests.
    """
    neo4j_ok = check_neo4j_health(infrastructure_ports)
    milvus_ok = check_milvus_health(infrastructure_ports)

    if not neo4j_ok:
        pytest.skip(
            f"Neo4j not available on port {infrastructure_ports['neo4j_http']}. "
            "Start with: ./scripts/test_integration.sh up"
        )
    if not milvus_ok:
        pytest.skip(
            f"Milvus not available on port {infrastructure_ports['milvus_health']}. "
            "Start with: ./scripts/test_integration.sh up"
        )


@pytest.fixture(scope="module")
def http_client(sophia_url: str, verify_sophia) -> httpx.Client:
    """Create an HTTP client for making requests to Sophia."""
    with httpx.Client(base_url=sophia_url, timeout=30.0) as client:
        yield client


@pytest.fixture(scope="module")
def hcg_client(neo4j_uri: str, neo4j_username: str, neo4j_password: str, verify_infrastructure):
    """Create HCG client for direct database verification."""
    from sophia.hcg_client import HCGClient

    client = HCGClient(
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password,
    )
    yield client
    client.close()
