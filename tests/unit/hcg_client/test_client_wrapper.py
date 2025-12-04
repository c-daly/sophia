"""Unit tests for the Sophia HCG client wrapper."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator
from unittest.mock import MagicMock

import pytest

from sophia.hcg_client.client import HCGClient


pytestmark = pytest.mark.unit


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> HCGClient:
    """Instantiate HCGClient with a mocked Neo4j driver."""

    mock_driver = MagicMock()
    mock_driver.verify_connectivity.return_value = True

    def mock_driver_factory(*_args, **_kwargs) -> MagicMock:
        return mock_driver

    monkeypatch.setattr(
        "logos_hcg.client.GraphDatabase.driver",
        mock_driver_factory,
    )
    return HCGClient(
        neo4j_uri="bolt://test",
        neo4j_username="neo4j",
        neo4j_password="password",
    )


def test_add_node_runs_shacl_validation(
    monkeypatch: pytest.MonkeyPatch, client: HCGClient
) -> None:
    """add_node should validate inputs and write via shared helper."""
    validator = MagicMock()
    validator.validate_node.return_value = (True, [])
    monkeypatch.setattr(client, "_validator", validator)

    execute = MagicMock(return_value=[{"id": "node-1"}])
    monkeypatch.setattr(client, "_execute_query", execute)

    result = client.add_node("node-1", "concept", {"name": "Test"})

    assert result == "node-1"
    validator.validate_node.assert_called_once()
    execute.assert_called_once()


def test_add_node_raises_on_validation_error(
    monkeypatch: pytest.MonkeyPatch, client: HCGClient
) -> None:
    """Invalid nodes should raise ValueError before Cypher executes."""
    validator = MagicMock()
    validator.validate_node.return_value = (False, ["missing type"])
    monkeypatch.setattr(client, "_validator", validator)

    with pytest.raises(ValueError, match="missing type"):
        client.add_node("bad", "", {})


def test_add_edge_uses_validator(
    monkeypatch: pytest.MonkeyPatch, client: HCGClient
) -> None:
    """Edges also run through SHACL validation."""
    validator = MagicMock()
    validator.validate_edge.return_value = (True, [])
    monkeypatch.setattr(client, "_validator", validator)

    execute = MagicMock(return_value=[{"id": "edge-1"}])
    monkeypatch.setattr(client, "_execute_query", execute)

    result = client.add_edge("edge-1", "a", "b", "relates_to")

    assert result == "edge-1"
    validator.validate_edge.assert_called_once()
    execute.assert_called_once()


def test_get_node_returns_none_when_missing(
    monkeypatch: pytest.MonkeyPatch, client: HCGClient
) -> None:
    """get_node should return None when Neo4j has no matching record."""
    monkeypatch.setattr(client, "_execute_read", lambda *args, **kwargs: [])

    assert client.get_node("unknown") is None


def test_delete_node_reports_deleted(
    monkeypatch: pytest.MonkeyPatch, client: HCGClient
) -> None:
    """delete_node exposes count returned by Cypher."""
    monkeypatch.setattr(
        client, "_execute_query", lambda *args, **kwargs: [{"deleted": 1}]
    )

    assert client.delete_node("node-1") is True


def test_health_check_uses_session(
    monkeypatch: pytest.MonkeyPatch, client: HCGClient
) -> None:
    """health_check should report Neo4j health and Milvus health (True when not configured)."""
    # Ensure no Milvus env vars interfere with test
    monkeypatch.delenv("MILVUS_HOST", raising=False)
    monkeypatch.delenv("MILVUS_PORT", raising=False)
    # Clear milvus config on client
    monkeypatch.setattr(client, "_milvus_host", None)
    monkeypatch.setattr(client, "_milvus_port", None)

    @contextmanager
    def fake_session() -> Iterator[MagicMock]:
        mock_session = MagicMock()
        mock_session.run.return_value.single.return_value = {"ok": 1}
        yield mock_session

    monkeypatch.setattr(client, "_session", lambda: fake_session())

    health = client.health_check()

    # Milvus returns True when not configured (no host/port set)
    assert health == {"neo4j": True, "milvus": True}
