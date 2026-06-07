"""Unit tests for batch query methods on the Sophia HCG client."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sophia.hcg_client.client import HCGClient


pytestmark = pytest.mark.unit


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> HCGClient:
    """Instantiate HCGClient with a mocked Neo4j driver."""

    # Clear environment variables to isolate unit tests
    monkeypatch.delenv("MILVUS_HOST", raising=False)
    monkeypatch.delenv("MILVUS_PORT", raising=False)

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


# --- get_nodes_batch tests ---


def test_get_nodes_batch_returns_correct_nodes(
    monkeypatch: pytest.MonkeyPatch, client: HCGClient
) -> None:
    """get_nodes_batch should return decoded node dicts for each uuid."""
    mock_read = MagicMock(
        return_value=[
            {
                "uuid": "aaa",
                "name": "Alpha",
                "type": "concept",
                "props": {
                    "uuid": "aaa",
                    "name": "Alpha",
                    "type": "concept",
                    "custom": "val1",
                },
            },
            {
                "uuid": "bbb",
                "name": "Beta",
                "type": "entity",
                "props": {
                    "uuid": "bbb",
                    "name": "Beta",
                    "type": "entity",
                    "custom": "val2",
                },
            },
        ]
    )
    monkeypatch.setattr(client, "_execute_read", mock_read)

    result = client.get_nodes_batch(["aaa", "bbb"])

    assert len(result) == 2
    assert result[0]["uuid"] == "aaa"
    assert result[0]["name"] == "Alpha"
    assert result[0]["type"] == "concept"
    assert result[0]["properties"] == {"custom": "val1"}
    assert result[1]["uuid"] == "bbb"
    assert result[1]["name"] == "Beta"
    assert result[1]["properties"] == {"custom": "val2"}

    # Verify _execute_read was called once with IN-based query
    mock_read.assert_called_once()
    query = mock_read.call_args[0][0]
    assert "n.uuid IN $uuids" in query
    assert "n.relation IS NULL" in query


def test_get_nodes_batch_empty_input(client: HCGClient) -> None:
    """get_nodes_batch should return empty list for empty input."""
    result = client.get_nodes_batch([])
    assert result == []


# --- find_nodes_by_names tests ---


def test_find_nodes_by_names_returns_correct_mapping(
    monkeypatch: pytest.MonkeyPatch, client: HCGClient
) -> None:
    """find_nodes_by_names should return a name->node dict."""
    mock_read = MagicMock(
        return_value=[
            {
                "uuid": "aaa",
                "name": "Alpha",
                "type": "concept",
                "props": {
                    "uuid": "aaa",
                    "name": "Alpha",
                    "type": "concept",
                    "source": "test",
                },
            },
            {
                "uuid": "bbb",
                "name": "Beta",
                "type": "entity",
                "props": {
                    "uuid": "bbb",
                    "name": "Beta",
                    "type": "entity",
                },
            },
        ]
    )
    monkeypatch.setattr(client, "_execute_read", mock_read)

    result = client.find_nodes_by_names(["Alpha", "Beta"])

    assert len(result) == 2
    assert "Alpha" in result
    assert "Beta" in result
    assert result["Alpha"]["uuid"] == "aaa"
    assert result["Alpha"]["properties"] == {"source": "test"}
    assert result["Beta"]["uuid"] == "bbb"
    assert result["Beta"]["properties"] == {}

    # Verify _execute_read was called once with IN-based query
    mock_read.assert_called_once()
    query = mock_read.call_args[0][0]
    assert "n.name IN $names" in query
    assert "n.relation IS NULL" in query


def test_find_nodes_by_names_empty_input(client: HCGClient) -> None:
    """find_nodes_by_names should return empty dict for empty input."""
    result = client.find_nodes_by_names([])
    assert result == {}


# --- get_members_of_type tests ---


def test_get_members_of_type_returns_members(
    monkeypatch: pytest.MonkeyPatch, client: HCGClient
) -> None:
    """get_members_of_type returns decoded member nodes for a type via IS_A edge."""
    mock_read = MagicMock(
        return_value=[
            {
                "uuid": "aaa",
                "name": "Alpha",
                "type": "concept",
                "props": {
                    "uuid": "aaa",
                    "name": "Alpha",
                    "type": "concept",
                    "custom": "val1",
                },
            },
            {
                "uuid": "bbb",
                "name": "Beta",
                "type": "entity",
                "props": {
                    "uuid": "bbb",
                    "name": "Beta",
                    "type": "entity",
                    "custom": "val2",
                },
            },
        ]
    )
    monkeypatch.setattr(client, "_execute_read", mock_read)

    result = client.get_members_of_type("type-123")

    assert len(result) == 2
    assert result[0]["uuid"] == "aaa"
    assert result[0]["name"] == "Alpha"
    assert result[0]["type"] == "concept"
    assert result[0]["properties"] == {"custom": "val1"}
    assert result[1]["uuid"] == "bbb"
    assert result[1]["name"] == "Beta"
    assert result[1]["properties"] == {"custom": "val2"}
    # membership is the edge now: no top-level type_uuid key is returned
    assert "type_uuid" not in result[0]
    assert "type_uuid" not in result[1]

    # Verify the query joins through the instance->type IS_A edge
    # (FROM=member, TO=type) and binds type_uuid.
    mock_read.assert_called_once()
    query, params = mock_read.call_args[0]
    assert "relation: 'IS_A'" in query
    assert "[:TO]->" in query
    assert "[:FROM]->" in query
    assert "$type_uuid" in query
    assert params == {"type_uuid": "type-123"}


def test_get_members_of_type_empty(
    monkeypatch: pytest.MonkeyPatch, client: HCGClient
) -> None:
    """get_members_of_type returns an empty list when the type has no members."""
    mock_read = MagicMock(return_value=[])
    monkeypatch.setattr(client, "_execute_read", mock_read)

    result = client.get_members_of_type("type-empty")

    assert result == []
    mock_read.assert_called_once()
    params = mock_read.call_args[0][1]
    assert params == {"type_uuid": "type-empty"}
