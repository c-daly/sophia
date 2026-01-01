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


def test_add_node_runs_shacl_validation(
    monkeypatch: pytest.MonkeyPatch, client: HCGClient
) -> None:
    """add_node should validate inputs and write via shared helper."""
    validator = MagicMock()
    validator.validate_node.return_value = (True, [])
    monkeypatch.setattr(client, "_validator", validator)

    execute = MagicMock(return_value=[{"uuid": "node-1"}])
    monkeypatch.setattr(client, "_execute_query", execute)

    result = client.add_node(
        uuid="node-1",
        name="Test Node",
        node_type="concept",
        ancestors=["parent", "root"],
        is_type_definition=False,
        properties={"custom": "value"},
    )

    assert result == "node-1"
    validator.validate_node.assert_called_once()
    execute.assert_called_once()


def test_add_node_raises_on_empty_uuid(client: HCGClient) -> None:
    """Empty uuid should raise ValueError before validation."""
    with pytest.raises(ValueError, match="uuid cannot be empty"):
        client.add_node(
            uuid="",
            name="Test",
            node_type="concept",
            ancestors=[],
        )


def test_add_node_raises_on_empty_name(client: HCGClient) -> None:
    """Empty name should raise ValueError before validation."""
    with pytest.raises(ValueError, match="name cannot be empty"):
        client.add_node(
            uuid="test-1",
            name="",
            node_type="concept",
            ancestors=[],
        )


def test_add_node_raises_on_empty_node_type(client: HCGClient) -> None:
    """Empty node_type should raise ValueError before validation."""
    with pytest.raises(ValueError, match="node_type cannot be empty"):
        client.add_node(
            uuid="test-1",
            name="Test",
            node_type="",
            ancestors=[],
        )


def test_add_node_raises_on_validation_error(
    monkeypatch: pytest.MonkeyPatch, client: HCGClient
) -> None:
    """Invalid nodes should raise ValueError before Cypher executes."""
    validator = MagicMock()
    validator.validate_node.return_value = (False, ["validation error"])
    monkeypatch.setattr(client, "_validator", validator)

    with pytest.raises(ValueError, match="validation error"):
        client.add_node(
            uuid="bad",
            name="Bad Node",
            node_type="concept",
            ancestors=[],
        )


def test_add_node_legacy_warns_deprecation(
    monkeypatch: pytest.MonkeyPatch, client: HCGClient
) -> None:
    """add_node_legacy should emit DeprecationWarning."""
    validator = MagicMock()
    validator.validate_node.return_value = (True, [])
    monkeypatch.setattr(client, "_validator", validator)

    execute = MagicMock(return_value=[{"uuid": "legacy-1"}])
    monkeypatch.setattr(client, "_execute_query", execute)

    with pytest.warns(DeprecationWarning, match="add_node_legacy is deprecated"):
        result = client.add_node_legacy(
            node_id="legacy-1",
            node_type="concept",
            properties={"name": "Legacy Node"},
        )

    assert result == "legacy-1"


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


# --- Ancestor auto-computation tests ---


def test_get_type_ancestors_returns_ancestors_from_type_definition(
    client: HCGClient,
) -> None:
    """_get_type_ancestors returns ancestors from matching type definition."""
    client._execute_read = MagicMock(
        return_value=[{"ancestors": ["physical_entity", "entity"]}]
    )

    ancestors = client._get_type_ancestors("object")

    assert ancestors == ["physical_entity", "entity"]
    client._execute_read.assert_called_once()
    call_args = client._execute_read.call_args
    assert "name: $node_type" in call_args[0][0]
    assert call_args[0][1]["node_type"] == "object"


def test_get_type_ancestors_returns_empty_when_no_type_definition(
    client: HCGClient,
) -> None:
    """_get_type_ancestors returns empty list when type definition not found."""
    client._execute_read = MagicMock(return_value=[])

    ancestors = client._get_type_ancestors("unknown_type")

    assert ancestors == []


def test_get_type_ancestors_returns_empty_when_ancestors_null(
    client: HCGClient,
) -> None:
    """_get_type_ancestors returns empty list when ancestors field is null."""
    client._execute_read = MagicMock(return_value=[{"ancestors": None}])

    ancestors = client._get_type_ancestors("some_type")

    assert ancestors == []


def test_add_node_auto_computes_ancestors_for_instance(
    client: HCGClient,
) -> None:
    """Instance nodes get ancestors auto-computed as [node_type] + type_def.ancestors."""
    client._get_type_ancestors = MagicMock(return_value=["physical_entity", "entity"])
    client._execute_query = MagicMock(return_value=[{"uuid": "generated-uuid"}])

    client.add_node(
        name="Red Block",
        node_type="object",
    )

    client._get_type_ancestors.assert_called_once_with("object")
    call_args = client._execute_query.call_args
    params = call_args[0][1]
    assert params["ancestors"] == ["object", "physical_entity", "entity"]


def test_add_node_type_definition_does_not_auto_compute_ancestors(
    client: HCGClient,
) -> None:
    """Type definitions don't auto-compute ancestors - use empty if not provided."""
    client._get_type_ancestors = MagicMock()
    client._execute_query = MagicMock(return_value=[{"uuid": "type-def-uuid"}])

    client.add_node(
        name="object",
        node_type="physical_entity",
        is_type_definition=True,
    )

    client._get_type_ancestors.assert_not_called()
    call_args = client._execute_query.call_args
    params = call_args[0][1]
    assert params["ancestors"] == []


def test_add_node_uses_provided_ancestors_when_given(
    client: HCGClient,
) -> None:
    """When ancestors are explicitly provided, they are used as-is."""
    client._get_type_ancestors = MagicMock()
    client._execute_query = MagicMock(return_value=[{"uuid": "custom-uuid"}])

    custom_ancestors = ["custom_parent", "custom_grandparent"]

    client.add_node(
        name="Custom Node",
        node_type="custom_type",
        ancestors=custom_ancestors,
    )

    client._get_type_ancestors.assert_not_called()
    call_args = client._execute_query.call_args
    params = call_args[0][1]
    assert params["ancestors"] == custom_ancestors


def test_add_node_generates_uuid_when_not_provided(
    client: HCGClient,
) -> None:
    """UUID is auto-generated when not provided."""
    client._get_type_ancestors = MagicMock(return_value=[])
    client._execute_query = MagicMock(return_value=[{"uuid": "auto-generated"}])

    client.add_node(
        name="Test Node",
        node_type="test_type",
    )

    call_args = client._execute_query.call_args
    params = call_args[0][1]
    assert len(params["uuid"]) == 36
    assert params["uuid"].count("-") == 4
