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
        )


def test_add_node_raises_on_empty_name(client: HCGClient) -> None:
    """Empty name should raise ValueError before validation."""
    with pytest.raises(ValueError, match="name cannot be empty"):
        client.add_node(
            uuid="test-1",
            name="",
            node_type="concept",
        )


def test_add_node_raises_on_empty_node_type(client: HCGClient) -> None:
    """Empty node_type should raise ValueError before validation."""
    with pytest.raises(ValueError, match="node_type cannot be empty"):
        client.add_node(
            uuid="test-1",
            name="Test",
            node_type="",
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
    """Edges also run through SHACL validation and execute reified Cypher."""
    validator = MagicMock()
    validator.validate_edge.return_value = (True, [])
    monkeypatch.setattr(client, "_validator", validator)

    execute = MagicMock(return_value=[{"uuid": "edge-1"}])
    monkeypatch.setattr(client, "_execute_query", execute)

    result = client.add_edge("a", "b", "RELATES_TO", edge_uuid="edge-1")

    assert result == "edge-1"
    validator.validate_edge.assert_called_once()
    execute.assert_called_once()
    # Verify the Cypher creates reified edge with :FROM/:TO
    query = execute.call_args[0][0]
    assert "MERGE (edge:Node" in query
    assert "[:FROM]" in query
    assert "[:TO]" in query


def test_add_edge_with_bidirectional(
    monkeypatch: pytest.MonkeyPatch, client: HCGClient
) -> None:
    """add_edge should pass bidirectional flag in the edge props."""
    validator = MagicMock()
    validator.validate_edge.return_value = (True, [])
    monkeypatch.setattr(client, "_validator", validator)

    execute = MagicMock(return_value=[{"uuid": "edge-2"}])
    monkeypatch.setattr(client, "_execute_query", execute)

    result = client.add_edge(
        "a", "b", "RELATED_TO", edge_uuid="edge-2", bidirectional=True
    )

    assert result == "edge-2"
    params = execute.call_args[0][1]
    assert params["props"]["bidirectional"] is True


def test_get_node_returns_none_when_missing(
    monkeypatch: pytest.MonkeyPatch, client: HCGClient
) -> None:
    """get_node should return None when Neo4j has no matching record."""
    monkeypatch.setattr(client, "_execute_read", lambda *args, **kwargs: [])

    assert client.get_node("unknown") is None


def test_get_node_excludes_edge_nodes(
    monkeypatch: pytest.MonkeyPatch, client: HCGClient
) -> None:
    """get_node query should filter out edge nodes (WHERE n.relation IS NULL)."""
    calls = []

    def mock_execute_read(query, params):
        calls.append(query)
        return []

    monkeypatch.setattr(client, "_execute_read", mock_execute_read)

    client.get_node("some-uuid")

    assert len(calls) == 1
    assert "n.relation IS NULL" in calls[0]


def test_delete_node_reports_deleted(
    monkeypatch: pytest.MonkeyPatch, client: HCGClient
) -> None:
    """delete_node cleans up edge nodes then deletes the content node."""
    calls = []

    def mock_execute(query, params=None):
        calls.append(query)
        if "count(n)" in query:
            return [{"deleted": 1}]
        return []

    monkeypatch.setattr(client, "_execute_query", mock_execute)

    assert client.delete_node("node-1") is True
    # Should have two queries: cleanup edges, then delete node
    assert len(calls) == 2
    assert "edge.source = $uuid OR edge.target = $uuid" in calls[0]


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


# --- Node creation tests ---


def test_add_node_generates_uuid_when_not_provided(
    monkeypatch: pytest.MonkeyPatch, client: HCGClient
) -> None:
    """UUID is auto-generated when not provided."""
    mock_execute = MagicMock(
        side_effect=[
            [],  # dedup lookup returns no existing node
            [{"uuid": "auto-generated"}],  # MERGE query
        ]
    )
    monkeypatch.setattr(client, "_execute_query", mock_execute)

    client.add_node(
        name="Test Node",
        node_type="test_type",
    )

    # Second call is the MERGE — check that a proper uuid was generated
    merge_call = mock_execute.call_args_list[1]
    params = merge_call[0][1]
    assert len(params["uuid"]) == 36
    assert params["uuid"].count("-") == 4


def test_add_node_no_ancestors_or_is_type_definition(
    monkeypatch: pytest.MonkeyPatch, client: HCGClient
) -> None:
    """add_node should not send ancestors or is_type_definition to Neo4j."""
    mock_execute = MagicMock(return_value=[{"uuid": "node-1"}])
    monkeypatch.setattr(client, "_execute_query", mock_execute)

    client.add_node(
        uuid="node-1",
        name="Test Node",
        node_type="concept",
    )

    call_args = mock_execute.call_args
    query = call_args[0][0]
    params = call_args[0][1]

    assert "ancestors" not in query
    assert "is_type_definition" not in query
    assert "ancestors" not in params
    assert "is_type_definition" not in params


# --- Provenance metadata tests ---


def test_add_node_with_provenance(
    monkeypatch: pytest.MonkeyPatch, client: HCGClient
) -> None:
    """add_node should accept and store provenance metadata."""
    mock_execute = MagicMock(return_value=[{"uuid": "prov-uuid"}])
    monkeypatch.setattr(client, "_execute_query", mock_execute)

    client.add_node(
        uuid="prov-uuid",
        name="Provenance Test",
        node_type="test_type",
        source="planner",
        derivation="imagined",
        confidence=0.85,
        tags=["simulation", "test"],
        links={"plan_id": "plan_123", "process_ids": ["proc_1"]},
    )

    call_args = mock_execute.call_args
    params = call_args[0][1]
    props = params["properties"]

    assert props["source"] == "planner"
    assert props["derivation"] == "imagined"
    assert props["confidence"] == 0.85
    # Lists with primitive items stay as lists; dicts get JSON-serialized
    assert props["tags"] == ["simulation", "test"]
    # Dicts are serialized as JSON strings with __LOGOS_JSON__ prefix
    assert "__LOGOS_JSON__:" in props["links"]
    assert "plan_123" in props["links"]
    assert "created" in props
    assert "updated" in props


def test_add_node_default_provenance(
    monkeypatch: pytest.MonkeyPatch, client: HCGClient
) -> None:
    """add_node should apply default provenance when not provided."""
    mock_execute = MagicMock(return_value=[{"uuid": "default-uuid"}])
    monkeypatch.setattr(client, "_execute_query", mock_execute)

    client.add_node(
        uuid="default-uuid",
        name="Default Provenance Test",
        node_type="test_type",
    )

    call_args = mock_execute.call_args
    params = call_args[0][1]
    props = params["properties"]

    assert props["source"] == "unknown"
    assert props["derivation"] == "observed"
    # confidence is not included when None (not present in properties)
    assert "confidence" not in props
    # Empty lists stay as lists; empty dicts get JSON-serialized
    assert props["tags"] == []
    assert props["links"] == "__LOGOS_JSON__:{}"
    assert "created" in props
    assert "updated" in props


def test_update_node_updates_timestamp(
    client: HCGClient,
) -> None:
    """update_node should set updated timestamp."""
    client._execute_query = MagicMock(return_value=[{"uuid": "existing-uuid"}])

    result = client.update_node(
        uuid="existing-uuid",
        properties={"confidence": 0.95},
    )

    assert result == "existing-uuid"
    call_args = client._execute_query.call_args
    params = call_args[0][1]

    # Verify the updated timestamp is in the properties being set
    assert "updated" in params["properties"]
    # Verify confidence is in properties
    assert params["properties"]["confidence"] == 0.95


def test_update_node_raises_on_missing_node(
    client: HCGClient,
) -> None:
    """update_node should raise ValueError for non-existent nodes."""
    # Empty result means node not found
    client._execute_query = MagicMock(return_value=[])

    with pytest.raises(ValueError, match="not found"):
        client.update_node(
            uuid="nonexistent-uuid",
            properties={"confidence": 0.5},
        )


# --- get_subgraph tests ---


def test_get_subgraph_empty_uuids(
    monkeypatch: pytest.MonkeyPatch, client: HCGClient
) -> None:
    """get_subgraph with empty list returns empty result."""
    result = client.get_subgraph([])
    assert result == {"nodes": [], "edges": []}


def test_get_subgraph_fetches_nodes_and_edges(
    monkeypatch: pytest.MonkeyPatch, client: HCGClient
) -> None:
    """get_subgraph queries both content nodes and connecting edge nodes."""
    calls = []

    def mock_execute_read(query, params):
        calls.append(query)
        if "n.relation IS NULL" in query:
            return [
                {
                    "uuid": "a",
                    "name": "A",
                    "type": "concept",
                    "props": {"uuid": "a", "name": "A", "type": "concept"},
                },
            ]
        elif "edge.relation IS NOT NULL" in query:
            return []
        return []

    monkeypatch.setattr(client, "_execute_read", mock_execute_read)

    result = client.get_subgraph(["a", "b"])

    assert len(calls) == 2
    assert len(result["nodes"]) == 1
    assert result["nodes"][0]["uuid"] == "a"
