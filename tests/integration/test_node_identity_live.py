"""Integration test (#148): entity identity is embedding-based, never name-based.

Direct ``HCGClient`` -> live Neo4j. Proves against the real graph what the unit
tests assert against mocks: two ``add_node`` calls with the same name+type but
no explicit uuid create TWO distinct nodes -- the literal name is not an
identity key. An explicit uuid still upserts a single node (MERGE on uuid).

Requires Neo4j; skips when unavailable. Created nodes are removed in teardown.
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def hcg():
    from sophia.hcg_client import HCGClient

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "logosdev")
    try:
        client = HCGClient(neo4j_uri=uri, neo4j_username=user, neo4j_password=password)
        client._execute_query("RETURN 1 AS ok", {})  # connectivity probe
    except Exception as exc:  # pragma: no cover - infra-dependent
        pytest.skip(f"Neo4j not available at {uri}: {exc}")
    yield client
    client.close()


def test_same_name_type_creates_distinct_nodes_in_graph(hcg):
    """Two omitted-uuid mentions of the same name+type are distinct graph nodes.

    The name string is never a merge key (#148): each call mints its own uuid4,
    so the graph ends up holding two nodes, not one.
    """
    name = f"dedup_probe_{uuid4().hex}"
    created: list[str] = []
    try:
        uuid_a = hcg.add_node(name=name, node_type="entity", source="test_148")
        uuid_b = hcg.add_node(name=name, node_type="entity", source="test_148")
        created = [uuid_a, uuid_b]

        # Distinct identities: name is not a merge key.
        assert uuid_a != uuid_b

        rows = hcg._execute_query(
            "MATCH (n:Node {name: $name, type: $type}) RETURN n.uuid AS uuid",
            {"name": name, "type": "entity"},
        )
        uuids = sorted(r["uuid"] for r in rows)
        assert uuids == sorted([uuid_a, uuid_b]), uuids
        assert len(uuids) == 2
    finally:
        for u in created:
            try:
                hcg.delete_node(u)
            except Exception:
                pass


def test_explicit_uuid_upserts_single_node_in_graph(hcg):
    """An explicit uuid still MERGEs to a single node across repeated calls."""
    name = f"upsert_probe_{uuid4().hex}"
    node_uuid = f"fixed_{uuid4().hex}"
    try:
        u1 = hcg.add_node(
            name=name, node_type="entity", uuid=node_uuid, source="test_148"
        )
        u2 = hcg.add_node(
            name=name, node_type="entity", uuid=node_uuid, source="test_148"
        )
        assert u1 == u2 == node_uuid

        rows = hcg._execute_query(
            "MATCH (n:Node {uuid: $uuid}) RETURN n.uuid AS uuid",
            {"uuid": node_uuid},
        )
        assert len(rows) == 1
    finally:
        try:
            hcg.delete_node(node_uuid)
        except Exception:
            pass
