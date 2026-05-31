"""Integration test: delete_edge removes only the reified edge, not its endpoints.

Direct ``HCGClient`` -> Neo4j. Proves the #149 retype path (``mint_type``
dropping a member's stale ``IS_A`` edge) detaches the member from its old
parent type *without* deleting either endpoint node. In particular the parent
type-definition node must survive so its other members and centroid stay
intact -- ``delete_edge`` deletes the edge instance, never the edge's
definition/target.

Requires Neo4j; the ``hcg_client`` fixture (tests/integration/conftest.py)
skips the test when infrastructure is unavailable.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

pytestmark = pytest.mark.integration


def test_delete_edge_removes_edge_but_not_endpoint_nodes(hcg_client):
    """delete_edge(edge_uuid) drops the one IS_A edge between two nodes and
    leaves both the member and the parent type-definition node in place."""
    member_uuid = f"test_member_{uuid4().hex}"
    type_uuid = f"test_type_def_{uuid4().hex}"

    # Mirror the production retype shape: an entity member and a type-definition
    # parent, linked by an IS_A edge (exactly what mint_type creates).
    hcg_client.add_node(
        name="test_member",
        node_type="entity",
        uuid=member_uuid,
        source="test_delete_edge",
    )
    hcg_client.add_node(
        name="test_parent_type",
        node_type="type_definition",
        uuid=type_uuid,
        properties={"is_type_definition": True, "ancestors": ["root"]},
        source="test_delete_edge",
    )
    edge_uuid = hcg_client.add_edge(member_uuid, type_uuid, "IS_A")

    try:
        # Precondition: the IS_A edge exists, member -> parent type.
        before = hcg_client.query_edges_from(member_uuid)
        assert any(
            e.get("id") == edge_uuid and e.get("relation") == "IS_A" for e in before
        ), before

        # Act.
        assert hcg_client.delete_edge(edge_uuid) is True

        # The edge instance is gone (query_edges_from would still surface a
        # merely-detached edge node, so its absence proves the node was deleted).
        after = hcg_client.query_edges_from(member_uuid)
        assert not any(e.get("relation") == "IS_A" for e in after), after

        # ...but BOTH endpoint nodes survive -- crucially the type definition,
        # which keeps its other members and centroid (#149).
        member = hcg_client.get_node(member_uuid)
        parent = hcg_client.get_node(type_uuid)
        assert member is not None and member["uuid"] == member_uuid
        assert parent is not None and parent["uuid"] == type_uuid
    finally:
        # Tidy up the nodes we created (the edge is already deleted above).
        hcg_client.delete_node(member_uuid)
        hcg_client.delete_node(type_uuid)
