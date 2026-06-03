"""Tests for node identity: uuid minting is embedding-based, never name-based.

#148: Sophia is non-linguistic. ``add_node`` must NOT treat name+type as an
identity key -- two entities that happen to share a name+type are distinct
nodes unless an explicit uuid says otherwise. Name-based deduplication used to
live here (a name+type MATCH that reused the existing uuid); it was removed
because the literal name string is *text used as a key*, and entity identity is
decided in embedding space by the upstream resolver, not by string equality.
Homonyms collapsing is an accepted edge case of the embedding resolver, not of
a name lookup. These tests pin the new contract: omitting the uuid always mints
a fresh uuid4 with no name lookup.
"""

from unittest.mock import MagicMock, patch

from sophia.hcg_client.client import HCGClient


class TestNodeIdentity:
    def _make_client(self):
        """Build an HCGClient with mocked Neo4j driver."""
        with patch.object(HCGClient, "__init__", lambda self: None):
            client = HCGClient()
        client._driver = MagicMock()
        client._validator = MagicMock()
        client._validator.validate_node.return_value = (True, [])
        client._database = "neo4j"
        return client

    def test_omitting_uuid_mints_fresh_uuid_without_name_lookup(self):
        """Without a uuid, add_node mints a fresh uuid4 and performs no name
        lookup -- a single MERGE call, never a name+type MATCH."""
        client = self._make_client()
        client._execute_query = MagicMock(return_value=[{"uuid": "ignored"}])

        client.add_node(name="Ireland", node_type="location")

        # Exactly one query: the MERGE. No name+type lookup precedes it.
        assert client._execute_query.call_count == 1
        query = client._execute_query.call_args_list[0][0][0]
        assert "MERGE" in query
        params = client._execute_query.call_args_list[0][0][1]
        # A real uuid4 was generated for the node, not derived from the name.
        assert len(params["uuid"]) == 36
        assert params["uuid"].count("-") == 4

    def test_same_name_and_type_get_distinct_uuids(self):
        """Two nodes sharing name+type are distinct identities (#148): the
        literal name is never used as a merge key. Each omitted-uuid call mints
        its own uuid4 -- one MERGE per call, distinct uuids."""
        client = self._make_client()
        client._execute_query = MagicMock(return_value=[{"uuid": "ignored"}])

        client.add_node(name="Mercury", node_type="entity")
        client.add_node(name="Mercury", node_type="entity")

        # One MERGE per call -- no preceding name lookup.
        assert client._execute_query.call_count == 2
        first = client._execute_query.call_args_list[0][0][1]["uuid"]
        second = client._execute_query.call_args_list[1][0][1]["uuid"]
        assert first != second

    def test_explicit_uuid_skips_lookup(self):
        """An explicit uuid is used verbatim with a single MERGE call."""
        client = self._make_client()
        client._execute_query = MagicMock(return_value=[{"uuid": "explicit-uuid"}])

        result = client.add_node(
            name="Ireland", node_type="location", uuid="explicit-uuid"
        )

        assert result == "explicit-uuid"
        assert client._execute_query.call_count == 1
        params = client._execute_query.call_args_list[0][0][1]
        assert params["uuid"] == "explicit-uuid"
