"""Tests for node deduplication on name+type."""

from unittest.mock import MagicMock, patch

from sophia.hcg_client.client import HCGClient


class TestNodeDedup:

    def _make_client(self):
        """Build an HCGClient with mocked Neo4j driver."""
        with patch.object(HCGClient, "__init__", lambda self: None):
            client = HCGClient()
        client._driver = MagicMock()
        client._validator = MagicMock()
        client._validator.validate_node.return_value = (True, [])
        client._database = "neo4j"
        return client

    def test_reuses_uuid_for_existing_name_type(self):
        """When add_node is called without a uuid and a node with the same
        name+type already exists, the existing uuid should be reused."""
        client = self._make_client()

        # Mock _execute_query to return existing node on lookup, then
        # return uuid on MERGE
        existing_uuid = "existing-uuid-123"
        client._execute_query = MagicMock(
            side_effect=[
                [{"uuid": existing_uuid}],  # lookup query
                [{"uuid": existing_uuid}],  # MERGE query
            ]
        )

        result = client.add_node(name="Ireland", node_type="location")

        assert result == existing_uuid
        # First call is the lookup
        lookup_call = client._execute_query.call_args_list[0]
        assert "MATCH" in lookup_call[0][0]
        assert lookup_call[0][1]["name"] == "Ireland"
        assert lookup_call[0][1]["type"] == "location"

    def test_generates_new_uuid_when_no_existing_node(self):
        """When no node with name+type exists, a new uuid is generated."""
        client = self._make_client()

        client._execute_query = MagicMock(
            side_effect=[
                [],  # lookup returns nothing
                [{"uuid": "new-uuid"}],  # MERGE query
            ]
        )

        result = client.add_node(name="NewPlace", node_type="location")

        assert result is not None
        # First call is the lookup, second is MERGE
        assert len(client._execute_query.call_args_list) == 2

    def test_explicit_uuid_skips_lookup(self):
        """When uuid is explicitly provided, no lookup is performed."""
        client = self._make_client()

        client._execute_query = MagicMock(return_value=[{"uuid": "explicit-uuid"}])

        result = client.add_node(
            name="Ireland", node_type="location", uuid="explicit-uuid"
        )

        assert result == "explicit-uuid"
        # Only one call — no lookup, just MERGE
        assert client._execute_query.call_count == 1
