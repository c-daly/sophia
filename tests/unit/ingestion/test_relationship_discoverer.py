"""Tests for cross-cluster relationship discovery."""

from unittest.mock import MagicMock

from sophia.ingestion.relationship_discoverer import RelationshipDiscoverer


class TestRelationshipDiscoverer:
    def test_find_cross_cluster_candidates(self):
        """Nodes close to a node but in different type clusters are candidates."""
        mock_milvus = MagicMock()
        # Searching non-own collections returns a close node
        mock_milvus.search_similar.return_value = [
            {"uuid": "node-in-other-cluster", "score": 0.3},
        ]
        mock_milvus.find_nearest_types.return_value = [
            {"uuid": "type_process", "score": 0.8},  # far from own centroid
        ]

        discoverer = RelationshipDiscoverer(milvus=mock_milvus)
        candidates = discoverer.find_candidates(
            embedding=[0.1] * 384,
            own_type="Entity",
            top_k=5,
        )

        assert len(candidates) >= 1
        assert candidates[0]["uuid"] == "node-in-other-cluster"

    def test_skips_own_type_cluster(self):
        """Nodes from the query's own type cluster are excluded."""
        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = [
            {"uuid": "same-cluster-node", "score": 0.1},
        ]

        discoverer = RelationshipDiscoverer(milvus=mock_milvus)
        discoverer.find_candidates(
            embedding=[0.1] * 384,
            own_type="Entity",
            top_k=5,
        )

        # Entity was skipped, so only Concept/State/Process were searched
        called_types = [
            c.kwargs["node_type"] for c in mock_milvus.search_similar.call_args_list
        ]
        assert "Entity" not in called_types
        assert len(called_types) == 3
