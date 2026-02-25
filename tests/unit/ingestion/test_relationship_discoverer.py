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

    def test_filter_boundary_nodes(self):
        """Nodes closer to their own centroid than to query are filtered out."""
        mock_milvus = MagicMock()
        # Node is close to query but also very close to its own centroid
        mock_milvus.search_similar.return_value = [
            {"uuid": "boundary-node", "score": 0.4},
        ]
        mock_milvus.find_nearest_types.return_value = [
            {"uuid": "type_concept", "score": 0.1},  # very close to own centroid
        ]

        discoverer = RelationshipDiscoverer(milvus=mock_milvus)
        candidates = discoverer.find_candidates(
            embedding=[0.1] * 384,
            own_type="Entity",
            top_k=5,
        )

        assert len(candidates) == 0  # filtered out
