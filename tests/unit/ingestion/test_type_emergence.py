"""Tests for type emergence detection via variance monitoring and k-means."""

from unittest.mock import MagicMock

from sophia.ingestion.type_emergence import TypeEmergenceDetector


class TestTypeEmergenceDetector:

    def test_no_split_when_variance_low(self):
        """Low variance type should not trigger a split."""
        mock_milvus = MagicMock()
        mock_hcg = MagicMock()
        mock_hcg.get_node.return_value = {
            "uuid": "type_location",
            "properties": {"member_count": 20, "centroid_variance": 0.1},
        }

        detector = TypeEmergenceDetector(
            milvus=mock_milvus,
            hcg=mock_hcg,
            variance_threshold=0.5,
        )
        result = detector.check_type("type_location")
        assert result is None  # no split needed

    def test_split_detected_when_variance_high(self):
        """High variance with two clear sub-clusters triggers a split."""
        mock_milvus = MagicMock()
        mock_hcg = MagicMock()
        mock_hcg.get_node.return_value = {
            "uuid": "type_state",
            "properties": {"member_count": 30, "centroid_variance": 0.8},
        }

        # Return embeddings for k-means to split
        cluster_a = [[0.0] * 384] * 15
        cluster_b = [[1.0] * 384] * 15
        mock_milvus.get_all_embeddings.return_value = cluster_a + cluster_b

        detector = TypeEmergenceDetector(
            milvus=mock_milvus,
            hcg=mock_hcg,
            variance_threshold=0.5,
        )
        result = detector.check_type("type_state")

        assert result is not None
        assert len(result.sub_clusters) == 2
        assert result.should_split is True
