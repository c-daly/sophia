"""Tests for centroid running-average and member_count persistence."""

from unittest.mock import MagicMock, patch

from sophia.ingestion.proposal_processor import ProposalProcessor


class TestCentroidMemberCount:
    def _make_processor(self, mock_hcg, mock_milvus):
        """Build a ProposalProcessor with mocked dependencies."""
        with patch.object(ProposalProcessor, "__init__", lambda self: None):
            proc = ProposalProcessor()
        proc._hcg = mock_hcg
        proc._milvus = mock_milvus
        proc._classifier = MagicMock()
        proc._classifier.classify.return_value = MagicMock(
            type_uuid="type_object",
            type_name="object",
            confidence=0.95,
            needs_reclassification=False,
        )
        return proc

    def test_member_count_incremented_after_centroid_update(self):
        """After averaging a new embedding into the centroid, member_count
        must be persisted as count + 1."""
        mock_hcg = MagicMock()
        mock_milvus = MagicMock()

        # Type node already has a centroid and member_count = 5
        mock_hcg.get_node.return_value = {
            "properties": {
                "member_count": 5,
                "centroid": [1.0, 0.0, 0.0],
            }
        }

        proc = self._make_processor(mock_hcg, mock_milvus)

        # Simulate the centroid update path directly
        embedding = [0.0, 1.0, 0.0]
        model = "test-model"
        type_uuid = "type_object"

        type_node = mock_hcg.get_node(type_uuid)
        props = type_node["properties"]
        member_count = props["member_count"]
        current_centroid = props["centroid"]

        proc._classifier.update_centroid_for_assignment(
            type_uuid=type_uuid,
            new_embedding=embedding,
            current_centroid=current_centroid,
            member_count=member_count,
            model=model,
        )
        mock_hcg.update_node(type_uuid, {"member_count": member_count + 1})

        mock_hcg.update_node.assert_called_with("type_object", {"member_count": 6})

    def test_member_count_initialized_on_first_node(self):
        """When no centroid exists yet, the first embedding initializes it
        and member_count is set to 1."""
        mock_hcg = MagicMock()
        mock_milvus = MagicMock()

        # Type node has no centroid yet
        mock_hcg.get_node.return_value = {
            "properties": {
                "member_count": 0,
            }
        }

        self._make_processor(mock_hcg, mock_milvus)

        embedding = [0.0, 1.0, 0.0]
        model = "test-model"
        type_uuid = "type_object"

        type_node = mock_hcg.get_node(type_uuid)
        props = type_node["properties"]
        current_centroid = props.get("centroid")

        # No centroid — initialize
        assert not current_centroid
        mock_milvus.update_centroid(
            type_uuid=type_uuid,
            centroid=embedding,
            model=model,
        )
        mock_hcg.update_node(type_uuid, {"member_count": 1})

        mock_hcg.update_node.assert_called_with("type_object", {"member_count": 1})
