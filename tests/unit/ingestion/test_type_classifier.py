"""Tests for Sophia's embedding-based type classifier."""

from unittest.mock import MagicMock

from sophia.ingestion.type_classifier import TypeClassifier


class TestTypeClassifier:
    """Test suite for TypeClassifier."""

    def _make_classifier(self, milvus=None, hcg=None):
        if hcg is None:
            hcg = MagicMock()
            # Type-defs carry a clean `name`; default it to the uuid minus
            # "type_" so existing assertions hold while still exercising the
            # get_node lookup the classifier now performs.
            hcg.get_node.side_effect = lambda uuid: {"name": uuid.removeprefix("type_")}
        return TypeClassifier(
            milvus=milvus or MagicMock(),
            hcg=hcg,
        )

    def test_classify_high_confidence(self):
        """Close to a centroid => high confidence assignment."""
        mock_milvus = MagicMock()
        mock_milvus.find_nearest_types.return_value = [
            {"uuid": "type_location", "score": 0.1},
            {"uuid": "type_concept", "score": 0.9},
        ]
        classifier = self._make_classifier(milvus=mock_milvus)

        result = classifier.classify([0.1] * 384)

        assert result.type_uuid == "type_location"
        assert result.type_name == "location"
        assert result.confidence > 0.8
        assert result.needs_reclassification is False

    def test_classify_uses_type_def_clean_name_not_uuid(self):
        """A minted type's hex-suffixed uuid must not leak into the name; the
        type-definition's clean `name` is used (#505 suffixed-name fix)."""
        mock_milvus = MagicMock()
        mock_milvus.find_nearest_types.return_value = [
            {"uuid": "type_organism_7ea1d37c", "score": 0.05},
        ]
        mock_hcg = MagicMock()
        mock_hcg.get_node.return_value = {"name": "organism"}
        classifier = self._make_classifier(milvus=mock_milvus, hcg=mock_hcg)

        result = classifier.classify([0.1] * 384)

        assert result.type_uuid == "type_organism_7ea1d37c"
        assert result.type_name == "organism"  # NOT "organism_7ea1d37c"

    def test_classify_falls_back_to_stripped_uuid_when_node_missing(self):
        """If the type-def node can't be read, fall back to the stripped uuid."""
        mock_milvus = MagicMock()
        mock_milvus.find_nearest_types.return_value = [
            {"uuid": "type_object", "score": 0.05},
        ]
        mock_hcg = MagicMock()
        mock_hcg.get_node.return_value = None
        classifier = self._make_classifier(milvus=mock_milvus, hcg=mock_hcg)

        result = classifier.classify([0.1] * 384)

        assert result.type_name == "object"

    def test_classify_low_confidence_ambiguous(self):
        """Between two centroids => low confidence, flagged."""
        mock_milvus = MagicMock()
        mock_milvus.find_nearest_types.return_value = [
            {"uuid": "type_location", "score": 0.45},
            {"uuid": "type_concept", "score": 0.50},
        ]
        classifier = self._make_classifier(milvus=mock_milvus)

        result = classifier.classify([0.1] * 384)

        assert result.type_uuid == "type_location"
        assert result.confidence < 0.5
        assert result.needs_reclassification is True

    def test_classify_no_centroids(self):
        """No centroids in Milvus => fallback to 'entity' with zero confidence."""
        mock_milvus = MagicMock()
        mock_milvus.find_nearest_types.return_value = []
        classifier = self._make_classifier(milvus=mock_milvus)

        result = classifier.classify([0.1] * 384)

        assert result.type_uuid == "type_entity"
        assert result.type_name == "entity"
        assert result.confidence == 0.0
        assert result.needs_reclassification is True

    def test_update_centroid_incremental(self):
        """Centroid updates incrementally after node assignment."""
        mock_milvus = MagicMock()
        mock_hcg = MagicMock()
        mock_hcg.get_node.return_value = {
            "uuid": "type_location",
            "properties": {"member_count": 10},
        }
        classifier = self._make_classifier(milvus=mock_milvus, hcg=mock_hcg)

        classifier.update_centroid_for_assignment(
            type_uuid="type_location",
            new_embedding=[1.0] * 384,
            current_centroid=[0.0] * 384,
            member_count=10,
            model="all-MiniLM-L6-v2",
        )

        mock_milvus.update_centroid.assert_called_once()
        call_args = mock_milvus.update_centroid.call_args
        new_centroid = (
            call_args.kwargs.get("centroid")
            or call_args[1].get("centroid")
            or call_args[0][1]
        )
        # (0.0 * 10 + 1.0) / 11 ≈ 0.0909
        assert abs(new_centroid[0] - (1.0 / 11)) < 0.001
