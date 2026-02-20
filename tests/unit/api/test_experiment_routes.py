"""Tests for experiment tracking routes."""

import pytest
from unittest.mock import MagicMock, patch


pytestmark = pytest.mark.unit


class TestListExperiments:
    def test_returns_all_runs(self):
        from sophia.api.experiment_routes import list_experiments
        import asyncio

        mock_hcg = MagicMock()
        mock_hcg.list_all_nodes.return_value = [
            {
                "uuid": "run-1",
                "name": "run_abc",
                "type": "experiment_run",
                "properties": {
                    "ner_provider": "spacy",
                    "embedding_provider": "all-MiniLM-L6-v2",
                    "total_duration_ms": 20.0,
                    "entity_count": 3,
                    "edge_count": 1,
                    "experiment_tags": ["baseline"],
                },
            },
        ]

        with patch("sophia.api.experiment_routes._get_hcg", return_value=mock_hcg):
            result = asyncio.get_event_loop().run_until_complete(
                list_experiments(
                    ner_provider=None, embedding_provider=None, tag=None, limit=50
                )
            )

        assert len(result) == 1
        assert result[0]["uuid"] == "run-1"

    def test_filters_by_ner_provider(self):
        from sophia.api.experiment_routes import list_experiments
        import asyncio

        mock_hcg = MagicMock()
        mock_hcg.list_all_nodes.return_value = [
            {
                "uuid": "run-1",
                "name": "run_a",
                "type": "experiment_run",
                "properties": {"ner_provider": "spacy", "embedding_provider": "m1"},
            },
            {
                "uuid": "run-2",
                "name": "run_b",
                "type": "experiment_run",
                "properties": {"ner_provider": "openai", "embedding_provider": "m1"},
            },
        ]

        with patch("sophia.api.experiment_routes._get_hcg", return_value=mock_hcg):
            result = asyncio.get_event_loop().run_until_complete(
                list_experiments(
                    ner_provider="spacy", embedding_provider=None, tag=None, limit=50
                )
            )

        assert len(result) == 1
        assert result[0]["uuid"] == "run-1"

    def test_filters_by_tag(self):
        from sophia.api.experiment_routes import list_experiments
        import asyncio

        mock_hcg = MagicMock()
        mock_hcg.list_all_nodes.return_value = [
            {
                "uuid": "run-1",
                "name": "run_a",
                "type": "experiment_run",
                "properties": {"experiment_tags": ["baseline"]},
            },
            {
                "uuid": "run-2",
                "name": "run_b",
                "type": "experiment_run",
                "properties": {"experiment_tags": ["v2-ner"]},
            },
        ]

        with patch("sophia.api.experiment_routes._get_hcg", return_value=mock_hcg):
            result = asyncio.get_event_loop().run_until_complete(
                list_experiments(
                    ner_provider=None, embedding_provider=None, tag="v2-ner", limit=50
                )
            )

        assert len(result) == 1
        assert result[0]["uuid"] == "run-2"


class TestCompareExperiments:
    def test_groups_by_ner_provider(self):
        from sophia.api.experiment_routes import compare_experiments
        import asyncio

        mock_hcg = MagicMock()
        mock_hcg.list_all_nodes.return_value = [
            {
                "uuid": "r1",
                "name": "r1",
                "type": "experiment_run",
                "properties": {
                    "ner_provider": "spacy",
                    "total_duration_ms": 20.0,
                    "entity_count": 3,
                    "edge_count": 1,
                    "ner_duration_ms": 10.0,
                    "embedding_duration_ms": 8.0,
                },
            },
            {
                "uuid": "r2",
                "name": "r2",
                "type": "experiment_run",
                "properties": {
                    "ner_provider": "spacy",
                    "total_duration_ms": 30.0,
                    "entity_count": 5,
                    "edge_count": 2,
                    "ner_duration_ms": 15.0,
                    "embedding_duration_ms": 12.0,
                },
            },
            {
                "uuid": "r3",
                "name": "r3",
                "type": "experiment_run",
                "properties": {
                    "ner_provider": "openai",
                    "total_duration_ms": 100.0,
                    "entity_count": 10,
                    "edge_count": 5,
                    "ner_duration_ms": 80.0,
                    "embedding_duration_ms": 15.0,
                },
            },
        ]

        with patch("sophia.api.experiment_routes._get_hcg", return_value=mock_hcg):
            result = asyncio.get_event_loop().run_until_complete(
                compare_experiments(group_by="ner_provider", limit=100)
            )

        assert len(result) == 2
        spacy_group = next(r for r in result if r["provider"] == "spacy")
        assert spacy_group["run_count"] == 2
        assert spacy_group["avg_duration_ms"] == 25.0


class TestGetExperimentEntities:
    def test_returns_produced_entities(self):
        from sophia.api.experiment_routes import get_experiment_entities
        import asyncio

        mock_hcg = MagicMock()
        mock_hcg.query_edges_from.return_value = [
            {"relation": "PRODUCED", "target_uuid": "node-1"},
            {"relation": "IS_A", "target_uuid": "type-def"},
        ]
        mock_hcg.get_node.return_value = {
            "uuid": "node-1",
            "name": "Paris",
            "type": "location",
        }

        with patch("sophia.api.experiment_routes._get_hcg", return_value=mock_hcg):
            result = asyncio.get_event_loop().run_until_complete(
                get_experiment_entities(run_id="run-1")
            )

        assert len(result) == 1
        assert result[0]["name"] == "Paris"
