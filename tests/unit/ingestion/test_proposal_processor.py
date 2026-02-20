"""Tests for ProposalProcessor -- cognitive intake of Hermes proposals."""

from unittest.mock import MagicMock


class TestProposalProcessor:
    def test_ingests_proposed_nodes(self):
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        mock_hcg.add_node.return_value = "new-uuid"
        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = []

        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)
        result = processor.process(
            {
                "proposal_id": "p1",
                "proposed_nodes": [
                    {
                        "name": "Paris",
                        "type": "GPE",
                        "embedding": [0.1] * 384,
                        "embedding_id": "emb-1",
                        "dimension": 384,
                        "model": "all-MiniLM-L6-v2",
                        "properties": {},
                    }
                ],
                "document_embedding": {
                    "embedding": [0.5] * 384,
                    "embedding_id": "doc-1",
                    "dimension": 384,
                    "model": "all-MiniLM-L6-v2",
                },
                "raw_text": "Tell me about Paris",
                "source_service": "hermes",
                "confidence": 0.7,
                "metadata": {},
            }
        )

        assert len(result["stored_node_ids"]) == 1
        # add_node called twice: once for entity, once for type_definition
        assert mock_hcg.add_node.call_count == 2
        mock_milvus.upsert_embedding.assert_called()

    def test_returns_relevant_context(self):
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        mock_hcg.get_node.return_value = {
            "uuid": "existing-uuid",
            "name": "France",
            "type": "entity",
            "properties": {"capital": "Paris"},
        }
        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = [
            {"uuid": "existing-uuid", "score": 0.15},
        ]

        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)
        result = processor.process(
            {
                "proposal_id": "p1",
                "proposed_nodes": [],
                "document_embedding": {
                    "embedding": [0.5] * 384,
                    "embedding_id": "doc-1",
                    "dimension": 384,
                    "model": "all-MiniLM-L6-v2",
                },
                "raw_text": "Tell me about France",
                "source_service": "hermes",
                "confidence": 0.7,
                "metadata": {},
            }
        )

        assert len(result["relevant_context"]) >= 1
        assert result["relevant_context"][0]["name"] == "France"

    def test_skips_creation_when_similar_entity_exists(self):
        """Sophia should not create a duplicate when a similar node exists."""
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        mock_hcg.get_node.return_value = {
            "uuid": "existing-paris",
            "name": "Paris",
            "type": "location",
            "properties": {},
        }
        mock_milvus = MagicMock()
        # Document-level search returns nothing, but entity-level returns a match
        mock_milvus.search_similar.side_effect = [
            [],  # Entity doc search
            [],  # Concept doc search
            [],  # State doc search
            [],  # Process doc search
            [
                {"uuid": "existing-paris", "score": 0.2}
            ],  # entity match (below threshold)
        ]

        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)
        result = processor.process(
            {
                "proposal_id": "p1",
                "proposed_nodes": [
                    {
                        "name": "Paris",
                        "type": "location",
                        "embedding": [0.1] * 384,
                        "embedding_id": "emb-1",
                        "dimension": 384,
                        "model": "all-MiniLM-L6-v2",
                        "properties": {},
                    }
                ],
                "document_embedding": {
                    "embedding": [0.5] * 384,
                    "embedding_id": "doc-1",
                    "dimension": 384,
                    "model": "all-MiniLM-L6-v2",
                },
                "raw_text": "Tell me about Paris",
                "source_service": "hermes",
                "confidence": 0.7,
                "metadata": {},
            }
        )

        # Node should NOT be created -- existing match found
        assert result["stored_node_ids"] == []
        mock_hcg.add_node.assert_not_called()
        # Existing node should appear in context
        assert any(
            c["node_uuid"] == "existing-paris" for c in result["relevant_context"]
        )

    def test_creates_experiment_run_when_pipeline_present(self):
        """experiment_run node created when metadata.pipeline exists."""
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        mock_hcg.add_node.return_value = "new-uuid"
        mock_hcg.add_edge.return_value = "edge-uuid"
        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = []

        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)
        result = processor.process(
            {
                "proposal_id": "p1",
                "proposed_nodes": [
                    {
                        "name": "Paris",
                        "type": "location",
                        "embedding": [0.1] * 384,
                        "embedding_id": "emb-1",
                        "dimension": 384,
                        "model": "all-MiniLM-L6-v2",
                        "properties": {"start": 0, "end": 5},
                    }
                ],
                "document_embedding": None,
                "raw_text": "Paris",
                "source_service": "hermes",
                "confidence": 0.7,
                "metadata": {
                    "pipeline": {
                        "ner_provider": "spacy",
                        "embedding_provider": "all-MiniLM-L6-v2",
                        "ner_duration_ms": 10.5,
                        "relation_duration_ms": 2.0,
                        "embedding_duration_ms": 5.0,
                        "total_duration_ms": 17.5,
                        "entity_count": 1,
                        "edge_count": 0,
                    },
                    "experiment_tags": ["baseline"],
                },
            }
        )

        assert result["experiment_run_id"] is not None
        # add_node called 3 times: entity, type_definition, experiment_run
        assert mock_hcg.add_node.call_count == 3
        # Last call should be the experiment_run node
        run_call = mock_hcg.add_node.call_args_list[2]
        assert (
            run_call.kwargs.get("node_type")
            or run_call[1].get("node_type") == "experiment_run"
        )

    def test_no_experiment_run_without_pipeline(self):
        """No experiment_run node when metadata.pipeline is absent."""
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        mock_hcg.add_node.return_value = "new-uuid"
        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = []

        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)
        result = processor.process(
            {
                "proposal_id": "p1",
                "proposed_nodes": [
                    {
                        "name": "Paris",
                        "type": "location",
                        "embedding": [0.1] * 384,
                        "embedding_id": "emb-1",
                        "dimension": 384,
                        "model": "all-MiniLM-L6-v2",
                        "properties": {},
                    }
                ],
                "document_embedding": None,
                "raw_text": "Paris",
                "source_service": "hermes",
                "confidence": 0.7,
                "metadata": {},
            }
        )

        assert result["experiment_run_id"] is None
        # add_node called twice: entity + type_definition, no experiment_run
        assert mock_hcg.add_node.call_count == 2

    def test_skips_empty_name_nodes(self):
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = []

        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)
        result = processor.process(
            {
                "proposal_id": "p1",
                "proposed_nodes": [
                    {
                        "name": "",
                        "type": "X",
                        "embedding": [0.1] * 384,
                        "embedding_id": "e1",
                        "dimension": 384,
                        "model": "m",
                        "properties": {},
                    }
                ],
                "document_embedding": None,
                "raw_text": "",
                "source_service": "hermes",
                "confidence": 0.7,
                "metadata": {},
            }
        )

        assert result["stored_node_ids"] == []
        mock_hcg.add_node.assert_not_called()

    def test_edge_neo4j_fallback_lookup(self):
        """Edges should look up unresolved names in Neo4j before skipping."""
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        mock_hcg.add_node.return_value = "new-uuid"
        mock_hcg.add_edge.return_value = "edge-uuid"
        # find_node_by_name returns a match for the target
        mock_hcg.find_node_by_name.return_value = {"uuid": "neo4j-target-uuid"}
        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = []

        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)
        result = processor.process(
            {
                "proposal_id": "p1",
                "proposed_nodes": [
                    {
                        "name": "Paris",
                        "type": "location",
                        "embedding": [0.1] * 384,
                        "embedding_id": "emb-1",
                        "dimension": 384,
                        "model": "all-MiniLM-L6-v2",
                        "properties": {},
                    }
                ],
                "proposed_edges": [
                    {
                        "source_name": "Paris",
                        "target_name": "France",
                        "relation": "LOCATED_IN",
                        "confidence": 0.8,
                    }
                ],
                "document_embedding": None,
                "raw_text": "Paris is in France",
                "source_service": "hermes",
                "confidence": 0.7,
                "metadata": {},
            }
        )

        # Should have resolved "France" via Neo4j fallback
        assert len(result["stored_edge_ids"]) == 1
        mock_hcg.find_node_by_name.assert_called_with("France")

    def test_edge_properties_cannot_overwrite_reserved_keys(self):
        """Untrusted properties must not overwrite uuid, source, target, relation."""
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        mock_hcg.add_node.return_value = "new-uuid"
        mock_hcg.add_edge.return_value = "edge-uuid"
        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = []

        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)
        result = processor.process(
            {
                "proposal_id": "p1",
                "proposed_nodes": [
                    {
                        "name": "A",
                        "type": "entity",
                        "embedding": [0.1] * 384,
                        "embedding_id": "e1",
                        "dimension": 384,
                        "model": "m",
                        "properties": {},
                    },
                    {
                        "name": "B",
                        "type": "entity",
                        "embedding": [0.2] * 384,
                        "embedding_id": "e2",
                        "dimension": 384,
                        "model": "m",
                        "properties": {},
                    },
                ],
                "proposed_edges": [
                    {
                        "source_name": "A",
                        "target_name": "B",
                        "relation": "KNOWS",
                        "confidence": 0.9,
                        "properties": {
                            "uuid": "EVIL",
                            "source": "EVIL",
                            "target": "EVIL",
                            "relation": "EVIL",
                            "safe_key": "safe_value",
                        },
                    }
                ],
                "document_embedding": None,
                "raw_text": "",
                "source_service": "hermes",
                "confidence": 0.7,
                "metadata": {},
            }
        )

        assert len(result["stored_edge_ids"]) == 1
        # Verify properties passed to add_edge do not contain reserved keys
        # Find the KNOWS edge call
        knows_call = None
        for c in mock_hcg.add_edge.call_args_list:
            if c.kwargs.get("relation") == "KNOWS" or (
                len(c.args) > 2 and c.args[2] == "KNOWS"
            ):
                knows_call = c
                break
        assert knows_call is not None
        props = knows_call.kwargs.get("properties", {})
        assert "uuid" not in props
        assert "source" not in props
        assert "target" not in props
        assert "relation" not in props
        assert props.get("safe_key") == "safe_value"
