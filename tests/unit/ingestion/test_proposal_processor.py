"""Tests for ProposalProcessor -- cognitive intake of Hermes proposals."""

from unittest.mock import MagicMock


class TestProposalProcessor:
    def test_ingests_proposed_nodes(self):
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        mock_hcg.add_node.return_value = "new-uuid"
        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = []
        mock_milvus.find_nearest_types.return_value = [
            {"uuid": "type_entity", "score": 0.1},
        ]

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
        mock_hcg.get_nodes_batch.return_value = [
            {
                "uuid": "existing-uuid",
                "name": "France",
                "type": "entity",
                "properties": {"capital": "Paris"},
            },
        ]
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
        mock_hcg.get_nodes_batch.return_value = []
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
        mock_milvus.find_nearest_types.return_value = [
            {"uuid": "type_location", "score": 0.1},
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
        mock_milvus.find_nearest_types.return_value = [
            {"uuid": "type_location", "score": 0.1},
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
        mock_milvus.find_nearest_types.return_value = [
            {"uuid": "type_entity", "score": 0.1},
        ]

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

    def test_process_uses_type_classifier(self):
        """Sophia classifies node type via centroid, ignoring Hermes type hint."""
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        mock_hcg.add_node.return_value = "uuid-1"
        mock_hcg.get_node.return_value = None

        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = []  # no dedup match
        mock_milvus.find_nearest_types.return_value = [
            {"uuid": "type_location", "score": 0.1},
        ]

        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)
        proposal = {
            "proposal_id": "test-1",
            "source_service": "hermes",
            "confidence": 0.8,
            "raw_text": "Dublin is a city",
            "proposed_nodes": [
                {
                    "name": "Dublin",
                    "type": "state",  # Hermes says state -- should be ignored
                    "embedding": [0.1] * 384,
                    "embedding_id": "emb-1",
                    "dimension": 384,
                    "model": "all-MiniLM-L6-v2",
                    "properties": {"start": 0, "end": 6},
                }
            ],
            "proposed_edges": [],
            "document_embedding": {
                "embedding": [0.2] * 384,
                "embedding_id": "doc-1",
                "dimension": 384,
                "model": "all-MiniLM-L6-v2",
            },
        }

        processor.process(proposal)

        # Verify add_node was called with Sophia's classification, not Hermes's
        add_node_call = mock_hcg.add_node.call_args_list[0]
        assert add_node_call.kwargs.get("node_type") == "location"  # NOT "state"

    # -- Parallel context search + batch hydration tests --

    def test_context_search_uses_batch_hydration(self):
        """Context phase must call get_nodes_batch instead of per-match get_node."""
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        mock_hcg.get_nodes_batch.return_value = [
            {"uuid": "uuid-1", "name": "Alpha", "type": "entity", "properties": {}},
            {"uuid": "uuid-2", "name": "Beta", "type": "concept", "properties": {}},
        ]
        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = [
            {"uuid": "uuid-1", "score": 0.1},
            {"uuid": "uuid-2", "score": 0.2},
        ]

        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)
        result = processor.process(
            {
                "proposal_id": "p-batch-hydrate",
                "proposed_nodes": [],
                "document_embedding": {
                    "embedding": [0.5] * 384,
                    "embedding_id": "doc-1",
                    "dimension": 384,
                    "model": "all-MiniLM-L6-v2",
                },
                "raw_text": "test",
                "source_service": "hermes",
                "confidence": 0.7,
                "metadata": {},
            }
        )

        # get_nodes_batch should have been called (not individual get_node calls
        # for context hydration)
        mock_hcg.get_nodes_batch.assert_called_once()
        # The batch call should include all unique match uuids
        call_args = mock_hcg.get_nodes_batch.call_args[0][0]
        assert "uuid-1" in call_args
        assert "uuid-2" in call_args
        # Both nodes should appear in context
        ctx_uuids = {c["node_uuid"] for c in result["relevant_context"]}
        assert "uuid-1" in ctx_uuids
        assert "uuid-2" in ctx_uuids

    def test_context_search_parallel(self):
        """All 4 searchable collections must be searched and all results appear."""
        from sophia.ingestion.proposal_processor import (
            ProposalProcessor,
            SEARCHABLE_COLLECTIONS,
        )

        mock_hcg = MagicMock()
        # Each collection returns 1 unique match
        nodes = [
            {"uuid": f"uuid-{i}", "name": f"Node{i}", "type": coll.lower(), "properties": {}}
            for i, coll in enumerate(SEARCHABLE_COLLECTIONS)
        ]
        mock_hcg.get_nodes_batch.return_value = nodes

        mock_milvus = MagicMock()
        # Each collection search returns 1 match with a unique uuid
        mock_milvus.search_similar.side_effect = [
            [{"uuid": f"uuid-{i}", "score": 0.1 * (i + 1)}]
            for i in range(len(SEARCHABLE_COLLECTIONS))
        ]

        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)
        result = processor.process(
            {
                "proposal_id": "p-parallel",
                "proposed_nodes": [],
                "document_embedding": {
                    "embedding": [0.5] * 384,
                    "embedding_id": "doc-1",
                    "dimension": 384,
                    "model": "all-MiniLM-L6-v2",
                },
                "raw_text": "test parallel",
                "source_service": "hermes",
                "confidence": 0.7,
                "metadata": {},
            }
        )

        # All 4 collections should have been searched
        assert mock_milvus.search_similar.call_count == len(SEARCHABLE_COLLECTIONS)
        # All 4 results should appear in context
        ctx_uuids = {c["node_uuid"] for c in result["relevant_context"]}
        for i in range(len(SEARCHABLE_COLLECTIONS)):
            assert f"uuid-{i}" in ctx_uuids, (
                f"uuid-{i} missing from context: {ctx_uuids}"
            )

    def test_context_search_handles_collection_failure(self):
        """If one collection search raises, others still return results."""
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        mock_hcg.get_nodes_batch.return_value = [
            {"uuid": "uuid-ok", "name": "OkNode", "type": "concept", "properties": {}},
        ]

        mock_milvus = MagicMock()
        # Entity search raises, Concept returns a match, State/Process return empty
        mock_milvus.search_similar.side_effect = [
            RuntimeError("Milvus timeout"),  # Entity fails
            [{"uuid": "uuid-ok", "score": 0.1}],  # Concept succeeds
            [],  # State empty
            [],  # Process empty
        ]

        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)
        result = processor.process(
            {
                "proposal_id": "p-fail-one",
                "proposed_nodes": [],
                "document_embedding": {
                    "embedding": [0.5] * 384,
                    "embedding_id": "doc-1",
                    "dimension": 384,
                    "model": "all-MiniLM-L6-v2",
                },
                "raw_text": "test failure",
                "source_service": "hermes",
                "confidence": 0.7,
                "metadata": {},
            }
        )

        # Despite Entity collection failure, Concept result should appear
        assert len(result["relevant_context"]) == 1
        assert result["relevant_context"][0]["node_uuid"] == "uuid-ok"

    def test_context_search_empty_matches(self):
        """When no matches are found, get_nodes_batch should not be called."""
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = []  # no matches in any collection

        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)
        result = processor.process(
            {
                "proposal_id": "p-empty",
                "proposed_nodes": [],
                "document_embedding": {
                    "embedding": [0.5] * 384,
                    "embedding_id": "doc-1",
                    "dimension": 384,
                    "model": "all-MiniLM-L6-v2",
                },
                "raw_text": "test empty",
                "source_service": "hermes",
                "confidence": 0.7,
                "metadata": {},
            }
        )

        # No matches -> get_nodes_batch should not be called
        mock_hcg.get_nodes_batch.assert_not_called()
        assert result["relevant_context"] == []
