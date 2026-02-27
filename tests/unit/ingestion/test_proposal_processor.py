"""Tests for ProposalProcessor -- cognitive intake of Hermes proposals."""

from unittest.mock import MagicMock, call


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
        # Embeddings should be batched, not individually upserted
        mock_milvus.upsert_embedding.assert_not_called()
        mock_milvus.batch_upsert_embeddings.assert_called()

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


class TestBatchEmbeddings:
    """Tests for batch embedding upsert behavior in ProposalProcessor."""

    def _make_processor(self, mock_hcg=None, mock_milvus=None):
        """Helper to create a processor with sensible mock defaults."""
        if mock_hcg is None:
            mock_hcg = MagicMock()
        if mock_milvus is None:
            mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = []
        mock_milvus.find_nearest_types.return_value = [
            {"uuid": "type_entity", "score": 0.1},
        ]
        from sophia.ingestion.proposal_processor import ProposalProcessor

        return ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus), mock_hcg, mock_milvus

    def _make_node(self, name, node_type="entity", embedding=None, model="all-MiniLM-L6-v2"):
        """Helper to create a proposed node dict."""
        return {
            "name": name,
            "type": node_type,
            "embedding": embedding or [0.1] * 384,
            "embedding_id": "emb-" + name,
            "dimension": 384,
            "model": model,
            "properties": {},
        }

    def _make_proposal(self, nodes=None, edges=None, doc_embedding=None):
        """Helper to create a proposal dict."""
        return {
            "proposal_id": "test-batch",
            "proposed_nodes": nodes or [],
            "proposed_edges": edges or [],
            "document_embedding": doc_embedding,
            "raw_text": "test text",
            "source_service": "hermes",
            "confidence": 0.7,
            "metadata": {},
        }

    def test_node_embeddings_batched_not_individual(self):
        """Individual upsert_embedding should never be called; batch_upsert_embeddings used instead."""
        mock_hcg = MagicMock()
        mock_hcg.add_node.return_value = "uuid-1"
        processor, _, mock_milvus = self._make_processor(mock_hcg=mock_hcg)

        proposal = self._make_proposal(nodes=[self._make_node("Alpha")])
        processor.process(proposal)

        mock_milvus.upsert_embedding.assert_not_called()
        mock_milvus.batch_upsert_embeddings.assert_called()

    def test_multiple_nodes_different_collections_get_separate_batches(self):
        """Nodes classified into different collections should produce separate batch calls."""
        mock_hcg = MagicMock()
        mock_hcg.add_node.side_effect = ["uuid-entity", "type-def-1", "uuid-concept", "type-def-2"]
        mock_hcg.get_node.return_value = None

        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = []
        mock_milvus.find_nearest_types.side_effect = [
            [{"uuid": "type_entity", "score": 0.1}],
            [{"uuid": "type_concept", "score": 0.1}],
        ]

        from sophia.ingestion.proposal_processor import ProposalProcessor

        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)

        proposal = self._make_proposal(nodes=[
            self._make_node("Alpha", node_type="entity"),
            self._make_node("Beta", node_type="concept"),
        ])
        processor.process(proposal)

        batch_calls = mock_milvus.batch_upsert_embeddings.call_args_list
        called_collections = set()
        for c in batch_calls:
            nt = c.kwargs.get("node_type")
            if nt is None and c.args:
                nt = c.args[0]
            called_collections.add(nt)
        assert "Entity" in called_collections
        assert "Concept" in called_collections

    def test_edge_embeddings_batched_to_edge_collection(self):
        """Edge embeddings should be batched into the Edge collection."""
        mock_hcg = MagicMock()
        mock_hcg.add_node.side_effect = ["uuid-a", "type-def-a", "uuid-b", "type-def-b"]
        mock_hcg.add_edge.return_value = "edge-uuid-1"
        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = []
        mock_milvus.find_nearest_types.return_value = [
            {"uuid": "type_entity", "score": 0.1},
        ]

        from sophia.ingestion.proposal_processor import ProposalProcessor

        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)

        proposal = self._make_proposal(
            nodes=[
                self._make_node("A"),
                self._make_node("B"),
            ],
            edges=[
                {
                    "source_name": "A",
                    "target_name": "B",
                    "relation": "RELATED_TO",
                    "confidence": 0.8,
                    "embedding": [0.3] * 384,
                    "model": "all-MiniLM-L6-v2",
                },
            ],
        )
        processor.process(proposal)

        mock_milvus.upsert_embedding.assert_not_called()

        batch_calls = mock_milvus.batch_upsert_embeddings.call_args_list
        edge_calls = [c for c in batch_calls
                      if (c.kwargs.get("node_type") or (c.args[0] if c.args else None)) == "Edge"]
        assert len(edge_calls) == 1
        edge_batch = edge_calls[0].kwargs.get("embeddings")
        if edge_batch is None and len(edge_calls[0].args) > 1:
            edge_batch = edge_calls[0].args[1]
        assert len(edge_batch) == 1
        assert edge_batch[0]["uuid"] == "edge-uuid-1"
        assert edge_batch[0]["model"] == "all-MiniLM-L6-v2"

    def test_batch_upsert_failure_still_returns_results(self):
        """If batch_upsert_embeddings raises, process() should still return stored IDs."""
        mock_hcg = MagicMock()
        mock_hcg.add_node.return_value = "uuid-1"
        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = []
        mock_milvus.find_nearest_types.return_value = [
            {"uuid": "type_entity", "score": 0.1},
        ]
        mock_milvus.batch_upsert_embeddings.side_effect = RuntimeError("Milvus connection lost")

        from sophia.ingestion.proposal_processor import ProposalProcessor

        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)

        proposal = self._make_proposal(nodes=[self._make_node("Gamma")])
        result = processor.process(proposal)

        assert "uuid-1" in result["stored_node_ids"]
        assert "stored_edge_ids" in result

    def test_no_embeddings_skips_batch_call(self):
        """If no nodes have embeddings, batch_upsert_embeddings should not be called."""
        mock_hcg = MagicMock()
        mock_hcg.add_node.return_value = "uuid-1"
        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = []

        from sophia.ingestion.proposal_processor import ProposalProcessor

        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)

        node_no_embedding = {
            "name": "NoEmbed",
            "type": "entity",
            "embedding": None,
            "embedding_id": "e1",
            "dimension": 384,
            "model": "m",
            "properties": {},
        }
        proposal = self._make_proposal(nodes=[node_no_embedding])
        processor.process(proposal)

        mock_milvus.batch_upsert_embeddings.assert_not_called()

    def test_batch_contains_correct_embedding_data(self):
        """Verify the batch payload has the right uuid, embedding, and model."""
        mock_hcg = MagicMock()
        mock_hcg.add_node.side_effect = ["uuid-X", "type-def-X"]
        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = []
        mock_milvus.find_nearest_types.return_value = [
            {"uuid": "type_entity", "score": 0.1},
        ]

        from sophia.ingestion.proposal_processor import ProposalProcessor

        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)

        test_embedding = [0.42] * 384
        proposal = self._make_proposal(nodes=[
            self._make_node("X", embedding=test_embedding, model="test-model"),
        ])
        processor.process(proposal)

        batch_calls = mock_milvus.batch_upsert_embeddings.call_args_list
        assert len(batch_calls) >= 1
        found = False
        for c in batch_calls:
            node_type = c.kwargs.get("node_type")
            if node_type is None and c.args:
                node_type = c.args[0]
            if node_type == "Entity":
                embeddings_arg = c.kwargs.get("embeddings")
                if embeddings_arg is None and len(c.args) > 1:
                    embeddings_arg = c.args[1]
                assert len(embeddings_arg) == 1
                assert embeddings_arg[0]["uuid"] == "uuid-X"
                assert embeddings_arg[0]["embedding"] == test_embedding
                assert embeddings_arg[0]["model"] == "test-model"
                found = True
                break
        assert found, "No batch_upsert_embeddings call found for Entity collection"
