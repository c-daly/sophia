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

    # -- Deferred centroid update tests --

    def test_centroid_updates_deferred(self):
        """3 nodes sharing the same type -> get_node for the type called once,
        update_node for the type called once (batched after node loop)."""
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        # add_node returns distinct uuids for the 3 entity nodes + type_definition merges
        node_uuids = iter(["uuid-a", "uuid-b", "uuid-c"])
        def add_node_side_effect(**kwargs):
            if kwargs.get("node_type") == "type_definition":
                return kwargs.get("uuid", "type_entity")
            return next(node_uuids)
        mock_hcg.add_node.side_effect = add_node_side_effect

        # get_node is called during centroid flush -- return type node with existing centroid
        mock_hcg.get_node.return_value = {
            "uuid": "type_entity",
            "name": "entity",
            "type": "type_definition",
            "properties": {
                "member_count": 10,
                "centroid": [0.5] * 384,
            },
        }

        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = []  # no dedup matches
        mock_milvus.find_nearest_types.return_value = [
            {"uuid": "type_entity", "score": 0.1},
        ]

        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)
        result = processor.process(
            {
                "proposal_id": "p-batch",
                "proposed_nodes": [
                    {
                        "name": f"Node{i}",
                        "type": "entity",
                        "embedding": [0.1 * i] * 384,
                        "embedding_id": f"emb-{i}",
                        "dimension": 384,
                        "model": "all-MiniLM-L6-v2",
                        "properties": {},
                    }
                    for i in range(1, 4)
                ],
                "proposed_edges": [],
                "document_embedding": None,
                "raw_text": "",
                "source_service": "hermes",
                "confidence": 0.7,
                "metadata": {},
            }
        )

        assert len(result["stored_node_ids"]) == 3

        # get_node should be called exactly once for the type_uuid during centroid flush
        # (not 3 times -- once per node as in the old code)
        type_get_calls = [
            c for c in mock_hcg.get_node.call_args_list
            if c.args == ("type_entity",) or c.kwargs.get("uuid") == "type_entity"
        ]
        assert len(type_get_calls) == 1, (
            f"Expected 1 get_node call for type_entity, got {len(type_get_calls)}"
        )

        # update_node for the type should be called once with the final member_count
        type_update_calls = [
            c for c in mock_hcg.update_node.call_args_list
            if c.args[0] == "type_entity"
        ]
        assert len(type_update_calls) == 1, (
            f"Expected 1 update_node call for type_entity, got {len(type_update_calls)}"
        )
        # Final member_count should be 10 + 3 = 13
        assert type_update_calls[0].args[1]["member_count"] == 13

    def test_centroid_first_node_initializes(self):
        """When there is no current centroid, update_centroid should be called
        to initialize it with the first embedding."""
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        mock_hcg.add_node.return_value = "uuid-new"
        # Type node has no centroid yet
        mock_hcg.get_node.return_value = {
            "uuid": "type_entity",
            "name": "entity",
            "type": "type_definition",
            "properties": {
                "member_count": 0,
            },
        }

        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = []
        mock_milvus.find_nearest_types.return_value = [
            {"uuid": "type_entity", "score": 0.1},
        ]

        embedding = [0.3] * 384
        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)
        result = processor.process(
            {
                "proposal_id": "p-init",
                "proposed_nodes": [
                    {
                        "name": "FirstNode",
                        "type": "entity",
                        "embedding": embedding,
                        "embedding_id": "emb-init",
                        "dimension": 384,
                        "model": "all-MiniLM-L6-v2",
                        "properties": {},
                    }
                ],
                "proposed_edges": [],
                "document_embedding": None,
                "raw_text": "",
                "source_service": "hermes",
                "confidence": 0.7,
                "metadata": {},
            }
        )

        assert len(result["stored_node_ids"]) == 1
        # update_centroid should be called to initialize the centroid
        mock_milvus.update_centroid.assert_called_once_with(
            type_uuid="type_entity",
            centroid=embedding,
            model="all-MiniLM-L6-v2",
        )

    def test_centroid_failure_does_not_block(self):
        """A centroid update failure must not affect stored_node_ids."""
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        mock_hcg.add_node.return_value = "uuid-ok"
        # Make get_node raise during centroid flush
        mock_hcg.get_node.side_effect = RuntimeError("Neo4j down")

        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = []
        mock_milvus.find_nearest_types.return_value = [
            {"uuid": "type_entity", "score": 0.1},
        ]

        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)
        result = processor.process(
            {
                "proposal_id": "p-fail",
                "proposed_nodes": [
                    {
                        "name": "SafeNode",
                        "type": "entity",
                        "embedding": [0.2] * 384,
                        "embedding_id": "emb-safe",
                        "dimension": 384,
                        "model": "all-MiniLM-L6-v2",
                        "properties": {},
                    }
                ],
                "proposed_edges": [],
                "document_embedding": None,
                "raw_text": "",
                "source_service": "hermes",
                "confidence": 0.7,
                "metadata": {},
            }
        )

        # Node should still be stored even though centroid update failed
        assert result["stored_node_ids"] == ["uuid-ok"]

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
