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
        # Embeddings should be batched, not individually upserted
        mock_milvus.upsert_embedding.assert_not_called()
        mock_milvus.batch_upsert_embeddings.assert_called()

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

    def test_same_entity_twice_in_one_ingest_yields_one_node(self):
        """A repeated entity within a single ingest must not double the node count.

        Acceptance criterion for #148: when the same entity is proposed twice in
        one proposal -- with (near-)identical embeddings -- exactly one graph
        node is created. The Milvus dedup search cannot catch this because the
        first node's embedding is only flushed to the index AFTER the node loop
        completes, so ``search_similar`` returns no match for the second mention
        (modeled here as []). Dedup must instead compare the second mention's
        embedding against the in-process embeddings of nodes already created in
        this same ingest (L2 distance below ENTITY_MATCH_THRESHOLD => reuse).
        Identity is embedding-based, never name-based: Sophia is non-linguistic.
        """
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        # Distinct uuids per add_node call: the bug mints a second entity uuid.
        mock_hcg.add_node.side_effect = [f"uuid-{i}" for i in range(10)]
        mock_hcg.get_node.return_value = None

        mock_milvus = MagicMock()
        # Empty index throughout: the first node's embedding has not been flushed
        # yet when the second mention is processed, so the search misses it. The
        # in-process embedding dedup must catch it instead.
        mock_milvus.search_similar.return_value = []
        mock_milvus.find_nearest_types.return_value = [
            {"uuid": "type_entity", "score": 0.1},
        ]

        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)
        node = {
            "name": "Plate Tectonics",
            "type": "entity",
            "embedding": [0.1] * 384,
            "embedding_id": "emb-1",
            "dimension": 384,
            "model": "all-MiniLM-L6-v2",
            "properties": {},
        }
        result = processor.process(
            {
                "proposal_id": "p-dup",
                "source_service": "hermes",
                "confidence": 0.7,
                "raw_text": "Plate tectonics, plate tectonics.",
                # Same entity (identical embedding) proposed twice in one ingest.
                "proposed_nodes": [dict(node), dict(node)],
                "proposed_edges": [],
            }
        )

        # Exactly one entity node, despite two mentions.
        assert len(result["stored_node_ids"]) == 1, (
            f"expected 1 entity node, got {len(result['stored_node_ids'])}: "
            f"{result['stored_node_ids']}"
        )
        # The repeat resolves to the already-created node's uuid (no new uuid).
        entity_add_calls = [
            c
            for c in mock_hcg.add_node.call_args_list
            if c.kwargs.get("name") == "Plate Tectonics"
        ]
        assert len(entity_add_calls) == 1, (
            f"add_node called {len(entity_add_calls)} times for the repeated "
            "entity; expected exactly 1"
        )

    def test_same_entity_twice_without_embeddings_yields_two_nodes(self):
        """Embedding-less repeats are NOT deduped within ingest -- by design.

        A node with no embedding carries no meaning signal, so within-ingest
        dedup (which is embedding-based, #148) deliberately does not collapse it.
        Two embedding-less mentions of the same name therefore create TWO nodes;
        any later consolidation is the downstream resolver's job, not the
        recorder's. This test pins that intent so it is explicit rather than a
        silent regression -- and guards against any name-based fallback creeping
        back in (Sophia is non-linguistic; identity is never name-based).
        """
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        mock_hcg.add_node.side_effect = [f"uuid-{i}" for i in range(10)]
        mock_hcg.get_node.return_value = None

        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = []
        mock_milvus.find_nearest_types.return_value = [
            {"uuid": "type_entity", "score": 0.1},
        ]

        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)
        # No "embedding" key: these mentions have no meaning signal.
        node = {
            "name": "Plate Tectonics",
            "type": "entity",
            "properties": {},
        }
        result = processor.process(
            {
                "proposal_id": "p-dup-noemb",
                "source_service": "hermes",
                "confidence": 0.7,
                "raw_text": "Plate tectonics, plate tectonics.",
                "proposed_nodes": [dict(node), dict(node)],
                "proposed_edges": [],
            }
        )

        # Two nodes: embedding-less repeats are deferred to the resolver.
        assert len(result["stored_node_ids"]) == 2, (
            f"expected 2 nodes (no within-ingest dedup without embeddings), got "
            f"{len(result['stored_node_ids'])}: {result['stored_node_ids']}"
        )
        entity_add_calls = [
            c
            for c in mock_hcg.add_node.call_args_list
            if c.kwargs.get("name") == "Plate Tectonics"
        ]
        assert len(entity_add_calls) == 2, (
            f"add_node called {len(entity_add_calls)} times for the repeated "
            "embedding-less entity; expected exactly 2 (no name-based dedup)"
        )

    def test_similar_embeddings_in_different_collections_not_merged(self):
        """Cross-collection nodes must not be merged by in-process dedup (#151).

        Two mentions with (near-)identical embeddings but classified into
        different collections (Entity vs Concept) are distinct identities. The
        persisted Milvus dedup (2a) is collection-scoped, so the in-process pass
        must be too -- batch membership alone must not collapse two nodes that
        separate batches would keep apart.
        """
        from types import SimpleNamespace

        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        mock_hcg.add_node.side_effect = [f"uuid-{i}" for i in range(10)]
        mock_hcg.get_node.return_value = None

        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = []

        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)
        # Force two different collections for identical embeddings.
        processor._classifier = MagicMock()
        processor._classifier.classify.side_effect = [
            SimpleNamespace(
                type_name="entity",
                type_uuid="type_entity",
                confidence=0.9,
                needs_reclassification=False,
            ),
            SimpleNamespace(
                type_name="concept",
                type_uuid="type_concept",
                confidence=0.9,
                needs_reclassification=False,
            ),
        ]

        emb = [0.1] * 384
        result = processor.process(
            {
                "proposal_id": "p-xcoll",
                "source_service": "hermes",
                "confidence": 0.7,
                "raw_text": "x",
                "proposed_nodes": [
                    {
                        "name": "Alpha",
                        "type": "entity",
                        "embedding": list(emb),
                        "embedding_id": "e1",
                        "dimension": 384,
                        "model": "m",
                        "properties": {},
                    },
                    {
                        "name": "Beta",
                        "type": "concept",
                        "embedding": list(emb),
                        "embedding_id": "e2",
                        "dimension": 384,
                        "model": "m",
                        "properties": {},
                    },
                ],
                "proposed_edges": [],
            }
        )

        # Identical embeddings but different collections => two distinct nodes.
        assert len(result["stored_node_ids"]) == 2, result["stored_node_ids"]

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
        """Edges should batch-resolve unresolved names via find_nodes_by_names."""
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        mock_hcg.add_node.return_value = "new-uuid"
        mock_hcg.add_edge.return_value = "edge-uuid"
        # Batch resolution returns a match for "France"
        mock_hcg.find_nodes_by_names.return_value = {
            "France": {"uuid": "neo4j-target-uuid"},
        }
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

        # Should have resolved "France" via batch find_nodes_by_names
        assert len(result["stored_edge_ids"]) == 1
        mock_hcg.find_nodes_by_names.assert_called_once()
        # Individual find_node_by_name should NOT be called
        mock_hcg.find_node_by_name.assert_not_called()

    def test_edge_batch_name_resolution(self):
        """Unresolved names trigger find_nodes_by_names batch call, not individual find_node_by_name."""
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        mock_hcg.add_node.return_value = "new-uuid"
        mock_hcg.add_edge.return_value = "edge-uuid"
        # Batch resolves both "France" and "Europe"
        mock_hcg.find_nodes_by_names.return_value = {
            "France": {"uuid": "france-uuid"},
            "Europe": {"uuid": "europe-uuid"},
        }
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
                    },
                    {
                        "source_name": "France",
                        "target_name": "Europe",
                        "relation": "PART_OF",
                        "confidence": 0.7,
                    },
                ],
                "document_embedding": None,
                "raw_text": "Paris is in France, part of Europe",
                "source_service": "hermes",
                "confidence": 0.7,
                "metadata": {},
            }
        )

        # Both edges should be created
        assert len(result["stored_edge_ids"]) == 2
        # Batch call made once with both unresolved names
        mock_hcg.find_nodes_by_names.assert_called_once()
        call_args = mock_hcg.find_nodes_by_names.call_args[0][0]
        assert set(call_args) == {"France", "Europe"}
        # No individual lookups
        mock_hcg.find_node_by_name.assert_not_called()

    def test_edge_batch_resolution_failure_graceful(self):
        """Batch resolution failure should skip unresolved edges, not crash."""
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        mock_hcg.add_node.return_value = "new-uuid"
        mock_hcg.add_edge.return_value = "edge-uuid"
        # Batch resolution raises an exception
        mock_hcg.find_nodes_by_names.side_effect = Exception("Neo4j connection failed")
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

        # Edge should be skipped (France unresolved), but no crash
        assert result["stored_edge_ids"] == []
        # Nodes should still be processed fine
        assert len(result["stored_node_ids"]) == 1

    def test_edge_no_unresolved_names(self):
        """When all edge names are already in name_to_uuid, find_nodes_by_names is not called."""
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
                    }
                ],
                "document_embedding": None,
                "raw_text": "",
                "source_service": "hermes",
                "confidence": 0.7,
                "metadata": {},
            }
        )

        # Both A and B were just created, so no batch resolution needed
        assert len(result["stored_edge_ids"]) == 1
        mock_hcg.find_nodes_by_names.assert_not_called()

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
            c
            for c in mock_hcg.get_node.call_args_list
            if c.args == ("type_entity",) or c.kwargs.get("uuid") == "type_entity"
        ]
        assert (
            len(type_get_calls) == 1
        ), f"Expected 1 get_node call for type_entity, got {len(type_get_calls)}"

        # update_node for the type should be called once with the final member_count
        type_update_calls = [
            c for c in mock_hcg.update_node.call_args_list if c.args[0] == "type_entity"
        ]
        assert (
            len(type_update_calls) == 1
        ), f"Expected 1 update_node call for type_entity, got {len(type_update_calls)}"
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

        return (
            ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus),
            mock_hcg,
            mock_milvus,
        )

    def _make_node(
        self, name, node_type="entity", embedding=None, model="all-MiniLM-L6-v2"
    ):
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
        mock_hcg.add_node.side_effect = [
            "uuid-entity",
            "type-def-1",
            "uuid-concept",
            "type-def-2",
        ]
        mock_hcg.get_node.return_value = None

        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = []
        mock_milvus.find_nearest_types.side_effect = [
            [{"uuid": "type_entity", "score": 0.1}],
            [{"uuid": "type_concept", "score": 0.1}],
        ]

        from sophia.ingestion.proposal_processor import ProposalProcessor

        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)

        # Distinct embeddings: these are different entities in different
        # collections. (The default _make_node embedding is identical for every
        # node, which the within-ingest embedding dedup would correctly collapse
        # into one node -- defeating this test's purpose of exercising separate
        # per-collection batches.)
        proposal = self._make_proposal(
            nodes=[
                self._make_node("Alpha", node_type="entity", embedding=[0.1] * 384),
                self._make_node("Beta", node_type="concept", embedding=[0.9] * 384),
            ]
        )
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
        edge_calls = [
            c
            for c in batch_calls
            if (c.kwargs.get("node_type") or (c.args[0] if c.args else None)) == "Edge"
        ]
        assert len(edge_calls) == 1
        edge_batch = edge_calls[0].kwargs.get("embeddings")
        if edge_batch is None and len(edge_calls[0].args) > 1:
            edge_batch = edge_calls[0].args[1]
        assert len(edge_batch) == 1
        assert edge_batch[0]["uuid"] == "edge-uuid-1"
        assert edge_batch[0]["model"] == "all-MiniLM-L6-v2"

    def test_batch_upsert_failure_propagates(self):
        """A failed Milvus embedding write must NOT be silently swallowed.

        Regression for #146: previously the flush wrapped batch_upsert in a
        warn-only except, so a failed write was swallowed and ingestion
        reported success while hcg_*_embeddings stayed empty. The failure must
        now propagate as EmbeddingPersistenceError so the caller sees it.
        """
        import pytest

        mock_hcg = MagicMock()
        mock_hcg.add_node.return_value = "uuid-1"
        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = []
        mock_milvus.find_nearest_types.return_value = [
            {"uuid": "type_entity", "score": 0.1},
        ]
        mock_milvus.batch_upsert_embeddings.side_effect = RuntimeError(
            "Milvus connection lost"
        )

        from sophia.ingestion.proposal_processor import (
            EmbeddingPersistenceError,
            ProposalProcessor,
        )

        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)

        proposal = self._make_proposal(nodes=[self._make_node("Gamma")])

        with pytest.raises(EmbeddingPersistenceError) as exc_info:
            processor.process(proposal)

        # The failing collection and its underlying error are surfaced.
        assert "Entity" in exc_info.value.failures
        assert "Milvus connection lost" in exc_info.value.failures["Entity"]
        # The write was actually attempted, not skipped.
        mock_milvus.batch_upsert_embeddings.assert_called()
        # Partial graph writes are rolled back so the batch is cleanly retryable
        # (an orphaned node with no embedding would otherwise dup on retry).
        mock_hcg.delete_node.assert_called_once_with("uuid-1")

    def test_rollback_also_deletes_edges(self):
        """On embedding failure, stored edges are rolled back too, not just nodes.

        Edges are reified as Node entities; an edge between two pre-existing nodes
        isn't removed by deleting the batch's new nodes, so it must be deleted by
        its own uuid (gemini asked to roll back stored_edge_ids).
        """
        import pytest

        mock_hcg = MagicMock()
        mock_hcg.add_node.return_value = "node-1"
        mock_hcg.add_edge.return_value = "edge-1"
        mock_hcg.find_nodes_by_names.return_value = {"France": {"uuid": "france-uuid"}}
        mock_milvus = MagicMock()
        mock_milvus.search_similar.return_value = []
        mock_milvus.find_nearest_types.return_value = [
            {"uuid": "type_location", "score": 0.1},
        ]
        mock_milvus.batch_upsert_embeddings.side_effect = RuntimeError("Milvus down")

        from sophia.ingestion.proposal_processor import (
            EmbeddingPersistenceError,
            ProposalProcessor,
        )

        processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)
        proposal = {
            "proposal_id": "p-rollback",
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
        }

        with pytest.raises(EmbeddingPersistenceError):
            processor.process(proposal)

        deleted = {c.args[0] for c in mock_hcg.delete_node.call_args_list}
        assert "node-1" in deleted, "stored node not rolled back"
        assert "edge-1" in deleted, "stored edge not rolled back"

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
        proposal = self._make_proposal(
            nodes=[
                self._make_node("X", embedding=test_embedding, model="test-model"),
            ]
        )
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
            {
                "uuid": f"uuid-{i}",
                "name": f"Node{i}",
                "type": coll.lower(),
                "properties": {},
            }
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
            assert (
                f"uuid-{i}" in ctx_uuids
            ), f"uuid-{i} missing from context: {ctx_uuids}"

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


class TestProposalProcessorEventBus:
    def test_proposal_processor_accepts_event_bus(self):
        """ProposalProcessor accepts optional event_bus parameter."""
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        mock_milvus = MagicMock()
        mock_event_bus = MagicMock()
        processor = ProposalProcessor(mock_hcg, mock_milvus, event_bus=mock_event_bus)
        assert processor._event_bus is mock_event_bus

    def test_proposal_processor_event_bus_defaults_to_none(self):
        """ProposalProcessor works without event_bus."""
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        mock_milvus = MagicMock()
        processor = ProposalProcessor(mock_hcg, mock_milvus)
        assert processor._event_bus is None

    def test_process_publishes_batch_event(self):
        """process() publishes a proposal_processed event via EventBus."""
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        mock_milvus = MagicMock()
        mock_event_bus = MagicMock()

        # Configure mocks to let process() run through node creation path
        mock_hcg.add_node.return_value = "node-uuid-1"
        mock_hcg.add_edge.return_value = "edge-uuid-1"
        mock_hcg.get_node.return_value = {
            "uuid": "type_entity",
            "name": "entity",
            "properties": {"member_count": 1, "centroid": [0.1] * 384},
        }
        mock_hcg.get_nodes_batch.return_value = []
        mock_hcg.find_nodes_by_names.return_value = {}
        mock_milvus.search_similar.return_value = []
        mock_milvus.find_nearest_types.return_value = [
            {"uuid": "type_entity", "score": 0.1},
        ]
        mock_milvus.batch_upsert_embeddings.return_value = None

        processor = ProposalProcessor(mock_hcg, mock_milvus, event_bus=mock_event_bus)

        proposal = {
            "proposal_id": "p-event-test",
            "proposed_nodes": [
                {
                    "name": "Alice",
                    "type": "entity",
                    "embedding": [0.1] * 384,
                    "model": "all-MiniLM-L6-v2",
                    "properties": {},
                }
            ],
            "proposed_edges": [],
            "document_embedding": {
                "embedding": [0.5] * 384,
                "embedding_id": "doc-1",
                "dimension": 384,
                "model": "all-MiniLM-L6-v2",
            },
            "raw_text": "test event publishing",
            "source_service": "hermes",
            "confidence": 0.7,
            "metadata": {},
        }

        processor.process(proposal)

        mock_event_bus.publish.assert_called_once()
        channel, event = mock_event_bus.publish.call_args[0]
        assert channel == "logos:sophia:proposal_processed"
        assert event["event_type"] == "proposal_processed"
        assert event["source"] == "sophia"
        assert "payload" in event
        payload = event["payload"]
        assert "affected_node_uuids" in payload
        assert "stored_node_ids" in payload
        assert "stored_edge_ids" in payload
        assert "new_types" in payload
        assert "updated_types" in payload
        assert "node-uuid-1" in payload["stored_node_ids"]

    def test_process_no_event_bus_no_publish(self):
        """process() works without event_bus -- no publish call."""
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        mock_milvus = MagicMock()

        mock_hcg.add_node.return_value = "node-uuid-1"
        mock_hcg.get_nodes_batch.return_value = []
        mock_milvus.search_similar.return_value = []
        mock_milvus.find_nearest_types.return_value = [
            {"uuid": "type_entity", "score": 0.1},
        ]

        processor = ProposalProcessor(mock_hcg, mock_milvus)

        proposal = {
            "proposal_id": "p-no-bus",
            "proposed_nodes": [
                {
                    "name": "Bob",
                    "type": "entity",
                    "embedding": [0.2] * 384,
                    "model": "all-MiniLM-L6-v2",
                    "properties": {},
                }
            ],
            "proposed_edges": [],
            "document_embedding": {},
            "raw_text": "test no event bus",
            "source_service": "hermes",
            "confidence": 0.7,
        }

        # Should not raise even without event_bus
        result = processor.process(proposal)
        assert "stored_node_ids" in result

    def test_process_publish_failure_does_not_break_processing(self):
        """If EventBus.publish() raises, process() still returns normally."""
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        mock_milvus = MagicMock()
        mock_event_bus = MagicMock()
        mock_event_bus.publish.side_effect = RuntimeError("Redis down")

        mock_hcg.add_node.return_value = "node-uuid-1"
        mock_hcg.get_nodes_batch.return_value = []
        mock_milvus.search_similar.return_value = []
        mock_milvus.find_nearest_types.return_value = [
            {"uuid": "type_entity", "score": 0.1},
        ]

        processor = ProposalProcessor(mock_hcg, mock_milvus, event_bus=mock_event_bus)

        proposal = {
            "proposal_id": "p-fail-bus",
            "proposed_nodes": [
                {
                    "name": "Charlie",
                    "type": "entity",
                    "embedding": [0.3] * 384,
                    "model": "all-MiniLM-L6-v2",
                    "properties": {},
                }
            ],
            "proposed_edges": [],
            "document_embedding": {},
            "raw_text": "test publish failure",
            "source_service": "hermes",
            "confidence": 0.7,
        }

        result = processor.process(proposal)
        assert "stored_node_ids" in result
        assert "node-uuid-1" in result["stored_node_ids"]
        mock_event_bus.publish.assert_called_once()


class TestProposalProcessorRedisSnapshot:
    def _make_processor_with_redis(
        self, mock_hcg=None, mock_milvus=None, mock_event_bus=None, mock_redis=None
    ):
        """Helper to create a processor with Redis and sensible mock defaults."""
        from sophia.ingestion.proposal_processor import ProposalProcessor

        if mock_hcg is None:
            mock_hcg = MagicMock()
        if mock_milvus is None:
            mock_milvus = MagicMock()
        if mock_event_bus is None:
            mock_event_bus = MagicMock()
        if mock_redis is None:
            mock_redis = MagicMock()

        mock_hcg.add_node.return_value = "new-uuid"
        mock_hcg.add_edge.return_value = "edge-uuid"
        mock_hcg.get_node.return_value = {
            "uuid": "type_entity",
            "name": "entity",
            "properties": {"member_count": 1, "centroid": [0.1] * 384},
        }
        mock_hcg.get_nodes_batch.return_value = []
        mock_hcg.find_nodes_by_names.return_value = {}
        mock_milvus.search_similar.return_value = []
        mock_milvus.find_nearest_types.return_value = [
            {"uuid": "type_entity", "score": 0.1},
        ]
        mock_milvus.batch_upsert_embeddings.return_value = None

        processor = ProposalProcessor(
            mock_hcg, mock_milvus, event_bus=mock_event_bus, redis_client=mock_redis
        )
        return processor, mock_hcg, mock_milvus, mock_event_bus, mock_redis

    def _make_proposal(self, nodes=None, edges=None):
        """Helper to create a minimal proposal."""
        return {
            "proposal_id": "p-redis-test",
            "proposed_nodes": nodes
            or [
                {
                    "name": "TestNode",
                    "type": "entity",
                    "embedding": [0.1] * 384,
                    "model": "all-MiniLM-L6-v2",
                    "properties": {},
                }
            ],
            "proposed_edges": edges or [],
            "document_embedding": {},
            "raw_text": "redis test",
            "source_service": "hermes",
            "confidence": 0.7,
        }

    def test_process_writes_type_snapshot_to_redis(self):
        """process() writes full type list to Redis key logos:ontology:types."""
        import json

        processor, mock_hcg, _, _, mock_redis = self._make_processor_with_redis()

        # Set up the get_all_type_definitions return for type snapshot
        mock_hcg.get_all_type_definitions.return_value = [
            {
                "uuid": "type_entity",
                "name": "entity",
                "properties": {"member_count": 5},
            },
            {
                "uuid": "type_concept",
                "name": "concept",
                "properties": {"member_count": 3},
            },
        ]

        processor.process(self._make_proposal())

        # Verify Redis write
        mock_redis.set.assert_called_once()
        key = mock_redis.set.call_args[0][0]
        assert key == "logos:ontology:types"
        value = json.loads(mock_redis.set.call_args[0][1])
        assert isinstance(value, dict)
        assert "entity" in value
        assert value["entity"]["uuid"] == "type_entity"
        assert value["entity"]["member_count"] == 5
        assert "concept" in value
        assert value["concept"]["uuid"] == "type_concept"
        assert value["concept"]["member_count"] == 3

    def test_process_no_redis_client_skips_snapshot(self):
        """process() skips type snapshot when redis_client is None."""
        from sophia.ingestion.proposal_processor import ProposalProcessor

        mock_hcg = MagicMock()
        mock_milvus = MagicMock()
        mock_hcg.add_node.return_value = "new-uuid"
        mock_hcg.get_nodes_batch.return_value = []
        mock_milvus.search_similar.return_value = []
        mock_milvus.find_nearest_types.return_value = [
            {"uuid": "type_entity", "score": 0.1},
        ]

        processor = ProposalProcessor(mock_hcg, mock_milvus)

        # Should not raise even without redis_client
        result = processor.process(self._make_proposal())
        assert "stored_node_ids" in result
        # get_all_type_definitions should NOT be called for type snapshot when no redis
        mock_hcg.get_all_type_definitions.assert_not_called()

    def test_process_redis_write_failure_does_not_break_processing(self):
        """If Redis write fails, process() still returns results."""
        processor, mock_hcg, _, _, mock_redis = self._make_processor_with_redis()

        mock_hcg.get_all_type_definitions.return_value = [
            {
                "uuid": "type_entity",
                "name": "entity",
                "properties": {"member_count": 1},
            },
        ]
        mock_redis.set.side_effect = RuntimeError("Redis connection lost")

        result = processor.process(self._make_proposal())
        assert "stored_node_ids" in result
        assert "new-uuid" in result["stored_node_ids"]

    def test_process_hcg_query_failure_does_not_break_processing(self):
        """If HCG type query fails, process() still returns results."""
        processor, mock_hcg, _, _, mock_redis = self._make_processor_with_redis()

        mock_hcg.get_all_type_definitions.side_effect = RuntimeError("Neo4j down")

        result = processor.process(self._make_proposal())
        assert "stored_node_ids" in result
        assert "new-uuid" in result["stored_node_ids"]
        # Redis should not have been written to
        mock_redis.set.assert_not_called()

    def test_type_snapshot_skips_nameless_records(self):
        """Type records with empty names are excluded from the snapshot."""
        import json

        processor, mock_hcg, _, _, mock_redis = self._make_processor_with_redis()

        mock_hcg.get_all_type_definitions.return_value = [
            {
                "uuid": "type_entity",
                "name": "entity",
                "properties": {"member_count": 2},
            },
            {"uuid": "type_empty", "name": "", "properties": {"member_count": 0}},
        ]

        processor.process(self._make_proposal())

        value = json.loads(mock_redis.set.call_args[0][1])
        assert "entity" in value
        assert "" not in value
        assert len(value) == 1
