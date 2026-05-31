"""#505: ProposalProcessor records Hermes' NER type as a provenance hint."""

from unittest.mock import MagicMock


def test_hermes_type_hint_persisted():
    from sophia.ingestion.proposal_processor import ProposalProcessor

    mock_hcg = MagicMock()
    mock_hcg.add_node.return_value = "new-uuid"
    mock_milvus = MagicMock()
    mock_milvus.search_similar.return_value = []
    mock_milvus.find_nearest_types.return_value = [
        {"uuid": "type_entity", "score": 0.1}
    ]

    processor = ProposalProcessor(hcg_client=mock_hcg, milvus_sync=mock_milvus)
    processor.process(
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

    entity_calls = [
        c for c in mock_hcg.add_node.call_args_list if c.kwargs.get("name") == "Paris"
    ]
    assert entity_calls, "expected an add_node call for the Paris entity"
    props = entity_calls[0].kwargs["properties"]
    assert props["hermes_type_hint"] == "GPE"
