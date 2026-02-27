import pytest

pytest.importorskip("openai", reason="openai not installed")

import numpy as np
from unittest.mock import patch, MagicMock
from sophia.experiments.agents.embedding import EmbeddingAgent, make_embedding_agent


def test_embedding_agent_returns_vector():
    """EmbeddingAgent.process takes text, returns numpy array."""
    fake_embedding = [0.1] * 1536
    agent = EmbeddingAgent(model="text-embedding-3-small", dim=1536)

    with patch.object(agent, "_client") as mock_client:
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=fake_embedding)]
        mock_client.embeddings.create.return_value = mock_response

        result = agent.process("hello world")

    assert isinstance(result, np.ndarray)
    assert result.shape == (1536,)


def test_embedding_agent_dim_configurable():
    agent = EmbeddingAgent(model="text-embedding-3-small", dim=768)
    assert agent.dim == 768


def test_factory_creates_agent():
    agent = make_embedding_agent({"model": "text-embedding-3-small", "dim": 1536})
    assert isinstance(agent, EmbeddingAgent)
    assert agent.dim == 1536
