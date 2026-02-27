import numpy as np
from openai import OpenAI


class EmbeddingAgent:
    """Takes text, returns embedding vector via OpenAI API."""

    def __init__(self, model: str, dim: int):
        self.model = model
        self.dim = dim
        self._client: OpenAI | None = None

    def process(self, input_data: str) -> np.ndarray:
        if self._client is None:
            self._client = OpenAI()
        response = self._client.embeddings.create(
            model=self.model,
            input=input_data,
            dimensions=self.dim,
        )
        return np.array(response.data[0].embedding, dtype=np.float64)


def make_embedding_agent(config: dict) -> EmbeddingAgent:
    return EmbeddingAgent(
        model=config.get("model", "text-embedding-3-small"),
        dim=config.get("dim", 1536),
    )
