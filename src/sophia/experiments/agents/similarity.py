import numpy as np


class SimilarityAgent:
    """Computes cosine similarity between two embedding vectors."""

    def process(self, input_data: dict) -> float:
        a = input_data["a"]
        b = input_data["b"]
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(dot / norm)


def make_similarity_agent(config: dict) -> SimilarityAgent:
    return SimilarityAgent()
