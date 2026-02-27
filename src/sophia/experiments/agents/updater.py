from __future__ import annotations

import numpy as np

from sophia.experiments.agents.matrix import MatrixAgent


class MatrixUpdateAgent:
    """Updates the emotional state matrix using a persona entry embedding.

    Naive column selection: highest absolute value in the embedding.
    Update: column_j = tanh(column_j + alpha * embedding)
    """

    def __init__(self, matrix_agent: MatrixAgent, alpha: float = 0.01):
        self.matrix_agent = matrix_agent
        self.alpha = alpha

    def process(self, input_data: np.ndarray) -> np.ndarray:
        """Update the matrix and return the input unchanged."""
        col_idx = int(np.argmax(np.abs(input_data)))
        matrix = self.matrix_agent._matrix
        matrix[:, col_idx] = np.tanh(matrix[:, col_idx] + self.alpha * input_data)
        return input_data


def make_update_agent(
    config: dict, matrix_agent: MatrixAgent | None = None
) -> MatrixUpdateAgent:
    if matrix_agent is None:
        raise ValueError("matrix_agent required")
    return MatrixUpdateAgent(
        matrix_agent=matrix_agent,
        alpha=config.get("alpha", 0.01),
    )


class OuterProductUpdateAgent:
    """Updates the emotional state matrix using outer product of embedding.

    Update: M = tanh(M + alpha * outer(embedding, embedding))
    Spreads influence proportionally across all dimensions.
    """

    def __init__(self, matrix_agent: MatrixAgent, alpha: float = 0.01):
        self.matrix_agent = matrix_agent
        self.alpha = alpha

    def process(self, input_data: np.ndarray) -> np.ndarray:
        """Update the matrix with outer product and return the input unchanged."""
        matrix = self.matrix_agent._matrix
        self.matrix_agent._matrix = np.tanh(
            matrix + self.alpha * np.outer(input_data, input_data)
        )
        return input_data


def make_outer_product_update_agent(
    config: dict, matrix_agent: MatrixAgent | None = None
) -> OuterProductUpdateAgent:
    if matrix_agent is None:
        raise ValueError("matrix_agent required")
    return OuterProductUpdateAgent(
        matrix_agent=matrix_agent,
        alpha=config.get("alpha", 0.01),
    )
