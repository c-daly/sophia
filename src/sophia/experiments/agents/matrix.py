from __future__ import annotations

import numpy as np
from typing import Any


class MatrixAgent:
    """Emotional state matrix. Filters embeddings via (M + I) * embedding.

    StatefulAgent: supports get_state, snapshot, reset.
    """

    def __init__(self, dim: int, std: float = 0.01, seed: int = 0):
        self.dim = dim
        self._rng = np.random.default_rng(seed)
        self._matrix = self._rng.normal(0, std, (dim, dim))
        self._identity = np.eye(dim)

    def process(self, input_data: np.ndarray) -> np.ndarray:
        """Filter embedding through emotional state: (M + I) * embedding."""
        return (self._matrix + self._identity) @ input_data  # type: ignore[no-any-return]

    def get_state(self) -> np.ndarray:
        return self._matrix.copy()  # type: ignore[no-any-return]

    def reset(self, state: Any = None) -> None:
        if state is not None:
            self._matrix = state.copy()
        else:
            self._matrix = np.zeros((self.dim, self.dim))

    def snapshot(self) -> np.ndarray:
        return self._matrix.copy()


def make_matrix_agent(config: dict) -> MatrixAgent:
    return MatrixAgent(
        dim=config.get("dim", 1536),
        std=config.get("std", 0.01),
        seed=config.get("seed", 0),
    )
