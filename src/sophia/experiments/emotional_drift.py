"""First experiment: does the emotional state matrix cause meaningful drift?

Hypothesis: filtering embeddings through a matrix shaped by emotionally-charged
experiences causes neutral inputs to drift toward that emotional character.
"""

import numpy as np

from sophia.experiments.agents.embedding import EmbeddingAgent
from sophia.experiments.agents.matrix import MatrixAgent
from sophia.experiments.agents.updater import MatrixUpdateAgent
from sophia.experiments.agents.similarity import SimilarityAgent

# Emotionally distinct input texts
ANGRY_TEXTS = [
    "I am furious about this situation and cannot believe it happened",
    "This is completely unacceptable and makes me incredibly angry",
    "I am enraged by the incompetence and negligence on display",
    "This infuriating problem keeps getting worse and nobody cares",
    "I am livid and fed up with being ignored and dismissed",
]

CURIOUS_TEXTS = [
    "I wonder how this mechanism actually works under the hood",
    "That is a fascinating pattern I have never noticed before",
    "What would happen if we approached this from a completely different angle",
    "I am intrigued by the unexpected connection between these ideas",
    "How does this phenomenon emerge from such simple rules",
]

NEUTRAL_TEXTS = [
    "The meeting is scheduled for three o'clock on Tuesday",
    "Please update the spreadsheet with the quarterly numbers",
    "The package arrived and has been placed on your desk",
    "The temperature today is expected to be around sixty degrees",
    "The report summarizes the findings from last month",
]


def run_emotional_drift_experiment(
    seed: int = 42,
    matrix_std: float = 0.01,
    alpha: float = 0.01,
    dim: int = 1536,
    emotion: str = "angry",
) -> dict:
    """Run the emotional drift experiment.

    1. Generate embeddings for emotional and neutral text
    2. Snapshot the initial matrix
    3. Update the matrix with emotional embeddings
    4. Filter neutral embeddings through baseline (initial) and updated matrix
    5. Compare: are filtered neutral embeddings closer to emotional embeddings?
    """
    # Create agents
    embedder = EmbeddingAgent(model="text-embedding-3-small", dim=dim)
    matrix = MatrixAgent(dim=dim, std=matrix_std, seed=seed)
    updater = MatrixUpdateAgent(matrix_agent=matrix, alpha=alpha)
    similarity = SimilarityAgent()

    # Select emotion texts
    emotion_texts = ANGRY_TEXTS if emotion == "angry" else CURIOUS_TEXTS

    # Generate embeddings
    emotion_embeddings = [embedder.process(t) for t in emotion_texts]
    neutral_embeddings = [embedder.process(t) for t in NEUTRAL_TEXTS]

    # Compute emotional centroid (average of emotional embeddings)
    emotion_centroid = np.mean(emotion_embeddings, axis=0)

    # Snapshot baseline matrix
    matrix_before = matrix.snapshot()

    # Baseline: filter neutral embeddings through initial matrix
    baseline_filtered = [matrix.process(e) for e in neutral_embeddings]
    baseline_similarities = [
        similarity.process({"a": f, "b": emotion_centroid}) for f in baseline_filtered
    ]

    # Update matrix with emotional embeddings
    for emb in emotion_embeddings:
        updater.process(emb)

    # Snapshot updated matrix
    matrix_after = matrix.snapshot()

    # Filter neutral embeddings through updated matrix
    updated_filtered = [matrix.process(e) for e in neutral_embeddings]
    filtered_similarities = [
        similarity.process({"a": f, "b": emotion_centroid}) for f in updated_filtered
    ]

    return {
        "emotion": emotion,
        "seed": seed,
        "alpha": alpha,
        "matrix_std": matrix_std,
        "dim": dim,
        "baseline_similarities": baseline_similarities,
        "filtered_similarities": filtered_similarities,
        "baseline_mean": float(np.mean(baseline_similarities)),
        "filtered_mean": float(np.mean(filtered_similarities)),
        "drift": float(np.mean(filtered_similarities) - np.mean(baseline_similarities)),
        "matrix_before": matrix_before,
        "matrix_after": matrix_after,
        "emotion_texts": emotion_texts,
        "neutral_texts": NEUTRAL_TEXTS,
    }
