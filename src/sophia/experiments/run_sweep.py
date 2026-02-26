"""Parameter sweep for emotional drift experiment.

Compares single-column vs outer-product matrix updates.

Usage: poetry run python -m sophia.experiments.run_sweep
"""

import numpy as np
from sophia.experiments.agents.embedding import EmbeddingAgent
from sophia.experiments.agents.matrix import MatrixAgent
from sophia.experiments.agents.updater import MatrixUpdateAgent, OuterProductUpdateAgent
from sophia.experiments.agents.similarity import SimilarityAgent
from sophia.experiments.emotional_drift import ANGRY_TEXTS, CURIOUS_TEXTS, NEUTRAL_TEXTS


def embed_all(embedder, texts):
    return [embedder.process(t) for t in texts]


def run_trial(emotional_embs, neutral_embs, emotional_centroid, dim, seed, alpha, std, updater_cls):
    matrix = MatrixAgent(dim=dim, std=std, seed=seed)
    updater = updater_cls(matrix_agent=matrix, alpha=alpha)

    baseline_sims = [
        cosine(matrix.process(n), emotional_centroid) for n in neutral_embs
    ]

    incremental = []
    for e in emotional_embs:
        updater.process(e)
        sims = [cosine(matrix.process(n), emotional_centroid) for n in neutral_embs]
        incremental.append(float(np.mean(sims)))

    final_sims = [
        cosine(matrix.process(n), emotional_centroid) for n in neutral_embs
    ]

    matrix_state = matrix.get_state()
    nonzero_cols = int(np.sum(np.any(np.abs(matrix_state) > 1e-10, axis=0)))
    return {
        "baseline_mean": float(np.mean(baseline_sims)),
        "final_mean": float(np.mean(final_sims)),
        "drift": float(np.mean(final_sims) - np.mean(baseline_sims)),
        "incremental_means": incremental,
        "matrix_norm": float(np.linalg.norm(matrix_state, "fro")),
        "matrix_max": float(np.max(np.abs(matrix_state))),
        "matrix_nonzero_cols": nonzero_cols,
    }


def cosine(a, b):
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / n) if n > 0 else 0.0


def main():
    dim = 1536
    seed = 42
    embedder = EmbeddingAgent(model="text-embedding-3-small", dim=dim)

    print("Generating embeddings (one-time API cost)...")
    angry_embs = embed_all(embedder, ANGRY_TEXTS)
    curious_embs = embed_all(embedder, CURIOUS_TEXTS)
    neutral_embs = embed_all(embedder, NEUTRAL_TEXTS)
    angry_centroid = np.mean(angry_embs, axis=0)
    curious_centroid = np.mean(curious_embs, axis=0)

    print("\n=== EMBEDDING SPACE BASELINE ===")
    print(f"Angry centroid vs neutral centroid:   {cosine(angry_centroid, np.mean(neutral_embs, axis=0)):.6f}")
    print(f"Curious centroid vs neutral centroid:  {cosine(curious_centroid, np.mean(neutral_embs, axis=0)):.6f}")
    print(f"Angry centroid vs curious centroid:    {cosine(angry_centroid, curious_centroid):.6f}")

    # Side-by-side comparison
    print("\n=== SINGLE-COLUMN vs OUTER-PRODUCT: ALPHA SWEEP (angry, std=0.0) ===")
    print(f"{'alpha':>8} | {'--- Single Column ---':>44} | {'--- Outer Product ---':>44}")
    print(f"{'':>8} | {'drift':>12} {'mat_norm':>10} {'mat_max':>10} {'cols':>6} | {'drift':>12} {'mat_norm':>10} {'mat_max':>10} {'cols':>6}")
    for alpha in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        sc = run_trial(angry_embs, neutral_embs, angry_centroid, dim, seed, alpha, 0.0, MatrixUpdateAgent)
        op = run_trial(angry_embs, neutral_embs, angry_centroid, dim, seed, alpha, 0.0, OuterProductUpdateAgent)
        print(f"{alpha:>8.3f} | {sc['drift']:>+12.6f} {sc['matrix_norm']:>10.4f} {sc['matrix_max']:>10.6f} {sc['matrix_nonzero_cols']:>6} | {op['drift']:>+12.6f} {op['matrix_norm']:>10.4f} {op['matrix_max']:>10.6f} {op['matrix_nonzero_cols']:>6}")

    print("\n=== SINGLE-COLUMN vs OUTER-PRODUCT: ALPHA SWEEP (curious, std=0.0) ===")
    print(f"{'alpha':>8} | {'--- Single Column ---':>44} | {'--- Outer Product ---':>44}")
    print(f"{'':>8} | {'drift':>12} {'mat_norm':>10} {'mat_max':>10} {'cols':>6} | {'drift':>12} {'mat_norm':>10} {'mat_max':>10} {'cols':>6}")
    for alpha in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        sc = run_trial(curious_embs, neutral_embs, curious_centroid, dim, seed, alpha, 0.0, MatrixUpdateAgent)
        op = run_trial(curious_embs, neutral_embs, curious_centroid, dim, seed, alpha, 0.0, OuterProductUpdateAgent)
        print(f"{alpha:>8.3f} | {sc['drift']:>+12.6f} {sc['matrix_norm']:>10.4f} {sc['matrix_max']:>10.6f} {sc['matrix_nonzero_cols']:>6} | {op['drift']:>+12.6f} {op['matrix_norm']:>10.4f} {op['matrix_max']:>10.6f} {op['matrix_nonzero_cols']:>6}")

    # Incremental comparison
    print("\n=== INCREMENTAL DRIFT COMPARISON (alpha=0.1, std=0.0, angry) ===")
    sc = run_trial(angry_embs, neutral_embs, angry_centroid, dim, seed, 0.1, 0.0, MatrixUpdateAgent)
    op = run_trial(angry_embs, neutral_embs, angry_centroid, dim, seed, 0.1, 0.0, OuterProductUpdateAgent)
    print(f"{'sample':>8} | {'single_col drift':>18} | {'outer_prod drift':>18}")
    for i in range(len(sc['incremental_means'])):
        sc_d = sc['incremental_means'][i] - sc['baseline_mean']
        op_d = op['incremental_means'][i] - op['baseline_mean']
        print(f"{i+1:>8} | {sc_d:>+18.6f} | {op_d:>+18.6f}")

    # Cross-emotion test: train on angry, measure drift toward curious
    print("\n=== CROSS-EMOTION TEST (train angry, measure curious drift, alpha=0.1) ===")
    print("Does training on anger accidentally make things more curious?")
    for updater_name, updater_cls in [("single_col", MatrixUpdateAgent), ("outer_prod", OuterProductUpdateAgent)]:
        matrix = MatrixAgent(dim=dim, std=0.0, seed=seed)
        updater = updater_cls(matrix_agent=matrix, alpha=0.1)

        # baseline similarities to both centroids
        angry_base = np.mean([cosine(matrix.process(n), angry_centroid) for n in neutral_embs])
        curious_base = np.mean([cosine(matrix.process(n), curious_centroid) for n in neutral_embs])

        # train on angry
        for e in angry_embs:
            updater.process(e)

        angry_after = np.mean([cosine(matrix.process(n), angry_centroid) for n in neutral_embs])
        curious_after = np.mean([cosine(matrix.process(n), curious_centroid) for n in neutral_embs])

        print(f"  {updater_name}: angry_drift={angry_after - angry_base:+.6f}  curious_drift={curious_after - curious_base:+.6f}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
