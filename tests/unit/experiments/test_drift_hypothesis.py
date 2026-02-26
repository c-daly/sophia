"""Tests validating the emotional drift hypothesis with synthetic embeddings.

No OpenAI API needed — uses controlled synthetic vectors to verify the
matrix update + filter pipeline produces meaningful, directional drift.
"""
import numpy as np
import pytest
from sophia.experiments.agents.matrix import MatrixAgent
from sophia.experiments.agents.updater import MatrixUpdateAgent
from sophia.experiments.agents.similarity import SimilarityAgent


def make_cluster(center: np.ndarray, n: int, spread: float, rng: np.random.Generator) -> list[np.ndarray]:
    """Generate n vectors clustered around center with given spread."""
    return [center + rng.normal(0, spread, center.shape) for _ in range(n)]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


class TestDriftDirection:
    """Drift should move neutral embeddings TOWARD the emotional cluster."""

    def test_drift_toward_emotional_centroid(self):
        """After updating matrix with emotional vectors, filtering neutral
        vectors should increase their similarity to the emotional centroid."""
        dim = 64
        rng = np.random.default_rng(42)

        # Create distinct clusters: emotional in one region, neutral in another
        emotional_center = np.zeros(dim)
        emotional_center[:16] = 1.0  # emotional signal in first 16 dims
        neutral_center = np.zeros(dim)
        neutral_center[32:48] = 1.0  # neutral signal in different dims

        emotional_vecs = make_cluster(emotional_center, 10, 0.1, rng)
        neutral_vecs = make_cluster(neutral_center, 10, 0.1, rng)
        emotional_centroid = np.mean(emotional_vecs, axis=0)

        matrix = MatrixAgent(dim=dim, std=0.0, seed=0)  # start at zero (identity)
        updater = MatrixUpdateAgent(matrix_agent=matrix, alpha=0.5)

        # Baseline similarities
        baseline_sims = [cosine_sim(matrix.process(n), emotional_centroid) for n in neutral_vecs]

        # Update matrix with emotional vectors
        for e in emotional_vecs:
            updater.process(e)

        # Post-update similarities
        updated_sims = [cosine_sim(matrix.process(n), emotional_centroid) for n in neutral_vecs]

        baseline_mean = np.mean(baseline_sims)
        updated_mean = np.mean(updated_sims)
        drift = updated_mean - baseline_mean

        assert drift > 0, f"Expected positive drift, got {drift:.6f}"

    def test_drift_not_random_direction(self):
        """Drift should be specifically toward the emotional cluster,
        not toward an unrelated cluster."""
        dim = 64
        rng = np.random.default_rng(42)

        emotional_center = np.zeros(dim)
        emotional_center[:16] = 1.0
        neutral_center = np.zeros(dim)
        neutral_center[32:48] = 1.0
        unrelated_center = np.zeros(dim)
        unrelated_center[48:64] = 1.0  # third, unrelated region

        emotional_vecs = make_cluster(emotional_center, 10, 0.1, rng)
        neutral_vecs = make_cluster(neutral_center, 5, 0.1, rng)
        emotional_centroid = np.mean(emotional_vecs, axis=0)
        unrelated_centroid = unrelated_center.copy()

        matrix = MatrixAgent(dim=dim, std=0.0, seed=0)
        updater = MatrixUpdateAgent(matrix_agent=matrix, alpha=0.5)

        for e in emotional_vecs:
            updater.process(e)

        # Measure drift toward emotional vs unrelated
        drift_toward_emotional = []
        drift_toward_unrelated = []
        for n in neutral_vecs:
            filtered = matrix.process(n)
            drift_toward_emotional.append(cosine_sim(filtered, emotional_centroid) - cosine_sim(n, emotional_centroid))
            drift_toward_unrelated.append(cosine_sim(filtered, unrelated_centroid) - cosine_sim(n, unrelated_centroid))

        mean_emotional_drift = np.mean(drift_toward_emotional)
        mean_unrelated_drift = np.mean(drift_toward_unrelated)

        assert mean_emotional_drift > mean_unrelated_drift, (
            f"Drift toward emotional ({mean_emotional_drift:.6f}) should exceed "
            f"drift toward unrelated ({mean_unrelated_drift:.6f})"
        )


class TestDriftScaling:
    """Drift magnitude should scale with controllable parameters."""

    def test_higher_alpha_more_drift(self):
        dim = 64
        rng = np.random.default_rng(42)

        emotional_center = np.zeros(dim)
        emotional_center[:16] = 1.0
        neutral_center = np.zeros(dim)
        neutral_center[32:48] = 1.0

        emotional_vecs = make_cluster(emotional_center, 10, 0.1, rng)
        neutral_vecs = make_cluster(neutral_center, 5, 0.1, np.random.default_rng(99))
        emotional_centroid = np.mean(emotional_vecs, axis=0)

        drifts = []
        for alpha in [0.01, 0.1, 0.5]:
            matrix = MatrixAgent(dim=dim, std=0.0, seed=0)
            updater = MatrixUpdateAgent(matrix_agent=matrix, alpha=alpha)
            for e in emotional_vecs:
                updater.process(e)
            sims = [cosine_sim(matrix.process(n), emotional_centroid) for n in neutral_vecs]
            baseline_sims = [cosine_sim(n, emotional_centroid) for n in neutral_vecs]
            drifts.append(np.mean(sims) - np.mean(baseline_sims))

        assert drifts[0] < drifts[1] < drifts[2], f"Drift should increase with alpha: {drifts}"

    def test_more_training_more_drift(self):
        dim = 64
        rng = np.random.default_rng(42)

        emotional_center = np.zeros(dim)
        emotional_center[:16] = 1.0
        neutral_center = np.zeros(dim)
        neutral_center[32:48] = 1.0

        all_emotional = make_cluster(emotional_center, 30, 0.1, rng)
        neutral_vecs = make_cluster(neutral_center, 5, 0.1, np.random.default_rng(99))

        drifts = []
        for n_train in [3, 10, 30]:
            emotional_centroid = np.mean(all_emotional[:n_train], axis=0)
            matrix = MatrixAgent(dim=dim, std=0.0, seed=0)
            updater = MatrixUpdateAgent(matrix_agent=matrix, alpha=0.1)
            for e in all_emotional[:n_train]:
                updater.process(e)
            sims = [cosine_sim(matrix.process(n), emotional_centroid) for n in neutral_vecs]
            baseline_sims = [cosine_sim(n, emotional_centroid) for n in neutral_vecs]
            drifts.append(np.mean(sims) - np.mean(baseline_sims))

        assert drifts[0] < drifts[2], f"More training should produce more drift: {drifts}"


class TestReproducibility:
    """Same seed = same results."""

    def test_deterministic_with_same_seed(self):
        dim = 64

        def run_once(seed):
            rng = np.random.default_rng(seed)
            center = np.zeros(dim)
            center[:16] = 1.0
            vecs = make_cluster(center, 5, 0.1, rng)
            matrix = MatrixAgent(dim=dim, std=0.01, seed=seed)
            updater = MatrixUpdateAgent(matrix_agent=matrix, alpha=0.1)
            for v in vecs:
                updater.process(v)
            return matrix.get_state()

        state1 = run_once(42)
        state2 = run_once(42)
        np.testing.assert_array_equal(state1, state2)

    def test_different_seed_different_results(self):
        dim = 64

        def run_once(seed):
            rng = np.random.default_rng(seed)
            center = np.zeros(dim)
            center[:16] = 1.0
            vecs = make_cluster(center, 5, 0.1, rng)
            matrix = MatrixAgent(dim=dim, std=0.01, seed=seed)
            updater = MatrixUpdateAgent(matrix_agent=matrix, alpha=0.1)
            for v in vecs:
                updater.process(v)
            return matrix.get_state()

        state1 = run_once(42)
        state2 = run_once(99)
        assert not np.allclose(state1, state2)


class TestMatrixBounds:
    """Matrix values should stay bounded regardless of input."""

    def test_matrix_stays_bounded_after_many_updates(self):
        dim = 32
        rng = np.random.default_rng(42)
        matrix = MatrixAgent(dim=dim, std=0.0, seed=0)
        updater = MatrixUpdateAgent(matrix_agent=matrix, alpha=1.0)

        # Hammer the matrix with 100 updates
        for _ in range(100):
            v = rng.normal(0, 1, dim)
            updater.process(v)

        state = matrix.get_state()
        assert np.all(state >= -1.0), f"Min value: {state.min()}"
        assert np.all(state <= 1.0), f"Max value: {state.max()}"

    def test_extreme_inputs_dont_break_bounds(self):
        dim = 32
        matrix = MatrixAgent(dim=dim, std=0.0, seed=0)
        updater = MatrixUpdateAgent(matrix_agent=matrix, alpha=1000.0)

        extreme = np.ones(dim) * 1e6
        updater.process(extreme)

        state = matrix.get_state()
        assert np.all(state >= -1.0)
        assert np.all(state <= 1.0)
