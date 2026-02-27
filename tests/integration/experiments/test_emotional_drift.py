import pytest

pytest.importorskip("openai", reason="openai not installed")

import numpy as np
from sophia.experiments.emotional_drift import run_emotional_drift_experiment


@pytest.mark.integration
def test_emotional_drift_runs_to_completion():
    """Smoke test: the experiment runs and produces artifacts."""
    artifacts = run_emotional_drift_experiment(
        seed=42,
        matrix_std=0.01,
        alpha=0.01,
        dim=1536,
    )
    assert "baseline_similarities" in artifacts
    assert "filtered_similarities" in artifacts
    assert "matrix_before" in artifacts
    assert "matrix_after" in artifacts
    assert len(artifacts["baseline_similarities"]) > 0
    assert len(artifacts["filtered_similarities"]) > 0


@pytest.mark.integration
def test_emotional_drift_matrix_changes():
    """The matrix should be different after processing emotional inputs."""
    artifacts = run_emotional_drift_experiment(
        seed=42,
        matrix_std=0.01,
        alpha=0.01,
        dim=1536,
    )
    assert not np.allclose(artifacts["matrix_before"], artifacts["matrix_after"])
