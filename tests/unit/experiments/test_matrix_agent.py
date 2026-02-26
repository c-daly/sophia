import numpy as np
from sophia.experiments.agents.matrix import MatrixAgent, make_matrix_agent


def test_matrix_init_gaussian():
    agent = MatrixAgent(dim=4, std=0.01, seed=42)
    state = agent.get_state()
    assert state.shape == (4, 4)
    assert np.abs(state.mean()) < 0.1
    assert state.std() > 0


def test_matrix_init_reproducible():
    a = MatrixAgent(dim=4, std=0.01, seed=42)
    b = MatrixAgent(dim=4, std=0.01, seed=42)
    np.testing.assert_array_equal(a.get_state(), b.get_state())


def test_matrix_filter_identity_when_zeros():
    """Zero matrix = identity filter via (M + I) * v."""
    agent = MatrixAgent(dim=4, std=0.0, seed=0)
    v = np.array([1.0, 2.0, 3.0, 4.0])
    result = agent.process(v)
    np.testing.assert_array_almost_equal(result, v)


def test_matrix_filter_modifies_embedding():
    agent = MatrixAgent(dim=4, std=0.5, seed=42)
    v = np.array([1.0, 2.0, 3.0, 4.0])
    result = agent.process(v)
    assert not np.allclose(result, v)


def test_matrix_snapshot_and_reset():
    agent = MatrixAgent(dim=4, std=0.01, seed=42)
    snap = agent.snapshot()
    agent._matrix += 999
    agent.reset(snap)
    np.testing.assert_array_equal(agent.get_state(), snap)


def test_factory():
    agent = make_matrix_agent({"dim": 8, "std": 0.01, "seed": 99})
    assert agent.get_state().shape == (8, 8)
