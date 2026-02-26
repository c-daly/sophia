import numpy as np
from sophia.experiments.agents.matrix import MatrixAgent
from sophia.experiments.agents.updater import MatrixUpdateAgent, make_update_agent


def test_update_modifies_one_column():
    matrix_agent = MatrixAgent(dim=4, std=0.0, seed=0)  # zero matrix
    updater = MatrixUpdateAgent(matrix_agent=matrix_agent, alpha=0.1)

    embedding = np.array([0.1, 0.9, 0.3, 0.2])  # dim 1 has highest abs
    updater.process(embedding)

    state = matrix_agent.get_state()
    # Column 1 should be modified (highest abs value dimension)
    assert not np.allclose(state[:, 1], 0.0)
    # Other columns should be unchanged
    np.testing.assert_array_equal(state[:, 0], np.zeros(4))
    np.testing.assert_array_equal(state[:, 2], np.zeros(4))
    np.testing.assert_array_equal(state[:, 3], np.zeros(4))


def test_update_bounded_by_tanh():
    matrix_agent = MatrixAgent(dim=4, std=0.0, seed=0)
    updater = MatrixUpdateAgent(matrix_agent=matrix_agent, alpha=100.0)  # huge alpha

    embedding = np.array([0.1, 0.9, 0.3, 0.2])
    updater.process(embedding)

    state = matrix_agent.get_state()
    assert np.all(state >= -1.0)
    assert np.all(state <= 1.0)


def test_update_alpha_scales_magnitude():
    m1 = MatrixAgent(dim=4, std=0.0, seed=0)
    m2 = MatrixAgent(dim=4, std=0.0, seed=0)

    u1 = MatrixUpdateAgent(matrix_agent=m1, alpha=0.01)
    u2 = MatrixUpdateAgent(matrix_agent=m2, alpha=0.1)

    embedding = np.array([0.1, 0.9, 0.3, 0.2])
    u1.process(embedding)
    u2.process(embedding)

    # Larger alpha = bigger change
    assert np.abs(m2.get_state()).sum() > np.abs(m1.get_state()).sum()


def test_factory():
    matrix_agent = MatrixAgent(dim=4, std=0.0, seed=0)
    updater = make_update_agent({"alpha": 0.05}, matrix_agent=matrix_agent)
    assert updater.alpha == 0.05
