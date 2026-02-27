import numpy as np
from sophia.experiments.agents.matrix import MatrixAgent
from sophia.experiments.agents.updater import (
    MatrixUpdateAgent,
    OuterProductUpdateAgent,
    make_outer_product_update_agent,
    make_update_agent,
)


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


def test_outer_product_modifies_multiple_columns():
    matrix_agent = MatrixAgent(dim=4, std=0.0, seed=0)
    updater = OuterProductUpdateAgent(matrix_agent=matrix_agent, alpha=0.1)

    embedding = np.array([0.5, 0.9, 0.3, 0.7])
    updater.process(embedding)

    state = matrix_agent.get_state()
    # All columns should be modified (outer product touches everything)
    for j in range(4):
        assert not np.allclose(state[:, j], 0.0), f"Column {j} should be modified"


def test_outer_product_bounded_by_tanh():
    matrix_agent = MatrixAgent(dim=4, std=0.0, seed=0)
    updater = OuterProductUpdateAgent(matrix_agent=matrix_agent, alpha=100.0)

    embedding = np.array([0.5, 0.9, 0.3, 0.7])
    updater.process(embedding)

    state = matrix_agent.get_state()
    assert np.all(state >= -1.0)
    assert np.all(state <= 1.0)


def test_outer_product_proportional_to_embedding():
    """Columns corresponding to larger embedding values should change more."""
    matrix_agent = MatrixAgent(dim=4, std=0.0, seed=0)
    updater = OuterProductUpdateAgent(matrix_agent=matrix_agent, alpha=0.1)

    embedding = np.array([0.1, 0.9, 0.01, 0.5])  # dim 1 >> dim 2
    updater.process(embedding)

    state = matrix_agent.get_state()
    col1_change = np.abs(state[:, 1]).sum()
    col2_change = np.abs(state[:, 2]).sum()
    assert (
        col1_change > col2_change
    ), "Higher embedding value should cause larger column change"


def test_outer_product_factory():
    matrix_agent = MatrixAgent(dim=4, std=0.0, seed=0)
    updater = make_outer_product_update_agent(
        {"alpha": 0.05}, matrix_agent=matrix_agent
    )
    assert updater.alpha == 0.05
