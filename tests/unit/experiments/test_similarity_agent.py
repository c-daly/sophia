import numpy as np
from sophia.experiments.agents.similarity import SimilarityAgent, make_similarity_agent


def test_identical_vectors_similarity_1():
    agent = SimilarityAgent()
    v = np.array([1.0, 0.0, 0.0])
    result = agent.process({"a": v, "b": v})
    assert abs(result - 1.0) < 1e-6


def test_orthogonal_vectors_similarity_0():
    agent = SimilarityAgent()
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    result = agent.process({"a": a, "b": b})
    assert abs(result) < 1e-6


def test_opposite_vectors_similarity_neg1():
    agent = SimilarityAgent()
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([-1.0, 0.0, 0.0])
    result = agent.process({"a": a, "b": b})
    assert abs(result - (-1.0)) < 1e-6


def test_factory():
    agent = make_similarity_agent({})
    assert isinstance(agent, SimilarityAgent)
