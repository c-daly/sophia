"""Tests for the structural neighbor-relation signature (#505)."""

from __future__ import annotations

from collections import Counter

from sophia.maintenance.structural_signature import (
    build_signature,
    signature_similarity,
)


def test_build_signature_counts_relation_neighbor_pairs():
    neighbors = [
        {"relation": "MOVED_TO", "neighbor_type": "location"},
        {"relation": "LOCATED_ON", "neighbor_type": "object"},
        {"relation": "MOVED_TO", "neighbor_type": "location"},
    ]
    sig = build_signature(neighbors)
    assert sig == Counter({("MOVED_TO", "location"): 2, ("LOCATED_ON", "object"): 1})


def test_build_signature_skips_incomplete_neighbors():
    neighbors = [{"relation": "X"}, {"neighbor_type": "t"}, {}]
    assert build_signature(neighbors) == Counter()


def test_similarity_identical_is_one():
    a = Counter({("DEFINED_AS", "concept"): 1})
    assert signature_similarity(a, a) == 1.0


def test_similarity_disjoint_is_zero():
    a = Counter({("DEFINED_AS", "concept"): 1})
    b = Counter({("MOVED_TO", "location"): 1})
    assert signature_similarity(a, b) == 0.0


def test_similarity_partial_overlap_weighted_jaccard():
    # Counts matter (weighted): intersection = min(2,1) = 1, union = max(2,1)+max(1,0) = 3.
    a = Counter({("X", "t"): 2, ("Y", "t"): 1})
    b = Counter({("X", "t"): 1})
    assert abs(signature_similarity(a, b) - (1 / 3)) < 1e-9


def test_empty_signatures_similarity_is_zero():
    assert signature_similarity(Counter(), Counter()) == 0.0
