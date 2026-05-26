"""Structural signal for emergence: a node's neighbor-relation signature (#505).

A node is characterised by the multiset of (relation_type, neighbor_type) pairs
on its incident edges. Two nodes that connect to the same kinds of neighbours via
the same relations are structurally similar.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def build_signature(neighbors: list[dict[str, Any]]) -> Counter:
    """Multiset of (relation_type, neighbor_type) pairs for a node."""
    return Counter(
        (n["relation"], n["neighbor_type"])
        for n in neighbors
        if n.get("relation") and n.get("neighbor_type")
    )


def signature_similarity(a: Counter, b: Counter) -> float:
    """Weighted Jaccard similarity of two signatures in [0, 1]."""
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    intersection = sum(min(a[k], b[k]) for k in keys)
    union = sum(max(a[k], b[k]) for k in keys)
    return intersection / union if union else 0.0
