"""Tests for the relation-vocabulary rollup handler (sophia#192).

Orchestration is unit-tested with injected dependencies (embed / synonym /
hcg). The handler: reads the descriptive-relation vocabulary, clusters the
labels by embedding so true synonyms co-locate, asks Hermes to name synonym
groups per cluster, and consolidates via rename_relation above a confidence
gate. Reserved typing relations never reach it.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from sophia.maintenance.hermes_naming import RelationSynonymGroup
from sophia.maintenance.relation_rollup_handler import RelationRollupHandler


def _hcg(vocab):
    hcg = MagicMock()
    hcg.get_relation_vocabulary.return_value = [
        {"relation": r, "edge_count": c} for r, c in vocab
    ]
    hcg.rename_relation.return_value = 1
    return hcg


def _embed_two_clusters(labels):
    # HAULS/DRAGS/CARRIES -> cluster A (x-axis); FAST/QUICK -> cluster B (y-axis)
    A = {"HAULS", "DRAGS", "CARRIES"}
    return [([1.0, 0.0] if l in A else [0.0, 1.0]) for l in labels]


def test_consolidates_synonym_group_via_rename():
    hcg = _hcg([("HAULS", 1), ("DRAGS", 1), ("CARRIES", 5)])

    def synonym_fn(preds, context=None):
        return [RelationSynonymGroup("CARRIES", ["HAULS", "DRAGS", "CARRIES"], 0.9)]

    handler = RelationRollupHandler(
        hcg, embed_fn=lambda ls: [[1.0, 0.0]] * len(ls), synonym_fn=synonym_fn
    )
    summary = handler.run()

    renamed = {c.args for c in hcg.rename_relation.call_args_list}
    assert ("HAULS", "CARRIES") in renamed
    assert ("DRAGS", "CARRIES") in renamed
    # the canonical is never renamed onto itself
    assert ("CARRIES", "CARRIES") not in renamed
    assert summary["groups_applied"] == 1
    assert summary["edges_renamed"] == 2


def test_confidence_gate_skips_low_confidence_groups():
    hcg = _hcg([("HAULS", 1), ("CARRIES", 5)])

    def synonym_fn(preds, context=None):
        return [RelationSynonymGroup("CARRIES", ["HAULS", "CARRIES"], 0.3)]

    handler = RelationRollupHandler(
        hcg,
        embed_fn=lambda ls: [[1.0, 0.0]] * len(ls),
        synonym_fn=synonym_fn,
        min_confidence=0.6,
    )
    summary = handler.run()
    hcg.rename_relation.assert_not_called()
    assert summary["groups_applied"] == 0


def test_clusters_route_only_co_located_labels_to_one_synonym_call():
    hcg = _hcg([("HAULS", 1), ("DRAGS", 1), ("CARRIES", 5), ("FAST", 1), ("QUICK", 1)])
    seen_batches = []

    def synonym_fn(preds, context=None):
        seen_batches.append(set(preds))
        return []

    handler = RelationRollupHandler(
        hcg, embed_fn=_embed_two_clusters, synonym_fn=synonym_fn, cluster_threshold=0.5
    )
    handler.run()
    # two clusters -> the haul-group and the fast-group never share a batch
    assert {"HAULS", "DRAGS", "CARRIES"} in seen_batches
    assert {"FAST", "QUICK"} in seen_batches
    for batch in seen_batches:
        assert not ({"CARRIES"} & batch and {"FAST"} & batch)


def test_singleton_clusters_are_not_sent():
    hcg = _hcg([("ALONE", 1), ("HAULS", 1), ("CARRIES", 2)])
    calls = []

    def synonym_fn(preds, context=None):
        calls.append(list(preds))
        return []

    # ALONE sits on its own axis -> its own cluster of 1, never sent
    def embed(ls):
        m = {"ALONE": [0, 0, 1], "HAULS": [1, 0, 0], "CARRIES": [1, 0, 0]}
        return [m[l] for l in ls]

    RelationRollupHandler(
        hcg, embed_fn=embed, synonym_fn=synonym_fn, cluster_threshold=0.5
    ).run()
    assert all("ALONE" not in c for c in calls)


def test_empty_vocabulary_is_a_noop():
    hcg = _hcg([])
    handler = RelationRollupHandler(
        hcg, embed_fn=lambda ls: [], synonym_fn=lambda p, context=None: []
    )
    summary = handler.run()
    assert summary == {"groups_applied": 0, "edges_renamed": 0, "clusters": 0}


def test_embed_failure_aborts_cleanly():
    hcg = _hcg([("HAULS", 1), ("CARRIES", 2)])

    def boom(ls):
        raise RuntimeError("hermes embed down")

    handler = RelationRollupHandler(
        hcg, embed_fn=boom, synonym_fn=lambda p, context=None: []
    )
    summary = handler.run()  # must not raise
    assert summary["edges_renamed"] == 0
    hcg.rename_relation.assert_not_called()
