"""needs_reclassification is a work-queue flag for #504 type correction.

First-pass classify() only ever yields True ("needs reclassifying") -- the
placement is provisional. False ("nowhere better right now") and None ("hit a
cycle / something unusual -> flag for resolution") are written only by #504.
"""

from __future__ import annotations

from sophia.ingestion.type_classifier import MAX_DISTANCE, TypeClassifier


class _Milvus:
    def __init__(self, results):
        self._results = results

    def find_nearest_types(self, query_embedding, top_k):
        return list(self._results)


def _classify(results):
    return TypeClassifier(milvus=_Milvus(results), hcg=None).classify([0.1, 0.2])


def test_confident_match_still_flags_true():
    # Even a confident placement is provisional on first pass -> True.
    a = _classify([{"uuid": "type_dog_aa", "score": 0.2}])
    assert a.needs_reclassification is True
    assert a.confidence >= 0.5


def test_low_confidence_flags_true():
    a = _classify([{"uuid": "type_dog_aa", "score": 1.4}])
    assert a.needs_reclassification is True


def test_ambiguous_runner_up_flags_true():
    a = _classify(
        [
            {"uuid": "type_dog_aa", "score": 0.40},
            {"uuid": "type_wolf_bb", "score": 0.45},
        ]
    )
    assert a.needs_reclassification is True


def test_beyond_max_distance_flags_true():
    a = _classify([{"uuid": "type_dog_aa", "score": MAX_DISTANCE + 0.5}])
    assert a.needs_reclassification is True


def test_no_candidates_flags_true_and_falls_back_to_entity():
    a = _classify([])
    assert a.needs_reclassification is True
    assert a.type_name == "entity"


def test_first_pass_never_writes_false_or_none():
    # The other two ternary values are #504's to write, never first-pass.
    for results in (
        [{"uuid": "type_x", "score": 0.1}],
        [{"uuid": "type_x", "score": 1.9}],
        [],
    ):
        assert _classify(results).needs_reclassification is True
