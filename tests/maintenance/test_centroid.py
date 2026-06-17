"""Tests for incremental + reconcile type-centroid maintenance."""

from __future__ import annotations

from sophia.maintenance.centroid import (
    bump_centroid,
    online_mean_add,
    reconcile_centroids,
)


# --------------------------------------------------------------- online_mean_add
def test_online_mean_init_when_no_centroid():
    """No prior centroid (or n<=0) -> plain mean of the added vectors."""
    assert online_mean_add(None, 0, [[2.0, 4.0], [4.0, 8.0]]) == [3.0, 6.0]
    assert online_mean_add([], 5, [[1.0, 1.0]]) == [1.0, 1.0]


def test_online_mean_add_folds_into_running_mean():
    """centroid of n vectors + one more == correct new mean."""
    # mean of 2 vectors is [2,2]; add [8,8] -> (([2,2]*2)+[8,8])/3 = [4,4]
    assert online_mean_add([2.0, 2.0], 2, [[8.0, 8.0]]) == [4.0, 4.0]


def test_online_mean_equivalent_to_full_mean():
    """Incremental folding reproduces the full batch mean exactly."""
    embs = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    c = None
    n = 0
    for e in embs:
        c = online_mean_add(c, n, [e])
        n += 1
    full = [sum(e[i] for e in embs) / len(embs) for i in range(2)]
    assert c == full == [3.0, 4.0]


def test_online_mean_empty_add_is_noop():
    assert online_mean_add([1.0, 1.0], 3, []) == [1.0, 1.0]
    assert online_mean_add([1.0, 1.0], 3, [None]) == [1.0, 1.0]


# ------------------------------------------------------------------ FakeMilvus
class FakeMilvus:
    def __init__(self, centroids=None, content=None):
        # centroids: {uuid: (vec, model)} ; content: {uuid: (vec, model)}
        self.centroids = dict(centroids or {})
        self.content = dict(content or {})
        self.writes = []

    def get_embedding(self, node_type, uuid):
        src = self.centroids if node_type == "TypeCentroid" else self.content
        if uuid in src:
            v, m = src[uuid]
            return {"uuid": uuid, "embedding": v, "embedding_model": m}
        return None

    def update_centroid(self, type_uuid, centroid, model):
        self.centroids[type_uuid] = (centroid, model)
        self.writes.append((type_uuid, centroid, model))


# ----------------------------------------------------------------- bump_centroid
def test_bump_folds_member_into_existing_centroid():
    mv = FakeMilvus(centroids={"t": ([2.0, 2.0], "text-embedding-3-large")})
    bump_centroid(mv, "t", [[8.0, 8.0]], n=2)
    assert mv.centroids["t"] == ([4.0, 4.0], "text-embedding-3-large")


def test_bump_initialises_when_no_centroid_using_passed_model():
    mv = FakeMilvus()
    bump_centroid(mv, "new", [[4.0, 4.0], [6.0, 6.0]], n=0, model="m2")
    assert mv.centroids["new"] == ([5.0, 5.0], "m2")


def test_bump_skips_rather_than_corrupts_when_count_unknown():
    """Existing centroid + n<=0 (count unavailable) -> skip, don't re-init to
    just the new members' mean (that would lose the existing population)."""
    mv = FakeMilvus(centroids={"t": ([2.0, 2.0], "m")})
    bump_centroid(mv, "t", [[8.0, 8.0]], n=0)
    assert mv.centroids["t"] == ([2.0, 2.0], "m")  # unchanged
    assert mv.writes == []


def test_bump_is_fail_soft():
    # No milvus / no uuid / no embeddings -> no error, no write.
    bump_centroid(None, "t", [[1.0]], n=1)
    mv = FakeMilvus()
    bump_centroid(mv, "", [[1.0]], n=1)
    bump_centroid(mv, "t", [], n=1)
    assert mv.writes == []


# ------------------------------------------------------------- reconcile_centroids
class FakeHCG:
    def __init__(self, types, members):
        self._types = types  # list of {uuid,name,...}
        self._members = members  # {type_uuid: [member_uuid,...]}

    def get_all_type_definitions(self):
        return self._types

    def _execute_read(self, query, params):
        return [{"uuid": u} for u in self._members.get(params["u"], [])]


def test_reconcile_writes_content_types_skips_roots_and_nodata():
    types = [
        {"uuid": "engine", "name": "engine"},
        {"uuid": "type_entity", "name": "entity"},  # protected root
        {"uuid": "ghost", "name": "ghost"},  # members lack embeddings
    ]
    members = {"engine": ["m1", "m2"], "type_entity": ["e1"], "ghost": ["g1"]}
    content = {
        "m1": ([0.0, 2.0], "text-embedding-3-large"),
        "m2": ([2.0, 0.0], "text-embedding-3-large"),
        # g1 has no embedding -> ghost skipped
    }
    hcg = FakeHCG(types, members)
    mv = FakeMilvus(content=content)

    stats = reconcile_centroids(hcg, mv)

    assert mv.centroids["engine"] == ([1.0, 1.0], "text-embedding-3-large")
    assert "type_entity" not in mv.centroids  # protected, skipped
    assert "ghost" not in mv.centroids  # no member embeddings, skipped
    assert stats["written"] == 1
    assert stats["skipped_protected"] == 1
    assert stats["skipped_no_embeddings"] == 1


def test_reconcile_skips_mixed_model():
    types = [{"uuid": "mix", "name": "mix"}]
    members = {"mix": ["a", "b"]}
    content = {"a": ([1.0], "model-x"), "b": ([2.0], "model-y")}
    hcg = FakeHCG(types, members)
    mv = FakeMilvus(content=content)
    stats = reconcile_centroids(hcg, mv)
    assert "mix" not in mv.centroids
    assert stats["skipped_mixed_model"] == 1


def test_reconcile_fail_soft_on_none():
    assert reconcile_centroids(None, None)["written"] == 0


def test_online_mean_add_raises_on_dim_mismatch():
    """online_mean_add raises AssertionError when existing centroid dim != new-vec dim."""
    import pytest

    with pytest.raises(AssertionError, match="dimension mismatch"):
        online_mean_add([1.0, 2.0], 3, [[1.0, 2.0, 3.0]])  # centroid dim=2, vec dim=3
