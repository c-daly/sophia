"""Tests for the Sophia -> Hermes name_cluster client (#505)."""

from __future__ import annotations

from collections import Counter

import httpx

import sophia.maintenance.hermes_naming as hn
from sophia.maintenance.emergence_types import (
    EmergentCluster,
    Member,
    TypeClusterResult,
)


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _cluster() -> EmergentCluster:
    m = Member(
        uuid="u1",
        name="derivative",
        embedding=[0.1],
        signature=Counter({("DEFINED_AS", "concept"): 1}),
        current_type="entity",
        hermes_type_hint="concept",
        neighbors=[],
    )
    return EmergentCluster(members=[m])


def test_name_cluster_posts_and_parses(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, json=json, headers=headers)
        return _FakeResp(
            {"label": "concept", "description": "ideas", "confidence": 0.8}
        )

    monkeypatch.setattr(hn.httpx, "post", fake_post)

    result = hn.name_cluster(
        _cluster(),
        candidates=["object", "concept"],
        hermes_url="http://hermes:17000",
        token="t",
    )

    assert result is not None
    assert result.label == "concept" and result.confidence == 0.8
    assert captured["url"].endswith("/name-cluster")
    assert captured["headers"]["Authorization"] == "Bearer t"
    assert captured["json"]["candidates"] == ["object", "concept"]
    assert captured["json"]["members"][0]["hermes_type_hint"] == "concept"
    # The member's uuid travels as `id` so Hermes can flag it back by id (#504).
    assert captured["json"]["members"][0]["id"] == "u1"
    # No outliers flagged -> empty removed list, not an error.
    assert result.removed == []


def test_name_cluster_parses_removed_outliers(monkeypatch):
    """Member ids Hermes flags as outliers come back on NameResult.removed (#504)."""

    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResp({"label": "tree", "confidence": 0.9, "removed": ["u7", "u9"]})

    monkeypatch.setattr(hn.httpx, "post", fake_post)

    result = hn.name_cluster(
        _cluster(), candidates=[], hermes_url="http://h", token="t"
    )
    assert result is not None
    assert result.removed == ["u7", "u9"]


def test_name_cluster_returns_none_on_error(monkeypatch):
    def fake_post(*args, **kwargs):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(hn.httpx, "post", fake_post)

    assert (
        hn.name_cluster(_cluster(), candidates=[], hermes_url="http://h", token="t")
        is None
    )


def test_name_cluster_samples_down_large_clusters(monkeypatch):
    members = [
        Member(
            uuid=f"u{i}",
            name=f"n{i}",
            embedding=[float(i)],
            signature=Counter(),
            current_type="entity",
            hermes_type_hint=None,
            neighbors=[],
        )
        for i in range(20)
    ]
    cluster = EmergentCluster(members=members)
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(json=json)
        return _FakeResp({"label": "x", "confidence": 0.9})

    monkeypatch.setattr(hn.httpx, "post", fake_post)
    hn.name_cluster(
        cluster,
        candidates=[],
        hermes_url="http://h",
        token="t",
        max_members=5,
    )
    assert len(captured["json"]["members"]) == 5


def test_name_cluster_sends_all_when_under_max(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(json=json)
        return _FakeResp({"label": "x", "confidence": 0.9})

    monkeypatch.setattr(hn.httpx, "post", fake_post)
    hn.name_cluster(
        _cluster(),
        candidates=[],
        hermes_url="http://h",
        token="t",
        max_members=50,
    )
    assert len(captured["json"]["members"]) == 1


def test_type_cluster_posts_and_parses(monkeypatch):
    """Hermes /type-cluster returns a FLAT verdict (#127/#199): top-level name +
    parent + residual_ids -- no `groups` envelope, no chain/assign_to."""
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, json=json, headers=headers)
        return _FakeResp(
            {
                "name": "vehicle",
                "parent": "object",
                "residual_ids": ["m3"],
                "over_specified": False,
                "raw_partition_ok": True,
            }
        )

    monkeypatch.setattr(hn.httpx, "post", fake_post)

    result = hn.type_cluster(
        _cluster(),
        hermes_url="http://hermes:17000",
        token="t",
    )

    assert result == TypeClusterResult(
        name="vehicle", parent="object", residual_ids=["m3"]
    )
    assert captured["url"].endswith("/type-cluster")
    assert captured["headers"]["Authorization"] == "Bearer t"
    # Type catalog is server-side for /type-cluster: no candidates are sent.
    assert "candidates" not in captured["json"]
    # /type-cluster members carry no `type` field (unlike /name-cluster).
    member = captured["json"]["members"][0]
    assert "type" not in member
    assert member["id"] == "u1"
    assert member["name"] == "derivative"
    assert member["hermes_type_hint"] == "concept"


def test_type_cluster_reuse_has_null_parent(monkeypatch):
    """parent=null => `name` is an existing type to REUSE; parent stays None."""

    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResp({"name": "vehicle", "parent": None, "residual_ids": []})

    monkeypatch.setattr(hn.httpx, "post", fake_post)

    result = hn.type_cluster(_cluster(), hermes_url="http://h", token="t")
    assert result == TypeClusterResult(name="vehicle", parent=None, residual_ids=[])


def test_type_cluster_missing_parent_key_is_none(monkeypatch):
    """An omitted `parent` key is treated as null (reuse), not an error."""

    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResp({"name": "concept", "residual_ids": []})

    monkeypatch.setattr(hn.httpx, "post", fake_post)
    result = hn.type_cluster(_cluster(), hermes_url="http://h", token="t")
    assert result == TypeClusterResult(name="concept", parent=None, residual_ids=[])


def test_type_cluster_no_name_returns_none(monkeypatch):
    """No top-level `name` => failure (None) with a warning."""

    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResp(
            {"parent": "object", "residual_ids": [], "raw_partition_ok": True}
        )

    monkeypatch.setattr(hn.httpx, "post", fake_post)
    # Spy on the module logger directly rather than via caplog: in a full-suite
    # run the app logging config can disable propagation to caplog root handler,
    # so the (emitted) warning is not captured -- a CI-only flake. The contract
    # is result is None AND a warning was logged.
    warnings: list[tuple] = []
    monkeypatch.setattr(hn.logger, "warning", lambda *a, **k: warnings.append((a, k)))
    assert hn.type_cluster(_cluster(), hermes_url="http://h", token="t") is None
    assert warnings


def test_type_cluster_empty_name_returns_none(monkeypatch):
    """A whitespace-only `name` is a failure (None) with a warning."""

    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResp({"name": "   ", "parent": "object", "residual_ids": []})

    monkeypatch.setattr(hn.httpx, "post", fake_post)
    warnings: list[tuple] = []
    monkeypatch.setattr(hn.logger, "warning", lambda *a, **k: warnings.append((a, k)))
    result = hn.type_cluster(_cluster(), hermes_url="http://h", token="t")
    assert result is None
    assert warnings


def test_type_cluster_returns_none_on_non_dict_json(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResp(["not", "a", "dict"])

    monkeypatch.setattr(hn.httpx, "post", fake_post)
    assert hn.type_cluster(_cluster(), hermes_url="http://h", token="t") is None


def test_type_cluster_returns_none_on_connect_error(monkeypatch):
    def fake_post(*args, **kwargs):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(hn.httpx, "post", fake_post)
    assert hn.type_cluster(_cluster(), hermes_url="http://h", token="t") is None


def test_type_cluster_returns_none_on_http_status_error(monkeypatch):
    class _FakeBadStatusResp:
        def raise_for_status(self):
            raise httpx.HTTPError("500 server error")

        def json(self):
            return {"name": "vehicle", "parent": None}

    def fake_post(*args, **kwargs):
        return _FakeBadStatusResp()

    monkeypatch.setattr(hn.httpx, "post", fake_post)
    assert hn.type_cluster(_cluster(), hermes_url="http://h", token="t") is None


def test_type_cluster_null_parent_yields_no_parent(monkeypatch):
    """A JSON null `parent` must resolve to parent=None, never the string "None"."""

    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResp({"name": "vehicle", "parent": None, "residual_ids": []})

    monkeypatch.setattr(hn.httpx, "post", fake_post)
    result = hn.type_cluster(_cluster(), hermes_url="http://h", token="t")
    assert result == TypeClusterResult(name="vehicle", parent=None, residual_ids=[])


def test_type_cluster_non_list_residual_ids_ignored(monkeypatch):
    """A non-list residual_ids (e.g. a bare string from a serialisation glitch)
    must not be iterated char-by-char; it is treated as no residuals."""

    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResp({"name": "vehicle", "parent": "object", "residual_ids": "m3"})

    monkeypatch.setattr(hn.httpx, "post", fake_post)
    result = hn.type_cluster(_cluster(), hermes_url="http://h", token="t")
    assert result == TypeClusterResult(name="vehicle", parent="object", residual_ids=[])


def test_type_cluster_whitespace_parent_treated_as_none(monkeypatch):
    """A whitespace-only `parent` collapses to None (reuse), not a bogus parent
    name that would force a spurious graft."""

    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResp({"name": "vehicle", "parent": "   ", "residual_ids": []})

    monkeypatch.setattr(hn.httpx, "post", fake_post)
    result = hn.type_cluster(_cluster(), hermes_url="http://h", token="t")
    assert result == TypeClusterResult(name="vehicle", parent=None, residual_ids=[])
