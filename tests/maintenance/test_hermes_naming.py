"""Tests for the Sophia -> Hermes name_cluster client (#505)."""

from __future__ import annotations

from collections import Counter

import httpx

import sophia.maintenance.hermes_naming as hn
from sophia.maintenance.emergence_types import EmergentCluster, Member


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
