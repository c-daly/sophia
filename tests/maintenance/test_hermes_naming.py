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
    """A NEW group: name + parent chain index 1 + TOP-LEVEL residual_ids parsed out
    of the hermes v2 groups envelope, not a flat top-level name."""
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, json=json, headers=headers)
        return _FakeResp(
            {
                "groups": [
                    {
                        "assign_to": "NEW",
                        "name": "vehicle",
                        "chain": ["vehicle", "object", "entity"],
                        "member_ids": ["m1", "m2"],
                    }
                ],
                "residual_ids": ["m3"],
                "raw_partition_ok": True,
            }
        )

    monkeypatch.setattr(hn.httpx, "post", fake_post)

    result = hn.type_cluster(
        _cluster(),
        hermes_url="http://hermes:17000",
        token="t",
    )

    # name == the group name; parent == chain index 1 (proposed guaranteed-existing
    # super); residual_ids come from the TOP level, not the group.
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


def test_type_cluster_existing_type_reuse_has_no_parent(monkeypatch):
    """An EXISTING-type group (assign_to is a uuid, not NEW) proposes no parent:
    the handler re-points members onto the existing same-name type rather than
    minting, so chain index 1 is ignored and parent is None."""

    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResp(
            {
                "groups": [
                    {
                        "assign_to": "11111111-2222-3333-4444-555555555555",
                        "name": "vehicle",
                        "chain": ["vehicle"],
                    }
                ],
                "residual_ids": [],
            }
        )

    monkeypatch.setattr(hn.httpx, "post", fake_post)

    result = hn.type_cluster(_cluster(), hermes_url="http://h", token="t")
    assert result is not None
    assert result.name == "vehicle"
    assert result.parent is None
    assert result.residual_ids == []


def test_type_cluster_new_group_single_element_chain_has_no_parent(monkeypatch):
    """A NEW group whose chain holds only the type itself (no super) gives None."""

    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResp(
            {"groups": [{"assign_to": "NEW", "name": "concept", "chain": ["concept"]}]}
        )

    monkeypatch.setattr(hn.httpx, "post", fake_post)

    result = hn.type_cluster(_cluster(), hermes_url="http://h", token="t")
    assert result is not None
    assert result.name == "concept"
    assert result.parent is None
    assert result.residual_ids == []


def test_type_cluster_no_groups_key_returns_none(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResp({"residual_ids": [], "raw_partition_ok": True})

    monkeypatch.setattr(hn.httpx, "post", fake_post)
    assert hn.type_cluster(_cluster(), hermes_url="http://h", token="t") is None


def test_type_cluster_empty_groups_returns_none(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResp({"groups": [], "residual_ids": []})

    monkeypatch.setattr(hn.httpx, "post", fake_post)
    assert hn.type_cluster(_cluster(), hermes_url="http://h", token="t") is None


def test_type_cluster_empty_name_returns_none(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResp(
            {"groups": [{"assign_to": "NEW", "name": "", "chain": ["", "entity"]}]}
        )

    monkeypatch.setattr(hn.httpx, "post", fake_post)
    # Spy on the module logger directly rather than via caplog: in a full-suite
    # run the app logging config can disable propagation to caplog root handler,
    # so the (emitted) warning is not captured -- a CI-only flake. The contract
    # is result is None AND a warning was logged.
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


def test_type_cluster_group_missing_name_returns_none(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResp(
            {"groups": [{"assign_to": "NEW", "chain": ["vehicle"]}], "residual_ids": []}
        )

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
            return {"groups": [{"assign_to": "NEW", "name": "vehicle"}]}

    def fake_post(*args, **kwargs):
        return _FakeBadStatusResp()

    monkeypatch.setattr(hn.httpx, "post", fake_post)
    assert hn.type_cluster(_cluster(), hermes_url="http://h", token="t") is None


def test_type_cluster_null_chain_entry_yields_no_parent(monkeypatch):
    """A JSON null at chain[1] must NOT become the literal string "None" as a
    parent name; it should resolve to parent=None."""

    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResp(
            {
                "groups": [
                    {
                        "assign_to": "NEW",
                        "name": "vehicle",
                        "chain": ["vehicle", None, "entity"],
                    }
                ],
                "residual_ids": [],
            }
        )

    monkeypatch.setattr(hn.httpx, "post", fake_post)
    result = hn.type_cluster(_cluster(), hermes_url="http://h", token="t")
    assert result == TypeClusterResult(name="vehicle", parent=None, residual_ids=[])


def test_type_cluster_non_list_residual_ids_ignored(monkeypatch):
    """A non-list residual_ids (e.g. a bare string from a serialisation glitch)
    must not be iterated char-by-char; it is treated as no residuals."""

    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResp(
            {
                "groups": [
                    {
                        "assign_to": "NEW",
                        "name": "vehicle",
                        "chain": ["vehicle", "object"],
                    }
                ],
                "residual_ids": "m3",
            }
        )

    monkeypatch.setattr(hn.httpx, "post", fake_post)
    result = hn.type_cluster(_cluster(), hermes_url="http://h", token="t")
    assert result == TypeClusterResult(name="vehicle", parent="object", residual_ids=[])


def test_type_cluster_whitespace_assign_to_treated_as_new(monkeypatch):
    """A whitespace-only assign_to is a missing value, not an existing-type
    reuse: it must default to NEW so the parent is taken from the chain."""

    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResp(
            {
                "groups": [
                    {
                        "assign_to": "   ",
                        "name": "vehicle",
                        "chain": ["vehicle", "object", "entity"],
                    }
                ],
                "residual_ids": [],
            }
        )

    monkeypatch.setattr(hn.httpx, "post", fake_post)
    result = hn.type_cluster(_cluster(), hermes_url="http://h", token="t")
    assert result == TypeClusterResult(name="vehicle", parent="object", residual_ids=[])


def test_type_cluster_multiple_groups_uses_first_and_warns(monkeypatch):
    """Hermes v2 guarantees one group; if more arrive, use the first but log a
    warning so the contract violation is observable."""

    def fake_post(url, json=None, headers=None, timeout=None):
        return _FakeResp(
            {
                "groups": [
                    {
                        "assign_to": "NEW",
                        "name": "vehicle",
                        "chain": ["vehicle", "object"],
                    },
                    {"assign_to": "NEW", "name": "boat", "chain": ["boat", "object"]},
                ],
                "residual_ids": [],
            }
        )

    monkeypatch.setattr(hn.httpx, "post", fake_post)
    warnings: list[tuple] = []
    monkeypatch.setattr(hn.logger, "warning", lambda *a, **k: warnings.append((a, k)))
    result = hn.type_cluster(_cluster(), hermes_url="http://h", token="t")
    assert result == TypeClusterResult(name="vehicle", parent="object", residual_ids=[])
    assert warnings  # the >1-group contract violation was logged


def test_relation_synonyms_posts_and_parses(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(url=url, json=json, headers=headers)
        return _FakeResp(
            {"groups": [{"canonical": "CARRIES", "members": ["HAULS", "CARRIES"],
                         "confidence": 0.9}]}
        )

    monkeypatch.setattr(hn.httpx, "post", fake_post)
    groups = hn.relation_synonyms(
        ["HAULS", "CARRIES"], hermes_url="http://hermes:17000", token="t"
    )
    assert len(groups) == 1
    assert groups[0].canonical == "CARRIES"
    assert groups[0].members == ("HAULS", "CARRIES")
    assert captured["url"].endswith("/relation-synonyms")
    assert captured["headers"]["Authorization"] == "Bearer t"


def test_relation_synonyms_omits_auth_when_token_empty(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.update(headers=headers)
        return _FakeResp({"groups": []})

    monkeypatch.setattr(hn.httpx, "post", fake_post)
    hn.relation_synonyms(["A", "B"], hermes_url="http://h", token="")
    # empty token must not produce an illegal "Bearer " header
    assert "Authorization" not in captured["headers"]


def test_relation_synonyms_under_two_predicates_is_noop():
    assert hn.relation_synonyms(["ONLY_ONE"], hermes_url="http://h", token="t") == []


def test_relation_synonyms_returns_empty_on_error(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(hn.httpx, "post", fake_post)
    assert hn.relation_synonyms(["A", "B"], hermes_url="http://h", token="t") == []
