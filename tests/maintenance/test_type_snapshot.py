"""Tests for the shared positional type-snapshot writer."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from sophia.maintenance.type_snapshot import REDIS_KEY, publish_type_snapshot


def _hcg(records, found=None):
    hcg = MagicMock()
    hcg.get_all_type_definitions.return_value = records
    # publish_type_snapshot resolves missing graftable realms by name; default to
    # an empty dict so existing tests (no realms in the graph) stay unaffected.
    hcg.find_nodes_by_names.return_value = found or {}
    return hcg


def test_publish_always_includes_graftable_realms_even_without_members() -> None:
    """The three graftable realms (entity/concept/process) are always in the
    catalog so the naming LLM always has a valid parent -- even on a cold graph
    where they have no members and the positional pass omits them."""
    redis = MagicMock()
    hcg = _hcg(
        # positional pass returns only entity (has members) + a content type
        [
            {"uuid": "u-entity", "name": "entity", "properties": {"member_count": 7}},
            {
                "uuid": "u-organelle",
                "name": "organelle",
                "properties": {"member_count": 3},
            },
        ],
        # concept/process have no members -> must be resolved by name
        found={
            "entity": {"uuid": "u-entity"},
            "concept": {"uuid": "u-concept"},
            "process": {"uuid": "u-process"},
        },
    )

    publish_type_snapshot(hcg, redis)

    _, payload = redis.set.call_args[0]
    snap = json.loads(payload)
    assert {"entity", "concept", "process"} <= set(snap)
    assert "organelle" in snap
    # entity keeps its real member_count from the positional pass
    assert snap["entity"]["member_count"] == 7
    # a realm resolved by name (no members) is published with member_count 0
    assert snap["concept"]["member_count"] == 0


def test_publish_excludes_structural_roots() -> None:
    """node/root are structural scaffolding, never graftable parents -- they must
    not be published to the catalog (hermes bars them, and they pollute it)."""
    redis = MagicMock()
    hcg = _hcg(
        [
            {"uuid": "u-root", "name": "root", "properties": {"member_count": 1}},
            {"uuid": "u-node", "name": "node", "properties": {"member_count": 5}},
            {"uuid": "u-engine", "name": "engine", "properties": {"member_count": 2}},
        ],
        found={
            "entity": {"uuid": "u-e"},
            "concept": {"uuid": "u-c"},
            "process": {"uuid": "u-p"},
        },
    )

    publish_type_snapshot(hcg, redis)

    _, payload = redis.set.call_args[0]
    snap = json.loads(payload)
    assert "node" not in snap and "root" not in snap
    assert "engine" in snap


def test_writes_name_keyed_snapshot() -> None:
    """Snapshot is keyed by type NAME with {uuid, member_count} values."""
    redis = MagicMock()
    hcg = _hcg(
        [
            {"uuid": "u-engine", "name": "engine", "properties": {"member_count": 5}},
            {"uuid": "u-protein", "name": "protein", "properties": {"member_count": 3}},
        ]
    )

    count = publish_type_snapshot(hcg, redis)

    assert count == 2
    key, payload = redis.set.call_args[0]
    assert key == REDIS_KEY
    assert json.loads(payload) == {
        "engine": {"uuid": "u-engine", "member_count": 5},
        "protein": {"uuid": "u-protein", "member_count": 3},
    }


def test_skips_records_without_a_name() -> None:
    """Nameless rows are dropped rather than written under an empty key."""
    redis = MagicMock()
    hcg = _hcg(
        [
            {"uuid": "u-x", "name": "", "properties": {"member_count": 1}},
            {"uuid": "u-engine", "name": "engine", "properties": {}},
        ]
    )

    count = publish_type_snapshot(hcg, redis)

    assert count == 1
    _, payload = redis.set.call_args[0]
    assert json.loads(payload) == {"engine": {"uuid": "u-engine", "member_count": 0}}


def test_skips_reserved_underscore_names() -> None:
    """Reserved/internal scaffolding (`_`-prefixed) is never published (#152)."""
    redis = MagicMock()
    hcg = _hcg(
        [
            {
                "uuid": "u-res",
                "name": "_reserved_node",
                "properties": {"member_count": 9},
            },
            {"uuid": "u-cog", "name": "_cognition", "properties": {"member_count": 4}},
            {"uuid": "u-engine", "name": "engine", "properties": {"member_count": 5}},
        ]
    )

    count = publish_type_snapshot(hcg, redis)

    assert count == 1
    _, payload = redis.set.call_args[0]
    assert json.loads(payload) == {"engine": {"uuid": "u-engine", "member_count": 5}}


def test_skips_non_string_name_without_crashing() -> None:
    """A malformed non-string name is dropped, not raised on .startswith."""
    redis = MagicMock()
    hcg = _hcg(
        [
            {"uuid": "u-bad", "name": 123, "properties": {"member_count": 1}},
            {"uuid": "u-engine", "name": "engine", "properties": {"member_count": 5}},
        ]
    )

    count = publish_type_snapshot(hcg, redis)

    assert count == 1
    _, payload = redis.set.call_args[0]
    assert json.loads(payload) == {"engine": {"uuid": "u-engine", "member_count": 5}}


def test_missing_properties_defaults_member_count_to_zero() -> None:
    """A record with no properties dict still serialises with member_count 0."""
    redis = MagicMock()
    hcg = _hcg([{"uuid": "u-engine", "name": "engine"}])

    publish_type_snapshot(hcg, redis)

    _, payload = redis.set.call_args[0]
    assert json.loads(payload) == {"engine": {"uuid": "u-engine", "member_count": 0}}


def test_fail_soft_on_none_redis() -> None:
    """No Redis handle -> no-op returning 0, never touching the graph."""
    hcg = _hcg([{"uuid": "u", "name": "engine", "properties": {}}])
    assert publish_type_snapshot(hcg, None) == 0
    hcg.get_all_type_definitions.assert_not_called()


def test_fail_soft_on_none_hcg() -> None:
    """No HCG client -> no-op returning 0, never touching Redis."""
    redis = MagicMock()
    assert publish_type_snapshot(None, redis) == 0
    redis.set.assert_not_called()


def test_fail_soft_on_graph_error() -> None:
    """A graph/Redis exception is swallowed and reported as 0 (sophia#195)."""
    redis = MagicMock()
    hcg = MagicMock()
    hcg.get_all_type_definitions.side_effect = RuntimeError("neo4j down")
    assert publish_type_snapshot(hcg, redis) == 0
    redis.set.assert_not_called()


def test_name_collision_logs_warning() -> None:
    """Same-name types produce a warning; the second clobbers the first.

    Uses unittest.mock to capture the warning directly on the module logger so
    the test is not sensitive to whether a parent logger (e.g. "sophia") has
    propagate=False (which breaks caplog when API tests run in the same session).
    """
    from unittest.mock import patch

    redis = MagicMock()
    hcg = _hcg(
        [
            {"uuid": "u-alpha", "name": "engine", "properties": {"member_count": 2}},
            {"uuid": "u-beta", "name": "engine", "properties": {"member_count": 7}},
        ]
    )

    with patch("sophia.maintenance.type_snapshot.logger") as mock_logger:
        count = publish_type_snapshot(hcg, redis)

    # Only one key written (collision); last wins
    assert count == 1
    _, payload = redis.set.call_args[0]
    written = json.loads(payload)
    assert written["engine"]["uuid"] == "u-beta"
    # Warning was emitted
    mock_logger.warning.assert_called_once()
    warning_args = mock_logger.warning.call_args[0]
    assert "collision" in warning_args[0]
