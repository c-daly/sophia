"""Tests for the type_emergence orchestration handler (#505)."""

from __future__ import annotations

from collections import Counter

from sophia.maintenance.config import MaintenanceConfig
from sophia.maintenance.emergence_handler import EmergenceHandler
from sophia.maintenance.emergence_types import Member, NameResult


def _members():
    phys = [
        Member(
            uuid=f"p{i}",
            name=f"p{i}",
            embedding=[0.0 + i * 0.01, 0.0],
            signature=Counter({("MOVED_TO", "location"): 1}),
            current_type="entity",
            hermes_type_hint="object",
            neighbors=[],
        )
        for i in range(4)
    ]
    con = [
        Member(
            uuid=f"c{i}",
            name=f"c{i}",
            embedding=[9.0 + i * 0.01, 9.0],
            signature=Counter({("DEFINED_AS", "concept"): 1}),
            current_type="entity",
            hermes_type_hint="concept",
            neighbors=[],
        )
        for i in range(4)
    ]
    return phys + con


def test_handler_mints_named_clusters_and_publishes():
    minted, published = [], []

    def fake_name(cluster, candidates, hermes_url, token):
        label = "object" if cluster.members[0].uuid.startswith("p") else "concept"
        return NameResult(label=label, description="", confidence=0.9)

    def fake_mint(cluster, name, hcg, milvus, source_cluster_id):
        minted.append(name.label)
        return f"type_{name.label}"

    class EB:
        def publish(self, channel, msg):
            published.append((channel, msg))

    handler = EmergenceHandler(
        config=MaintenanceConfig(),
        hcg=object(),
        milvus=object(),
        event_bus=EB(),
        hermes_url="http://h",
        token="t",
        load_members=lambda u: _members(),
        name_fn=fake_name,
        mint_fn=fake_mint,
        candidates_fn=lambda: ["object", "location", "concept"],
    )
    handler.run(type_uuid="type_entity")

    assert set(minted) == {"object", "concept"}
    assert len(published) == 2  # one ontology-change event per minted type


def test_handler_skips_low_confidence():
    minted = []

    def fake_name(cluster, candidates, hermes_url, token):
        return NameResult(label="x", description="", confidence=0.1)

    handler = EmergenceHandler(
        config=MaintenanceConfig(),
        hcg=object(),
        milvus=object(),
        event_bus=None,
        hermes_url="http://h",
        token="t",
        load_members=lambda u: _members(),
        name_fn=fake_name,
        mint_fn=lambda *a, **k: minted.append(1),
        candidates_fn=lambda: [],
    )
    handler.run(type_uuid="type_entity")

    assert minted == []  # all below hermes_confidence_floor (0.5)


def test_handler_feeds_minted_labels_into_later_candidates():
    # Each successful mint should append its label to the live candidate list
    # so a later cluster in the same run sees it (no within-run same-label blind
    # spot).
    seen_candidates = []

    def fake_name(cluster, candidates, hermes_url, token):
        seen_candidates.append(list(candidates))
        label = "object" if cluster.members[0].uuid.startswith("p") else "concept"
        return NameResult(label=label, description="", confidence=0.9)

    def fake_mint(cluster, name, hcg, milvus, source_cluster_id):
        return f"type_{name.label}_x"

    handler = EmergenceHandler(
        config=MaintenanceConfig(),
        hcg=object(),
        milvus=object(),
        event_bus=None,
        hermes_url="http://h",
        token="t",
        load_members=lambda u: _members(),
        name_fn=fake_name,
        mint_fn=fake_mint,
        candidates_fn=lambda: ["location"],
    )
    handler.run(type_uuid="type_entity")

    # First call sees only the seed; the second call sees the first mint's label.
    assert seen_candidates[0] == ["location"]
    assert seen_candidates[1][-1] in {"object", "concept"}
    assert len(seen_candidates[1]) == 2
