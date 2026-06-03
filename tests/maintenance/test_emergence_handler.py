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

    def fake_mint(cluster, name, hcg, milvus, source_cluster_id, **kwargs):
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


def test_handler_isolates_failing_cluster():
    """A mint failure on one cluster must not abort the rest of the run."""
    minted = []

    def fake_name(cluster, candidates, hermes_url, token):
        label = "object" if cluster.members[0].uuid.startswith("p") else "concept"
        return NameResult(label=label, description="", confidence=0.9)

    def flaky_mint(cluster, name, hcg, milvus, source_cluster_id, **kwargs):
        if name.label == "object":
            raise RuntimeError("transient HCG write error")
        minted.append(name.label)
        return f"type_{name.label}"

    handler = EmergenceHandler(
        config=MaintenanceConfig(),
        hcg=object(),
        milvus=object(),
        event_bus=None,
        hermes_url="http://h",
        token="t",
        load_members=lambda u: _members(),
        name_fn=fake_name,
        mint_fn=flaky_mint,
        candidates_fn=lambda: ["object", "location", "concept"],
    )

    # Must not raise even though the "object" cluster's mint blows up.
    handler.run(type_uuid="type_entity")

    # The failing cluster is skipped; the other cluster still mints.
    assert minted == ["concept"]


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

    def fake_mint(cluster, name, hcg, milvus, source_cluster_id, **kwargs):
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


def test_handler_mints_under_entity_via_hierarchy(monkeypatch):
    """Emergence must (a) drive minting off the hierarchy roll-up and (b) parent
    every minted type under `type_entity` with the entity lineage (#505)."""
    from sophia.maintenance import emergence_handler as eh
    from sophia.maintenance.emergence_clustering import HierarchyNode
    from sophia.maintenance.emergence_types import EmergentCluster

    seen_members, mint_kwargs = [], []

    def fake_hierarchy(members, *, min_cluster_size, variance_threshold):
        # Two top-level hierarchy nodes, one per disjoint group of members.
        phys = [m for m in members if m.uuid.startswith("p")]
        con = [m for m in members if m.uuid.startswith("c")]
        return [
            HierarchyNode(members=phys, centroid=[0.0, 0.0]),
            HierarchyNode(members=con, centroid=[9.0, 9.0]),
        ]

    monkeypatch.setattr(eh, "find_emergent_hierarchy", fake_hierarchy)

    def fake_name(cluster, candidates, hermes_url, token):
        assert isinstance(cluster, EmergentCluster)
        seen_members.append([m.uuid for m in cluster.members])
        label = "object" if cluster.members[0].uuid.startswith("p") else "concept"
        return NameResult(label=label, description="", confidence=0.9)

    def fake_mint(cluster, name, hcg, milvus, source_cluster_id, **kwargs):
        mint_kwargs.append(kwargs)
        return f"type_{name.label}_x"

    published = []

    class EB:
        def publish(self, channel, msg):
            published.append(msg)

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
        candidates_fn=lambda: [],
    )
    handler.run(type_uuid="type_entity")

    # Each top-level hierarchy node became its own EmergentCluster.
    assert seen_members == [["p0", "p1", "p2", "p3"], ["c0", "c1", "c2", "c3"]]
    # Every mint is parented under entity with the entity lineage.
    assert mint_kwargs == [
        {"parent_type_uuid": "type_entity", "parent_ancestors": ["root", "node"]},
        {"parent_type_uuid": "type_entity", "parent_ancestors": ["root", "node"]},
    ]
    # The published lineage descends root -> node -> entity.
    assert all(p["ancestors"] == ["root", "node", "entity"] for p in published)
