"""Tests for the type_emergence orchestration handler (#505)."""

from __future__ import annotations

from collections import Counter

from sophia.maintenance.config import MaintenanceConfig
from sophia.maintenance.emergence_handler import EmergenceHandler
from sophia.maintenance.emergence_types import Member, NameResult


class _NoMatchMilvus:
    """Milvus stub for which match-before-mint (#504) never finds an existing
    type, so every cluster mints fresh."""

    def find_nearest_types(self, centroid, top_k=1):
        return []

    def get_embedding(self, node_type, uuid):
        return None


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
        milvus=_NoMatchMilvus(),
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
        milvus=_NoMatchMilvus(),
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
        milvus=_NoMatchMilvus(),
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
        milvus=_NoMatchMilvus(),
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
        milvus=_NoMatchMilvus(),
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
    # Every mint is parented under entity with the entity lineage. Both nodes are
    # leaves (no children), so their members are retyped onto them.
    assert mint_kwargs == [
        {
            "parent_type_uuid": "type_entity",
            "parent_ancestors": ["root", "node"],
            "retype_members": True,
        },
        {
            "parent_type_uuid": "type_entity",
            "parent_ancestors": ["root", "node"],
            "retype_members": True,
        },
    ]
    # The published lineage descends root -> node -> entity.
    assert all(p["ancestors"] == ["root", "node", "entity"] for p in published)


def test_handler_mints_nested_hierarchy(monkeypatch):
    """An internal super-type node mints type-only (retype_members=False) and its
    leaf children mint *under it* (retype_members=True), so the tree nests (#505)."""
    from sophia.maintenance import emergence_handler as eh
    from sophia.maintenance.emergence_clustering import HierarchyNode

    leaf_a = HierarchyNode(
        members=[m for m in _members() if m.uuid.startswith("p")], centroid=[0.0, 0.0]
    )
    leaf_b = HierarchyNode(
        members=[m for m in _members() if m.uuid.startswith("c")], centroid=[9.0, 9.0]
    )
    supertype = HierarchyNode(
        members=leaf_a.members + leaf_b.members,
        centroid=[4.5, 4.5],
        children=[leaf_a, leaf_b],
    )
    monkeypatch.setattr(eh, "find_emergent_hierarchy", lambda *a, **k: [supertype])

    calls = []  # (label, parent_type_uuid, retype_members)

    def fake_name(cluster, candidates, hermes_url, token):
        if len(cluster.members) == 8:
            label = "science"
        elif cluster.members[0].uuid.startswith("p"):
            label = "object"
        else:
            label = "concept"
        return NameResult(label=label, description="", confidence=0.9)

    def fake_mint(
        cluster,
        name,
        hcg,
        milvus,
        source_cluster_id,
        *,
        parent_type_uuid,
        parent_ancestors,
        retype_members,
    ):
        calls.append((name.label, parent_type_uuid, retype_members))
        return f"type_{name.label}_x"

    handler = EmergenceHandler(
        config=MaintenanceConfig(),
        hcg=object(),
        milvus=_NoMatchMilvus(),
        event_bus=None,
        hermes_url="http://h",
        token="t",
        load_members=lambda u: _members(),
        name_fn=fake_name,
        mint_fn=fake_mint,
        candidates_fn=lambda: [],
    )
    handler.run(type_uuid="type_entity")

    # Super-type minted first, under entity, type-only (members live in leaves).
    assert calls[0] == ("science", "type_entity", False)
    # Leaves mint under the freshly-minted super-type and retype their members.
    assert ("object", "type_science_x", True) in calls
    assert ("concept", "type_science_x", True) in calls


def test_handler_reconciles_into_existing_type(monkeypatch):
    """A cluster whose centroid matches an existing type is retyped onto it; no
    duplicate type is minted (#504 match-before-mint)."""
    from sophia.maintenance import emergence_handler as eh
    from sophia.maintenance.emergence_clustering import HierarchyNode

    leaf = HierarchyNode(
        members=[m for m in _members() if m.uuid.startswith("p")], centroid=[1.0, 0.0]
    )
    monkeypatch.setattr(eh, "find_emergent_hierarchy", lambda *a, **k: [leaf])

    minted, retyped = [], {}

    class FakeMilvus:
        def find_nearest_types(self, centroid, top_k=1):
            return [{"uuid": "type_vehicle_abc", "score": 0.0}]

        def get_embedding(self, node_type, uuid):
            # Same direction as the cluster centroid -> cosine 1.0 (>= threshold).
            return {"uuid": uuid, "embedding": [2.0, 0.0]}

    class FakeHCG:
        def get_node(self, uuid):
            return {"uuid": uuid, "name": "vehicle", "properties": {"ancestors": ["root", "node", "entity"]}}

        def update_node(self, uuid, props):
            retyped[uuid] = props

    handler = EmergenceHandler(
        config=MaintenanceConfig(),
        hcg=FakeHCG(),
        milvus=FakeMilvus(),
        event_bus=None,
        hermes_url="http://h",
        token="t",
        load_members=lambda u: _members(),
        name_fn=lambda *a: NameResult(label="vehicle", description="", confidence=0.9),
        mint_fn=lambda *a, **k: minted.append(1) or "type_should_not_mint",
        candidates_fn=lambda: [],
    )
    handler.run(type_uuid="type_entity")

    assert minted == []  # reconciled into the existing type, not minted
    assert len(retyped) == 4  # the four "p" members retyped
    assert all(p["type_uuid"] == "type_vehicle_abc" for p in retyped.values())


def test_handler_subdivides_minted_type_nests_under_it(monkeypatch):
    """Re-emergence on an already-minted type parents new subtypes under *it* (with
    its stored ancestors), deepening the hierarchy instead of flattening (#505)."""
    from sophia.maintenance import emergence_handler as eh
    from sophia.maintenance.emergence_clustering import HierarchyNode

    leaf = HierarchyNode(
        members=[m for m in _members() if m.uuid.startswith("p")], centroid=[0.0, 0.0]
    )
    monkeypatch.setattr(eh, "find_emergent_hierarchy", lambda *a, **k: [leaf])

    mint_kwargs = []

    class FakeHCG:
        def get_node(self, uuid):
            return {"uuid": uuid, "name": "vehicle", "properties": {"ancestors": ["root", "node", "entity"]}}

        def update_node(self, *a, **k):
            pass

    def fake_mint(
        cluster, name, hcg, milvus, source_cluster_id, *, parent_type_uuid, parent_ancestors, **k
    ):
        mint_kwargs.append((parent_type_uuid, parent_ancestors))
        return "type_sedan_x"

    handler = EmergenceHandler(
        config=MaintenanceConfig(),
        hcg=FakeHCG(),
        milvus=_NoMatchMilvus(),
        event_bus=None,
        hermes_url="http://h",
        token="t",
        load_members=lambda u: _members(),
        name_fn=lambda *a: NameResult(label="sedan", description="", confidence=0.9),
        mint_fn=fake_mint,
        candidates_fn=lambda: [],
    )
    handler.run(type_uuid="type_vehicle_abc")

    # Parent is the subdivided type itself, with that type's ancestors.
    assert mint_kwargs == [("type_vehicle_abc", ["root", "node", "entity"])]
