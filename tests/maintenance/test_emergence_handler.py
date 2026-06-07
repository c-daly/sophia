"""Flat parent-driven drainage in the type_emergence handler (B1 T4b).

Embeddings only PROPOSE clusters; the graph ASSERTS placement via the parent
the LLM names (validated closed-world through ``placement``) or the realm root.
The pass is flat -- one placement per cluster, outliers left untouched in the
pool -- and centroids never decide placement.
"""

from __future__ import annotations

from collections import Counter

from sophia.maintenance import emergence_handler as eh
from sophia.maintenance.config import MaintenanceConfig
from sophia.maintenance.emergence_handler import EmergenceHandler
from sophia.maintenance.emergence_types import (
    EmergentCluster,
    Member,
    TypeClusterResult,
)


def _m(uuid: str) -> Member:
    return Member(
        uuid=uuid,
        name=uuid,
        embedding=[0.0, 0.0],
        signature=Counter(),
        current_type="entity",
        hermes_type_hint=None,
        neighbors=[],
    )


class _RecordingMilvus:
    """Records find_nearest_types calls so tests can prove centroids never drive
    placement; harmless stubs otherwise."""

    def __init__(self) -> None:
        self.find_nearest_calls = 0

    def find_nearest_types(self, *a, **k):
        self.find_nearest_calls += 1
        return []

    def get_embedding(self, *a, **k):
        return None

    def update_centroid(self, *a, **k):
        return None


class _FakeHCG:
    """Minimal HCG: a type catalog (list_all_nodes), node lookup, IS_A edges for
    realm walks, and an update_node recorder for retypes."""

    def __init__(self, type_defs, nodes=None, edges=None):
        self._type_defs = list(type_defs)
        self._nodes = dict(nodes or {})
        self._edges = dict(edges or {})
        self.updated: list[tuple[str, dict]] = []

    def list_all_nodes(self, node_type=None):
        if node_type == "type_definition":
            return list(self._type_defs)
        return []

    def get_node(self, uuid):
        return self._nodes.get(uuid)

    def query_edges_from(self, uuid):
        return list(self._edges.get(uuid, []))

    def update_node(self, uuid, props):
        self.updated.append((uuid, props))

    def register_type(self, uuid, name, parent_uuid):
        """Mirror a mint: add the type node + its single upward IS_A edge so a
        later realm walk resolves it (used by in-pass dedup)."""
        self._nodes[uuid] = {"uuid": uuid, "name": name}
        self._edges[uuid] = [{"relation": "IS_A", "target": parent_uuid}]


def _handler(hcg, milvus, name_fn, mint_fn, event_bus=None):
    return EmergenceHandler(
        config=MaintenanceConfig(),
        hcg=hcg,
        milvus=milvus,
        event_bus=event_bus,
        hermes_url="http://h",
        token="t",
        load_members=lambda u: [],
        name_fn=name_fn,
        mint_fn=mint_fn,
    )


def _entity_pool_hcg(extra_type_defs=(), extra_nodes=None, extra_edges=None):
    """An HCG whose only realm root is `entity`; the seed pool IS that root."""
    type_defs = [{"name": "entity", "uuid": "entity_root"}, *extra_type_defs]
    nodes = {"entity_root": {"uuid": "entity_root", "name": "entity"}}
    nodes.update(extra_nodes or {})
    return _FakeHCG(type_defs, nodes=nodes, edges=extra_edges or {})


def test_parent_drives_placement_not_centroid(monkeypatch):
    cluster = EmergentCluster(members=[_m("b0"), _m("b1"), _m("b2")])
    monkeypatch.setattr(eh, "find_emergent_clusters", lambda *a, **k: [cluster])

    hcg = _entity_pool_hcg(
        extra_type_defs=[{"name": "vehicle", "uuid": "vehicle_uuid"}],
        extra_nodes={"vehicle_uuid": {"uuid": "vehicle_uuid", "name": "vehicle"}},
        extra_edges={"vehicle_uuid": [{"relation": "IS_A", "target": "entity_root"}]},
    )
    milvus = _RecordingMilvus()
    mint_calls = []

    def fake_mint(cluster, name, hcg, milvus, source_cluster_id, **kwargs):
        mint_calls.append((name.label, kwargs))
        return "boat_uuid"

    handler = _handler(
        hcg,
        milvus,
        name_fn=lambda c: TypeClusterResult(
            name="boat", parent="vehicle", residual_ids=[]
        ),
        mint_fn=fake_mint,
    )
    handler.run(type_uuid="entity_root")

    assert len(mint_calls) == 1  # flat: exactly one placement, no sub-tree
    label, kwargs = mint_calls[0]
    assert label == "boat"
    # The LLM-named parent drives placement; the graph asserts via that parent.
    assert kwargs["parent_type_uuid"] == "vehicle_uuid"
    assert kwargs["placed_by"] == "parent_resolution"
    assert kwargs["retype_members"] is True
    # Centroids never decide placement.
    assert milvus.find_nearest_calls == 0


def test_one_cluster_one_flat_placement(monkeypatch):
    cluster = EmergentCluster(members=[_m("a0"), _m("a1")])
    monkeypatch.setattr(eh, "find_emergent_clusters", lambda *a, **k: [cluster])
    hcg = _entity_pool_hcg()
    mint_calls = []

    def fake_mint(cluster, name, hcg, milvus, source_cluster_id, **kwargs):
        mint_calls.append(name.label)
        return "x_uuid"

    handler = _handler(
        hcg,
        _RecordingMilvus(),
        name_fn=lambda c: TypeClusterResult(
            name="gadget", parent=None, residual_ids=[]
        ),
        mint_fn=fake_mint,
    )
    handler.run(type_uuid="entity_root")

    # One cluster -> exactly one mint; no recursive sub-tree minting.
    assert mint_calls == ["gadget"]


def test_reuse_on_in_realm_name_match(monkeypatch):
    cluster = EmergentCluster(members=[_m("v0"), _m("v1")])
    monkeypatch.setattr(eh, "find_emergent_clusters", lambda *a, **k: [cluster])
    hcg = _entity_pool_hcg(
        extra_type_defs=[{"name": "vehicle", "uuid": "vehicle_uuid"}],
        extra_nodes={"vehicle_uuid": {"uuid": "vehicle_uuid", "name": "vehicle"}},
        extra_edges={"vehicle_uuid": [{"relation": "IS_A", "target": "entity_root"}]},
    )
    minted = []
    handler = _handler(
        hcg,
        _RecordingMilvus(),
        name_fn=lambda c: TypeClusterResult(
            name="vehicle", parent=None, residual_ids=[]
        ),
        mint_fn=lambda *a, **k: minted.append(1),
    )
    handler.run(type_uuid="entity_root")

    # Null parent + same name already in-realm -> reuse (attach), never re-mint.
    # placed_by is "name_reuse" internally; the observable effect is the retype.
    assert minted == []
    assert {u for u, _ in hcg.updated} == {"v0", "v1"}
    assert all(p["type_uuid"] == "vehicle_uuid" for _, p in hcg.updated)
    assert all(p["type"] == "vehicle" for _, p in hcg.updated)


def test_cross_realm_name_match_not_reused(monkeypatch):
    cluster = EmergentCluster(members=[_m("x0"), _m("x1")])
    monkeypatch.setattr(eh, "find_emergent_clusters", lambda *a, **k: [cluster])
    # "vehicle" exists but is rooted in the CONCEPT realm -- not reusable for an
    # entity-realm pool.
    hcg = _entity_pool_hcg(
        extra_type_defs=[
            {"name": "concept", "uuid": "concept_root"},
            {"name": "vehicle", "uuid": "vehicle_concept_uuid"},
        ],
        extra_nodes={
            "concept_root": {"uuid": "concept_root", "name": "concept"},
            "vehicle_concept_uuid": {
                "uuid": "vehicle_concept_uuid",
                "name": "vehicle",
            },
        },
        extra_edges={
            "vehicle_concept_uuid": [{"relation": "IS_A", "target": "concept_root"}]
        },
    )
    mint_calls = []

    def fake_mint(cluster, name, hcg, milvus, source_cluster_id, **kwargs):
        mint_calls.append(kwargs)
        return "vehicle_entity_uuid"

    handler = _handler(
        hcg,
        _RecordingMilvus(),
        name_fn=lambda c: TypeClusterResult(
            name="vehicle", parent=None, residual_ids=[]
        ),
        mint_fn=fake_mint,
    )
    handler.run(type_uuid="entity_root")

    # The cross-realm name is NOT reused -> mint fresh under the entity realm root.
    assert len(mint_calls) == 1
    assert mint_calls[0]["parent_type_uuid"] == "entity_root"
    assert mint_calls[0]["placed_by"] == "root_fallback"
    assert hcg.updated == []  # no retype onto the cross-realm type


def test_mint_under_realm_root_fallback(monkeypatch):
    cluster = EmergentCluster(members=[_m("a0"), _m("a1")])
    monkeypatch.setattr(eh, "find_emergent_clusters", lambda *a, **k: [cluster])
    hcg = _entity_pool_hcg()
    mint_calls = []

    def fake_mint(cluster, name, hcg, milvus, source_cluster_id, **kwargs):
        mint_calls.append(kwargs)
        return "gadget_uuid"

    handler = _handler(
        hcg,
        _RecordingMilvus(),
        name_fn=lambda c: TypeClusterResult(
            name="gadget", parent=None, residual_ids=[]
        ),
        mint_fn=fake_mint,
    )
    handler.run(type_uuid="entity_root")

    # Null parent + name absent from the catalog -> mint under the realm root.
    assert len(mint_calls) == 1
    assert mint_calls[0]["parent_type_uuid"] == "entity_root"
    assert mint_calls[0]["placed_by"] == "root_fallback"


def test_unresolvable_parent_falls_back_to_realm_root(monkeypatch):
    cluster = EmergentCluster(members=[_m("a0"), _m("a1")])
    monkeypatch.setattr(eh, "find_emergent_clusters", lambda *a, **k: [cluster])
    hcg = _entity_pool_hcg()
    mint_calls = []

    def fake_mint(cluster, name, hcg, milvus, source_cluster_id, **kwargs):
        mint_calls.append(kwargs)
        return "gadget_uuid"

    # parent "spaceship" is not in the catalog -> closed-world None -> realm root.
    handler = _handler(
        hcg,
        _RecordingMilvus(),
        name_fn=lambda c: TypeClusterResult(
            name="gadget", parent="spaceship", residual_ids=[]
        ),
        mint_fn=fake_mint,
    )
    handler.run(type_uuid="entity_root")

    assert mint_calls[0]["parent_type_uuid"] == "entity_root"
    assert mint_calls[0]["placed_by"] == "root_fallback"


def test_outliers_stay_in_pool(monkeypatch):
    cluster = EmergentCluster(members=[_m("k0"), _m("k1"), _m("k2"), _m("out")])
    monkeypatch.setattr(eh, "find_emergent_clusters", lambda *a, **k: [cluster])
    hcg = _entity_pool_hcg()
    minted_members: dict[str, list[str]] = {}

    def fake_mint(cluster, name, hcg, milvus, source_cluster_id, **kwargs):
        minted_members[name.label] = [m.uuid for m in cluster.members]
        return "thing_uuid"

    handler = _handler(
        hcg,
        _RecordingMilvus(),
        name_fn=lambda c: TypeClusterResult(
            name="thing", parent=None, residual_ids=["out"]
        ),
        mint_fn=fake_mint,
    )
    handler.run(type_uuid="entity_root")

    # The outlier is excluded from the minted cohort...
    assert set(minted_members["thing"]) == {"k0", "k1", "k2"}
    assert "out" not in minted_members["thing"]
    # ...and is NOT retyped or parked -- it stays untouched in the pool.
    assert hcg.updated == []


def test_all_outliers_mints_nothing_and_leaves_members(monkeypatch):
    cluster = EmergentCluster(members=[_m("o0"), _m("o1")])
    monkeypatch.setattr(eh, "find_emergent_clusters", lambda *a, **k: [cluster])
    hcg = _entity_pool_hcg()
    minted = []
    handler = _handler(
        hcg,
        _RecordingMilvus(),
        name_fn=lambda c: TypeClusterResult(
            name="thing", parent=None, residual_ids=["o0", "o1"]
        ),
        mint_fn=lambda *a, **k: minted.append(1),
    )
    handler.run(type_uuid="entity_root")

    assert minted == []  # nothing minted when every member is an outlier
    assert hcg.updated == []  # members untouched, left in the pool


def test_no_name_leaves_cluster_in_pool(monkeypatch):
    cluster = EmergentCluster(members=[_m("a0"), _m("a1")])
    monkeypatch.setattr(eh, "find_emergent_clusters", lambda *a, **k: [cluster])
    hcg = _entity_pool_hcg()
    minted = []
    handler = _handler(
        hcg,
        _RecordingMilvus(),
        name_fn=lambda c: None,
        mint_fn=lambda *a, **k: minted.append(1),
    )
    handler.run(type_uuid="entity_root")

    assert minted == []
    assert hcg.updated == []


def test_in_pass_dedup_reuses_fresh_mint(monkeypatch):
    c1 = EmergentCluster(members=[_m("a0"), _m("a1")])
    c2 = EmergentCluster(members=[_m("b0"), _m("b1")])
    monkeypatch.setattr(eh, "find_emergent_clusters", lambda *a, **k: [c1, c2])
    hcg = _entity_pool_hcg()
    mint_calls = []

    def fake_mint(cluster, name, hcg, milvus, source_cluster_id, **kwargs):
        new_uuid = "widget_minted"
        hcg.register_type(new_uuid, name.label, kwargs["parent_type_uuid"])
        mint_calls.append(name.label)
        return new_uuid

    handler = _handler(
        hcg,
        _RecordingMilvus(),
        name_fn=lambda c: TypeClusterResult(
            name="widget", parent=None, residual_ids=[]
        ),
        mint_fn=fake_mint,
    )
    handler.run(type_uuid="entity_root")

    # First cluster mints "widget"; the second reuses that fresh uuid (attach).
    assert mint_calls == ["widget"]
    assert {u for u, _ in hcg.updated} == {"b0", "b1"}
    assert all(p["type_uuid"] == "widget_minted" for _, p in hcg.updated)


def test_mint_publishes_event_without_ancestors(monkeypatch):
    cluster = EmergentCluster(members=[_m("a0"), _m("a1")])
    monkeypatch.setattr(eh, "find_emergent_clusters", lambda *a, **k: [cluster])
    hcg = _entity_pool_hcg()
    published = []

    class EB:
        def publish(self, channel, msg):
            published.append((channel, msg))

    handler = _handler(
        hcg,
        _RecordingMilvus(),
        name_fn=lambda c: TypeClusterResult(
            name="gadget", parent=None, residual_ids=[]
        ),
        mint_fn=lambda *a, **k: "gadget_uuid",
        event_bus=EB(),
    )
    handler.run(type_uuid="entity_root")

    assert len(published) == 1
    channel, msg = published[0]
    assert channel == eh.ONTOLOGY_CHANGED_CHANNEL
    assert msg == {
        "type_uuid": "gadget_uuid",
        "name": "gadget",
        "parent_uuid": "entity_root",
        "placed_by": "root_fallback",
    }
    assert "ancestors" not in msg  # structure is the only membership (B1)


def test_handler_isolates_failing_cluster(monkeypatch):
    c1 = EmergentCluster(members=[_m("a0"), _m("a1")])
    c2 = EmergentCluster(members=[_m("b0"), _m("b1")])
    monkeypatch.setattr(eh, "find_emergent_clusters", lambda *a, **k: [c1, c2])
    hcg = _entity_pool_hcg()
    minted = []

    def flaky_mint(cluster, name, hcg, milvus, source_cluster_id, **kwargs):
        if cluster.members[0].uuid == "a0":
            raise RuntimeError("transient HCG write error")
        minted.append(name.label)
        return "ok_uuid"

    handler = _handler(
        hcg,
        _RecordingMilvus(),
        name_fn=lambda c: TypeClusterResult(
            name="a" if c.members[0].uuid == "a0" else "b",
            parent=None,
            residual_ids=[],
        ),
        mint_fn=flaky_mint,
    )
    handler.run(type_uuid="entity_root")  # must not raise

    assert minted == ["b"]  # the failing cluster is skipped; the other places


def test_no_clusters_is_a_noop(monkeypatch):
    monkeypatch.setattr(eh, "find_emergent_clusters", lambda *a, **k: [])
    minted = []
    handler = _handler(
        object(),
        _RecordingMilvus(),
        name_fn=lambda c: None,
        mint_fn=lambda *a, **k: minted.append(1),
    )
    handler.run(type_uuid="entity_root")  # no clusters -> returns before hcg use
    assert minted == []


def test_legacy_subtree_and_match_helpers_removed():
    # The per-pass hierarchy + centroid matching are gone for good (B1 T4b).
    assert not hasattr(EmergenceHandler, "_mint_subtree")
    assert not hasattr(EmergenceHandler, "_match_existing_type")
    assert not hasattr(EmergenceHandler, "_park_residuals")
    assert not hasattr(eh, "_UNSORTED_TYPE")
    assert not hasattr(eh, "_UNSORTED_TYPE_UUID")
