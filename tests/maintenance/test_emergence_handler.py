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
    realm walks, and the edge ops placement.reparent needs (B2/B3: membership is
    the instance->type IS_A edge, written via add_edge / delete_edge)."""

    def __init__(self, type_defs, nodes=None, edges=None):
        self._type_defs = list(type_defs)
        self._nodes = dict(nodes or {})
        self._edges = dict(edges or {})
        self.updated: list[tuple[str, dict]] = []
        self.added_edges: list[tuple[str, str, str, dict | None]] = []
        self.deleted_edges: list = []

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

    def add_edge(self, source_uuid, target_uuid, relation, properties=None, **kw):
        self.added_edges.append((source_uuid, target_uuid, relation, properties))
        if relation == "IS_A":
            # Keep the single-upward-pointer invariant in the fake adjacency.
            self._edges[source_uuid] = [
                {
                    "relation": "IS_A",
                    "target": target_uuid,
                    "id": f"e_{source_uuid}_{target_uuid}",
                }
            ]
        return "edge"

    def delete_edge(self, edge_id):
        self.deleted_edges.append(edge_id)
        return True

    def delete_edges_between(self, source, target, relation):
        self.deleted_edges.append((source, target, relation))
        return True

    def register_type(self, uuid, name, parent_uuid):
        """Mirror a mint: add the type node + its single upward IS_A edge so a
        later realm walk resolves it (used by in-pass dedup)."""
        self._nodes[uuid] = {"uuid": uuid, "name": name}
        self._edges[uuid] = [{"relation": "IS_A", "target": parent_uuid}]

    def membership_edges(self):
        """The instance->type IS_A edges drawn this pass, as (member, type)."""
        return [(s, t) for s, t, rel, _ in self.added_edges if rel == "IS_A"]


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


def _park(hcg, *uuids, parent="entity_root"):
    """Park members directly under a realm root (their pre-drainage state) so
    placement.reparent has a stale realm-root IS_A edge to drop."""
    for u in uuids:
        hcg._edges[u] = [{"relation": "IS_A", "target": parent, "id": f"e_{u}"}]
    return hcg


def test_parent_drives_placement_not_centroid(monkeypatch):
    cluster = EmergentCluster(members=[_m("b0"), _m("b1"), _m("b2")])
    monkeypatch.setattr(eh, "find_emergent_clusters", lambda *a, **k: [cluster])

    hcg = _entity_pool_hcg(
        extra_type_defs=[{"name": "vehicle", "uuid": "vehicle_uuid"}],
        extra_nodes={"vehicle_uuid": {"uuid": "vehicle_uuid", "name": "vehicle"}},
        extra_edges={"vehicle_uuid": [{"relation": "IS_A", "target": "entity_root"}]},
    )
    _park(hcg, "b0", "b1", "b2")
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
    # mint no longer retypes; drainage owns member placement via placement.reparent.
    assert kwargs["retype_members"] is False
    # Each fitting member's instance->type IS_A edge is re-pointed to the minted
    # type, carrying the type's placed_by; no type_uuid property is stamped.
    assert set(hcg.membership_edges()) == {
        ("b0", "boat_uuid"),
        ("b1", "boat_uuid"),
        ("b2", "boat_uuid"),
    }
    assert all(
        props == {"placed_by": "parent_resolution"}
        for _s, _t, rel, props in hcg.added_edges
        if rel == "IS_A"
    )
    assert hcg.updated == []
    # The stale realm-root edges were dropped (single upward pointer invariant).
    assert set(hcg.deleted_edges) == {"e_b0", "e_b1", "e_b2"}
    # Centroids never decide placement.
    assert milvus.find_nearest_calls == 0


def test_one_cluster_one_flat_placement(monkeypatch):
    cluster = EmergentCluster(members=[_m("a0"), _m("a1")])
    monkeypatch.setattr(eh, "find_emergent_clusters", lambda *a, **k: [cluster])
    hcg = _park(_entity_pool_hcg(), "a0", "a1")
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
    # Both members are re-pointed onto the minted type; no type_uuid stamp.
    assert set(hcg.membership_edges()) == {("a0", "x_uuid"), ("a1", "x_uuid")}
    assert hcg.updated == []


def test_reuse_on_in_realm_name_match(monkeypatch):
    cluster = EmergentCluster(members=[_m("v0"), _m("v1")])
    monkeypatch.setattr(eh, "find_emergent_clusters", lambda *a, **k: [cluster])
    hcg = _entity_pool_hcg(
        extra_type_defs=[{"name": "vehicle", "uuid": "vehicle_uuid"}],
        extra_nodes={"vehicle_uuid": {"uuid": "vehicle_uuid", "name": "vehicle"}},
        extra_edges={"vehicle_uuid": [{"relation": "IS_A", "target": "entity_root"}]},
    )
    _park(hcg, "v0", "v1")
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

    # Null parent + same name already in-realm -> reuse (re-point edges), never
    # re-mint. Members inherit the type's placed_by (name_reuse).
    assert minted == []
    assert set(hcg.membership_edges()) == {
        ("v0", "vehicle_uuid"),
        ("v1", "vehicle_uuid"),
    }
    assert all(
        props == {"placed_by": "name_reuse"}
        for _s, _t, rel, props in hcg.added_edges
        if rel == "IS_A"
    )
    # No type_uuid/type property stamp on members.
    assert hcg.updated == []


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
    _park(hcg, "x0", "x1")
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
    # Members are placed onto the freshly-minted entity-realm type, NOT the
    # cross-realm same-name type.
    assert set(hcg.membership_edges()) == {
        ("x0", "vehicle_entity_uuid"),
        ("x1", "vehicle_entity_uuid"),
    }
    assert hcg.updated == []  # no type_uuid/type stamp on members


def test_mint_under_realm_root_fallback(monkeypatch):
    cluster = EmergentCluster(members=[_m("a0"), _m("a1")])
    monkeypatch.setattr(eh, "find_emergent_clusters", lambda *a, **k: [cluster])
    hcg = _park(_entity_pool_hcg(), "a0", "a1")
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
    # Members re-pointed onto the minted type, carrying root_fallback.
    assert set(hcg.membership_edges()) == {
        ("a0", "gadget_uuid"),
        ("a1", "gadget_uuid"),
    }
    assert all(
        props == {"placed_by": "root_fallback"}
        for _s, _t, rel, props in hcg.added_edges
        if rel == "IS_A"
    )


def test_unresolvable_parent_falls_back_to_realm_root(monkeypatch):
    cluster = EmergentCluster(members=[_m("a0"), _m("a1")])
    monkeypatch.setattr(eh, "find_emergent_clusters", lambda *a, **k: [cluster])
    hcg = _park(_entity_pool_hcg(), "a0", "a1")
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
    # Members re-pointed onto the minted type under the realm root.
    assert set(hcg.membership_edges()) == {
        ("a0", "gadget_uuid"),
        ("a1", "gadget_uuid"),
    }


def test_outliers_stay_in_pool(monkeypatch):
    cluster = EmergentCluster(members=[_m("k0"), _m("k1"), _m("k2"), _m("out")])
    monkeypatch.setattr(eh, "find_emergent_clusters", lambda *a, **k: [cluster])
    hcg = _park(_entity_pool_hcg(), "k0", "k1", "k2", "out")
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
    # ...and is NOT re-pointed -- it keeps its realm-root edge (untouched) and
    # re-enters the next pass. Only the fitting cohort gets new instance->type
    # edges; no type_uuid/type stamp anywhere.
    assert set(hcg.membership_edges()) == {
        ("k0", "thing_uuid"),
        ("k1", "thing_uuid"),
        ("k2", "thing_uuid"),
    }
    assert "out" not in {member for member, _type in hcg.membership_edges()}
    assert "e_out" not in hcg.deleted_edges
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
    assert hcg.membership_edges() == []  # members untouched, left in the pool
    assert hcg.updated == []


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
    assert hcg.membership_edges() == []
    assert hcg.updated == []


def test_in_pass_dedup_reuses_fresh_mint(monkeypatch):
    c1 = EmergentCluster(members=[_m("a0"), _m("a1")])
    c2 = EmergentCluster(members=[_m("b0"), _m("b1")])
    monkeypatch.setattr(eh, "find_emergent_clusters", lambda *a, **k: [c1, c2])
    hcg = _park(_entity_pool_hcg(), "a0", "a1", "b0", "b1")
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

    # First cluster mints "widget"; the second reuses that fresh uuid (re-point).
    # Both cohorts' instance->type IS_A edges land on the one minted type.
    assert mint_calls == ["widget"]
    assert set(hcg.membership_edges()) == {
        ("a0", "widget_minted"),
        ("a1", "widget_minted"),
        ("b0", "widget_minted"),
        ("b1", "widget_minted"),
    }
    assert hcg.updated == []


def test_mint_publishes_event_without_ancestors(monkeypatch):
    cluster = EmergentCluster(members=[_m("a0"), _m("a1")])
    monkeypatch.setattr(eh, "find_emergent_clusters", lambda *a, **k: [cluster])
    hcg = _park(_entity_pool_hcg(), "a0", "a1")
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
    hcg = _park(_entity_pool_hcg(), "a0", "a1", "b0", "b1")
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
    # The failing cluster placed no member edges; the other one did.
    assert set(hcg.membership_edges()) == {("b0", "ok_uuid"), ("b1", "ok_uuid")}


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
    # The type_uuid-stamping retype helper is replaced by the edge-based
    # placement helper (B2/B3): membership is the instance->type IS_A edge.
    assert not hasattr(EmergenceHandler, "_attach_members")
    assert hasattr(EmergenceHandler, "_place_members")
