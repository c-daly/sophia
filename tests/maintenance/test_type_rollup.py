"""Tests for the type-level rollup handler (sophia#160)."""

from __future__ import annotations

from collections import Counter

from sophia.maintenance.config import MaintenanceConfig
from sophia.maintenance.emergence_clustering import HierarchyNode
from sophia.maintenance.emergence_types import Member, NameResult
from sophia.maintenance.type_rollup_handler import TypeRollupHandler


class FakeHCG:
    """In-memory type-def graph: nodes + reified IS_A edges, recording writes."""

    def __init__(self, type_defs, members=None, is_a=None, other_edges=None):
        # type_defs: list of {uuid,name,properties{ancestors,is_type_definition}}
        self.nodes = {td["uuid"]: td for td in type_defs}
        for m in members or []:  # entity members carry a type_uuid
            self.nodes[m["uuid"]] = m
        # edges: list of {id, source, target, relation}
        self.edges = list(is_a or []) + list(other_edges or [])
        self._eid = 0
        self.writes = []  # (op, *args)

    def get_all_type_definitions(self):
        return [
            n
            for n in self.nodes.values()
            if (n.get("properties") or {}).get("is_type_definition")
        ]

    def get_node(self, uuid):
        return self.nodes.get(uuid)

    def get_nodes_batch(self, uuids):
        return [self.nodes[u] for u in uuids if u in self.nodes]

    def list_all_edges(self, relation_type=None, limit=1000):
        return [
            e
            for e in self.edges
            if relation_type is None or e["relation"] == relation_type
        ]

    def query_edges_from(self, uuid):
        return [e for e in self.edges if e["source"] == uuid]

    def add_edge(self, source, target, relation, **kw):
        # MERGE semantics: dedupe on (source,target,relation)
        for e in self.edges:
            if (e["source"], e["target"], e["relation"]) == (source, target, relation):
                return e["id"]
        self._eid += 1
        eid = f"edge{self._eid}"
        self.edges.append(
            {"id": eid, "source": source, "target": target, "relation": relation}
        )
        self.writes.append(("add_edge", source, target, relation))
        return eid

    def delete_edge(self, edge_uuid):
        self.edges = [e for e in self.edges if e.get("id") != edge_uuid]
        self.writes.append(("delete_edge", edge_uuid))
        return True

    def delete_edges_between(self, source, target, relation):
        before = len(self.edges)
        self.edges = [
            e
            for e in self.edges
            if not (
                e["source"] == source
                and e["target"] == target
                and e["relation"] == relation
            )
        ]
        removed = before - len(self.edges)
        self.writes.append(("delete_edges_between", source, target, relation))
        return removed

    def update_node(self, uuid, properties=None):
        self.nodes.setdefault(uuid, {"uuid": uuid, "properties": {}})
        self.nodes[uuid].setdefault("properties", {}).update(properties or {})
        self.writes.append(("update_node", uuid, dict(properties or {})))
        return uuid


class FakeMilvus:
    def __init__(self, centroids):  # {uuid: vector}
        self.centroids = dict(centroids)

    def get_embedding(self, node_type, uuid):
        v = self.centroids.get(str(uuid))
        return {"uuid": uuid, "embedding": v, "embedding_model": "m"} if v else None

    def find_nearest_types(self, query_embedding, top_k=1):
        return []  # no pre-existing super-types -> always mint

    def update_centroid(self, type_uuid, centroid, model):
        self.centroids[type_uuid] = centroid


def _td(uuid, name, ancestors):
    return {
        "uuid": uuid,
        "name": name,
        "properties": {"is_type_definition": True, "ancestors": ancestors},
    }


def _cfg():
    return MaintenanceConfig()


def _handler(hcg, milvus, mint_calls=None, name_label="mathematics"):
    def name_fn(cluster, candidates, url, tok):
        return NameResult(label=name_label, description="", confidence=0.9)

    def mint_fn(
        cluster,
        name,
        *,
        hcg,
        milvus,
        source_cluster_id,
        parent_type_uuid,
        parent_ancestors,
        parent_name,
        retype_members,
    ):
        uuid = f"type_{name.label}_super01"
        hcg.nodes[uuid] = {
            "uuid": uuid,
            "name": name.label,
            "properties": {
                "is_type_definition": True,
                "ancestors": list(parent_ancestors) + [parent_name],
            },
        }
        hcg.add_edge(uuid, parent_type_uuid, "IS_A")
        milvus.update_centroid(uuid, cluster.embeddings[0], "m")
        if mint_calls is not None:
            mint_calls.append((name.label, parent_type_uuid, retype_members))
        return uuid

    return TypeRollupHandler(
        config=_cfg(),
        hcg=hcg,
        milvus=milvus,
        event_bus=None,
        hermes_url="http://h",
        token="t",
        name_fn=name_fn,
        mint_fn=mint_fn,
    )


def test_tier1_lifts_explicit_subsumption(monkeypatch):
    """A HAS_PART member edge between two types lifts child under parent as IS_A."""
    import sophia.maintenance.type_rollup_handler as tr

    # two leaf types, each flat under entity; one entity member each
    tds = [
        _td("type_anatomy_aa", "anatomy", ["root", "node", "entity"]),
        _td("type_insect_part_bb", "insect part", ["root", "node", "entity"]),
    ]
    members = [
        {"uuid": "e1", "type_uuid": "type_anatomy_aa"},
        {"uuid": "e2", "type_uuid": "type_insect_part_bb"},
    ]
    edges = [{"id": "r1", "source": "e1", "target": "e2", "relation": "HAS_PART"}]
    hcg = FakeHCG(tds, members=members, other_edges=edges)
    milvus = FakeMilvus(
        {"type_anatomy_aa": [1.0, 0.0], "type_insect_part_bb": [0.0, 1.0]}
    )
    monkeypatch.setattr(tr, "find_emergent_hierarchy", lambda *a, **k: [])  # tier 2 off
    _handler(hcg, milvus).run()
    # insect part (e2's type, the TARGET of HAS_PART) is the child of anatomy (e1's type)
    assert any(
        e["source"] == "type_insect_part_bb"
        and e["target"] == "type_anatomy_aa"
        and e["relation"] == "IS_A"
        for e in hcg.edges
    )
    assert hcg.nodes["type_insect_part_bb"]["properties"]["ancestors"] == [
        "root",
        "node",
        "entity",
        "anatomy",
    ]


def test_tier2_mints_supertype_and_reparents(monkeypatch):
    """Residual types cluster into a super-type; leaves re-parent under it."""
    import sophia.maintenance.type_rollup_handler as tr

    tds = [
        _td("type_calculus_aa", "calculus", ["root", "node", "entity"]),
        _td("type_algebra_bb", "algebra", ["root", "node", "entity"]),
    ]
    hcg = FakeHCG(tds)
    milvus = FakeMilvus({"type_calculus_aa": [1.0, 0.0], "type_algebra_bb": [0.9, 0.1]})
    # mock hierarchy: one super-node over the two leaves
    leaf = HierarchyNode(
        members=[
            Member(
                "type_calculus_aa",
                "calculus",
                [1.0, 0.0],
                Counter(),
                "type_definition",
                None,
                [],
                "m",
            ),
            Member(
                "type_algebra_bb",
                "algebra",
                [0.9, 0.1],
                Counter(),
                "type_definition",
                None,
                [],
                "m",
            ),
        ],
        centroid=[0.95, 0.05],
        children=[
            HierarchyNode(
                members=[
                    Member(
                        "type_calculus_aa",
                        "calculus",
                        [1.0, 0.0],
                        Counter(),
                        "type_definition",
                        None,
                        [],
                        "m",
                    )
                ],
                centroid=[1.0, 0.0],
            ),
            HierarchyNode(
                members=[
                    Member(
                        "type_algebra_bb",
                        "algebra",
                        [0.9, 0.1],
                        Counter(),
                        "type_definition",
                        None,
                        [],
                        "m",
                    )
                ],
                centroid=[0.9, 0.1],
            ),
        ],
    )
    monkeypatch.setattr(tr, "find_emergent_hierarchy", lambda *a, **k: [leaf])
    mint_calls = []
    _handler(hcg, milvus, mint_calls=mint_calls).run()
    # a super-type was minted (type-only) and both leaves now sit under it
    assert mint_calls == [("mathematics", "type_entity", False)]
    super_uuid = "type_mathematics_super01"
    for leaf_uuid in ("type_calculus_aa", "type_algebra_bb"):
        assert any(
            e["source"] == leaf_uuid
            and e["target"] == super_uuid
            and e["relation"] == "IS_A"
            for e in hcg.edges
        )
        assert hcg.nodes[leaf_uuid]["properties"]["ancestors"] == [
            "root",
            "node",
            "entity",
            "mathematics",
        ]


def test_rollup_is_idempotent(monkeypatch):
    """Second run with no structural change writes nothing (the convergence anchor)."""
    import sophia.maintenance.type_rollup_handler as tr

    tds = [
        _td("type_calculus_aa", "calculus", ["root", "node", "entity"]),
        _td("type_algebra_bb", "algebra", ["root", "node", "entity"]),
    ]
    hcg = FakeHCG(tds)
    milvus = FakeMilvus({"type_calculus_aa": [1.0, 0.0], "type_algebra_bb": [0.9, 0.1]})
    leaf = HierarchyNode(
        members=[
            Member(
                "type_calculus_aa",
                "calculus",
                [1.0, 0.0],
                Counter(),
                "type_definition",
                None,
                [],
                "m",
            ),
            Member(
                "type_algebra_bb",
                "algebra",
                [0.9, 0.1],
                Counter(),
                "type_definition",
                None,
                [],
                "m",
            ),
        ],
        centroid=[0.95, 0.05],
        children=[
            HierarchyNode(
                members=[
                    Member(
                        "type_calculus_aa",
                        "calculus",
                        [1.0, 0.0],
                        Counter(),
                        "type_definition",
                        None,
                        [],
                        "m",
                    )
                ],
                centroid=[1.0, 0.0],
            ),
            HierarchyNode(
                members=[
                    Member(
                        "type_algebra_bb",
                        "algebra",
                        [0.9, 0.1],
                        Counter(),
                        "type_definition",
                        None,
                        [],
                        "m",
                    )
                ],
                centroid=[0.9, 0.1],
            ),
        ],
    )
    monkeypatch.setattr(tr, "find_emergent_hierarchy", lambda *a, **k: [leaf])
    # First run: existing super-type so no mint; FakeMilvus.find_nearest_types must return it 2nd time
    h = _handler(hcg, milvus)
    h.run()
    # After run 1, make find_nearest_types return the minted super-type so run 2 reconciles (no new mint)
    super_uuid = "type_mathematics_super01"
    milvus.find_nearest_types = lambda q, top_k=1: [{"uuid": super_uuid, "score": 0.0}]
    hcg.writes.clear()
    _handler(hcg, milvus).run()  # second pass
    # idempotent: re-parent is a no-op, only allowed write is the (idempotent) add_edge MERGE which returns existing
    assert all(
        op != "update_node" for op, *_ in hcg.writes
    ), f"unexpected writes: {hcg.writes}"


def test_cycle_is_recorded_as_ambiguous_not_minted_as_edge():
    """A reparent that would close an IS_A cycle records AMBIGUOUS_SUBSUMPTION
    instead of the (false) IS_A edge -- the 'misunderstood relationship' rule (#160)."""
    # b IS_A a already exists -> b is a descendant of a.
    tds = [
        _td("type_a_aa", "a", ["root", "node", "entity"]),
        _td("type_b_bb", "b", ["root", "node", "entity", "a"]),
    ]
    is_a = [
        {"id": "r1", "source": "type_b_bb", "target": "type_a_aa", "relation": "IS_A"}
    ]
    hcg = FakeHCG(tds, is_a=is_a)
    milvus = FakeMilvus({"type_a_aa": [1.0, 0.0], "type_b_bb": [0.9, 0.1]})
    h = _handler(hcg, milvus)
    h._build_is_a_adjacency()
    # Try to make b the parent of a -> would close the 2-cycle a->b->a.
    h._reparent_one("type_a_aa", "type_b_bb", ["root", "node", "entity"], "b")
    # No cyclic IS_A edge a->b was created.
    assert not any(
        e["source"] == "type_a_aa"
        and e["target"] == "type_b_bb"
        and e["relation"] == "IS_A"
        for e in hcg.edges
    )
    # The pair is recorded as a misunderstood/ambiguous relationship (canonical order).
    lo, hi = sorted(("type_a_aa", "type_b_bb"))
    assert any(
        e["source"] == lo
        and e["target"] == hi
        and e["relation"] == "AMBIGUOUS_SUBSUMPTION"
        for e in hcg.edges
    )


def test_adjacency_ignores_instance_taxonomy():
    """_build_is_a_adjacency keeps only type-def<->type-def IS_A, so the cascade
    and cycle-check never walk into instance taxonomy (#160 Fix C)."""
    tds = [_td("type_x_aa", "x", ["root", "node", "entity"])]
    # one type-def IS_A and one instance-taxonomy IS_A (raw uuids, no type_ prefix)
    is_a = [
        {
            "id": "r1",
            "source": "type_x_aa",
            "target": "type_entity",
            "relation": "IS_A",
        },
        {
            "id": "r2",
            "source": "fish_inst",
            "target": "natural_resource_inst",
            "relation": "IS_A",
        },
    ]
    hcg = FakeHCG(tds, is_a=is_a)
    h = _handler(hcg, FakeMilvus({}))
    h._build_is_a_adjacency()
    assert h._children_of.get("type_entity") == ["type_x_aa"]
    assert "natural_resource_inst" not in h._children_of  # instance edge excluded


def test_name_reconcile_reuses_existing_type(monkeypatch):
    """Minting a super whose coined name already exists reuses that type-def
    rather than minting a duplicate-named one (#160 Fix D)."""
    import sophia.maintenance.type_rollup_handler as tr

    # An existing 'mathematics' type-def plus two leaves to cluster under it.
    tds = [
        _td("type_mathematics_existing", "mathematics", ["root", "node", "entity"]),
        _td("type_calculus_aa", "calculus", ["root", "node", "entity"]),
        _td("type_algebra_bb", "algebra", ["root", "node", "entity"]),
    ]
    hcg = FakeHCG(tds)
    milvus = FakeMilvus(
        {
            "type_mathematics_existing": [0.5, 0.5],
            "type_calculus_aa": [1.0, 0.0],
            "type_algebra_bb": [0.9, 0.1],
        }
    )
    leaf = HierarchyNode(
        members=[
            Member(
                "type_calculus_aa",
                "calculus",
                [1.0, 0.0],
                Counter(),
                "type_definition",
                None,
                [],
                "m",
            ),
            Member(
                "type_algebra_bb",
                "algebra",
                [0.9, 0.1],
                Counter(),
                "type_definition",
                None,
                [],
                "m",
            ),
        ],
        centroid=[0.95, 0.05],
        children=[
            HierarchyNode(
                members=[
                    Member(
                        "type_calculus_aa",
                        "calculus",
                        [1.0, 0.0],
                        Counter(),
                        "type_definition",
                        None,
                        [],
                        "m",
                    )
                ],
                centroid=[1.0, 0.0],
            ),
            HierarchyNode(
                members=[
                    Member(
                        "type_algebra_bb",
                        "algebra",
                        [0.9, 0.1],
                        Counter(),
                        "type_definition",
                        None,
                        [],
                        "m",
                    )
                ],
                centroid=[0.9, 0.1],
            ),
        ],
    )
    monkeypatch.setattr(tr, "find_emergent_hierarchy", lambda *a, **k: [leaf])
    mint_calls = []
    # name_fn coins 'mathematics' -> collides with the existing type-def.
    _handler(hcg, milvus, mint_calls=mint_calls, name_label="mathematics").run()
    # No new super minted; the existing type-def is reused as the parent.
    assert mint_calls == []
    for leaf_uuid in ("type_calculus_aa", "type_algebra_bb"):
        assert any(
            e["source"] == leaf_uuid
            and e["target"] == "type_mathematics_existing"
            and e["relation"] == "IS_A"
            for e in hcg.edges
        )


def test_reused_super_is_reparented_to_intended_parent(monkeypatch):
    """A super reused (by name) as a NESTED node must be re-parented to its
    intended parent, not left at its old position (#160 review must-fix)."""
    import sophia.maintenance.type_rollup_handler as tr

    # 'geometry' exists flat under entity; it will be reused as a nested super
    # under a freshly-minted 'mathematics'. 'trig'/'calculus' are leaves.
    tds = [
        _td("type_geometry_existing", "geometry", ["root", "node", "entity"]),
        _td("type_trig_aa", "trig", ["root", "node", "entity"]),
        _td("type_calculus_bb", "calculus", ["root", "node", "entity"]),
    ]
    hcg = FakeHCG(tds)
    milvus = FakeMilvus(
        {
            "type_geometry_existing": [0.5, 0.5],
            "type_trig_aa": [0.4, 0.6],
            "type_calculus_bb": [1.0, 0.0],
        }
    )

    def _m(uuid, name, vec):
        return Member(uuid, name, vec, Counter(), "type_definition", None, [], "m")

    trig, calc = _m("type_trig_aa", "trig", [0.4, 0.6]), _m(
        "type_calculus_bb", "calculus", [1.0, 0.0]
    )
    top = HierarchyNode(
        members=[trig, calc],
        centroid=[0.6, 0.4],
        children=[
            HierarchyNode(  # internal -> will reuse 'geometry' by name
                members=[trig],
                centroid=[0.4, 0.6],
                children=[HierarchyNode(members=[trig], centroid=[0.4, 0.6])],
            ),
            HierarchyNode(members=[calc], centroid=[1.0, 0.0]),  # leaf
        ],
    )
    monkeypatch.setattr(tr, "find_emergent_hierarchy", lambda *a, **k: [top])

    def name_fn(cluster, candidates, url, tok):
        names = {m.name for m in cluster.members}
        label = "mathematics" if "calculus" in names else "geometry"
        return NameResult(label=label, description="", confidence=0.9)

    mint_calls = []

    def mint_fn(
        cluster,
        name,
        *,
        hcg,
        milvus,
        source_cluster_id,
        parent_type_uuid,
        parent_ancestors,
        parent_name,
        retype_members,
    ):
        uuid = f"type_{name.label}_super01"
        hcg.nodes[uuid] = {
            "uuid": uuid,
            "name": name.label,
            "properties": {
                "is_type_definition": True,
                "ancestors": list(parent_ancestors) + [parent_name],
            },
        }
        hcg.add_edge(uuid, parent_type_uuid, "IS_A")
        milvus.update_centroid(uuid, cluster.embeddings[0], "m")
        mint_calls.append(name.label)
        return uuid

    TypeRollupHandler(
        config=_cfg(),
        hcg=hcg,
        milvus=milvus,
        event_bus=None,
        hermes_url="http://h",
        token="t",
        name_fn=name_fn,
        mint_fn=mint_fn,
    ).run()

    # only 'mathematics' minted; 'geometry' was reused, not duplicated
    assert mint_calls == ["mathematics"]
    math_uuid = "type_mathematics_super01"
    # the REUSED 'geometry' is now under 'mathematics' (not still under entity)
    assert any(
        e["source"] == "type_geometry_existing"
        and e["target"] == math_uuid
        and e["relation"] == "IS_A"
        for e in hcg.edges
    )
    assert hcg.nodes["type_geometry_existing"]["properties"]["ancestors"] == [
        "root",
        "node",
        "entity",
        "mathematics",
    ]
    # and trig sits under geometry with the full chain
    assert hcg.nodes["type_trig_aa"]["properties"]["ancestors"] == [
        "root",
        "node",
        "entity",
        "mathematics",
        "geometry",
    ]


def test_centroid_match_never_selects_a_cluster_member_as_its_own_super(monkeypatch):
    """A cluster centroid is the mean of its members, so its nearest existing
    type is frequently one of those members. Reusing a member as the cluster's
    super-type persists a peer-as-parent IS_A edge (member-of-X IS_A
    sibling-member-of-X) and a wrong ancestor cascade. The member-exclusion in
    `_match_existing_type` must skip such hits and mint a fresh super instead
    (greptile #161)."""
    import sophia.maintenance.type_rollup_handler as tr

    tds = [
        _td("type_calculus_aa", "calculus", ["root", "node", "entity"]),
        _td("type_algebra_bb", "algebra", ["root", "node", "entity"]),
    ]
    hcg = FakeHCG(tds)
    milvus = FakeMilvus({"type_calculus_aa": [1.0, 0.0], "type_algebra_bb": [0.9, 0.1]})
    # The cluster centroid [0.95, 0.05] is nearest to its OWN member
    # type_calculus_aa (cosine ~0.999, well above the 0.9 match threshold).
    # Without exclusion the handler would reuse that member as the super-type.
    milvus.find_nearest_types = lambda q, top_k=1: [
        {"uuid": "type_calculus_aa", "score": 0.0}
    ]
    leaf = HierarchyNode(
        members=[
            Member(
                "type_calculus_aa",
                "calculus",
                [1.0, 0.0],
                Counter(),
                "type_definition",
                None,
                [],
                "m",
            ),
            Member(
                "type_algebra_bb",
                "algebra",
                [0.9, 0.1],
                Counter(),
                "type_definition",
                None,
                [],
                "m",
            ),
        ],
        centroid=[0.95, 0.05],
        children=[
            HierarchyNode(
                members=[
                    Member(
                        "type_calculus_aa",
                        "calculus",
                        [1.0, 0.0],
                        Counter(),
                        "type_definition",
                        None,
                        [],
                        "m",
                    )
                ],
                centroid=[1.0, 0.0],
            ),
            HierarchyNode(
                members=[
                    Member(
                        "type_algebra_bb",
                        "algebra",
                        [0.9, 0.1],
                        Counter(),
                        "type_definition",
                        None,
                        [],
                        "m",
                    )
                ],
                centroid=[0.9, 0.1],
            ),
        ],
    )
    monkeypatch.setattr(tr, "find_emergent_hierarchy", lambda *a, **k: [leaf])
    mint_calls = []
    _handler(hcg, milvus, mint_calls=mint_calls).run()

    # A fresh super-type was minted (the member was NOT reused as the parent).
    assert mint_calls == [("mathematics", "type_entity", False)]
    super_uuid = "type_mathematics_super01"
    # No peer-as-parent edge: a member must never become its sibling's parent.
    assert not any(
        e["source"] == "type_algebra_bb"
        and e["target"] == "type_calculus_aa"
        and e["relation"] == "IS_A"
        for e in hcg.edges
    )
    # Both leaves sit under the freshly minted super-type instead.
    for leaf_uuid in ("type_calculus_aa", "type_algebra_bb"):
        assert any(
            e["source"] == leaf_uuid
            and e["target"] == super_uuid
            and e["relation"] == "IS_A"
            for e in hcg.edges
        )


def test_reparent_drops_stale_is_a_even_without_edge_id():
    """A type must never end up with two IS_A parents. When the stale edge was
    persisted without an id/uuid, _current_is_a returns a parent but no edge
    handle; delete_edge(None) would silently no-op. _reparent_one must fall
    back to a (source, target, relation) delete so the old parent is removed
    before the new IS_A is added (greptile #161)."""
    tds = [_td("type_child_aa", "child", ["root", "node", "entity"])]
    # Pre-existing IS_A edge to an OLD parent, persisted WITHOUT an id.
    stale = [
        {"source": "type_child_aa", "target": "type_oldparent_xx", "relation": "IS_A"}
    ]
    hcg = FakeHCG(tds, is_a=stale)
    milvus = FakeMilvus({"type_child_aa": [1.0, 0.0]})
    handler = _handler(hcg, milvus)
    # Mirror what run() would build: the child sits under the old parent.
    handler._children_of = {"type_oldparent_xx": ["type_child_aa"]}
    handler._name_of = {"type_child_aa": "child"}

    handler._reparent_one(
        "type_child_aa", "type_newparent_bb", ["root", "node", "entity"], "newparent"
    )

    is_a_targets = [
        e["target"]
        for e in hcg.edges
        if e["source"] == "type_child_aa" and e["relation"] == "IS_A"
    ]
    # Exactly one IS_A parent, and it is the new one (stale edge removed).
    assert is_a_targets == ["type_newparent_bb"]
    assert "type_oldparent_xx" not in is_a_targets
    # The id-less stale edge was removed via the triple-delete fallback.
    assert ("delete_edges_between", "type_child_aa", "type_oldparent_xx", "IS_A") in (
        hcg.writes
    )


def test_top_level_super_grafts_under_realm_root(monkeypatch):
    """B (concept/process population): when Hermes names a ``parent`` for a
    TOP-LEVEL super-type, the rollup roots it under that realm root (here
    ``concept``) instead of flat under ``entity``, and the realm chain flows
    down to the leaves. The rollup is the single hierarchy authority (#160)."""
    import sophia.maintenance.type_rollup_handler as tr

    tds = [
        _td("type_concept", "concept", ["root", "node"]),  # seeded realm root
        _td("type_calculus_aa", "calculus", ["root", "node", "entity"]),
        _td("type_algebra_bb", "algebra", ["root", "node", "entity"]),
    ]
    hcg = FakeHCG(tds)
    milvus = FakeMilvus({"type_calculus_aa": [1.0, 0.0], "type_algebra_bb": [0.9, 0.1]})

    def _m(uuid, name, vec):
        return Member(uuid, name, vec, Counter(), "type_definition", None, [], "m")

    leaf = HierarchyNode(
        members=[
            _m("type_calculus_aa", "calculus", [1.0, 0.0]),
            _m("type_algebra_bb", "algebra", [0.9, 0.1]),
        ],
        centroid=[0.95, 0.05],
        children=[
            HierarchyNode(
                members=[_m("type_calculus_aa", "calculus", [1.0, 0.0])],
                centroid=[1.0, 0.0],
            ),
            HierarchyNode(
                members=[_m("type_algebra_bb", "algebra", [0.9, 0.1])],
                centroid=[0.9, 0.1],
            ),
        ],
    )
    monkeypatch.setattr(tr, "find_emergent_hierarchy", lambda *a, **k: [leaf])

    mint_calls: list = []

    def name_fn(cluster, candidates, url, tok):
        # `concept` must be offered as a candidate so Hermes can graft under it.
        assert "concept" in candidates
        return NameResult(
            label="mathematics", description="", confidence=0.9, parent="concept"
        )

    def mint_fn(
        cluster,
        name,
        *,
        hcg,
        milvus,
        source_cluster_id,
        parent_type_uuid,
        parent_ancestors,
        parent_name,
        retype_members,
    ):
        uuid = f"type_{name.label}_super01"
        hcg.nodes[uuid] = {
            "uuid": uuid,
            "name": name.label,
            "properties": {
                "is_type_definition": True,
                "ancestors": list(parent_ancestors) + [parent_name],
            },
        }
        hcg.add_edge(uuid, parent_type_uuid, "IS_A")
        milvus.update_centroid(uuid, cluster.embeddings[0], "m")
        mint_calls.append((name.label, parent_type_uuid, retype_members))
        return uuid

    TypeRollupHandler(
        config=_cfg(),
        hcg=hcg,
        milvus=milvus,
        event_bus=None,
        hermes_url="http://h",
        token="t",
        name_fn=name_fn,
        mint_fn=mint_fn,
    ).run()

    # The super-type was rooted under `concept`, not `entity`.
    assert mint_calls == [("mathematics", "type_concept", False)]
    super_uuid = "type_mathematics_super01"
    assert any(
        e["source"] == super_uuid
        and e["target"] == "type_concept"
        and e["relation"] == "IS_A"
        for e in hcg.edges
    )
    assert hcg.nodes[super_uuid]["properties"]["ancestors"] == [
        "root",
        "node",
        "concept",
    ]
    # The realm chain flows down to the leaves.
    for leaf_uuid in ("type_calculus_aa", "type_algebra_bb"):
        assert hcg.nodes[leaf_uuid]["properties"]["ancestors"] == [
            "root",
            "node",
            "concept",
            "mathematics",
        ]


def test_centroid_match_never_reparents_a_seeded_realm_root(monkeypatch):
    """Blocker (review): the reuse path must never move a seeded structural root.
    A domain super-cluster whose centroid is nearest the seeded `concept`
    centroid must MINT a fresh super, not reuse type_concept (which _reparent_one
    would then move under the domain tree -- no cycle, so the cycle guard cannot
    stop it). Realm roots are not cluster members, so exclude=members can't catch
    this; the _PROTECTED_ROOT_UUIDS guard must."""
    import sophia.maintenance.type_rollup_handler as tr

    tds = [
        _td("type_concept", "concept", ["root", "node"]),  # seeded root w/ centroid
        _td("type_calculus_aa", "calculus", ["root", "node", "entity"]),
        _td("type_algebra_bb", "algebra", ["root", "node", "entity"]),
    ]
    hcg = FakeHCG(tds)
    milvus = FakeMilvus(
        {
            "type_concept": [0.95, 0.05],
            "type_calculus_aa": [1.0, 0.0],
            "type_algebra_bb": [0.9, 0.1],
        }
    )
    # The cluster centroid is nearest the seeded concept centroid.
    milvus.find_nearest_types = lambda q, top_k=1: [
        {"uuid": "type_concept", "score": 0.0}
    ]

    def _m(uuid, name, vec):
        return Member(uuid, name, vec, Counter(), "type_definition", None, [], "m")

    leaf = HierarchyNode(
        members=[
            _m("type_calculus_aa", "calculus", [1.0, 0.0]),
            _m("type_algebra_bb", "algebra", [0.9, 0.1]),
        ],
        centroid=[0.95, 0.05],
        children=[
            HierarchyNode(
                members=[_m("type_calculus_aa", "calculus", [1.0, 0.0])],
                centroid=[1.0, 0.0],
            ),
            HierarchyNode(
                members=[_m("type_algebra_bb", "algebra", [0.9, 0.1])],
                centroid=[0.9, 0.1],
            ),
        ],
    )
    monkeypatch.setattr(tr, "find_emergent_hierarchy", lambda *a, **k: [leaf])
    mint_calls: list = []
    # name_fn returns a normal label with no parent (no graft rebind interferes).
    _handler(hcg, milvus, mint_calls=mint_calls, name_label="mathematics").run()

    # The seeded concept root was NOT reused/reparented.
    assert not any(
        e["source"] == "type_concept" and e["relation"] == "IS_A" for e in hcg.edges
    )
    assert hcg.nodes["type_concept"]["properties"]["ancestors"] == ["root", "node"]
    # A fresh super was minted under entity instead.
    assert mint_calls == [("mathematics", "type_entity", False)]
