"""Unit tests for the shared placement module (naming-driven-typing B1 T1).

These tests pin the load-bearing invariants of ``placement.py`` (DESIGN §3/§6):
structure is the only representation of membership, so NO function may ever write
an ``ancestors`` property or do a descendant cascade; names resolve to uuids only
via the supplied ``uuid_by_name`` map (never a fabricated ``type_<name>`` slug);
guards deflect-and-record (cycle -> AMBIGUOUS_SUBSUMPTION) rather than force.
"""

from __future__ import annotations

from sophia.maintenance import placement


class FakeHCG:
    """Minimal in-memory reified-edge graph that records every write.

    Mirrors the real ``HCGClient`` surface the placement module touches:
    ``get_node`` / ``add_edge`` / ``delete_edge`` / ``delete_edges_between`` /
    ``query_edges_from``. ``update_node`` is recorded (never expected) so a test
    can prove the module writes no ``ancestors`` property.
    """

    def __init__(self, nodes, edges=None):
        # nodes: list of {uuid, name, properties{...}}
        self.nodes = {n["uuid"]: n for n in nodes}
        # edges: list of {id, source, target, relation, properties}
        self.edges = list(edges or [])
        self._eid = 0
        self.update_calls = []  # (uuid, properties) -- must never carry ancestors

    # ---- reads -----------------------------------------------------------
    def get_node(self, uuid):
        return self.nodes.get(uuid)

    def query_edges_from(self, uuid):
        return [dict(e) for e in self.edges if e.get("source") == uuid]

    # ---- writes ----------------------------------------------------------
    def add_edge(
        self,
        source,
        target,
        relation,
        edge_uuid=None,
        bidirectional=False,
        properties=None,
    ):
        # MERGE semantics: collapse a duplicate (source, target, relation).
        for e in self.edges:
            if (
                e["source"] == source
                and e["target"] == target
                and e["relation"] == relation
            ):
                e["properties"] = dict(properties or {})
                e["bidirectional"] = bidirectional
                return e["id"]
        self._eid += 1
        eid = edge_uuid or f"edge_{self._eid}"
        self.edges.append(
            {
                "id": eid,
                "source": source,
                "target": target,
                "relation": relation,
                "bidirectional": bidirectional,
                "properties": dict(properties or {}),
            }
        )
        return eid

    def delete_edge(self, edge_id):
        before = len(self.edges)
        self.edges = [e for e in self.edges if e["id"] != edge_id]
        return len(self.edges) < before

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
        return before - len(self.edges)

    def update_node(self, uuid, properties=None):
        self.update_calls.append((uuid, dict(properties or {})))
        return uuid

    # ---- test helpers ----------------------------------------------------
    def is_a_edges_from(self, source):
        return [
            e for e in self.edges if e["source"] == source and e["relation"] == "IS_A"
        ]

    def edges_of_relation(self, relation):
        return [e for e in self.edges if e["relation"] == relation]


def _node(uuid, name, node_type="type_definition"):
    return {"uuid": uuid, "name": name, "properties": {"type": node_type}}


def _seed_skeleton():
    """root <- node <- {entity, concept, process, cognition}; plus content types.

    vehicle IS_A entity (an in-domain published type); strategy IS_A cognition
    (a type whose IS_A walk never reaches a graftable realm).
    """
    nodes = [
        _node("u_root", "root"),
        _node("u_node", "node"),
        _node("u_entity", "entity"),
        _node("u_concept", "concept"),
        _node("u_process", "process"),
        _node("u_cognition", "cognition"),
        _node("u_vehicle", "vehicle"),
        _node("u_strategy", "strategy"),
    ]
    edges = [
        {
            "id": "e_node",
            "source": "u_node",
            "target": "u_root",
            "relation": "IS_A",
            "properties": {},
        },
        {
            "id": "e_entity",
            "source": "u_entity",
            "target": "u_node",
            "relation": "IS_A",
            "properties": {},
        },
        {
            "id": "e_concept",
            "source": "u_concept",
            "target": "u_node",
            "relation": "IS_A",
            "properties": {},
        },
        {
            "id": "e_process",
            "source": "u_process",
            "target": "u_node",
            "relation": "IS_A",
            "properties": {},
        },
        {
            "id": "e_cognition",
            "source": "u_cognition",
            "target": "u_node",
            "relation": "IS_A",
            "properties": {},
        },
        {
            "id": "e_vehicle",
            "source": "u_vehicle",
            "target": "u_entity",
            "relation": "IS_A",
            "properties": {},
        },
        {
            "id": "e_strategy",
            "source": "u_strategy",
            "target": "u_cognition",
            "relation": "IS_A",
            "properties": {},
        },
    ]
    uuid_by_name = {n["name"]: n["uuid"] for n in nodes}
    return nodes, edges, uuid_by_name


# --------------------------------------------------------------------------- #
# resolve_parent                                                              #
# --------------------------------------------------------------------------- #
def test_resolve_parent_returns_uuid_for_in_domain_type():
    nodes, edges, uuid_by_name = _seed_skeleton()
    hcg = FakeHCG(nodes, edges)
    assert (
        placement.resolve_parent("vehicle", uuid_by_name=uuid_by_name, hcg=hcg)
        == "u_vehicle"
    )


def test_resolve_parent_realm_root_itself_is_graftable():
    nodes, edges, uuid_by_name = _seed_skeleton()
    hcg = FakeHCG(nodes, edges)
    assert (
        placement.resolve_parent("entity", uuid_by_name=uuid_by_name, hcg=hcg)
        == "u_entity"
    )


def test_resolve_parent_unknown_name_is_closed_world_none():
    nodes, edges, uuid_by_name = _seed_skeleton()
    hcg = FakeHCG(nodes, edges)
    assert (
        placement.resolve_parent("spaceship", uuid_by_name=uuid_by_name, hcg=hcg)
        is None
    )


def test_resolve_parent_empty_is_none():
    nodes, edges, uuid_by_name = _seed_skeleton()
    hcg = FakeHCG(nodes, edges)
    assert placement.resolve_parent("", uuid_by_name=uuid_by_name, hcg=hcg) is None
    assert placement.resolve_parent("   ", uuid_by_name=uuid_by_name, hcg=hcg) is None


def test_resolve_parent_rejects_each_protected_name():
    nodes, edges, uuid_by_name = _seed_skeleton()
    hcg = FakeHCG(nodes, edges)
    for protected in ("root", "node", "cognition"):
        assert (
            placement.resolve_parent(protected, uuid_by_name=uuid_by_name, hcg=hcg)
            is None
        ), protected


def test_resolve_parent_rejects_underscore_prefixed():
    nodes, edges, uuid_by_name = _seed_skeleton()
    uuid_by_name = dict(uuid_by_name)
    uuid_by_name["_cognition"] = "u_cognition"
    hcg = FakeHCG(nodes, edges)
    assert (
        placement.resolve_parent("_cognition", uuid_by_name=uuid_by_name, hcg=hcg)
        is None
    )


def test_resolve_parent_rejects_reserved_prefixed():
    nodes, edges, uuid_by_name = _seed_skeleton()
    uuid_by_name = dict(uuid_by_name)
    uuid_by_name["reserved_agent"] = "u_reserved_agent"
    hcg = FakeHCG(nodes, edges)
    assert (
        placement.resolve_parent("reserved_agent", uuid_by_name=uuid_by_name, hcg=hcg)
        is None
    )


def test_resolve_parent_rejects_type_outside_graftable_realm():
    # strategy IS_A cognition -- its IS_A walk never reaches entity/concept/process.
    nodes, edges, uuid_by_name = _seed_skeleton()
    hcg = FakeHCG(nodes, edges)
    assert (
        placement.resolve_parent("strategy", uuid_by_name=uuid_by_name, hcg=hcg) is None
    )


def test_resolve_parent_never_fabricates_slug():
    # The name maps to a slug in NO map -> closed-world None, never type_<name>.
    nodes, edges, _ = _seed_skeleton()
    hcg = FakeHCG(nodes, edges)
    assert placement.resolve_parent("vehicle", uuid_by_name={}, hcg=hcg) is None


def test_resolve_parent_rejects_cross_realm_name_match():
    # Flat name->uuid catalog: the SAME name "vehicle" maps to two distinct types
    # -- one under the entity chain, one under the concept chain. The catalog
    # points at the concept one; an entity-realm cluster must NOT graft under it,
    # or entity content would root in concept and break IS_A uniformity (sec 3).
    nodes, edges, uuid_by_name = _seed_skeleton()
    nodes = nodes + [_node("u_vehicle_concept", "vehicle")]
    edges = edges + [
        {
            "id": "e_vehicle_concept",
            "source": "u_vehicle_concept",
            "target": "u_concept",
            "relation": "IS_A",
            "properties": {},
        }
    ]
    uuid_by_name = dict(uuid_by_name)
    uuid_by_name["vehicle"] = "u_vehicle_concept"  # catalog resolves to concept
    hcg = FakeHCG(nodes, edges)
    assert (
        placement.resolve_parent(
            "vehicle", uuid_by_name=uuid_by_name, hcg=hcg, realm="entity"
        )
        is None
    )


def test_resolve_parent_allows_same_realm_match():
    # vehicle IS_A entity: an entity-realm cluster may reuse a same-realm type.
    nodes, edges, uuid_by_name = _seed_skeleton()
    hcg = FakeHCG(nodes, edges)
    assert (
        placement.resolve_parent(
            "vehicle", uuid_by_name=uuid_by_name, hcg=hcg, realm="entity"
        )
        == "u_vehicle"
    )


def test_resolve_parent_no_realm_arg_skips_check():
    # realm=None (default): existing behavior is unchanged -- an in-domain parent
    # still resolves and no cross-realm check is applied.
    nodes, edges, uuid_by_name = _seed_skeleton()
    hcg = FakeHCG(nodes, edges)
    assert (
        placement.resolve_parent(
            "vehicle", uuid_by_name=uuid_by_name, hcg=hcg, realm=None
        )
        == "u_vehicle"
    )


# --------------------------------------------------------------------------- #
# realm_of                                                                    #
# --------------------------------------------------------------------------- #
def test_realm_of_walks_edges_to_realm():
    nodes, edges, _ = _seed_skeleton()
    hcg = FakeHCG(nodes, edges)
    assert placement.realm_of("u_vehicle", hcg=hcg) == "entity"


def test_realm_of_realm_root_returns_itself():
    nodes, edges, _ = _seed_skeleton()
    hcg = FakeHCG(nodes, edges)
    assert placement.realm_of("u_concept", hcg=hcg) == "concept"


def test_realm_of_none_when_chain_misses_realm():
    nodes, edges, _ = _seed_skeleton()
    hcg = FakeHCG(nodes, edges)
    assert placement.realm_of("u_strategy", hcg=hcg) is None


# --------------------------------------------------------------------------- #
# reparent                                                                    #
# --------------------------------------------------------------------------- #
def test_reparent_swings_edge_and_tags_placed_by():
    nodes, edges, _ = _seed_skeleton()
    nodes = nodes + [_node("u_car", "car")]
    edges = edges + [
        {
            "id": "e_car",
            "source": "u_car",
            "target": "u_entity",
            "relation": "IS_A",
            "properties": {},
        }
    ]
    hcg = FakeHCG(nodes, edges)
    children_of = {"u_entity": ["u_vehicle", "u_car"]}

    placement.reparent(
        "u_car",
        "u_vehicle",
        hcg=hcg,
        children_of=children_of,
        placed_by="parent_resolution",
    )

    # old IS_A removed, exactly one new IS_A car -> vehicle carrying placed_by
    car_edges = hcg.is_a_edges_from("u_car")
    assert len(car_edges) == 1
    assert car_edges[0]["target"] == "u_vehicle"
    assert car_edges[0]["properties"]["placed_by"] == "parent_resolution"
    assert hcg.delete_edge("e_car") is False  # already gone
    # children_of updated both sides
    assert "u_car" not in children_of["u_entity"]
    assert "u_car" in children_of["u_vehicle"]


def test_reparent_writes_no_ancestors_property():
    nodes, edges, _ = _seed_skeleton()
    nodes = nodes + [_node("u_car", "car")]
    edges = edges + [
        {
            "id": "e_car",
            "source": "u_car",
            "target": "u_entity",
            "relation": "IS_A",
            "properties": {},
        }
    ]
    hcg = FakeHCG(nodes, edges)
    children_of = {"u_entity": ["u_car"]}

    placement.reparent(
        "u_car",
        "u_vehicle",
        hcg=hcg,
        children_of=children_of,
        placed_by="parent_resolution",
    )

    # Structure is the only representation of membership: no per-node cache.
    assert hcg.update_calls == []
    assert not any("ancestors" in props for _, props in hcg.update_calls)
    assert not any("ancestors" in e["properties"] for e in hcg.edges)


def test_reparent_is_noop_when_already_correct():
    nodes, edges, _ = _seed_skeleton()
    nodes = nodes + [_node("u_car", "car")]
    edges = edges + [
        {
            "id": "e_car",
            "source": "u_car",
            "target": "u_vehicle",
            "relation": "IS_A",
            "properties": {"placed_by": "parent_resolution"},
        }
    ]
    hcg = FakeHCG(nodes, edges)
    children_of = {"u_vehicle": ["u_car"]}
    before = [dict(e) for e in hcg.edges]

    placement.reparent(
        "u_car",
        "u_vehicle",
        hcg=hcg,
        children_of=children_of,
        placed_by="parent_resolution",
    )

    assert hcg.edges == before  # untouched
    assert hcg.update_calls == []


def test_reparent_self_is_noop():
    nodes, edges, _ = _seed_skeleton()
    hcg = FakeHCG(nodes, edges)
    before = [dict(e) for e in hcg.edges]
    placement.reparent(
        "u_vehicle",
        "u_vehicle",
        hcg=hcg,
        children_of={},
        placed_by="parent_resolution",
    )
    assert hcg.edges == before


# --------------------------------------------------------------------------- #
# creates_cycle + deflect-and-record                                          #
# --------------------------------------------------------------------------- #
def test_creates_cycle_true_and_false():
    children_of = {"a": ["b"], "b": ["c"]}
    assert placement.creates_cycle("a", "c", children_of) is True
    assert placement.creates_cycle("a", "b", children_of) is True
    assert placement.creates_cycle("c", "a", children_of) is False
    assert placement.creates_cycle("a", "z", children_of) is False


def test_creates_cycle_terminates_on_corrupt_adjacency():
    children_of = {"a": ["b"], "b": ["a"]}  # already a loop
    assert placement.creates_cycle("a", "z", children_of) is False


def test_reparent_on_cycle_records_ambiguous_and_adds_no_is_a():
    nodes = [_node("u_a", "a"), _node("u_b", "b")]
    edges = [
        {
            "id": "e_b",
            "source": "u_b",
            "target": "u_a",
            "relation": "IS_A",
            "properties": {},
        }
    ]
    hcg = FakeHCG(nodes, edges)
    children_of = {"u_a": ["u_b"]}  # b is a descendant of a

    placement.reparent(
        "u_a",
        "u_b",
        hcg=hcg,
        children_of=children_of,
        placed_by="parent_resolution",
    )

    # no new IS_A edge from a; the original b -> a IS_A is untouched
    assert hcg.is_a_edges_from("u_a") == []
    ambiguous = hcg.edges_of_relation("AMBIGUOUS_SUBSUMPTION")
    assert len(ambiguous) == 1
    lo, hi = sorted(("u_a", "u_b"))
    assert ambiguous[0]["source"] == lo
    assert ambiguous[0]["target"] == hi
    assert ambiguous[0]["bidirectional"] is True
    assert ambiguous[0]["properties"]["detected_by"] == "placement"
    assert ambiguous[0]["properties"]["reason"] == "is_a_cycle"


def test_record_ambiguous_canonical_order():
    hcg = FakeHCG([_node("u_b", "b"), _node("u_a", "a")])
    placement.record_ambiguous("u_b", "u_a", hcg=hcg)
    edge = hcg.edges_of_relation("AMBIGUOUS_SUBSUMPTION")[0]
    assert (edge["source"], edge["target"]) == ("u_a", "u_b")  # lo, hi


# --------------------------------------------------------------------------- #
# attach                                                                       #
# --------------------------------------------------------------------------- #
def test_attach_loose_node_creates_single_is_a():
    nodes, edges, _ = _seed_skeleton()
    nodes = nodes + [_node("u_widget", "widget")]  # loose: no IS_A edge
    hcg = FakeHCG(nodes, edges)
    children_of = {}

    placement.attach(
        "u_widget",
        "u_entity",
        hcg=hcg,
        children_of=children_of,
        placed_by="name_reuse",
    )

    widget_edges = hcg.is_a_edges_from("u_widget")
    assert len(widget_edges) == 1
    assert widget_edges[0]["target"] == "u_entity"
    assert widget_edges[0]["properties"]["placed_by"] == "name_reuse"
    assert children_of["u_entity"] == ["u_widget"]
    assert hcg.update_calls == []


# --------------------------------------------------------------------------- #
# repark                                                                       #
# --------------------------------------------------------------------------- #
def test_repark_points_to_realm_root_with_root_fallback():
    nodes, edges, _ = _seed_skeleton()
    nodes = nodes + [_node("u_outlier", "outlier")]
    edges = edges + [
        {
            "id": "e_outlier",
            "source": "u_outlier",
            "target": "u_vehicle",
            "relation": "IS_A",
            "properties": {},
        }
    ]
    hcg = FakeHCG(nodes, edges)
    children_of = {"u_vehicle": ["u_outlier"]}

    placement.repark("u_outlier", "u_entity", hcg=hcg, children_of=children_of)

    outlier_edges = hcg.is_a_edges_from("u_outlier")
    assert len(outlier_edges) == 1
    assert outlier_edges[0]["target"] == "u_entity"
    assert outlier_edges[0]["properties"]["placed_by"] == "root_fallback"
    assert hcg.delete_edge("e_outlier") is False  # the stale edge is gone
    assert "u_outlier" not in children_of.get("u_vehicle", [])
    assert "u_outlier" in children_of["u_entity"]


# --------------------------------------------------------------------------- #
# placed_by vocabulary                                                         #
# --------------------------------------------------------------------------- #
def test_placed_by_vocabulary_is_pinned():
    assert placement.PLACED_BY_REASONS == frozenset(
        {"parent_resolution", "name_reuse", "root_fallback"}
    )
