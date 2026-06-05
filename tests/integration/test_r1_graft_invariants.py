"""R1 graft write-path invariants on a disposable live graph (#172).

THIS SUITE IS THE BLOCKING PROMOTION GATE for naming-driven typing (vault
sophia/plans/naming-driven-typing/SPEC.md, section 8 risk R1). The offline
experiment validates SEMANTIC quality only -- cycle prevention, idempotency,
same-norm dedup, and residual durability live in write paths the offline
harness never executes. A green offline verdict is therefore necessary but
NOT sufficient: production wiring of the typing pass stays blocked until this
suite passes against a live graph.

Four invariants (SPEC sections 5.9-5.13):

1. run-twice idempotency  -- the typing pass (mint + rollup) twice over the
   same inputs creates ZERO new type-defs and ZERO IS_A edge churn.
2. two-writer arbitration -- A-under-B then B-under-A subsumption claims
   resolve to ONE IS_A direction plus a recorded AMBIGUOUS_SUBSUMPTION edge;
   the live IS_A closure stays acyclic.
3. same-norm dedup        -- two DISTINCT clusters sharing a canonical name,
   the same proposed root, and centroids within the match band yield exactly
   ONE minted type-def.
4. residual durability    -- an evicted member is not re-included in the next
   cluster pull. UNIMPLEMENTED in production today (SPEC section 5.13): the
   eviction path filters flagged members out of the cluster but never retypes
   them, so the type_uuid pull re-includes them. The test xfails with that
   citation; remove the marker when production parks residuals durably.

Gating mirrors tests/integration/test_node_identity_live.py (integration
marker + skip when Neo4j is unreachable) plus an explicit R1_LIVE=1 env gate
so the suite never fires implicitly on a shared stack.

Scope of liveness: Neo4j is LIVE -- every node write, IS_A edge swing,
member retype, and AMBIGUOUS_SUBSUMPTION record below goes through the
production HCGClient write path. Milvus and the Hermes namer are in-process
fakes injected through the same seams build_emergence_handler /
build_type_rollup_handler use (and tests/maintenance mirrors): the four R1
invariants are GRAPH invariants, and the code under test --
EmergenceHandler, TypeRollupHandler, mint_type, _reparent_one, the dedup
predicate -- runs unmodified production code.

Disposable-graph policy: every uuid this suite creates embeds a per-test
namespace token, and the stub namer embeds the token in every label so
minted type uuids (type_<slug>_<hex8>) inherit it. Teardown DETACH-DELETEs
every node and reified edge node carrying the token. The only shared key
touched is the seeded type_entity realm root (the graft terminal), created
only when absent and removed again only if this suite created it.
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

from sophia.maintenance.config import MaintenanceConfig
from sophia.maintenance.emergence_handler import (
    EmergenceHandler,
    current_categories,
    load_type_members,
)
from sophia.maintenance.emergence_types import NameResult
from sophia.maintenance.type_minting import mint_type
from sophia.maintenance.type_rollup_handler import TypeRollupHandler

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("R1_LIVE") != "1",
        reason="R1 graft-invariant gate runs only with R1_LIVE=1 and live Neo4j",
    ),
]

_MODEL = "all-MiniLM-L6-v2"
_DIM = 8
_JITTER_AXIS = 6
# Axis layout for the synthetic embeddings: 0/1 = rollup super groups,
# 2/3 = rollup fine-group offsets, 4/5 = emergence member groups, 6 = jitter.


def _vec(
    main_axis: int,
    off_axis: int | None = None,
    off: float = 0.0,
    jitter: float = 0.0,
) -> list[float]:
    v = [0.0] * _DIM
    v[main_axis] = 1.0
    if off_axis is not None:
        v[off_axis] = off
    v[_JITTER_AXIS] = jitter
    return v


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class FakeMilvus:
    """In-memory vector store mirroring the tests/maintenance fakes.

    Serves the entity collection (member vectors) and TypeCentroid (minted
    or seeded centroids); find_nearest_types ranks the centroid store by
    cosine, exactly the contract the production handlers rely on.
    """

    def __init__(self) -> None:
        self.members: dict[str, dict] = {}
        self.centroids: dict[str, dict] = {}

    def get_embedding(self, node_type: str, uuid: str) -> dict | None:
        store = self.centroids if node_type == "TypeCentroid" else self.members
        row = store.get(uuid)
        return dict(row) if row else None

    def update_centroid(
        self, type_uuid: str, centroid: list[float], model: str
    ) -> None:
        self.centroids[type_uuid] = {
            "embedding": list(centroid),
            "model": model,
            "embedding_model": model,
        }

    def find_nearest_types(
        self, query_embedding: list[float], top_k: int = 1
    ) -> list[dict]:
        ranked = sorted(
            self.centroids.items(),
            key=lambda kv: -_cos(query_embedding, kv[1]["embedding"]),
        )
        return [{"uuid": u} for u, _ in ranked[:top_k]]


class NamespacedRollup(TypeRollupHandler):
    """Production rollup with a namespace-filtered LOADER.

    Production sweeps the whole type layer; on a shared dev graph that would
    re-parent type-defs owned by other processes. The filter is a READ-side
    isolation seam only -- every write below it (_tier1_explicit,
    _tier2_residual, _reparent_subtree, _reparent_one, mint_type) is
    unmodified production code against the live graph.
    """

    def __init__(self, *args, namespace: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._namespace = namespace

    def _load_type_layer(self) -> list[dict]:
        return [r for r in super()._load_type_layer() if self._namespace in r["uuid"]]


def _config() -> MaintenanceConfig:
    # variance_threshold is lowered so the compact synthetic fixtures clear
    # the junk-drawer pre-filter deterministically; everything else matches
    # the production defaults that matter to the invariants (notably the
    # 0.9 type_match_threshold dedup band).
    return MaintenanceConfig(
        variance_threshold=0.05,
        min_cluster_size=3,
        hermes_confidence_floor=0.5,
        type_match_threshold=0.9,
        rollup_min_cluster_size=2,
        rollup_min_supercluster_size=2,
    )


def _dominant_axis(vectors: list[list[float]]) -> int:
    centroid = [sum(col) / len(vectors) for col in zip(*vectors, strict=True)]
    return max(range(len(centroid)), key=lambda i: abs(centroid[i]))


def _axis_namer(labels_by_axis: dict[int, str], removed: list[str] | None = None):
    """Deterministic in-process namer: label = dominant centroid axis.

    Stands in for Hermes name_cluster through the injected name_fn seam; the
    same cluster always gets the same label, so re-runs exercise the dedup /
    no-op anchors rather than namer noise.
    """

    def name_fn(cluster, candidates, hermes_url, token):
        label = labels_by_axis[_dominant_axis(cluster.embeddings)]
        in_cluster = {m.uuid for m in cluster.members}
        rm = [u for u in (removed or []) if u in in_cluster]
        return NameResult(label=label, description="", confidence=0.9, removed=rm)

    return name_fn


def _emergence_handler(hcg, milvus, name_fn) -> EmergenceHandler:
    return EmergenceHandler(
        config=_config(),
        hcg=hcg,
        milvus=milvus,
        event_bus=None,
        hermes_url="http://stub.invalid",
        token="stub",
        load_members=lambda type_uuid: load_type_members(hcg, milvus, type_uuid),
        name_fn=name_fn,
        mint_fn=mint_type,
        candidates_fn=lambda: current_categories(hcg),
    )


def _rollup_handler(hcg, milvus, name_fn, namespace: str) -> NamespacedRollup:
    return NamespacedRollup(
        config=_config(),
        hcg=hcg,
        milvus=milvus,
        event_bus=None,
        hermes_url="http://stub.invalid",
        token="stub",
        name_fn=name_fn,
        mint_fn=mint_type,
        namespace=namespace,
    )


# ------------------------------------------------------------------ seeding


def _seed_type_def(
    hcg,
    uuid: str,
    name: str,
    ancestors: list[str],
    source: str,
    parent_uuid: str | None = None,
) -> str:
    hcg.add_node(
        name=name,
        node_type="type_definition",
        uuid=uuid,
        properties={"is_type_definition": True, "ancestors": list(ancestors)},
        source=source,
    )
    if parent_uuid is not None:
        hcg.add_edge(uuid, parent_uuid, "IS_A")
    return uuid


def _seed_member(
    hcg,
    milvus: FakeMilvus,
    ns: str,
    ident: str,
    type_uuid: str,
    embedding: list[float],
) -> str:
    uuid = f"{ns}-{ident}"
    hcg.add_node(name=f"{ns} {ident}", node_type="entity", uuid=uuid, source=f"r1-{ns}")
    # The production retype write: the authoritative current-type pointer.
    hcg.update_node(uuid, {"type": "entity", "type_uuid": type_uuid})
    milvus.members[uuid] = {"embedding": list(embedding), "model": _MODEL}
    return uuid


# ---------------------------------------------------------------- snapshots


def _ns_is_a_edges(hcg, token: str) -> set[tuple[str, str, str | None]]:
    """(source, target, edge-node uuid) for IS_A edges touching the namespace.

    Comparing edge-node uuids as well as endpoints catches delete-and-recreate
    churn that an endpoint-only set comparison would miss.
    """
    edges = hcg.list_all_edges(relation_type="IS_A", limit=100_000) or []
    return {
        (e["source"], e["target"], e.get("id"))
        for e in edges
        if token in (e.get("source") or "") or token in (e.get("target") or "")
    }


def _ns_type_defs(hcg, token: str) -> dict[str, list[str]]:
    """uuid -> stored ancestors for every namespaced type-def (count + churn)."""
    return {
        td["uuid"]: list((td.get("properties") or {}).get("ancestors") or [])
        for td in (hcg.get_all_type_definitions() or [])
        if token in td["uuid"]
    }


def _types_named(hcg, name: str) -> list[str]:
    return sorted(
        td["uuid"]
        for td in (hcg.get_all_type_definitions() or [])
        if td.get("name") == name
    )


def _has_is_a_cycle(hcg, token: str) -> bool:
    """Walk the LIVE IS_A edges touching the namespace; True if any cycle."""
    edges = hcg.list_all_edges(relation_type="IS_A", limit=100_000) or []
    adj: dict[str, list[str]] = {}
    nodes: set[str] = set()
    for e in edges:
        s, t = e.get("source") or "", e.get("target") or ""
        if token in s or token in t:
            adj.setdefault(s, []).append(t)
            nodes.update((s, t))
    state: dict[str, int] = {}

    def visit(node: str) -> bool:
        state[node] = 1
        for nxt in adj.get(node, []):
            if state.get(nxt) == 1:
                return True
            if state.get(nxt, 0) == 0 and visit(nxt):
                return True
        state[node] = 2
        return False

    return any(state.get(n, 0) == 0 and visit(n) for n in nodes)


# ----------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def hcg():
    from sophia.hcg_client import HCGClient

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "logosdev")
    try:
        client = HCGClient(neo4j_uri=uri, neo4j_username=user, neo4j_password=password)
        client._execute_query("RETURN 1 AS ok", {})  # connectivity probe
    except Exception as exc:  # pragma: no cover - infra-dependent
        pytest.skip(f"Neo4j not available at {uri}: {exc}")
    yield client
    client.close()


@pytest.fixture(scope="module")
def entity_root(hcg):
    """Ensure the seeded type_entity realm root (rollup graft terminal) exists.

    Created only when absent (routinely-wiped dev graphs); deleted again in
    teardown only if THIS suite created it, so a seeded shared root is never
    touched.
    """
    marker = "r1-gate-fixture"
    created = hcg.get_node("type_entity") is None
    if created:
        hcg.add_node(
            name="entity",
            node_type="type_definition",
            uuid="type_entity",
            properties={"is_type_definition": True, "ancestors": ["root", "node"]},
            source=marker,
        )
    yield "type_entity"
    if created:
        hcg._execute_query(
            "MATCH (n:Node {uuid: $u, source: $s}) DETACH DELETE n",
            {"u": "type_entity", "s": marker},
        )


@pytest.fixture
def ns(hcg):
    """Per-test namespace token + teardown of everything carrying it."""
    token = f"r1g{uuid4().hex[:8]}"
    yield token
    # Nodes carry the token in uuid or provenance source; reified edge nodes
    # carry it in their source/target endpoint uuids.
    hcg._execute_query(
        "MATCH (n:Node) WHERE n.uuid CONTAINS $t OR n.source CONTAINS $t "
        "OR n.target CONTAINS $t DETACH DELETE n",
        {"t": token},
    )


# -------------------------------------------------------------------- tests


def test_run_twice_zero_churn(hcg, entity_root, ns):
    """SPEC 5.11/5.12: the second identical typing pass writes NOTHING.

    Pass = emergence (mint from a junk-drawer type) + rollup (graft the flat
    type layer). Inputs: six members in two clusters under a disposable
    junk-drawer type, plus eight flat type-defs under entity in four fine
    groups that consolidate into two super groups. The first pass must mint
    two leaf types and two super-types and re-parent all eight flats; the
    second pass must mint zero type-defs and leave the IS_A edge set --
    including edge-node identity -- and all stored ancestors bit-identical.
    """
    milvus = FakeMilvus()
    em_ns, ro_ns = f"{ns}em", f"{ns}ro"

    # Emergence inputs: junk-drawer type + two member clusters (axes 4 / 5).
    seed_uuid = _seed_type_def(
        hcg, f"type_{em_ns}base", f"{em_ns} base", ["root", "node"], f"r1-{ns}"
    )
    for i in range(3):
        _seed_member(
            hcg, milvus, em_ns, f"w{i}", seed_uuid, _vec(4, jitter=0.01 * (i - 1))
        )
        _seed_member(
            hcg, milvus, em_ns, f"g{i}", seed_uuid, _vec(5, jitter=0.01 * (i - 1))
        )

    # Rollup inputs: eight flat type-defs under entity, four fine groups
    # (axis 0 or 1, offset on 2 or 3) pairing into two super groups.
    groups = [(0, 2, 0.3), (0, 2, -0.3), (1, 3, 0.3), (1, 3, -0.3)]
    flat_types: list[str] = []
    for gi, (main, off_axis, off) in enumerate(groups):
        for j in range(2):
            uuid = f"type_{ro_ns}t{gi}{j}"
            _seed_type_def(
                hcg,
                uuid,
                f"{ro_ns} t{gi}{j}",
                ["root", "node", "entity"],
                f"r1-{ns}",
                parent_uuid="type_entity",
            )
            milvus.update_centroid(
                type_uuid=uuid,
                centroid=_vec(main, off_axis, off, jitter=0.01 * (1 if j else -1)),
                model=_MODEL,
            )
            flat_types.append(uuid)

    emergence = _emergence_handler(
        hcg, milvus, _axis_namer({4: f"{em_ns} widget", 5: f"{em_ns} gadget"})
    )
    rollup = _rollup_handler(
        hcg,
        milvus,
        _axis_namer({0: f"{ro_ns} machines", 1: f"{ro_ns} creatures"}),
        namespace=ro_ns,
    )

    def typing_pass() -> None:
        emergence.run(seed_uuid)
        rollup.run()

    typing_pass()
    types_first = _ns_type_defs(hcg, ns)
    edges_first = _ns_is_a_edges(hcg, ns)

    # Sanity: the first pass really built structure (zero-churn must not be
    # vacuous). Minted uuids embed the namespaced label slug.
    minted = set(types_first) - {seed_uuid, *flat_types}
    leaf_mints = {u for u in minted if "_widget_" in u or "_gadget_" in u}
    super_mints = {u for u in minted if "_machines_" in u or "_creatures_" in u}
    assert len(leaf_mints) == 2, f"expected widget+gadget mints, got {sorted(minted)}"
    assert len(super_mints) == 2, f"expected two super mints, got {sorted(minted)}"
    for uuid in flat_types:
        (parent,) = {t for (s, t, _e) in edges_first if s == uuid}
        assert parent in super_mints, f"{uuid} not lifted under a minted super"

    # The gate: run the identical pass again.
    typing_pass()
    assert (
        _ns_type_defs(hcg, ns) == types_first
    ), "second pass must mint zero type-defs and leave ancestors untouched"
    assert (
        _ns_is_a_edges(hcg, ns) == edges_first
    ), "second pass must produce zero IS_A churn (count, edge set, edge ids)"


def test_two_writer_arbitration_no_loop(hcg, entity_root, ns):
    """SPEC 5.9: contradictory subsumption resolves without an IS_A loop.

    Equal-weight PART_OF evidence in both directions drives tier-1 to attempt
    A-under-B and B-under-A in one pass (the documented tie path). The first
    reparent wins; the second must hit _creates_cycle, record a single
    canonical AMBIGUOUS_SUBSUMPTION edge, and leave the live closure acyclic.
    A second pass over the same evidence must be a pure no-op (G-IDEM-3).
    """
    milvus = FakeMilvus()
    a_uuid = _seed_type_def(
        hcg,
        f"type_{ns}alpha",
        f"{ns} alpha",
        ["root", "node", "entity"],
        f"r1-{ns}",
        parent_uuid="type_entity",
    )
    b_uuid = _seed_type_def(
        hcg,
        f"type_{ns}beta",
        f"{ns} beta",
        ["root", "node", "entity"],
        f"r1-{ns}",
        parent_uuid="type_entity",
    )
    ma = _seed_member(hcg, milvus, ns, "ma", a_uuid, _vec(4))
    mb = _seed_member(hcg, milvus, ns, "mb", b_uuid, _vec(5))
    hcg.add_edge(ma, mb, "PART_OF")
    hcg.add_edge(mb, ma, "PART_OF")

    _rollup_handler(hcg, milvus, lambda *args: None, namespace=ns).run()

    is_a = _ns_is_a_edges(hcg, ns)
    between = {(s, t) for (s, t, _e) in is_a if {s, t} == {a_uuid, b_uuid}}
    assert len(between) == 1, f"exactly one arbitration winner expected: {between}"

    amb = [
        e
        for e in (
            hcg.list_all_edges(relation_type="AMBIGUOUS_SUBSUMPTION", limit=100_000)
            or []
        )
        if ns in (e.get("source") or "")
    ]
    assert len(amb) == 1, "the rejected direction must be recorded once"
    lo, hi = sorted((a_uuid, b_uuid))
    assert (amb[0]["source"], amb[0]["target"]) == (
        lo,
        hi,
    ), "AMBIGUOUS_SUBSUMPTION must use the canonical endpoint order"

    assert not _has_is_a_cycle(hcg, ns), "live IS_A closure must stay acyclic"

    # The loser was NOT force-reparented: it keeps its original root parent.
    (winner_child, winner_parent) = next(iter(between))
    loser = b_uuid if winner_child == a_uuid else a_uuid
    assert {t for (s, t, _e) in is_a if s == loser} == {"type_entity"}
    assert {t for (s, t, _e) in is_a if s == winner_child} == {winner_parent}

    # Second writer pass over the same evidence: zero churn, still one record.
    _rollup_handler(hcg, milvus, lambda *args: None, namespace=ns).run()
    assert _ns_is_a_edges(hcg, ns) == is_a
    amb_after = [
        e
        for e in (
            hcg.list_all_edges(relation_type="AMBIGUOUS_SUBSUMPTION", limit=100_000)
            or []
        )
        if ns in (e.get("source") or "")
    ]
    assert len(amb_after) == 1


def test_same_norm_dedup_single_mint(hcg, ns):
    """SPEC 5.11 G-IDEM-2: same canonical name + same root => ONE type-def.

    Two DISTINCT clusters (disjoint members, separate passes so the first is
    published before the second resolves) share a canonical name, the same
    proposed root, and centroids within the 0.9 match band. The second
    resolution must convert mint into reuse: exactly one type-def carries the
    name, and the second cluster is absorbed by the production retype write.
    """
    milvus = FakeMilvus()
    seed_uuid = _seed_type_def(
        hcg, f"type_{ns}base", f"{ns} base", ["root", "node"], f"r1-{ns}"
    )
    for i in range(3):
        _seed_member(
            hcg, milvus, ns, f"w{i}", seed_uuid, _vec(4, jitter=0.01 * (i - 1))
        )
        _seed_member(
            hcg, milvus, ns, f"g{i}", seed_uuid, _vec(5, jitter=0.01 * (i - 1))
        )

    handler = _emergence_handler(
        hcg, milvus, _axis_namer({4: f"{ns} widget", 5: f"{ns} gadget"})
    )
    handler.run(seed_uuid)

    widgets = _types_named(hcg, f"{ns} widget")
    assert len(widgets) == 1, "first pass mints the widget type exactly once"
    widget_uuid = widgets[0]
    types_after_first = _ns_type_defs(hcg, ns)

    # Second, DISTINCT cluster: fresh members, same canonical name, same
    # proposed root (the same junk-drawer parent), centroid within the band.
    fresh = [
        _seed_member(
            hcg, milvus, ns, f"w2{i}", seed_uuid, _vec(4, jitter=0.008 * (i - 1))
        )
        for i in range(3)
    ]
    for i in range(3):
        _seed_member(
            hcg, milvus, ns, f"g2{i}", seed_uuid, _vec(5, jitter=0.008 * (i - 1))
        )
    handler.run(seed_uuid)

    assert _types_named(hcg, f"{ns} widget") == [widget_uuid], (
        "same-canonical-name cluster must reuse the published type-def, "
        "not mint a type_<slug>_<hex8> twin"
    )
    assert (
        _ns_type_defs(hcg, ns) == types_after_first
    ), "the second pass must mint zero new type-defs"
    member_uuids = {n["uuid"] for n in (hcg.get_nodes_by_type_uuid(widget_uuid) or [])}
    assert (
        set(fresh) <= member_uuids
    ), "reused type must absorb the second cluster via the retype write"


@pytest.mark.xfail(
    reason=(
        "SPEC section 5.13 (residual durability) is UNIMPLEMENTED in "
        "production: the eviction path (emergence_handler.py, the "
        "Hermes-flagged outlier filter around lines 252-254) drops removed "
        "members from the cluster but never retypes them off the source "
        "type, so the next type_uuid pull re-includes them. This xfail IS "
        "the promotion gate: remove the marker once residuals are durably "
        "parked (retype to type_entity or an excluded unsorted sentinel; "
        "sophia#175). strict=True so an unexpected XPASS fails the suite "
        "loudly, forcing the marker off the moment production catches up."
    ),
    strict=True,
)
def test_evicted_member_not_reclustered(hcg, ns):
    """SPEC 5.13: an evicted member must not re-enter the next cluster pull.

    The namer flags one widget member as removed; the cluster mints without
    it (true today, asserted as sanity). The GATE assertion -- the very next
    candidate pull excludes the evictee -- fails against current production
    because eviction leaves the old type_uuid in place, hence the xfail.
    """
    milvus = FakeMilvus()
    seed_uuid = _seed_type_def(
        hcg, f"type_{ns}base", f"{ns} base", ["root", "node"], f"r1-{ns}"
    )
    widget_members = [
        _seed_member(
            hcg, milvus, ns, f"w{i}", seed_uuid, _vec(4, jitter=0.01 * (i - 1))
        )
        for i in range(4)
    ]
    for i in range(3):
        _seed_member(
            hcg, milvus, ns, f"g{i}", seed_uuid, _vec(5, jitter=0.01 * (i - 1))
        )
    evicted = widget_members[-1]

    handler = _emergence_handler(
        hcg,
        milvus,
        _axis_namer({4: f"{ns} widget", 5: f"{ns} gadget"}, removed=[evicted]),
    )
    handler.run(seed_uuid)

    # Sanity (holds today): the cluster minted without the evictee.
    widgets = _types_named(hcg, f"{ns} widget")
    assert len(widgets) == 1
    minted_members = {n["uuid"] for n in (hcg.get_nodes_by_type_uuid(widgets[0]) or [])}
    assert evicted not in minted_members
    assert set(widget_members[:-1]) <= minted_members

    # THE GATE (SPEC 5.13): the next candidate pull -- the exact production
    # source build_emergence_handler wires (load_type_members) -- must not
    # re-include the evicted member.
    repulled = {m.uuid for m in load_type_members(hcg, milvus, seed_uuid)}
    assert evicted not in repulled, (
        "evicted member still carries the junk-drawer type_uuid and would be "
        "re-clustered by the next pass (SPEC section 5.13 residual durability)"
    )
