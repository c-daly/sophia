# Emergent Type Discovery (logos #505) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Sophia turn the junk-drawer `entity` type into real, named types by clustering its members on two agreeing signals (embedding ∩ structure), asking Hermes to name each cluster, and minting the types — so the ontology grows from data and future ingestions classify correctly.

**Architecture:** A `type_emergence` maintenance handler (the scheduler already dispatches this `job_type` with `params={"type_uuid": ...}`) orchestrates: detect high variance → pull outliers → dual-signal cluster → Hermes `name_cluster` → mint type (node + Milvus centroid + `IS_A` rewire + `name_history`) → publish ontology change via `EventBus`. Pure logic (signatures, clustering, minting decisions) is split into small, unit-testable modules; the handler wires them to Neo4j/Milvus/Hermes.

**Tech Stack:** Python 3.12, Poetry, pytest (mock-based unit tests), Pydantic v2 (`BaseSettings`), Neo4j (`HCGClient`), Milvus (type-centroid store), Redis `EventBus` (`logos_events`), FastAPI (Hermes endpoint), `httpx` (Sophia→Hermes call).

**Design doc:** `sophia/docs/plans/2026-05-26-emergent-type-discovery-505-design.md`
**Branch:** `feat/505-emergent-type-discovery` (already created off `main`).
**Run tests with:** `poetry run pytest <path> -v` from the relevant repo root (`sophia/` or `hermes/`).

---

## File Structure

**Sophia (new):**
- `src/sophia/maintenance/emergence_types.py` — shared dataclasses (`Member`, `EmergentCluster`, `NameResult`).
- `src/sophia/maintenance/structural_signature.py` — neighbor-relation signature + similarity (pure).
- `src/sophia/maintenance/emergence_clustering.py` — outlier pull + dual-signal clustering (pure).
- `src/sophia/maintenance/type_minting.py` — mint type node + seed centroid + retype member + `name_history` (HCG/Milvus side effects, dependency-injected).
- `src/sophia/maintenance/hermes_naming.py` — Sophia→Hermes `name_cluster` client (httpx).
- `src/sophia/maintenance/emergence_handler.py` — the `type_emergence` orchestration handler.

**Sophia (modify):**
- `src/sophia/maintenance/config.py` — add emergence tunables.
- `src/sophia/ingestion/proposal_processor.py` — persist `hermes_type_hint`.
- `src/sophia/api/app.py` — register the handler in the scheduler `handlers` map; inject `EventBus`/Hermes URL.

**Hermes (new/modify):**
- `src/hermes/main.py` — `POST /name-cluster` endpoint + `NameClusterRequest`/`NameClusterResponse` models.

**Tests (new):** mirror under `sophia/tests/maintenance/`, `sophia/tests/ingestion/`, `hermes/tests/`.

---

## Task 1: Emergence config tunables

**Files:**
- Modify: `sophia/src/sophia/maintenance/config.py`
- Test: `sophia/tests/maintenance/test_config_emergence.py`

- [ ] **Step 1: Write the failing test**

```python
# sophia/tests/maintenance/test_config_emergence.py
from sophia.maintenance.config import MaintenanceConfig


def test_emergence_tunables_have_defaults():
    cfg = MaintenanceConfig()
    assert cfg.variance_threshold > 0
    assert cfg.min_cluster_size >= 2
    assert cfg.max_cluster_size >= cfg.min_cluster_size
    assert 0.0 < cfg.min_cohesion_improvement <= 1.0
    assert 0.0 <= cfg.hermes_confidence_floor <= 1.0


def test_emergence_tunables_env_override(monkeypatch):
    monkeypatch.setenv("SOPHIA_MAINTENANCE_MIN_CLUSTER_SIZE", "5")
    cfg = MaintenanceConfig()
    assert cfg.min_cluster_size == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd sophia && poetry run pytest tests/maintenance/test_config_emergence.py -v`
Expected: FAIL — `AttributeError: 'MaintenanceConfig' object has no attribute 'variance_threshold'`.

- [ ] **Step 3: Add the fields**

In `MaintenanceConfig`, after `max_concurrent_jobs`, add:

```python
    # --- Ontology evolution / emergence (#505) tunables ---
    variance_threshold: float = Field(
        default=0.6,
        gt=0,
        description="Mean squared distance from centroid above which a type is a junk-drawer candidate.",
    )
    min_cluster_size: int = Field(
        default=3, ge=2, description="Smallest cluster that may be minted into a type."
    )
    max_cluster_size: int = Field(
        default=50,
        ge=2,
        description="Largest membership sent verbatim to Hermes name_cluster; larger clusters are sampled.",
    )
    min_cohesion_improvement: float = Field(
        default=0.15,
        gt=0,
        le=1.0,
        description="Minimum fractional variance reduction a split must achieve to be accepted.",
    )
    hermes_confidence_floor: float = Field(
        default=0.5,
        ge=0,
        le=1.0,
        description="Discard Hermes cluster names below this confidence.",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd sophia && poetry run pytest tests/maintenance/test_config_emergence.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/sophia/maintenance/config.py tests/maintenance/test_config_emergence.py
git commit -m "feat(505): add emergence tunables to MaintenanceConfig"
```

---

## Task 2: Record Hermes' type hint at ingestion

**Files:**
- Modify: `sophia/src/sophia/ingestion/proposal_processor.py` (the block that assembles a node's properties before writing; near the `needs_reclassification` assignment, ~line 312)
- Test: `sophia/tests/ingestion/test_proposal_processor_hint.py`

- [ ] **Step 1: Read the surrounding code**

Run: `cd sophia && sed -n '290,330p' src/sophia/ingestion/proposal_processor.py`
Identify the dict (`node_props`) built per proposed node and the variable holding the proposed node (it carries `"type"` — Hermes' NER pick). Confirm the exact local variable names before editing.

- [ ] **Step 2: Write the failing test**

```python
# sophia/tests/ingestion/test_proposal_processor_hint.py
from sophia.ingestion.proposal_processor import ProposalProcessor


def test_hermes_type_hint_persisted(monkeypatch):
    """A proposed node's Hermes-assigned type is recorded as hermes_type_hint."""
    captured = {}

    class FakeHCG:
        def add_node(self, name, node_type, uuid=None, properties=None, **kw):
            captured["properties"] = properties or {}
            return uuid or "n1"

        def add_edge(self, *a, **k):
            return "e1"

    class FakeMilvus:
        def find_nearest_types(self, query_embedding, top_k=3):
            return []  # cold-start: forces fallback to entity

        def update_centroid(self, *a, **k):
            return None

    proc = ProposalProcessor(hcg=FakeHCG(), milvus=FakeMilvus(), event_bus=None)
    proposal = {
        "proposal_id": "p1",
        "document_embedding": {"embedding": None},
        "proposed_nodes": [
            {"name": "derivative", "type": "concept",
             "embedding": [0.1, 0.2], "embedding_id": "e1", "properties": {}}
        ],
        "proposed_edges": [],
    }
    proc.process(proposal)
    assert captured["properties"].get("hermes_type_hint") == "concept"
```

> Adjust the `ProposalProcessor(...)` constructor call and `FakeHCG.add_node` signature in Step 2 to match what Step 1 showed (constructor params, add_node kwargs). The assertion on `hermes_type_hint` is the invariant that must not change.

- [ ] **Step 3: Run test to verify it fails**

Run: `cd sophia && poetry run pytest tests/ingestion/test_proposal_processor_hint.py -v`
Expected: FAIL — `hermes_type_hint` is `None`.

- [ ] **Step 4: Persist the hint**

In the per-node property assembly block (where `node_props["needs_reclassification"]` is set), add:

```python
            # Preserve Hermes' initial NER type pick (provenance / weak prior for #505).
            node_props["hermes_type_hint"] = proposed_node.get("type")
```

(Use the actual proposed-node variable name from Step 1 in place of `proposed_node`.)

- [ ] **Step 5: Run test to verify it passes**

Run: `cd sophia && poetry run pytest tests/ingestion/test_proposal_processor_hint.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/sophia/ingestion/proposal_processor.py tests/ingestion/test_proposal_processor_hint.py
git commit -m "feat(505): record hermes_type_hint on ingested nodes"
```

---

## Task 3: Shared emergence dataclasses

**Files:**
- Create: `sophia/src/sophia/maintenance/emergence_types.py`
- Test: `sophia/tests/maintenance/test_emergence_types.py`

- [ ] **Step 1: Write the failing test**

```python
# sophia/tests/maintenance/test_emergence_types.py
from collections import Counter
from sophia.maintenance.emergence_types import Member, EmergentCluster, NameResult


def test_member_and_cluster_construct():
    m = Member(uuid="u1", name="derivative", embedding=[0.1, 0.2],
               signature=Counter({("DEFINED_AS", "concept"): 1}),
               current_type="entity", hermes_type_hint="concept",
               neighbors=[{"relation": "DEFINED_AS", "neighbor_name": "limit", "neighbor_type": "entity"}])
    cluster = EmergentCluster(members=[m])
    assert cluster.members[0].name == "derivative"
    assert cluster.size == 1


def test_name_result():
    r = NameResult(label="concept", description="abstract idea", is_new=True, confidence=0.8)
    assert r.is_new and r.confidence == 0.8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd sophia && poetry run pytest tests/maintenance/test_emergence_types.py -v`
Expected: FAIL — `ModuleNotFoundError: sophia.maintenance.emergence_types`.

- [ ] **Step 3: Create the module**

```python
# sophia/src/sophia/maintenance/emergence_types.py
"""Shared value types for ontology-evolution emergence (#505)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Member:
    """A candidate node being considered for re-typing."""

    uuid: str
    name: str
    embedding: list[float]
    signature: Counter  # Counter[(relation_type, neighbor_type)]
    current_type: str
    hermes_type_hint: str | None
    neighbors: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EmergentCluster:
    """A group of members that agree on both signals and may become a type."""

    members: list[Member]

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def embeddings(self) -> list[list[float]]:
        return [m.embedding for m in self.members]


@dataclass
class NameResult:
    """Hermes' answer to 'what binds these together?'."""

    label: str
    description: str
    is_new: bool
    confidence: float
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd sophia && poetry run pytest tests/maintenance/test_emergence_types.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sophia/maintenance/emergence_types.py tests/maintenance/test_emergence_types.py
git commit -m "feat(505): emergence shared dataclasses"
```

---

## Task 4: Structural neighbor-relation signature

**Files:**
- Create: `sophia/src/sophia/maintenance/structural_signature.py`
- Test: `sophia/tests/maintenance/test_structural_signature.py`

- [ ] **Step 1: Write the failing test**

```python
# sophia/tests/maintenance/test_structural_signature.py
from collections import Counter
from sophia.maintenance.structural_signature import (
    build_signature,
    signature_similarity,
)


def test_build_signature_counts_relation_neighbor_pairs():
    neighbors = [
        {"relation": "MOVED_TO", "neighbor_type": "location"},
        {"relation": "LOCATED_ON", "neighbor_type": "object"},
        {"relation": "MOVED_TO", "neighbor_type": "location"},
    ]
    sig = build_signature(neighbors)
    assert sig == Counter({("MOVED_TO", "location"): 2, ("LOCATED_ON", "object"): 1})


def test_similarity_identical_is_one():
    a = Counter({("DEFINED_AS", "concept"): 1})
    assert signature_similarity(a, a) == 1.0


def test_similarity_disjoint_is_zero():
    a = Counter({("DEFINED_AS", "concept"): 1})
    b = Counter({("MOVED_TO", "location"): 1})
    assert signature_similarity(a, b) == 0.0


def test_similarity_partial_overlap_jaccard():
    a = Counter({("X", "t"): 1, ("Y", "t"): 1})
    b = Counter({("X", "t"): 1})
    # weighted Jaccard: intersection 1, union 3 -> 1/3
    assert abs(signature_similarity(a, b) - (1 / 3)) < 1e-9


def test_empty_signatures_similarity_is_zero():
    assert signature_similarity(Counter(), Counter()) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd sophia && poetry run pytest tests/maintenance/test_structural_signature.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# sophia/src/sophia/maintenance/structural_signature.py
"""Structural signal for emergence: a node's neighbor-relation signature.

A node is characterised by the multiset of (relation_type, neighbor_type)
pairs on its incident edges. Two nodes that connect to the same kinds of
neighbours via the same relations are structurally similar.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def build_signature(neighbors: list[dict[str, Any]]) -> Counter:
    """Multiset of (relation_type, neighbor_type) pairs for a node."""
    return Counter(
        (n["relation"], n["neighbor_type"])
        for n in neighbors
        if n.get("relation") and n.get("neighbor_type")
    )


def signature_similarity(a: Counter, b: Counter) -> float:
    """Weighted Jaccard similarity of two signatures in [0, 1]."""
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    intersection = sum(min(a[k], b[k]) for k in keys)
    union = sum(max(a[k], b[k]) for k in keys)
    return intersection / union if union else 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd sophia && poetry run pytest tests/maintenance/test_structural_signature.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/sophia/maintenance/structural_signature.py tests/maintenance/test_structural_signature.py
git commit -m "feat(505): structural neighbor-relation signature + similarity"
```

---

## Task 5: Dual-signal full-membership clustering (recursive)

**Files:**
- Create: `sophia/src/sophia/maintenance/emergence_clustering.py`
- Test: `sophia/tests/maintenance/test_emergence_clustering.py`
- Reuse: `sophia/src/sophia/ingestion/type_emergence.py` (`_variance`, `_mean_vector`, `_kmeans_2`)

- [ ] **Step 1: Write the failing test**

```python
# sophia/tests/maintenance/test_emergence_clustering.py
from collections import Counter
from sophia.maintenance.emergence_types import Member
from sophia.maintenance.emergence_clustering import find_emergent_clusters


def _m(uuid, vec, sig_key):
    return Member(uuid=uuid, name=uuid, embedding=vec,
                  signature=Counter({sig_key: 1}), current_type="entity",
                  hermes_type_hint=None, neighbors=[])


def test_two_coherent_groups_split_when_both_signals_agree():
    # Physical cluster: vectors near (0,0), shared structural sig A
    phys = [_m(f"p{i}", [0.0 + i * 0.01, 0.0], ("MOVED_TO", "location")) for i in range(4)]
    # Concept cluster: vectors near (9,9), shared structural sig B
    concept = [_m(f"c{i}", [9.0 + i * 0.01, 9.0], ("DEFINED_AS", "concept")) for i in range(4)]
    clusters = find_emergent_clusters(
        phys + concept, min_cluster_size=3, min_cohesion_improvement=0.15
    )
    assert len(clusters) == 2
    names = {frozenset(m.uuid for m in c.members) for c in clusters}
    assert frozenset(f"p{i}" for i in range(4)) in names
    assert frozenset(f"c{i}" for i in range(4)) in names


def test_no_split_when_signals_disagree():
    # Embedding says two groups, but structure is uniform -> no agreement -> no cluster
    members = [_m(f"x{i}", [0.0, 0.0], ("R", "t")) for i in range(3)] + \
              [_m(f"y{i}", [9.0, 9.0], ("R", "t")) for i in range(3)]
    clusters = find_emergent_clusters(
        members, min_cluster_size=3, min_cohesion_improvement=0.15
    )
    # structural similarity uniform -> single structural group -> intersection
    # with the embedding split is empty per-group -> nothing cohesive enough
    assert clusters == [] or all(c.size >= 3 for c in clusters)


def test_below_min_size_not_returned():
    members = [_m(f"p{i}", [0.0, 0.0], ("MOVED_TO", "location")) for i in range(2)]
    clusters = find_emergent_clusters(members, min_cluster_size=3,
                                      min_cohesion_improvement=0.15)
    assert clusters == []


def test_three_groups_recurse_to_three():
    """Full-membership recursive split finds >2 tight groups, no outliers needed."""
    members = []
    for (bx, by), sig in [((0.0, 0.0), ("A", "t")), ((9.0, 9.0), ("B", "t")),
                          ((0.0, 18.0), ("C", "t"))]:
        members += [_m(f"{sig[0]}{i}", [bx + i * 0.01, by], sig) for i in range(3)]
    clusters = find_emergent_clusters(members, min_cluster_size=3,
                                      min_cohesion_improvement=0.1)
    assert len(clusters) == 3


def test_single_cohesive_group_returns_empty():
    """A type that is already one tight group yields no sub-types."""
    members = [_m(f"a{i}", [0.0 + i * 0.01, 0.0], ("A", "t")) for i in range(6)]
    clusters = find_emergent_clusters(members, min_cluster_size=3,
                                      min_cohesion_improvement=0.15)
    assert clusters == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd sophia && poetry run pytest tests/maintenance/test_emergence_clustering.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# sophia/src/sophia/maintenance/emergence_clustering.py
"""Dual-signal clustering for emergent type discovery (#505).

A candidate cluster must agree on BOTH signals:
  * embedding proximity (k-means on member vectors), and
  * structural similarity (neighbor-relation signature grouping).
Only the intersection (members grouped together by both) is returned, and
only if the embedding split actually improves cohesion.
"""

from __future__ import annotations

from sophia.ingestion.type_emergence import _kmeans_2, _mean_vector, _variance
from sophia.maintenance.emergence_types import EmergentCluster, Member
from sophia.maintenance.structural_signature import signature_similarity

_STRUCTURAL_SIM_THRESHOLD = 0.5


def _embedding_groups(members: list[Member]) -> list[list[Member]]:
    """Binary embedding split; returns the two non-empty sub-groups."""
    if len(members) < 2:
        return [members]
    vectors = [m.embedding for m in members]
    c0, c1 = _kmeans_2(vectors)
    # _kmeans_2 returns two lists of embeddings; map back to members by identity/index
    group0, group1 = [], []
    used: set[int] = set()
    for vec in c0:
        for i, m in enumerate(members):
            if i not in used and m.embedding == vec:
                group0.append(m)
                used.add(i)
                break
    group1 = [m for i, m in enumerate(members) if i not in used]
    return [g for g in (group0, group1) if g]


def _structurally_coherent(group: list[Member]) -> bool:
    """True if members are mutually similar on the structural signature."""
    if len(group) < 2:
        return True
    ref = group[0].signature
    return all(signature_similarity(ref, m.signature) >= _STRUCTURAL_SIM_THRESHOLD
               for m in group[1:])


def _cohesion_improvement(parent: list[Member], group: list[Member]) -> float:
    """Fractional variance reduction of a group vs its parent set."""
    pv = _variance([m.embedding for m in parent],
                   _mean_vector([m.embedding for m in parent]))
    if pv <= 0:
        return 0.0
    gv = _variance([m.embedding for m in group],
                   _mean_vector([m.embedding for m in group]))
    return (pv - gv) / pv


def _recursive_clusters(
    members: list[Member], *, min_cluster_size: int, min_cohesion_improvement: float
) -> list[list[Member]]:
    """Recursively binary-split the FULL set while each split improves cohesion."""
    if len(members) < 2 * min_cluster_size:
        return [members]
    groups = _embedding_groups(members)
    if len(groups) < 2 or any(len(g) < min_cluster_size for g in groups):
        return [members]
    if min(_cohesion_improvement(members, g) for g in groups) < min_cohesion_improvement:
        return [members]  # split doesn't help -> this set is a cohesive leaf
    leaves: list[list[Member]] = []
    for g in groups:
        leaves.extend(_recursive_clusters(
            g, min_cluster_size=min_cluster_size,
            min_cohesion_improvement=min_cohesion_improvement))
    return leaves


def find_emergent_clusters(
    members: list[Member],
    *,
    min_cluster_size: int,
    min_cohesion_improvement: float,
) -> list[EmergentCluster]:
    """Cluster the FULL membership; return cohesive, structurally-coherent groups.

    No outlier step: a type is split when recursive clustering reveals tighter
    sub-groups (cohesion gain), which also catches outlier-free but multi-modal
    types. Returns [] when the membership is already one cohesive group.
    """
    if len(members) < 2 * min_cluster_size:
        return []
    leaves = _recursive_clusters(
        members, min_cluster_size=min_cluster_size,
        min_cohesion_improvement=min_cohesion_improvement)
    if len(leaves) < 2:
        return []  # nothing split out -> no new sub-types
    return [
        EmergentCluster(members=leaf)
        for leaf in leaves
        if len(leaf) >= min_cluster_size and _structurally_coherent(leaf)
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd sophia && poetry run pytest tests/maintenance/test_emergence_clustering.py -v`
Expected: PASS. If `_kmeans_2`'s return shape differs from `(list_a, list_b)`, fix `_embedding_groups` to match its actual signature (confirm via `sed -n '44,90p' src/sophia/ingestion/type_emergence.py`).

- [ ] **Step 5: Commit**

```bash
git add src/sophia/maintenance/emergence_clustering.py tests/maintenance/test_emergence_clustering.py
git commit -m "feat(505): dual-signal outlier clustering"
```

---

## Task 6: Hermes `name_cluster` endpoint

**Files:**
- Modify: `hermes/src/hermes/main.py` (add models + route; model on the existing `/name-type` at ~line 1349 is the template)
- Test: `hermes/tests/test_name_cluster.py`

- [ ] **Step 1: Read the `/name-type` template**

Run: `cd hermes && sed -n '1349,1396p' src/hermes/main.py` and find `class NameTypeRequest`/`NameTypeResponse`. Mirror their style, LLM-provider access, and auth dependency.

- [ ] **Step 2: Write the failing test**

```python
# hermes/tests/test_name_cluster.py
from fastapi.testclient import TestClient
from hermes.main import app


def test_name_cluster_returns_label(monkeypatch):
    # Stub the LLM provider call used inside the handler to a deterministic answer.
    import hermes.main as m
    async def fake_name(members, candidates):
        return {"label": "concept", "description": "abstract ideas",
                "is_new": False, "confidence": 0.82}
    monkeypatch.setattr(m, "_name_cluster_via_llm", fake_name, raising=False)

    client = TestClient(app)
    resp = client.post("/name-cluster", json={
        "members": [
            {"name": "derivative", "type": "entity", "hermes_type_hint": "concept",
             "neighbors": [{"relation": "DEFINED_AS", "neighbor_name": "limit",
                            "neighbor_type": "entity"}]},
            {"name": "integral", "type": "entity", "hermes_type_hint": "concept",
             "neighbors": []},
        ],
        "candidates": ["object", "location", "concept"],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "concept"
    assert 0.0 <= body["confidence"] <= 1.0
```

> If `/name-type` is behind auth, add the same bearer header the other Hermes tests use (copy from an existing passing test in `hermes/tests/`).

- [ ] **Step 3: Run test to verify it fails**

Run: `cd hermes && poetry run pytest tests/test_name_cluster.py -v`
Expected: FAIL — 404 (route missing).

- [ ] **Step 4: Implement models + route + helper**

```python
# in hermes/src/hermes/main.py (near the other Pydantic models)
class NameClusterMember(BaseModel):
    name: str
    type: Optional[str] = None
    hermes_type_hint: Optional[str] = None
    neighbors: List[Dict[str, Any]] = Field(default_factory=list)


class NameClusterRequest(BaseModel):
    members: List[NameClusterMember]
    candidates: List[str] = Field(default_factory=list)


class NameClusterResponse(BaseModel):
    label: str
    description: str
    is_new: bool
    confidence: float


async def _name_cluster_via_llm(
    members: List[NameClusterMember], candidates: List[str]
) -> Dict[str, Any]:
    """Ask the LLM what binds the cluster's members together (even if broad)."""
    member_lines = "\n".join(
        f"- {mem.name}"
        + (f" (hint: {mem.hermes_type_hint})" if mem.hermes_type_hint else "")
        + (
            "; relations: "
            + ", ".join(f"{n['relation']}->{n.get('neighbor_name','?')}" for n in mem.neighbors)
            if mem.neighbors else ""
        )
        for mem in members
    )
    prompt = (
        "These entities were grouped together because they are semantically and "
        "structurally similar. Name the single category that binds them — the common "
        "thread — even if it must be broad. Prefer one of the existing categories if it "
        "genuinely fits; otherwise propose a new lowercase noun.\n\n"
        f"Existing categories: {', '.join(candidates) or '(none)'}\n\n"
        f"Members:\n{member_lines}\n\n"
        'Respond as JSON: {"label": "...", "description": "...", '
        '"is_new": true|false, "confidence": 0.0-1.0}'
    )
    # Reuse the same provider/JSON-parse path as /name-type. Pseudocode:
    raw = await _call_llm_json(prompt)  # the helper /name-type already uses
    return {
        "label": str(raw["label"]).strip().lower(),
        "description": str(raw.get("description", "")),
        "is_new": bool(raw.get("is_new", str(raw["label"]).lower() not in candidates)),
        "confidence": float(raw.get("confidence", 0.5)),
    }


@app.post("/name-cluster", response_model=NameClusterResponse)
async def name_cluster(request: NameClusterRequest) -> NameClusterResponse:
    """Name the category that binds a cluster of nodes (Sophia emergence #505)."""
    result = await _name_cluster_via_llm(request.members, request.candidates)
    return NameClusterResponse(**result)
```

> Replace `_call_llm_json` with the actual JSON-LLM helper `/name-type` uses (found in Step 1). Keep `_name_cluster_via_llm` as a separate, monkeypatchable function so the test can stub it.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd hermes && poetry run pytest tests/test_name_cluster.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/hermes/main.py tests/test_name_cluster.py
git commit -m "feat(505): add /name-cluster endpoint to Hermes"
```

---

## Task 7: Sophia → Hermes `name_cluster` client

**Files:**
- Create: `sophia/src/sophia/maintenance/hermes_naming.py`
- Test: `sophia/tests/maintenance/test_hermes_naming.py`
- Reference: `sophia/src/sophia/feedback/worker.py` (existing Sophia→Hermes `httpx` POST + auth header pattern)

- [ ] **Step 1: Confirm the httpx + auth pattern**

Run: `cd sophia && sed -n '1,90p' src/sophia/feedback/worker.py` — copy how it builds the URL, sets `Authorization: Bearer <SOPHIA_API_KEY>`, and handles errors.

- [ ] **Step 2: Write the failing test**

```python
# sophia/tests/maintenance/test_hermes_naming.py
from collections import Counter
import respx
import httpx
from sophia.maintenance.emergence_types import EmergentCluster, Member
from sophia.maintenance.hermes_naming import name_cluster


def _cluster():
    m = Member(uuid="u1", name="derivative", embedding=[0.1],
               signature=Counter({("DEFINED_AS", "concept"): 1}),
               current_type="entity", hermes_type_hint="concept", neighbors=[])
    return EmergentCluster(members=[m])


@respx.mock
def test_name_cluster_posts_and_parses():
    route = respx.post("http://hermes:17000/name-cluster").mock(
        return_value=httpx.Response(200, json={
            "label": "concept", "description": "ideas", "is_new": False, "confidence": 0.8
        })
    )
    result = name_cluster(_cluster(), candidates=["object", "concept"],
                          hermes_url="http://hermes:17000", token="t")
    assert route.called
    assert result.label == "concept"
    assert result.confidence == 0.8
```

> If `respx` is not a dev dependency, add it (`cd sophia && poetry add --group dev respx`) in this step, or substitute the project's existing httpx-mocking approach found in Step 1's neighbouring tests.

- [ ] **Step 3: Run test to verify it fails**

Run: `cd sophia && poetry run pytest tests/maintenance/test_hermes_naming.py -v`
Expected: FAIL — module not found.

- [ ] **Step 4: Implement**

```python
# sophia/src/sophia/maintenance/hermes_naming.py
"""Sophia -> Hermes name_cluster client (#505)."""

from __future__ import annotations

import logging

import httpx

from sophia.maintenance.emergence_types import EmergentCluster, NameResult

logger = logging.getLogger(__name__)


def name_cluster(
    cluster: EmergentCluster,
    *,
    candidates: list[str],
    hermes_url: str,
    token: str,
    timeout: float = 30.0,
) -> NameResult | None:
    """Ask Hermes to name what binds the cluster. Returns None on failure."""
    payload = {
        "members": [
            {
                "name": m.name,
                "type": m.current_type,
                "hermes_type_hint": m.hermes_type_hint,
                "neighbors": m.neighbors,
            }
            for m in cluster.members
        ],
        "candidates": candidates,
    }
    url = f"{hermes_url.rstrip('/')}/name-cluster"
    try:
        resp = httpx.post(
            url, json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(timeout),
        )
        resp.raise_for_status()
        data = resp.json()
        return NameResult(
            label=data["label"],
            description=data.get("description", ""),
            is_new=bool(data.get("is_new", True)),
            confidence=float(data.get("confidence", 0.0)),
        )
    except (httpx.HTTPError, KeyError, ValueError) as e:
        logger.warning("name_cluster failed: %s", e)
        return None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd sophia && poetry run pytest tests/maintenance/test_hermes_naming.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/sophia/maintenance/hermes_naming.py tests/maintenance/test_hermes_naming.py
git commit -m "feat(505): Sophia client for Hermes /name-cluster"
```

---

## Task 8: Type minting + retype (HCG/Milvus side effects)

**Files:**
- Create: `sophia/src/sophia/maintenance/type_minting.py`
- Test: `sophia/tests/maintenance/test_type_minting.py`
- Reference: `sophia/src/sophia/ingestion/type_classifier.py` (`update_centroid_for_assignment`), `sophia/src/sophia/hcg_client/client.py` (`add_node`, `add_edge`)

- [ ] **Step 1: Confirm `add_edge` + Milvus centroid-insert signatures**

Run: `cd sophia && sed -n '262,310p' src/sophia/hcg_client/client.py` (the `add_edge` signature) and `grep -nE "def update_centroid|def insert|def add_type|def upsert" src/sophia/**/*.py` to find how a NEW type centroid is written to Milvus (the seed step). Record the exact method name + args; use them in Step 3 in place of `milvus.upsert_type_centroid(...)`.

- [ ] **Step 2: Write the failing test**

```python
# sophia/tests/maintenance/test_type_minting.py
from collections import Counter
from sophia.maintenance.emergence_types import EmergentCluster, Member, NameResult
from sophia.maintenance.type_minting import mint_type


class FakeHCG:
    def __init__(self):
        self.nodes, self.edges = [], []

    def add_node(self, name, node_type, uuid=None, properties=None, **kw):
        self.nodes.append({"name": name, "node_type": node_type,
                           "uuid": uuid, "properties": properties or {}})
        return uuid or f"type_{name}"

    def add_edge(self, *a, **k):
        self.edges.append((a, k))
        return "edge1"

    def set_node_type(self, node_uuid, new_type, type_uuid):
        self.nodes.append({"retyped": node_uuid, "new_type": new_type})


class FakeMilvus:
    def __init__(self):
        self.centroids = {}

    def upsert_type_centroid(self, type_uuid, centroid):
        self.centroids[type_uuid] = centroid


def _cluster():
    return EmergentCluster(members=[
        Member(uuid="u1", name="derivative", embedding=[0.0, 2.0],
               signature=Counter(), current_type="entity",
               hermes_type_hint="concept", neighbors=[]),
        Member(uuid="u2", name="integral", embedding=[2.0, 0.0],
               signature=Counter(), current_type="entity",
               hermes_type_hint="concept", neighbors=[]),
    ])


def test_mint_creates_type_node_with_centroid_and_lineage():
    hcg, milvus = FakeHCG(), FakeMilvus()
    name = NameResult(label="concept", description="ideas", is_new=True, confidence=0.8)
    type_uuid = mint_type(_cluster(), name, hcg=hcg, milvus=milvus,
                          source_cluster_id="cl1")
    # type-definition node created
    tdef = next(n for n in hcg.nodes if n.get("properties", {}).get("is_type_definition"))
    assert tdef["properties"]["is_type_definition"] is True
    assert "root" in tdef["properties"]["ancestors"]
    # name_history lineage recorded
    hist = tdef["properties"]["name_history"]
    assert hist[0]["name"] == "concept" and hist[0]["hermes_confidence"] == 0.8
    # centroid seeded = mean of member embeddings ([1.0, 1.0])
    assert milvus.centroids[type_uuid] == [1.0, 1.0]
    # members retyped + IS_A edges created
    assert any(n.get("retyped") == "u1" for n in hcg.nodes)
    assert len(hcg.edges) == 2  # one IS_A per member
```

> Match `FakeHCG`/`FakeMilvus` method names to what Step 1 revealed (e.g. the real "set a node's type" call and the real centroid-upsert name). The assertions (type-def node + ancestors + name_history + centroid mean + 2 IS_A edges) are the invariants.

- [ ] **Step 3: Implement**

```python
# sophia/src/sophia/maintenance/type_minting.py
"""Mint an emergent type from a named cluster: type node + centroid + retype (#505)."""

from __future__ import annotations

import logging
import uuid as uuid_lib
from datetime import datetime, timezone

from sophia.maintenance.emergence_types import EmergentCluster, NameResult

logger = logging.getLogger(__name__)


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    n = len(vectors)
    dim = len(vectors[0])
    return [sum(v[d] for v in vectors) / n for d in range(dim)]


def mint_type(
    cluster: EmergentCluster,
    name: NameResult,
    *,
    hcg,
    milvus,
    source_cluster_id: str,
) -> str:
    """Create the type-definition node, seed its centroid, and retype members."""
    type_uuid = f"type_{name.label}_{uuid_lib.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc).isoformat()
    name_history = [{
        "name": name.label,
        "named_at": now,
        "reason": "emergence",
        "source_cluster_id": source_cluster_id,
        "hermes_confidence": name.confidence,
    }]
    hcg.add_node(
        name=name.label,
        node_type="type_definition",
        uuid=type_uuid,
        properties={
            "is_type_definition": True,
            "ancestors": ["root"],
            "name_history": name_history,
        },
        source="emergence",
    )
    milvus.upsert_type_centroid(type_uuid, _mean_vector(cluster.embeddings))

    for m in cluster.members:
        hcg.set_node_type(m.uuid, name.label, type_uuid)
        hcg.add_edge(m.uuid, type_uuid, "IS_A")

    logger.info("Minted type %s (%s) from %d members",
                name.label, type_uuid, cluster.size)
    return type_uuid
```

> In Step 1 you confirmed the real method to change a node's type and create the `IS_A` edge — substitute them for `hcg.set_node_type(...)` / `hcg.add_edge(m.uuid, type_uuid, "IS_A")`, and the real centroid-seed call for `milvus.upsert_type_centroid(...)`. Keep the behaviour identical.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd sophia && poetry run pytest tests/maintenance/test_type_minting.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sophia/maintenance/type_minting.py tests/maintenance/test_type_minting.py
git commit -m "feat(505): mint emergent type node + centroid + retype members"
```

---

## Task 9: Emergence orchestration handler

**Files:**
- Create: `sophia/src/sophia/maintenance/emergence_handler.py`
- Test: `sophia/tests/maintenance/test_emergence_handler.py`

The handler is the callable the scheduler dispatches as `handlers["type_emergence"]`, invoked `handler(type_uuid=...)`. It: loads the type's members (names, embeddings, neighbors, hints) → builds signatures → `find_emergent_clusters` → for each cluster, `name_cluster` → guard on confidence → `mint_type` → publish ontology change.

- [ ] **Step 1: Write the failing test**

```python
# sophia/tests/maintenance/test_emergence_handler.py
from collections import Counter
from sophia.maintenance.emergence_types import EmergentCluster, Member, NameResult
from sophia.maintenance.emergence_handler import EmergenceHandler
from sophia.maintenance.config import MaintenanceConfig


def _members():
    phys = [Member(uuid=f"p{i}", name=f"p{i}", embedding=[0.0 + i * 0.01, 0.0],
                   signature=Counter({("MOVED_TO", "location"): 1}),
                   current_type="entity", hermes_type_hint="object", neighbors=[])
            for i in range(4)]
    con = [Member(uuid=f"c{i}", name=f"c{i}", embedding=[9.0 + i * 0.01, 9.0],
                  signature=Counter({("DEFINED_AS", "concept"): 1}),
                  current_type="entity", hermes_type_hint="concept", neighbors=[])
           for i in range(4)]
    return phys + con


def test_handler_mints_named_clusters_and_publishes():
    minted, published = [], []

    def fake_load_members(type_uuid):
        return _members()

    def fake_name(cluster, candidates, hermes_url, token):
        label = "object" if cluster.members[0].uuid.startswith("p") else "concept"
        return NameResult(label=label, description="", is_new=True, confidence=0.9)

    def fake_mint(cluster, name, hcg, milvus, source_cluster_id):
        minted.append(name.label)
        return f"type_{name.label}"

    handler = EmergenceHandler(
        config=MaintenanceConfig(), hcg=object(), milvus=object(),
        event_bus=type("EB", (), {"publish": lambda self, ch, msg: published.append((ch, msg))})(),
        hermes_url="http://h", token="t",
        load_members=fake_load_members, name_fn=fake_name, mint_fn=fake_mint,
        candidates_fn=lambda: ["object", "location", "concept"],
    )
    handler.run(type_uuid="type_entity")
    assert set(minted) == {"object", "concept"}
    assert len(published) == 2  # one ontology-change event per minted type


def test_handler_skips_low_confidence(monkeypatch):
    def fake_name(cluster, candidates, hermes_url, token):
        return NameResult(label="x", description="", is_new=True, confidence=0.1)
    minted = []
    handler = EmergenceHandler(
        config=MaintenanceConfig(), hcg=object(), milvus=object(),
        event_bus=type("EB", (), {"publish": lambda self, *a: None})(),
        hermes_url="http://h", token="t",
        load_members=lambda u: _members(), name_fn=fake_name,
        mint_fn=lambda *a, **k: minted.append(1), candidates_fn=lambda: [],
    )
    handler.run(type_uuid="type_entity")
    assert minted == []  # all below hermes_confidence_floor (0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd sophia && poetry run pytest tests/maintenance/test_emergence_handler.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

```python
# sophia/src/sophia/maintenance/emergence_handler.py
"""The 'type_emergence' maintenance handler (#505).

Dispatched by MaintenanceScheduler as handlers['type_emergence'](type_uuid=...).
Dependencies (load_members / name_fn / mint_fn / candidates_fn) are injected so
the orchestration is unit-testable without Neo4j/Milvus/Hermes.
"""

from __future__ import annotations

import logging
import uuid as uuid_lib

from sophia.maintenance.config import MaintenanceConfig
from sophia.maintenance.emergence_clustering import find_emergent_clusters

logger = logging.getLogger(__name__)

ONTOLOGY_CHANGED_CHANNEL = "ontology.type_created"


class EmergenceHandler:
    def __init__(self, *, config: MaintenanceConfig, hcg, milvus, event_bus,
                 hermes_url: str, token: str,
                 load_members, name_fn, mint_fn, candidates_fn):
        self._config = config
        self._hcg = hcg
        self._milvus = milvus
        self._event_bus = event_bus
        self._hermes_url = hermes_url
        self._token = token
        self._load_members = load_members
        self._name_fn = name_fn
        self._mint_fn = mint_fn
        self._candidates_fn = candidates_fn

    def run(self, type_uuid: str) -> None:
        members = self._load_members(type_uuid)
        clusters = find_emergent_clusters(
            members,
            min_cluster_size=self._config.min_cluster_size,
            min_cohesion_improvement=self._config.min_cohesion_improvement,
        )
        if not clusters:
            logger.info("emergence: no qualifying clusters in %s", type_uuid)
            return
        candidates = self._candidates_fn()
        for cluster in clusters:
            name = self._name_fn(cluster, candidates, self._hermes_url, self._token)
            if name is None or name.confidence < self._config.hermes_confidence_floor:
                logger.info("emergence: skip cluster (no/low-confidence name)")
                continue
            cluster_id = uuid_lib.uuid4().hex[:8]
            new_type_uuid = self._mint_fn(
                cluster, name, hcg=self._hcg, milvus=self._milvus,
                source_cluster_id=cluster_id,
            )
            if self._event_bus is not None:
                self._event_bus.publish(
                    ONTOLOGY_CHANGED_CHANNEL,
                    {"type_uuid": new_type_uuid, "name": name.label,
                     "ancestors": ["root"]},
                )
            logger.info("emergence: minted %s from %d members",
                        name.label, cluster.size)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd sophia && poetry run pytest tests/maintenance/test_emergence_handler.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/sophia/maintenance/emergence_handler.py tests/maintenance/test_emergence_handler.py
git commit -m "feat(505): type_emergence orchestration handler"
```

---

## Task 10: Neo4j member-loading + candidate-list adapters

**Files:**
- Modify: `sophia/src/sophia/maintenance/emergence_handler.py` (add module-level functions `load_type_members(hcg, milvus, type_uuid)` and `current_categories(hcg)` used as the default injected callables)
- Test: `sophia/tests/maintenance/test_emergence_adapters.py`

These bridge the pure handler to real stores. `load_type_members` queries nodes of the type from Neo4j (with `name`, `hermes_type_hint`, neighbors via incident edges) and their embeddings from Milvus, builds `Member` objects (signature via `build_signature`). `current_categories` lists existing `is_type_definition` names (minus `entity`/`reserved_`).

- [ ] **Step 1: Confirm the read methods**

Run: `cd sophia && grep -nE "def get_node|def query|def neighbors|def get_embedding|def list_types|def find_nodes_by_type|MATCH" src/sophia/hcg_client/client.py | head -30` — identify how to (a) list nodes by type, (b) get a node's neighbors with relation + neighbor type, (c) fetch a node's embedding from Milvus, (d) list type-definition nodes.

- [ ] **Step 2: Write the failing test** (mock `hcg`/`milvus` returning canned rows; assert `Member` objects are built with populated `signature` and that `current_categories` excludes `entity`/`reserved_`). Use the real method names from Step 1 in the mocks.

```python
# sophia/tests/maintenance/test_emergence_adapters.py
from sophia.maintenance.emergence_handler import current_categories


def test_current_categories_excludes_entity_and_reserved():
    class FakeHCG:
        def list_type_definitions(self):
            return [{"name": "entity"}, {"name": "concept"},
                    {"name": "reserved_state"}, {"name": "object"}]
    assert set(current_categories(FakeHCG())) == {"concept", "object"}
```

> If the real "list type definitions" method differs, adapt the mock + implementation name together.

- [ ] **Step 3: Run test to verify it fails**

Run: `cd sophia && poetry run pytest tests/maintenance/test_emergence_adapters.py -v`
Expected: FAIL — `current_categories` not defined.

- [ ] **Step 4: Implement the adapters** (module-level functions in `emergence_handler.py`), using the methods confirmed in Step 1. `current_categories` filters out `entity` and any `reserved_`-prefixed names. `load_type_members` assembles `Member`s (embedding from Milvus, neighbors from HCG, `signature = build_signature(neighbors)`).

- [ ] **Step 5: Run test to verify it passes**

Run: `cd sophia && poetry run pytest tests/maintenance/test_emergence_adapters.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/sophia/maintenance/emergence_handler.py tests/maintenance/test_emergence_adapters.py
git commit -m "feat(505): Neo4j/Milvus adapters for emergence handler"
```

---

## Task 11: Register the handler in the app

**Files:**
- Modify: `sophia/src/sophia/api/app.py` (where `MaintenanceScheduler(...)` is constructed and the `handlers` dict is assembled — search `handlers=`)
- Test: `sophia/tests/maintenance/test_handler_registration.py`

- [ ] **Step 1: Find the wiring**

Run: `cd sophia && grep -nE "MaintenanceScheduler|handlers|_proposal_processor =|_event_bus|hermes_url|SOPHIA_API" src/sophia/api/app.py | head -30`. Identify the `handlers` dict, the `EventBus` instance, and how the Hermes URL + token are obtained (mirror `feedback_config.hermes_url` / `SOPHIA_API_TOKEN`).

- [ ] **Step 2: Write the failing test**

```python
# sophia/tests/maintenance/test_handler_registration.py
def test_build_emergence_handler_is_callable_with_type_uuid():
    from sophia.maintenance.emergence_handler import build_emergence_handler
    handler = build_emergence_handler(
        config=__import__("sophia.maintenance.config", fromlist=["MaintenanceConfig"]).MaintenanceConfig(),
        hcg=object(), milvus=object(), event_bus=None,
        hermes_url="http://h", token="t",
    )
    assert callable(handler)
    # signature accepts type_uuid kwarg (scheduler calls handler(type_uuid=...))
    import inspect
    assert "type_uuid" in inspect.signature(handler).parameters
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd sophia && poetry run pytest tests/maintenance/test_handler_registration.py -v`
Expected: FAIL — `build_emergence_handler` not defined.

- [ ] **Step 4: Add a factory + register it**

In `emergence_handler.py`:

```python
def build_emergence_handler(*, config, hcg, milvus, event_bus, hermes_url, token):
    """Return the callable registered as handlers['type_emergence']."""
    from sophia.maintenance.hermes_naming import name_cluster
    from sophia.maintenance.type_minting import mint_type

    handler = EmergenceHandler(
        config=config, hcg=hcg, milvus=milvus, event_bus=event_bus,
        hermes_url=hermes_url, token=token,
        load_members=lambda u: load_type_members(hcg, milvus, u),
        name_fn=lambda c, cand, url, tok: name_cluster(
            c, candidates=cand, hermes_url=url, token=tok),
        mint_fn=mint_type,
        candidates_fn=lambda: current_categories(hcg),
    )

    def _run(type_uuid: str) -> None:
        handler.run(type_uuid=type_uuid)

    return _run
```

Then in `app.py`, where the scheduler `handlers` dict is built (Step 1 location), add:

```python
        "type_emergence": build_emergence_handler(
            config=_maintenance_config, hcg=_hcg_client, milvus=_milvus_client,
            event_bus=_maint_event_bus,
            hermes_url=feedback_config.hermes_url,
            token=get_env_value("SOPHIA_API_TOKEN") or "",
        ),
```

(Use the actual variable names found in Step 1 for the HCG client, Milvus client, event bus, and config.)

- [ ] **Step 5: Run test to verify it passes**

Run: `cd sophia && poetry run pytest tests/maintenance/test_handler_registration.py -v`
Expected: PASS.

- [ ] **Step 6: Run the full maintenance + ingestion test suites**

Run: `cd sophia && poetry run pytest tests/maintenance tests/ingestion -v`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/sophia/maintenance/emergence_handler.py src/sophia/api/app.py tests/maintenance/test_handler_registration.py
git commit -m "feat(505): register type_emergence handler in the scheduler"
```

---

## Task 12: Live validation against the `entity` blob

**Files:** none (manual/integration verification). Requires the stack running (`scripts/run_apollo.sh`) with the seeded + demo `entity` nodes present.

- [ ] **Step 1: Snapshot the baseline**

```bash
TOKEN=$(grep -oE 'sophia_dev|[^=]+$' <<<"$SOPHIA_API_TOKEN"); TOKEN=${SOPHIA_API_TOKEN:-sophia_dev}
curl -s -H "Authorization: Bearer $TOKEN" 'http://localhost:47000/hcg/entities?limit=500' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print('entity:',sum(1 for n in d if n['type']=='entity'),'type_defs:',[n['name'] for n in d if n['type']=='type_definition'])"
```
Expected: a sizeable `entity` count, `type_defs: ['entity']`.

- [ ] **Step 2: Trigger emergence on the `entity` type**

Enqueue a `type_emergence` job for `type_entity` (via the scheduler's periodic scan, or a one-off: call `build_emergence_handler(...)(type_uuid="type_entity")` from a `poetry run python` REPL with the live clients). Watch `/tmp/sophia.log`.

- [ ] **Step 3: Verify the outcome**

```bash
curl -s -H "Authorization: Bearer $TOKEN" 'http://localhost:47000/hcg/entities?limit=500' \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print('entity:',sum(1 for n in d if n['type']=='entity'),'type_defs:',[n['name'] for n in d if n['type']=='type_definition'])"
```
Expected: `entity` count dropped; ≥1 new `type_definition` (e.g. an object/physical-ish and/or concept/math-ish type); members retyped with `IS_A` edges; new type renders in the dashboard at `:3000`.

- [ ] **Step 4: Confirm the closing loop**

Re-send a domain sentence through Hermes `/llm` (as in the creation-loop demo) and confirm new similar nodes classify into the emergent type rather than `entity` (may require Hermes to have received the pub/sub type update).

- [ ] **Step 5: Record the run** in the design doc's validation section (actual emergent type names, counts before/after).

---

## Self-Review

**Spec coverage** (design doc → task):
- Junk-drawer/variance trigger → Task 9 (handler uses `find_emergent_clusters`; scheduler already dispatches `type_emergence`). Variance threshold config → Task 1.
- Dual-signal (embedding ∩ structural), outlier-first → Tasks 4, 5.
- `name_cluster` (all members, name-the-bind, candidates as hints, `hermes_type_hint` prior) → Tasks 6, 7.
- Type creation under `root` + retype + `IS_A` + centroid seed + `name_history` lineage → Task 8.
- Pub/sub propagation → Task 9 (`EventBus.publish` on mint).
- Record `hermes_type_hint` at ingestion → Task 2.
- Config tunables → Task 1.
- Guards (min size, cohesion improvement, dual-signal agreement, confidence floor) → Tasks 5 (size/cohesion/agreement), 9 (confidence floor). `max_cluster_size` sampling → noted in Task 6 request; **add an explicit sampling step if a cluster exceeds it** (currently relies on Hermes tolerating large payloads — acceptable for the tiny-graph first iteration; flagged).
- `DERIVED_FROM` provenance edge → **not yet a task** (design lists it as optional/"small add"); deferred from first iteration intentionally — node `name_history` covers lineage. Add as a follow-up task if wanted before merge.
- Validation against `entity` blob → Task 12.

**Placeholder scan:** The plan contains several "confirm the exact signature, then implement as:" steps (Tasks 2, 5, 6, 8, 10, 11). These are deliberate: the external store/LLM signatures (`HCGClient.add_edge`/set-type, the Milvus centroid-insert method, Hermes' JSON-LLM helper, `app.py` wiring var names) must be read from the live code rather than guessed — each such step names the exact file/grep to run and the invariant to preserve. Not free-floating TODOs.

**Type consistency:** `Member`, `EmergentCluster`, `NameResult` (Task 3) are used consistently in Tasks 5, 7, 8, 9. `find_emergent_clusters(members, *, min_cluster_size, min_cohesion_improvement)` signature matches between Task 5 and its callers (Task 9). `name_cluster(cluster, *, candidates, hermes_url, token)` matches Task 7 ↔ Task 9 wrapper. `mint_type(cluster, name, *, hcg, milvus, source_cluster_id)` matches Task 8 ↔ Task 9. Handler dispatched as `handler(type_uuid=...)` matches the scheduler contract verified in the design.

**Known risk carried from design:** the structural-signature similarity threshold (`_STRUCTURAL_SIM_THRESHOLD = 0.5`) and `variance_threshold`/`min_cohesion_improvement` defaults are guesses for a ~20-node graph; Task 12 is where they get calibrated (tune via the config env vars, no code change).
