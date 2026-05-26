# Emergent Type Discovery (logos #505) — Design

**Date:** 2026-05-26
**Issue:** logos #505 — Ontology Evolution — Emergent Type Discovery (epic #499)
**Status:** design — pending review
**Components:** sophia (clustering, type creation, scheduler handler), hermes (`name_cluster` endpoint)

---

## Context & motivation

Sophia's KG **creation** arm works end-to-end today (verified 2026-05-26): text → Hermes NER/relation extraction → `POST /ingest/hermes_proposal` → `ProposalProcessor` writes `:Node`s + reified edges to Neo4j. But every ingested node lands as `type="entity"` with `needs_reclassification=True`.

The reason is a cold-start: `TypeClassifier` assigns types by nearest **type centroid** in Milvus, but the only type-definition node that exists is `entity` (`type_entity`). The 14 authored `ONTOLOGY_TYPES` in Hermes (`object, location, agent, process, action, concept, state, data, workspace, zone, goal, plan, capability`) have **no centroids** in Sophia, so every node falls back to `entity`. A live demo grew the graph to 22 `entity` nodes spanning two obviously-distinct domains (robot/physical-scene terms and calculus concepts) — a textbook junk drawer.

**#505 is the mechanism that resolves this:** detect junk-drawer types, cluster their members, have Hermes name what binds each cluster, and mint those as real types — so the ontology grows from the data and future ingestions classify correctly. This design covers a **first dual-signal iteration** validated against that live `entity` blob.

Relationship to siblings: **#504 (Type Correction)** is downstream — it corrects individual mistyped nodes against *existing* centroids, which only exist once #505 has created them. #505 is therefore built first. The two share centroid infrastructure.

## Goals / non-goals

**Goals**
- Detect when a type has become a junk drawer (high internal variance).
- Cluster its outlier members using **two agreeing signals** (embedding proximity ∩ structural relationship-pattern similarity).
- Ask Hermes to name the unifying category of each cluster (the bind), reusing an existing category label when one fits, otherwise coining a new (possibly broad) one.
- Mint the type (under `root`, peer of object/location), retype members with `IS_A` rewiring, seed its centroid, and keep a name/lineage history.
- Propagate the new type to Hermes via pub/sub so subsequent ingestions use it.

**Non-goals (this iteration)**
- #504 correction of already-confidently-typed nodes (separate story).
- Pure structural-only or embedding-only discovery — dual-signal is the bar.
- Multi-level hierarchy grouping ("group related types") — that's a later emergence round.
- Treating Hermes' per-node NER type hint as *authoritative* for typing — Sophia owns the vocabulary, and types are subject to clustering. The hint **is** recorded at ingestion (see "Enabling change" below) and used as a weak prior during naming, just not as the assigned type.

## The self-improving loop

1. Hermes extracts; unknowns fall to `entity` (the junk drawer).
2. Sophia ingests → `entity` grows and its variance climbs.
3. Sophia detects the high-variance type and pulls its outliers.
4. Sophia clusters outliers by **embedding ∩ structure**; only groups coherent in both proceed.
5. Sophia sends each candidate cluster's **full membership** to Hermes `name_cluster`; Hermes returns the binding label (existing or new, broad if necessary).
6. Sophia mints the type, retypes members (`type` + `IS_A` + incremental centroid update), records lineage.
7. Sophia publishes the ontology change; Hermes updates its type list.
8. Next similar ingestion classifies into the new type. The loop repeats, refining coarse types into finer ones over time (monotonic refinement — see Guards).

## Components

### 0. Enabling change — record Hermes' initial type recommendation (ingestion)
Hermes' proposal already carries a per-node `type` (its NER pick from the 14-vocab), but `ProposalProcessor` overwrites it with the centroid-derived type and discards the hint. Persist it instead as `hermes_type_hint` on the node (a one-property addition in `ProposalProcessor`), alongside the authoritative centroid `type`. This costs nothing at ingestion and gives emergence (and #504) a recorded prior — e.g. a cluster whose members were mostly hinted `concept` reinforces naming it `concept` — plus a debugging breadcrumb for *what Hermes originally thought*.

### 1. Trigger — junk-drawer detection
A maintenance handler registered on `MaintenanceScheduler` (job_type → handler). For each candidate type it computes cohesion = variance of member embeddings around the type centroid (reuse `type_emergence._variance`). A type is a junk-drawer candidate when `variance > variance_threshold` AND `member_count ≥ min_cluster_size`. `entity` is the first target. Runs on the periodic maintenance sweep and on the scheduler's existing threshold events.

### 2. Dual-signal, outlier-first clustering (scale-aware)
- **Outlier pull:** select members whose distance from the type centroid is anomalous; cluster only those, not the whole type.
- **Embedding signal:** cluster the outlier embeddings (extend the existing `type_emergence` k-means; choose k via cohesion gain / recursive k=2 until sub-clusters are cohesive).
- **Structural signal (net-new):** compute a per-node *neighbor-relation signature* from Neo4j — the multiset of `(relation_type, neighbor_type)` pairs incident to the node (e.g. `{MOVED_TO→location, LOCATED_ON→object}` vs `{DEFINED_AS→concept, MEASURES→concept}`). Group nodes by signature similarity.
- **Intersection:** a candidate cluster is a group of nodes that co-occur in both an embedding cluster and a structural cluster. Only the intersection proceeds to naming. This is the precision bar that prevents spurious types.

### 3. `name_cluster` — Hermes contract (net-new endpoint)
Sophia knows *that* the nodes belong together; Hermes says *what* they are.

- **Request:** all member nodes of the candidate cluster — `[{name, type (current), hermes_type_hint, neighbors: [{relation, neighbor_name, neighbor_type}]}]` — plus the current category list as optional hints. Each member's recorded `hermes_type_hint` travels as a weak prior. (If membership exceeds `max_cluster_size`, send a representative sample; default is all.)
- **Task:** name the single category that binds them, *even if broad*; prefer an existing category label when one genuinely fits; never refuse.
- **Response:** `{label, description, is_new: bool, confidence}`.

Adapt the existing `POST /name-type` into `POST /name-cluster` (or extend it). Because the cluster is a proper subset of the parent type, the returned label is necessarily more specific than the parent (`entity`) — so naming cannot regenerate junk-drawer breadth.

### 4. Type creation, retype, and lineage
- Create a type-definition node: `:Node {uuid, name, is_type_definition: true, ancestors: [root, …]}` directly under `root` (the level at which `object`/`location` sit; at cold-start these are the first real type nodes created). No `reserved_` prefix.
- Seed its Milvus centroid = mean of the cluster members' embeddings.
- Retype each member: set `type`, rewire the `IS_A` edge (`member → new_type`), and incrementally update both centroids (remove from parent, add to new) via the existing `TypeClassifier.update_centroid_for_assignment`.
- **Lineage / name history:** store on the type node `name_history: [{name, named_at, reason, source_cluster_id, hermes_confidence}]` (typed records). When a type splits out of another, add a `DERIVED_FROM` provenance edge between the type nodes so the ontology's evolution is navigable in-graph.

### 5. Propagation
Publish an ontology-changed event via the #501 ontology pub/sub with the new type (uuid, name, ancestors). Hermes consumes it and updates its type list, closing the loop.

### 6. Guards (anti-over-split / anti-thrash)
- `min_cluster_size` — don't mint from a handful of nodes.
- `min_cohesion_improvement` — the split must measurably reduce variance vs the parent, else skip.
- **Dual-signal agreement required** — embedding and structural clusters must overlap.
- `hermes_confidence_floor` — discard low-confidence names.
- **Idempotent** — re-running on an already-cohesive type is a no-op.
- **Monotonic refinement** — a minted type's *member set* is always a proper subset of the junk drawer it was split from, so repeated runs converge (coarse → fine) and cannot regenerate parent-level breadth or oscillate. (This is set containment of members, independent of the `IS_A` hierarchy placement under `root`.)
- **Audit trail** — every mint/retype decision (cluster membership, chosen label, members moved, centroid deltas) is logged for inspection and reversibility, distinct from the per-type `name_history`.

**Cleanliness is maintained over time, not enforced upfront.** Every mint and every (re)typed member assignment carries a confidence score; **periodic reclassification (#504)** re-evaluates assignments against centroids and moves mistyped nodes as the graph grows. So these guards only need to prevent *gross* errors (spurious types, over-splitting, junk-drawer regeneration) — not perfection. An imperfect emergent type or a borderline member is acceptable because the confidence signal plus the #504 sweep will correct it. #505 and #504 are a self-healing pair: #505 grows the ontology (tolerant of being approximately right), #504 keeps it clean.

### 7. Config (tunables)
Add to `MaintenanceConfig` (`sophia/maintenance/config.py`), no hardcoded thresholds:
`variance_threshold`, `min_cluster_size`, `max_cluster_size`, `min_cohesion_improvement`, `hermes_confidence_floor`.

## Reuse vs net-new

| Piece | Status |
|---|---|
| Variance / cohesion (`type_emergence._variance`) | reuse |
| Embedding k-means (`type_emergence._kmeans_2`) | reuse / extend |
| Incremental centroid update (`TypeClassifier.update_centroid_for_assignment`) | reuse |
| Milvus `find_nearest_types` / `update_centroid` | reuse |
| Node/edge writes (`HCGClient.add_node`/`add_edge`) | reuse |
| Scheduler handler seam + threshold events | reuse |
| Ontology pub/sub (#501) | reuse |
| **Structural neighbor-relation signature + similarity** | **net-new** |
| **Embedding ∩ structural intersection** | **net-new** |
| **`/name-cluster` Hermes endpoint** | **net-new (adapt `/name-type`)** |
| Type-definition node creation + `name_history` + `DERIVED_FROM` | net-new |
| Emergence orchestration handler | net-new |
| `MaintenanceConfig` emergence tunables | net-new |

## Validation plan

Run the handler against the live `entity` blob (22 nodes: robot/physical-scene + calculus). Expected: it detects `entity` as a junk drawer, the dual-signal clustering separates the physical-scene cluster from the concept/math cluster, Hermes names each (e.g. an object/physical-ish label and a concept/math-ish label), both are minted under `root` with seeded centroids, members are retyped with `IS_A` edges, `entity`'s count drops, and the new types render in the dashboard. Then re-send a similar sentence and confirm the new nodes classify into the emergent type rather than `entity` (the closing loop). Acceptance mirrors #505's AC.

## Open questions / risks

- **Structural signature definition** is the main algorithmic unknown — exact feature (relation-type multiset vs relation×neighbor-type), similarity metric, and threshold; sensitive on a tiny graph.
- **k selection** for the embedding clusters on small N.
- **Variance threshold calibration** on a ~20-node graph — may need generous defaults initially (hence config).
- Cluster→Hermes membership size vs context window (`max_cluster_size` + representative sampling).

## Dependencies & ordering

- **#501 Ontology Pub/Sub** — required for propagating new types (scaffolding present).
- **#504 Type Correction** — downstream; consumes the centroids #505 creates; shares centroid infra.
- Build order: #505 (this) before #504.
