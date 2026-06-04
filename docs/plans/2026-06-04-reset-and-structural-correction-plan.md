# Reset + Non-Linguistic Type Correction — Plan (2026-06-04)

## Why we're here

The live graph became too chaotic for *any* type-correction signal to bootstrap.
Four signals were tested empirically and all failed:

| signal | result |
|---|---|
| embedding distance | circular (it already placed the node) + noisy: `cue → innate_immune_barrier` |
| raw structural-signature overlap | cosine 0.04–0.15, wrong: `mammal → avian_taxon` |
| SVD/PCA latent factorization | top-128 components capture **5.7%** variance → no low-rank structure |
| dominant (highest-IDF) edge | every node's rarest edge is a `df=1` noise one-off (`DOMINANT_UNTIL`, `HAS_UNDESCRIBED_SPECIES_OF`) |

**Root cause:** the relational structure is noise *at the source* — Hermes over-extracting
idiosyncratic one-off predicates, junk/duplicate type names, fragmented features. There is no
signal to recover from this graph.

**But the design is sound.** The non-linguistic structural corrector was validated end-to-end on
a clean toy: `osteosarcoma` (mistyped `location`, embedding leaning `location` so distance can't
fix it) is corrected to `disease` purely from its wiring (`PRONE_TO`/`TREATED_BY`/`AFFECTS`), and
the ascend-to-cohesion step lands it at the right granularity. So: **reset to a clean baseline with
the improved pipeline, then build the validated corrector.**

## Phase 1 — Land outstanding PRs (in progress)

- **sophia#161** — #505 lineage + #160 rollup + #504 ternary `needs_reclassification`. Membership
  orphaning bug fixed (`type_uuid` must be `classification.type_uuid`, not name-rebuilt) + regression
  test. Review hardening: batch chunking, tie-guard/rollup-defer clarifications.
- **apollo#186** — explorer semantic layout. Review fix: scope the high snapshot limit to the
  explorer (shared hook default back to 200), camera `near:1` for depth precision.
- **hermes#116** — NER value/unit + `/name-cluster` + reasoning-model support. Review fix:
  `reasoning_effort` default `"none"` is invalid → `"low"`, omit on `"none"`.
- Merge once the review bots settle. Remaining red CI is **infra, not code**: the `sync-status`
  workflow 404s on every repo; hermes `Python lint & tests` red is pre-existing flaky JEPA tests
  (weights/env).

## Phase 2 — Reset (wipe + re-ingest)

1. ✅ Stash the abandoned v1 *distance* corrector so a fresh Sophia won't run it.
2. **Stop** Hermes + Sophia (via `run_apollo.sh`). Required: the running services are pre-#160 /
   pre-NER; the restart is the only way the improved pipeline goes live.
3. **Wipe** Neo4j + Milvus. Order matters — stop *before* wipe to avoid the stale-Milvus-handle bug.
4. **Start** Hermes + Sophia on the merged/branch code (improved NER, #160 rollup, ternary flag).
5. **Re-ingest a smallish, focused corpus** (a handful of coherent topics, ~50–100 blocks) for a
   clean, *measurable* baseline — not a huge dump. Scale up only once it looks clean.

## Phase 3 — Evaluate the fresh graph

Measure: relation noise (one-off junk predicates), neighbor-type quality, duplicate/junk type names,
isolated-node %, hierarchy depth. **The bet:** the improved NER cuts the relation-noise that poisoned
every correction signal. If relations + neighbor-types are clean enough, structural correction
becomes viable. **Caveat:** the emergence junk-*naming* gate is still unsolved, so expect some junk
type names regardless.

## Phase 4 — Build the non-linguistic structural corrector (logos#504)

Reuse the v1 scaffolding (queue, budget, batch-settle, scheduler loop, config) but replace `_decide`
with the validated structural design. **No Hermes** — correction is Sophia's non-linguistic job.

- **Detector** — `closer-to-another-centroid` (embedding) only *flags* suspects; it never decides.
- **Corrector** — the node's type-defining `IS_A`/`INSTANCE_OF` **parent** (identity anchor) +
  its `(relation, neighbor-type)` **structural signature** matched against per-type signature
  profiles (reuse `structural_signature.build_signature`; include both edge directions).
- **Confidence-gated granularity** — if the wiring is too confused for a confident fine type,
  **ascend the `IS_A` chain** (the #160 hierarchy) to the single *sufficient* parent where it
  coheres; precision below comes from **subtypes (depth)**, never parallel parents.
- **Multiple parent categories = a trigger** to resolve to single sufficiency (don't arbitrate-and-drop).
- **Homeless nodes** (no usable anchor, nowhere to ascend) — resolve **top-down**: compare to the
  top-level super-types below `entity`/`concept`; **slot** where it fits or **seed a new type** at
  that level. **Never dump at `entity`/`root`** (that's a regression).
- **Representation** — raw signature overlap was fragmented on the messy graph; if the fresh graph is
  still fragmented, build the type×`(relation,neighbor-type)` **matrix**, TF-IDF, and **factorize**
  to latent structural embeddings (cosine in latent space). On a clean graph the direct signature
  match may suffice — *validate first* (this is exactly what the live experiments measured).
- **Governance** — budget-bounded per run (the governor); **confidence informs the response**
  (whether/how precisely to act); the **curiosity budget** (spec §13.3) governs how much correction
  Sophia does (bounded, reward-shaped). `needs_reclassification` is the work queue: `True` = needs
  reclassifying, `False` = nowhere better (only #504 writes it), `None` = a type-cycle Hermes flagged
  for Sophia.
- **Validate on the fresh graph** before enabling unattended runs.

## Phase 5 — Co-evolution (ongoing)

Correction, rollup, and emergence tighten each other: the rollup deepens the chain to climb,
emergence mints the level that's missing, correction places nodes into what exists. The corrector's
effectiveness grows as the hierarchy matures.

## Open risks

- **Emergence junk-naming gate** is the genuinely-unsolved frontier (embedding variance doesn't
  discriminate in 3072-dim; Hermes confidence is not trusted; retain-don't-drop means junk is kept,
  not rejected). The structural corrector needs clean-enough neighbor-types to bite — that's the
  reset's bet.
- Bootstrapping on a near-flat fresh graph (ascend/homeless paths need *some* hierarchy to exist).

## Reference — design memories (vault: 10-projects/LOGOS/sophia/.memory/)

`reclassification-flag-semantics`, `type-correction-needs-structure-not-distance`,
`type-correction-is-non-linguistic-sophia-job`, `correction-ascends-to-cohesion-when-confused`,
`single-sufficient-parent-multi-is-trigger`, `correction-never-dumps-to-entity`,
`homeless-nodes-slot-or-seed-at-top-level`, `structural-correction-needs-latent-matrix-not-raw-overlap`,
`all-correction-signals-fail-on-current-graph`, `structural-corrector-validated-on-clean-toy`,
`confidence-informs-queue-response`, `curiosity-budget-governs-reclamation`,
`favorite-spot-can-be-orphan-connection`, `sophia-reclamation-north-star`,
`never-destroy-information-flag-and-defer`, `no-reliance-on-hermes-confidence`,
`retain-ambiguous-types-not-discard`, `type-cycle-null-is-pragmatic-handoff`.
