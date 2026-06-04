# Surprisal-Driven Typing & Metacognition — Architecture (2026-06-04)

**Status:** design / brainstorm consolidation. Not yet built. Captures a long
co-design session; supersedes the typing portion of
`2026-06-04-reset-and-structural-correction-plan.md`.

**One line:** Stop asking embedding geometry to carry the taxonomy. Type via a
catalog-aware naming pass, organize knowledge with an induction/contradiction
loop, and run one surprisal-driven attention mechanism over the world, the user,
and Sophia's own knowledge structure.

Source notes (vault `10-projects/LOGOS/sophia/.memory/`, all dated 2026-06-04):
naming-driven-typing-engine-brainstorm, typing-reject-overspecified-ceiling,
typing-favor-existing-type-reuse, typing-graft-as-subtype-placement,
typing-catalog-plumbing-feasibility, typing-confers-relevance-principle,
typing-closed-world-available-types, metacognitive-schema-induction-cognition-root,
hypotheses-promote-unifies-all-gates, inductive-reasoning-densifies-graph-engine,
contradiction-detection-refutation-half, curiosity-as-local-attention-field,
curiosity-equals-surprisal-unification, surprisal-extends-to-user-model,
structural-graph-metacognition-self-model, lossless-graph-compression-via-rules,
graph-theory-roles-same-sparsity-wall, cosine-cannot-separate-kind-from-related,
fresh-graph-eval-failure-modes, supergrouping-embedding-wall.

---

## 1. Why this exists — the empirical wall

The fresh re-ingest (102 six-domain sentences) produced 38 emergent types that
were coherent-ish but "a lot wrong." Diagnosing why, three independent tests all
hit the **same ceiling**:

| Method | Same-vs-cross-domain separation |
|---|---|
| Embedding cosine (text-embedding-3-large, 3072-dim) | gap ~+0.13 |
| 1-hop relation signature | gap +0.131 |
| Multi-hop graph roles (Weisfeiler–Leman) | +0.030 → +0.036 (recursion barely helps) |

Findings that drive the whole design:

- **Embeddings encode *association*, not *taxonomy*.** `tusk↔narwhal` = 0.50,
  `bee↔hive` = 0.66, but two same-kind animals from different sentences ≈ 0.28
  (barely above unrelated, 0.20). No cosine threshold separates "same kind" from
  "merely related" — and raising the threshold preferentially selects the
  *associated* (meronymic) pairs, making it worse. A near-SOTA model behaves this
  way *by design*; swapping models won't fix it.
- **The false linkage is meronymic:** clusters bundle a whole with its parts /
  products / habitat (`tusk` with `narwhal`, `pollen`/`nectar`/`hive` with `bee`)
  — and the graph *already extracted* the `PART_OF`/`PRODUCES` edges that say
  these are different kinds. The signal to separate them was present and unused.
- **The bottleneck is sparsity, not method.** ~1.5 edges/entity starves every
  unsupervised method, geometric or topological. Graph-role analysis is the right
  structural typer (0.80 purity in the few rich pockets) but is *deferred* — it
  needs density to work.

**Two levers:** (1) raise edges-per-entity (richer extraction + maturity +
inference), (2) type via the LLM naming pass, which reads the *words* and
sidesteps the thin graph. This doc is mostly lever (2) plus the loop that drives
lever (1) internally.

---

## 2. The architecture in one breath

One surprisal-driven loop, three domains it operates on:

- **world** — the logical graph (what's true)
- **other** — the user model (what the human expects)
- **self** — the structural graph (how Sophia's own knowledge is shaped)

Currency: **confidence / promotion** (evidential, *not* Hermes-reported).
Attention budget: **curiosity = surprisal**.

```
perceive → induce rules (conjecture) → infer & densify
        → spot contradictions (refute) → demote / refine / split
        → compress what's now explained → induce better
   attention routed by surprisal; hypotheses promoted/decayed by evidence
```

---

## 3. The typing engine (naming-driven)

Embeddings do **cheap, coarse candidate clustering** (over-clustering is fine).
Then **one catalog-aware Hermes pass per candidate** does the semantic work
embeddings can't. This is the typing fork resolved toward language — but at ~1
call per *cluster* (~38), not per entity, at the one step where language was
always legitimate (you can't have a non-linguistic *name*).

**Contract:**
```
in:  members       = [{id, name}]
     existing_types = [{type_id, name, chain}]   # K-nearest PUBLISHED types, embedding-retrieved
out: groups        = [{ assign_to: <type_id>|NEW, name, chain:[hypernyms→root], member_ids }]
     residual_ids  = [...]                        # fit no coherent group → junk-drawer
```

Composed rules (each maps to a memory note):

- **Name granularity, floor + ceiling.** Floor: closest hypernym that covers all
  members (`mammal`, not `organism`). Ceiling: reject over-specified names —
  if it takes a phrase / contains a conjunction (`...and related marine
  mammals`), it's reaching; that's also a **split trigger**. The same word-count/
  conjunction rule canonicalizes the noisy relation tail
  (`DIVES_AT_SPEED`→`DIVES`). Free, language-free string check.
- **Chain → hierarchy backbone.** The returned `bear→mammal→animal→…` chain is the
  `IS_A` backbone the corpus never states (only 1 extracted `IS_A` existed).
- **Generality = cohesion gate.** Can't name tightly → don't mint; split or hold.
  Replaces the broken embedding-variance gate.
- **Placement cascade:** reuse existing node → else graft as subtype of an
  existing parent (attach at deepest **available** ancestor) → else new branch at
  a root. Fixes "35 types flat under entity."
- **Three roots:** `entity / concept / process` (currently everything lands flat
  under `entity`; ~1/3 of the 38 are really concepts or processes). A split can
  route members to different roots.
- **Closed-world / available-only:** Hermes references only **published** types;
  `assign_to`/graft-parent must be in the provided set or `NEW`. Validate, reject
  hallucinated targets. Publish-timeliness is what makes this *converge* instead
  of churn duplicates.
- **Split & evict (re-segment):** returning members *with* the name turns evict
  into re-segment (1 group+residual = evict; 2+ = split). Guard: ids only, total
  partition, no hallucinated/lost members.

**Pairs with — does not replace:** the **structural `PART_OF` eviction rule**
(proven: evicted 17 part/product members from 9 of 38 clusters with zero
embeddings — pollination 6→3, etc.). Structure handles edge-grounded cases;
the namer reaches the edge-less embedding-only mixes.

---

## 4. Metacognition: schema induction under a `cognition` root

Sophia mines recurring subgraphs (motifs / frequent-subgraph mining), **names**
the pattern, and **reifies** it as a learned node — under a fourth realm,
`cognition` (self-knowledge), distinct from the three world-model realms.

- A recurring subgraph **shape** is a **process/schema** (`X carries Y → delivered
  to Z` = a transport schema) — populates the `process` root.
- Matching by shape aligns nodes by **functional role** (carrier/cargo/prey), not
  domain — which may be the *more useful* axis for a causal world model.
- Discovered schemas **are** the canonical relations, learned bottom-up
  (`CARRIES`/`HAULS`/`TRANSPORTS` → one `transport` cognition-node they
  `INSTANTIATE`). Canonicalization becomes emergent, not hand-coded.
- Division of labor: **structure discovers → language labels → cognition curates.**
- Recursive: patterns among cognition-nodes = insights about insights.

---

## 5. The epistemic lifecycle (one mechanism for every "is it real?")

Every gate in this doc — cohesion, over-specification, schema-validation,
type-vs-residual — is the **same** mechanism at different levels: a **hypothesis**
that accrues **evidential confidence** and is **promoted/consolidated** or
**decays**.

- Confidence is **evidential** (earned by recurrence + demonstrated utility), NOT
  Hermes self-reported (a standing project constraint).
- "Typing confers relevance": untyped nodes are inert backlog (not noise, not
  deleted) — they hold latent relations that later let them be typed. The catalog
  Hermes reads is the **typed structure only**.
- Hooks exist but partial: `confidence`/`type_confidence` fields,
  `consolidation`/`pruning` maintenance job-types. Schema induction would *ride
  and complete* this lifecycle.

---

## 6. The engine: induction ⇄ compression, one comparison

The rule engine produces a **prediction field**; compare to graph state:

| Rule predicts | Graph has | → |
|---|---|---|
| edge | **absent** | **infer** (densify — the one lever that raises the sparsity ceiling) |
| edge | present, consistent | **redundant** → drop losslessly (keep rule, regenerate on demand) |
| edge | present, **conflicting** | **contradiction** → refute (repair the rule) |
| — | present, **unpredicted** | **surprise** → keep / induce a new rule |

- Induction is **generative** (propagates existing regularities onto instances
  lacking the explicit edge) — unlike clustering, which can only read present
  signal. It densifies by inference and **raises** the ceiling.
- Compression is the dual: the graph converges to **rules + exceptions** (MDL).
  Re-observing a predicted fact doesn't grow the graph, just reinforces the rule's
  confidence. **Graph size tracks information content; local density maps
  understanding** (well-understood regions collapse to a rule + a few exceptions).
- Inferred facts carry `derivation:"inferred"`/`"entailed"` provenance — retractable.
- Distinguish **redundant** (derivable → lossless drop) from **useless** (one-off
  noise → *lossy* forgetting via decay). Don't conflate.

---

## 7. Attention: contradiction + surprise = surprisal = curiosity

- **Curiosity = surprisal** = −log P(observation | current model). Raised by
  **contradiction** (predicted-and-wrong → repair) *and* **surprise**
  (unpredicted → extend). One scalar drives **attention** (high → investigate),
  **retention** (high → keep), and **compression** (low → drop).
- It's a **local field**: a contradiction/surprise raises curiosity on the
  *implicated element(s)*, making the field a priority queue sorted by where the
  model is most wrong. **Homeostatic** — resolution lowers it; attention relaxes.
- A fact's life cycle is its curiosity temperature: **surprising (hot, kept) →
  explained by an induced rule (cools) → redundant (compressible).**
- Bounded by the **curiosity budget** (LOGOS spec §13.3), which rewards/penalizes
  outcomes → Sophia learns *which* surprises are worth her slots.
- Seed already in code: `AMBIGUOUS_SUBSUMPTION` (cycle → unresolved) and
  `needs_reclassification = None` ("has problems") already record a structural
  contradiction without pretending it's resolved.

---

## 8. Other & Self — the same loop, two more domains

- **Other (user model):** the user is another thing Sophia predicts. An **odd
  reaction** is surprisal on her *model of the user* → curiosity → investigate
  (was I wrong about the world, about what they wanted, or did I miscommunicate?).
  Theory of mind / interactive alignment; the existing **feedback system** is the
  channel; weight it heavily (the human is the most authoritative signal). Guard:
  "odd" is relative to an immature user-model; it's a *question*, not a verdict.
- **Self (structural graph):** run the whole loop on the **topology** (vs the
  logical content) for meta-insights about the *shape* of Sophia's knowledge.
  Two sources: (1) patterns in the structure (balanced tree vs hub-and-spokes,
  motifs), (2) **structure-vs-logic discrepancies** (a "type" that's structurally
  a leaf; an "instance" that's structurally a hub). **Not starved by sparsity** —
  topology is fully observable however thin the graph, and the discrepancy gives a
  meta-signal for *every* node, dense where semantic signal was sparse. Proof of
  concept: "35 types flat under `entity`" is a structural pathology (a star, not a
  tree) diagnosable with zero semantics.

---

## 9. Reality checks — what's built vs needs building

- ✅ **Structural `PART_OF` eviction** — prototyped, works (17 evictions / 9
  clusters cleaned).
- ⚠️ **Config bug fixed in this session:** `FeedbackConfig.hermes_url` defaulted to
  `:18000` but Hermes is on `:17000`, silently disabling all emergent-type naming
  (every cluster skipped: "name_cluster failed: Connection refused"). Default must
  be `17000` / `run_apollo.sh` must export `SOPHIA_FEEDBACK_HERMES_URL`.
- ⚠️ **Catalog plumbing partial:** Hermes `TypeRegistry` syncs `logos:ontology:types`
  from Redis (+ HTTP `ontology_client`), but the key holds only `entity` (emergent
  types not published, count stale) and is **flat** (no hierarchy). Needs: publish
  the typed structure; enrich with each type's chain/ancestors. Not new infra.
- ⚠️ **Confidence/promotion** — fields + `consolidation`/`pruning` job-types exist;
  reads as scaffolding, not a finished promotion engine.
- ❌ **Naming engine, induction, contradiction, curiosity field, structural
  metacognition** — design only.

## 10. Open questions / next steps

1. Validate the naming engine on the current 38 clusters (does the
   cohesion/over-specification gate flag the known-bad clusters? does reuse/graft
   reduce fragmentation?).
2. Publish emergent types (+ hierarchy) into `logos:ontology:types` — unblocks
   reuse/graft.
3. Add `concept` and `process` as real base type-def roots.
4. Decide the granularity target (floor/ceiling thresholds) for naming.
5. The whole loop is **density-gated**: induction is what unlocks density, so the
   build order likely is — fix naming + publish + roots first (clean types now),
   then induction/contradiction (densify + correct), then the metacognitive/self
   layer once there's enough structure to reflect on.
