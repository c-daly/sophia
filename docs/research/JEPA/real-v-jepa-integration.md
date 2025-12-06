# Real V-JEPA Integration Plan

## Purpose
Define the steps to replace the stub JEPA runner with a real V-JEPA model while keeping existing API contracts (`/simulate`, `/ingest/media`) and downstream consumers unchanged.

## Scope
- Covers integration, configuration toggles, observability, testing, licensing/security, and rollout.
- Assumes model/infra selection is tracked in issue #75 (see comments there for chosen checkpoint and hardware).

## Integration Approach
- Implement a production `JEPARunner` variant (e.g., `RealJEPARunner`) that conforms to the existing runner interface.
- Add a small projection head if the model’s native embedding dim differs from the current 768-dim visual/physics embeddings; keep output keys stable.
- Accept image/video inputs; if the model needs clips, assemble frame windows from uploads or simulation context. Map entity/sensor metadata to optional conditioning tokens without changing request schema.
- Preserve response structures: imagined states/processes, `overall_confidence`, and embedding IDs written to Neo4j/Milvus.
- Backward compatibility: default to stub runner unless explicitly enabled.

## Configuration & Toggles
- New config flag: `JEPA_BACKEND` with values `stub` (default) or `real`.
- Paths/URIs for weights: `JEPA_WEIGHTS_PATH` (local) or `JEPA_WEIGHTS_URI` (remote); fail fast if missing when `real` is selected.
- Device selection: `JEPA_DEVICE` (e.g., `cuda:0`, `cpu`), plus `JEPA_DTYPE` (e.g., `fp16`, `bf16`) when supported.
- Optional acceleration flags: `JEPA_TRT_ENABLED`, `JEPA_SDPA_ENABLED`.
- Health/readiness probes should reflect backend status (weights loaded, device available).

## Observability
- Metrics: model load time, inference latency (embedding + rollout), GPU util/memory, cache hit rate for weights, and error counts by type.
- Logging: model version/commit, chosen device/dtype, input resolution/frame count, and fallback decisions (e.g., stub fallback on error or missing GPU).
- Tracing hooks around `/simulate` and `/ingest/media` to attribute latency to JEPA calls.

## Testing Strategy
- Unit: interface contract tests for `RealJEPARunner` (shapes, required keys, projection head correctness).
- Integration: run with a small/mini checkpoint (or recorded fixtures) to validate `/simulate` and `/ingest/media` end-to-end; gate on `JEPA_BACKEND=real` and skip if GPU unavailable.
- E2E: short-rollout (k<=3) scenario to verify imagined state persistence and embedding storage in Neo4j/Milvus.
- CI: prefer a distilled/mini model; otherwise mark GPU-required jobs optional and keep stub as default in main pipeline.

## Security & Licensing
- Record license terms for the chosen checkpoint; confirm redistribution rights in containers.
- Avoid logging media content or embeddings; ensure no PII is stored in Neo4j/Milvus metadata.
- Verify dependencies are sourced from trusted registries; pin model commit/version.

## Rollout Plan
- Dev: enable `JEPA_BACKEND=real` behind a feature flag on a GPU node; validate latency and correctness.
- Staging: run canary pods with real backend; compare metrics vs stub, keep stub as fallback.
- Prod: progressive enablement; maintain a quick switch back to stub. Document SLOs for latency and error rates.
- Downstream impact: API contracts stay unchanged; no action required for Apollo/logos unless embedding dims change (avoid if possible).

## Research / PoC Phase
- Goal: validate feasibility and quality of a real V-JEPA backend before full integration.
- Scope: stand up a minimal runner that loads the chosen checkpoint, produces embeddings, and runs a short rollout (k<=3) on sample media and simulation contexts.
- Approach:
	- Start with the pluggable `JEPARunner` backend interface (stub is default); add a PoC backend in an isolated module without wiring to prod paths.
	- Use a small/distilled or reduced-resolution checkpoint if available; otherwise constrain inputs (224–336 px, short clips) to fit a single GPU.
	- Implement only the minimal projection head to map native model outputs to our 768-dim visual/physics embeddings; keep API shapes identical.
	- Drive PoC via a notebook or a focused CLI script that loads weights, runs `simulate`/`process_media_sample` equivalents, and dumps latency + sample outputs.
	- Record metrics: load time, per-call latency, GPU memory, embedding norms/statistics, and qualitative sanity (nearest-neighbor sanity on a small gallery if possible).
	- Add feature flag to invoke the PoC backend manually (e.g., `JEPA_BACKEND=poc`) without changing default behavior.
- Exit criteria: (1) embeddings produced end-to-end with stable shapes; (2) rollout returns confidences and imagined states; (3) latency and VRAM fit within target envelope for dev/staging; (4) no API contract changes required.

## Risks / Open Questions
- Availability of a “mini” checkpoint for CI; if none exists, rely on recorded fixtures for integration tests.
- Embedding dimensionality mismatch requiring projection—ensure it does not degrade semantics.
- GPU quota/latency under load; may need batching or queueing if throughput targets are missed.
