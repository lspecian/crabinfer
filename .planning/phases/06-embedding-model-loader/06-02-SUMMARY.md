---
phase: 06-embedding-model-loader
plan: 02
subsystem: serving
tags: [bert, embedding, encoder, paged-attention, engine-loop, worker-pool]

# Dependency graph
requires:
  - phase: 06-embedding-model-loader/06-01
    provides: BertEmbeddingRunner, NomicBertRunner, ModelArchitecture.is_embedding_only()

provides:
  - EngineHandle::new_embedding_only() constructor (no engine thread, no KV cache)
  - EngineHandle::embed() routes through real encoder forward pass for BERT/NomicBert
  - load_serving_engine() detects embedding-only models and returns early without PagedAttention
  - HfArchitectureProbe and ModelArchitecture made pub for cross-crate access

affects: [serving-engine, embeddings-endpoint, worker-pool]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Embedding-only bypass: detect encoding-only arch from config.json, create lightweight EngineHandle"
    - "Encoder embed path: Arc<Mutex<Box<dyn ModelRunner>>> for stateful encoder access through EngineHandle"

key-files:
  created: []
  modified:
    - crabinfer-core/src/serving/engine_loop.rs
    - crabinfer-core/src/serving/safetensors_loader.rs
    - crabinfer-server/src/lib.rs

key-decisions:
  - "embedding_model field added to EngineHandle as Arc<Mutex<Box<dyn ModelRunner>>> for thread-safe encoder access across cloned handles"
  - "new_embedding_only() creates a dummy channel and drops receiver so submit() returns Err(Shutdown) cleanly if accidentally called"
  - "embed() priority order: encoder model > embed_table > hash fallback — maintains backward compat for causal LMs"
  - "HfArchitectureProbe made pub (not pub(crate)) so server crate can detect embedding-only without re-reading config.json in a separate function"
  - "is_safetensors guard on embedding detection: GGUF BERT models are not supported and would not have an HfArchitectureProbe"

patterns-established:
  - "Encoder bypass pattern: check architecture.is_embedding_only() early in load_serving_engine, return WorkerPool before KV cache estimation"

requirements-completed: [EMBD-02]

# Metrics
duration: 8min
completed: 2026-04-04
---

# Phase 06 Plan 02: Embedding Model Wiring Summary

**Encoder-only BERT/NomicBert models bypass PagedAttention entirely via new EngineHandle::new_embedding_only() constructor, routing embed() through the real encoder forward pass instead of token-table lookup**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-04-04T22:00:00Z
- **Completed:** 2026-04-04T22:08:00Z
- **Tasks:** 1
- **Files modified:** 3

## Accomplishments

- Added `embedding_model: Option<Arc<Mutex<Box<dyn ModelRunner>>>>` field to `EngineHandle` for encoder-only model access
- Added `EngineHandle::new_embedding_only()` constructor that skips engine thread, KV cache allocation, and scheduler creation
- Updated `EngineHandle::embed()` with a new priority path: encoder model > embed_table > hash fallback
- Added `embed_with_encoder()` helper that creates a 1-D token ID tensor and calls `model.embed()` (full BERT/NomicBert forward pass)
- Updated `load_serving_engine()` to detect BERT/NomicBert via `ModelArchitecture::is_embedding_only()` and return early with a lightweight `WorkerPool`
- Made `HfArchitectureProbe` and `ModelArchitecture` `pub` in `safetensors_loader` for cross-crate access
- Added "bert"/"nomic_bert" cases to the chat template architecture mapping to prevent defaulting to "llama"

## Task Commits

1. **Task 1: Bypass PagedAttention for encoder-only models and wire embed() through real encoder** - `fed97e2` (feat)

## Files Created/Modified

- `crabinfer-core/src/serving/engine_loop.rs` - Added `embedding_model` field, `new_embedding_only()` constructor, updated `embed()` and added `embed_with_encoder()` helper
- `crabinfer-core/src/serving/safetensors_loader.rs` - Made `HfArchitectureProbe` and `ModelArchitecture` `pub` (was `pub(crate)`/private)
- `crabinfer-server/src/lib.rs` - Early return for embedding-only models in `load_serving_engine()`, bert entry in chat template mapping

## Decisions Made

- **`Arc<Mutex<Box<dyn ModelRunner>>>` for embedding model**: The Mutex serializes encoder calls (inference is stateful), and Arc allows the handle to be cloned (WorkerPool clones handles). An alternative was `Arc<Mutex<dyn ModelRunner>>` (unsized) but `Box<dyn ModelRunner>` is simpler.
- **Dropped receiver trick for request_tx**: Creating a channel and dropping the receiver means any accidental `submit()` calls return `Err(Shutdown)` cleanly without panicking. This is safer than an `Option<Sender>` which would require touching submit() throughout.
- **`is_safetensors` guard on detection**: GGUF BERT models are not supported and would not have a valid `HfArchitectureProbe` JSON structure. Guard prevents false positives on GGUF paths.
- **Double L2 normalization in `embed_with_encoder()`**: BertEmbeddingRunner and NomicBertRunner already L2 normalize internally, but the outer normalization in `embed_with_encoder()` is a no-op when norm=1 and acts as a safety net for future runners that might not normalize.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed `&*model` dereference through Box<dyn ModelRunner>**
- **Found during:** Task 1 (first build)
- **Issue:** `embed_with_encoder(&*model, ...)` where `model` is `MutexGuard<Box<dyn ModelRunner>>` — dereferencing gives `Box<dyn ModelRunner>` not `dyn ModelRunner`, causing E0277
- **Fix:** Changed to `model.as_ref()` which correctly gives `&dyn ModelRunner`
- **Files modified:** `crabinfer-core/src/serving/engine_loop.rs`
- **Verification:** Build passes, 832 lib tests pass
- **Committed in:** `fed97e2` (part of task commit)

**2. [Rule 3 - Blocking] Made HfArchitectureProbe pub for cross-crate access**
- **Found during:** Task 1 (planning the implementation)
- **Issue:** `HfArchitectureProbe` was private (`struct`, no pub) and `ModelArchitecture` was `pub(crate)` — not accessible from `crabinfer-server`
- **Fix:** Made both `pub` with appropriate doc comments explaining the cross-crate use case. Also added `#[derive(Default)]` to `HfArchitectureProbe` for `serde_json::from_str().unwrap_or_default()` fallback.
- **Files modified:** `crabinfer-core/src/serving/safetensors_loader.rs`
- **Verification:** Server references compile cleanly
- **Committed in:** `fed97e2` (part of task commit)

---

**Total deviations:** 2 auto-fixed (1 bug, 1 blocking)
**Impact on plan:** Both auto-fixes required for compilation. No scope creep — stayed exactly within the plan's intended changes.

## Issues Encountered

None beyond the two auto-fixed compilation issues above.

## Next Phase Readiness

- EMBD-02 requirement complete: `crabinfer serve --model nomic-ai/nomic-embed-text-v1.5` loads and serves real encoder embeddings via POST /v1/embeddings
- No KV cache or PagedAttention scheduler is created for BERT/NomicBert models
- All 832 crabinfer-core library tests and 74 crabinfer-server tests pass — no regressions
- Phase 06 is now complete (Plan 01: runners, Plan 02: wiring)

---
*Phase: 06-embedding-model-loader*
*Completed: 2026-04-04*
