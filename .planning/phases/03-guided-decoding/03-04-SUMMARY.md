---
phase: 03-guided-decoding
plan: "04"
subsystem: server/guided-endpoint
tags: [guided-decoding, api, endpoint, stop-tokens, logprobs]
dependency_graph:
  requires:
    - crabinfer-core/src/serving/guided.rs (GuidedConstraint, IndexCache, create_guided_state)
    - crabinfer-core/src/serving/worker_pool.rs (WorkerPool)
    - crabinfer-server/src/routes/openai.rs (finish_reason_to_openai, resolve_architecture)
  provides:
    - POST /v1/guided/completions endpoint
    - GuidedCompletionRequest and GuidedConstraintSpec types
    - WorkerPool::validate_constraint()
  affects:
    - crabinfer-core/src/serving/engine_loop.rs (stop token suppression during guided)
tech_stack:
  added: []
  patterns:
    - axum SSE streaming (reuse from openai.rs)
    - dry-run constraint validation via temporary IndexCache
key_files:
  created:
    - crabinfer-server/src/types/guided.rs
    - crabinfer-server/src/routes/guided.rs
  modified:
    - crabinfer-server/src/types/mod.rs
    - crabinfer-server/src/routes/mod.rs
    - crabinfer-server/src/routes/openai.rs
    - crabinfer-core/src/serving/worker_pool.rs
    - crabinfer-core/src/serving/engine_loop.rs
    - crabinfer-core/src/serving/guided.rs
decisions:
  - key: validate_constraint_via_tempvocab
    summary: "WorkerPool::validate_constraint() builds a temporary vocabulary per call using the engine's tokenizer — no caching, but correct. Acceptable since validation is once per request and failures are early-exit."
  - key: stop_token_suppression_placement
    summary: "Stop token suppression implemented inline in sample_and_distribute() using guided_states.contains_key() — no new message types or architectural changes needed."
metrics:
  duration: "7min"
  completed: "2026-04-04T15:07:20Z"
  tasks_completed: 2
  files_changed: 8
---

# Phase 03 Plan 04: Guided Completions Endpoint Summary

New unified `/v1/guided/completions` endpoint exposing constrained generation (regex + JSON Schema) via the serving engine, with configurable strict/graceful error behavior and correct stop-token suppression during guided decoding.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create guided endpoint types and route handler | f7330dd | types/guided.rs, routes/guided.rs, routes/mod.rs, types/mod.rs, openai.rs, worker_pool.rs, engine_loop.rs |
| 2 | Constraint overrides stop tokens and post-mask logprobs verification | 9716565 | guided.rs (tests) |

## What Was Built

**types/guided.rs:** `GuidedCompletionRequest` (extends chat completion with required `constraint` field and `strict_constraints: bool = true`) and `GuidedConstraintSpec` enum with `regex`/`json_schema` variants using serde `tag = "type"`.

**routes/guided.rs:** Full handler for `POST /v1/guided/completions` that:
- Converts `GuidedConstraintSpec` to `GuidedConstraint`
- Validates constraint via `WorkerPool::validate_constraint()` before submission
- Returns HTTP 400 with outlines-core error message when constraint is invalid and `strict_constraints = true` (default)
- Logs warning and proceeds unconstrained when `strict_constraints = false`
- Handles both streaming (SSE) and non-streaming responses
- Reuses `finish_reason_to_openai` and `resolve_architecture` from openai.rs

**WorkerPool::validate_constraint():** New method that builds a temporary vocabulary from the tokenizer and calls `create_guided_state()` as a dry-run. Returns `Ok(())` or an outlines-core error string. Added to worker_pool.rs without touching the engine loop channel protocol.

**engine_loop.rs stop token suppression:** Changed:
```rust
// Before
let should_stop = is_eos || seq.should_stop(token_id);

// After
let has_guided = self.guided_states.contains_key(&sched.seq_id);
let should_stop = is_eos || (!has_guided && seq.should_stop(token_id));
```
This prevents stop_token_ids from firing mid-sequence when the DFA is controlling generation.

**Logprobs comment:** Added documenting comment confirming logprobs are computed from post-mask logits (already the case from prior plans), making the behavior explicit.

**EngineHandle constructor fix:** Added missing `guided_cache_hits/misses/evictions/size` field initializations (pre-existing struct fields that had been added without corresponding constructor updates).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] EngineHandle missing guided cache metric field initializations**
- **Found during:** Task 1 compilation
- **Issue:** `EngineHandle::new()` was missing 4 fields (`guided_cache_hits`, `guided_cache_misses`, `guided_cache_evictions`, `guided_cache_size`) that had been added to the struct but not initialized in the constructor
- **Fix:** Added initialization of all 4 Arc<AtomicU64> fields in the `Ok(Self { ... })` block and in `ServingEngineInner` initialization
- **Files modified:** `crabinfer-core/src/serving/engine_loop.rs`
- **Commit:** f7330dd

**2. [Rule 2 - Missing functionality] WorkerPool::validate_constraint uses temporary vocab**
- **Context:** Plan said "use index_cache on the serving engine" but the index_cache is private to the engine loop thread. An alternative was needed.
- **Solution:** `WorkerPool::validate_constraint()` builds a temporary `IndexCache` from the tokenizer (which WorkerPool already exposes) each time validation is requested. This is slightly less efficient than reusing the engine's cache but requires no architectural changes and is called at most once per request.

## Self-Check

### Files Exist
- [x] `crabinfer-server/src/types/guided.rs`
- [x] `crabinfer-server/src/routes/guided.rs`

### Commits Exist
- [x] f7330dd
- [x] 9716565

## Self-Check: PASSED
