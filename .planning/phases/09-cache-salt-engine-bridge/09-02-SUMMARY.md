---
phase: 09-cache-salt-engine-bridge
plan: 02
subsystem: serving
tags: [worker_pool, cache_aware_routing, prefix_cache, tenant_isolation, fnv_hash, pcch02]

# Dependency graph
requires:
  - phase: 09-cache-salt-engine-bridge
    plan: 01
    provides: "BlockHash::from_tokens_salted registered in Sequence.block_hashes via engine_loop"
provides:
  - "WorkerPool::compute_prompt_hashes(prompt_tokens, salt) — 2-arg salt-aware signature"
  - "WorkerPool::best_prefix_worker(prompt_tokens, salt) — forwards salt to hash computation"
  - "WorkerPool::submit() extracts cache_salt from SamplingParams on CacheAware branch"
  - "test_cache_aware_routing_with_salt: 3-scenario PCCH-02 unit test"
affects:
  - 09-cache-salt-engine-bridge
  - serving/worker_pool

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Salt threading: extract at submit() boundary, thread through all hash computation sites as Option<&str>"
    - "None-salt backward compat: from_tokens_salted(chunk, prev_hash, None) == from_tokens(chunk, prev_hash)"

key-files:
  created: []
  modified:
    - crabinfer-core/src/serving/worker_pool.rs

key-decisions:
  - "compute_prompt_hashes and best_prefix_worker both accept Option<&str> salt; None preserves pre-PCCH-02 behavior"
  - "submit() extracts sampling_params.cache_salt.as_deref() at the CacheAware routing branch only — RoundRobin path is unchanged"
  - "from_tokens_salted is the only hash call in production worker_pool.rs; from_tokens remains in test code for backward-compat expectation construction"

patterns-established:
  - "Salt domain alignment: PCCH-01 (storage via engine_loop) and PCCH-02 (routing via worker_pool) both use the same from_tokens_salted with the same request cache_salt, ensuring routing hashes and block hashes live in the same hash domain"

requirements-completed:
  - PCCH-02

# Metrics
duration: 5min
completed: 2026-07-01
---

# Phase 9 Plan 02: Cache-Salt WorkerPool Routing Summary

**`WorkerPool::compute_prompt_hashes` and `best_prefix_worker` now accept `Option<&str>` salt so cache-aware routing computes salted hashes that align with the salted blocks registered by PCCH-01, closing tenant-aware prefix routing (PCCH-02)**

## Performance

- **Duration:** 5 min
- **Started:** 2026-07-01T21:17:45Z
- **Completed:** 2026-07-01T21:23:22Z
- **Tasks:** 2 (TDD: RED + GREEN)
- **Files modified:** 1

## Accomplishments
- Updated `compute_prompt_hashes` to accept `Option<&str>` salt and call `BlockHash::from_tokens_salted` instead of `from_tokens`
- Updated `best_prefix_worker` to accept and forward salt to `compute_prompt_hashes`
- Updated `submit()` on the `CacheAware` branch to extract `sampling_params.cache_salt.as_deref()` and pass it through
- Added `test_cache_aware_routing_with_salt` with 3 scenarios: different-salt diverges, same-salt is deterministic, None-salt matches pre-PCCH-02 behavior
- Updated 3 pre-existing test call sites to pass `None` salt (lines previously calling 1-arg form)
- Full 840-test library suite passes with zero failures

## Task Commits

Each task was committed atomically:

1. **Task 1: Add failing test (TDD RED)** - `9d3140d` (test)
2. **Task 2: Thread cache_salt through routing (TDD GREEN)** - `955158b` (feat)

## Files Created/Modified
- `crabinfer-core/src/serving/worker_pool.rs` — Updated `compute_prompt_hashes` and `best_prefix_worker` signatures; `submit()` CacheAware branch; 3 existing test call sites updated; new `test_cache_aware_routing_with_salt`

## Output Section Notes (from plan)

- **Pre-existing worker_pool call sites updated to pass None salt:** 3 (lines 356, 370, 387 in updated file — `test_compute_prompt_hashes_empty`, `test_compute_prompt_hashes_single_block`, `test_compute_prompt_hashes_multiple_blocks`)
- **Production call sites outside worker_pool.rs needing update:** None — both methods are `fn` (private, not `pub`); confirmed with `grep -rn "compute_prompt_hashes\|best_prefix_worker" crabinfer-core/src/` returning only worker_pool.rs hits
- **Test count delta:** 839 -> 840 (added `test_cache_aware_routing_with_salt`)
- **PCCH-01 + PCCH-02 alignment confirmed:** `register_completed_blocks` (09-01) calls `BlockHash::from_tokens_salted(tokens, prev, salt)` from `seq.sampling_params.cache_salt.as_deref()`; `compute_prompt_hashes` (09-02) calls `BlockHash::from_tokens_salted(chunk, prev_hash, salt)` with the same `sampling_params.cache_salt.as_deref()`. Both use the same salt value from the same request, so routing hashes and stored block hashes are in the same hash domain.

## Decisions Made
- `compute_prompt_hashes` and `best_prefix_worker` both accept `Option<&str>` salt; `None` preserves pre-PCCH-02 behavior
- `submit()` extracts `sampling_params.cache_salt.as_deref()` at the `CacheAware` branch only — `RoundRobin` path is unchanged (no hashing involved)
- `BlockHash::from_tokens` is not called from production worker_pool.rs code after this plan; it remains in test code only for inline backward-compat expectation construction

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

The `objc2` crate blocks `cargo check/build` on Linux when default features (Metal) are enabled. All test runs used `--no-default-features` to work around the platform guard, consistent with prior phases.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- PCCH-01 (storage) and PCCH-02 (routing) are both complete; Phase 9 is fully closed
- `09-VALIDATION.md` describes the multi-worker server / `/metrics` verification path for `/gsd:verify-work 9`
- No blockers

---
*Phase: 09-cache-salt-engine-bridge*
*Completed: 2026-07-01*
