---
phase: 03-guided-decoding
plan: 03
subsystem: serving
tags: [lru, prometheus, guided-decoding, metrics, outlines-core, atomics]

# Dependency graph
requires:
  - phase: 03-guided-decoding
    provides: IndexCache with HashMap, GuidedConstraint, GuidedState types from plans 01-02

provides:
  - LRU-backed IndexCache with configurable max_entries (default 256) and CacheStatsSnapshot
  - CacheStats atomic counters (hits, misses, evictions) on IndexCache
  - guided_cache_stats() method on EngineHandle and WorkerPool
  - Four Prometheus lines in GET /metrics: crabinfer_guided_cache_{hits,misses,size,evictions}

affects: [03-guided-decoding, serving, monitoring]

# Tech tracking
tech-stack:
  added: [lru::LruCache (was already in Cargo.toml for tokenizer cache, now used in guided.rs)]
  patterns:
    - "Shared Arc<AtomicU64> counters bridging background engine thread to EngineHandle (same as kv_blocks_used)"
    - "store() vs fetch_add() for absolute snapshot values vs delta increments"
    - "CacheStatsSnapshot: plain Copy struct for lock-free point-in-time reads"

key-files:
  created: []
  modified:
    - crabinfer-core/src/serving/guided.rs
    - crabinfer-core/src/serving/engine_loop.rs
    - crabinfer-core/src/serving/worker_pool.rs
    - crabinfer-server/src/state.rs
    - crabinfer-server/src/routes/health.rs

key-decisions:
  - "IndexCache::new(vocab, max_entries) + new_default(vocab) — explicit capacity API; new_default uses DEFAULT_MAX_ENTRIES=256"
  - "Shared Arc<AtomicU64> counters for guided cache stats (same cross-thread pattern as kv_blocks_used/num_waiting)"
  - "store() not fetch_add() for guided cache counters — IndexCache snapshots are absolute (not deltas)"
  - "guided_cache_stats() reads directly from EngineHandle's shared atomics — no locking needed at read path"

patterns-established:
  - "Absolute-value atomic counters (store) vs incremental (fetch_add): use store when source is already a cumulative snapshot"
  - "WorkerPool aggregates worker stats by summing hits/misses/evictions, size across all workers"

requirements-completed: [GDEC-04]

# Metrics
duration: 7min
completed: 2026-04-04
---

# Phase 03 Plan 03: Guided Decoding LRU Cache and Prometheus Metrics Summary

**LRU-bounded IndexCache (cap 256) with CacheStatsSnapshot wired to Prometheus /metrics via shared Arc atomics across the engine thread boundary**

## Performance

- **Duration:** 7 min
- **Started:** 2026-04-04T14:58:58Z
- **Completed:** 2026-04-04T15:06:10Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Replaced unbounded `HashMap` in `IndexCache` with `lru::LruCache` (configurable cap, default 256); oldest entry evicted when full
- Added `CacheStats` (AtomicU64 hits/misses/evictions) and `CacheStatsSnapshot` (Copy struct for lock-free reads); exposed via `IndexCache::stats()`
- Bridged per-thread cache stats to `EngineHandle` using four `Arc<AtomicU64>` (same pattern as `kv_blocks_used`), updated after each `create_guided_state()` call
- Exposed `guided_cache_stats()` on `EngineHandle` and `WorkerPool`; four `crabinfer_guided_cache_*` Prometheus lines now appear in GET /metrics

## Task Commits

Each task was committed atomically:

1. **Task 1: Replace IndexCache HashMap with LRU and add cache stats** - `46a8411` (feat)
2. **Task 2: Add guided cache Prometheus metrics to ServerMetrics and /metrics endpoint** - `fd229d1` (feat)

**Plan metadata:** (docs commit below)

## Files Created/Modified
- `crabinfer-core/src/serving/guided.rs` - LruCache replacing HashMap; CacheStats/CacheStatsSnapshot; new/new_default constructors; stats() method; 3 new tests (lru_eviction, cache_stats_tracking, default_max_entries)
- `crabinfer-core/src/serving/engine_loop.rs` - 4 new Arc<AtomicU64> fields on EngineHandle and ServingEngineInner; guided_cache_stats() method; store() update after create_guided_state()
- `crabinfer-core/src/serving/worker_pool.rs` - guided_cache_stats() aggregating across workers; updated IndexCache::new call to new_default
- `crabinfer-server/src/state.rs` - 4 guided cache AtomicU64 fields on ServerMetrics; update_guided_cache_stats() helper
- `crabinfer-server/src/routes/health.rs` - Prometheus block for crabinfer_guided_cache_{hits,misses,size,evictions}

## Decisions Made
- `new(vocab, max_entries)` + `new_default(vocab)` API: explicit capacity constructor keeps test code readable; default wraps it.
- Used `store()` (not `fetch_add()`) in engine_loop.rs for the shared guided cache counters because `IndexCache::stats()` already returns absolute cumulative values, not deltas.
- Read guided stats directly from `EngineHandle`'s shared atomics in `/metrics` (no round-trip to engine thread; same pattern as kv_cache metrics).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed 2 remaining IndexCache::new(vocab) call sites with old 1-arg signature**
- **Found during:** Task 1 (GREEN phase - compilation failures)
- **Issue:** `test_json_schema_constraint` and `test_regex_constraint` in guided.rs still used `IndexCache::new(vocab)` (1-arg). Also `worker_pool.rs` had the same issue.
- **Fix:** Updated all 3 sites to `IndexCache::new_default(vocab)`
- **Files modified:** guided.rs, worker_pool.rs
- **Verification:** `cargo test --lib -- guided::tests` — 9 tests pass
- **Committed in:** `46a8411` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 3 - blocking compile error)
**Impact on plan:** Required fix; call sites were created before the API signature was finalized. No scope creep.

## Issues Encountered
- Pre-existing `provider_integration` test binary fails to compile without `providers` feature (unrelated to this plan; tests are feature-gated but the test binary itself lacks the feature guard). Scoped out.

## Next Phase Readiness
- IndexCache is now memory-safe for production (bounded at 256 entries by default)
- Prometheus monitoring of guided decoding effectiveness is live
- Plan 03-04 can build on guided decoding infrastructure knowing cache is bounded and observable

---
*Phase: 03-guided-decoding*
*Completed: 2026-04-04*
