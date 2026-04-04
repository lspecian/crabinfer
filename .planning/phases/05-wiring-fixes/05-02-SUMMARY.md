---
phase: 05-wiring-fixes
plan: "02"
subsystem: serving/worker_pool
tags: [routing, cache-aware, worker-pool, prefix-cache]
dependency_graph:
  requires: []
  provides: [RoutingPolicy, WorkerPool::new_with_policy, WorkerPool::submit cache-aware]
  affects: [serving/mod.rs]
tech_stack:
  added: []
  patterns: [TDD red-green, FNV-1a prefix hash chaining, HashSet intersection count]
key_files:
  created: []
  modified:
    - crabinfer-core/src/serving/worker_pool.rs
    - crabinfer-core/src/serving/mod.rs
decisions:
  - "RoutingPolicy::RoundRobin remains the default for backward compatibility; new_with_policy() opt-in to CacheAware"
  - "block_size field on WorkerPool (default 16) mirrors KV cache block size for hash chunking"
  - "Single-worker CacheAware pool always returns index 0 — no hash computation needed"
  - "best_prefix_worker() falls back to round-robin on zero-match (prevents thundering herd on cold cache)"
metrics:
  duration: "3 min"
  completed_date: "2026-04-04"
  tasks_completed: 1
  files_modified: 2
---

# Phase 5 Plan 02: Cache-Aware WorkerPool Routing Summary

Cache-aware routing added to WorkerPool using BlockHash prefix matching: RoutingPolicy enum, compute_prompt_hashes/prefix_match_count helpers, and CacheAware submit() path that calls block_hashes() per worker and picks the best prefix match.

## What Was Built

Closes WORK-03 gap: `EngineHandle::block_hashes()` was already exposed but `WorkerPool::submit()` always used round-robin, wasting prefix cache hits in multi-worker deployments.

**New API:**
- `RoutingPolicy` enum: `RoundRobin` (default) | `CacheAware`
- `WorkerPool::new_with_policy(workers, policy, block_size)` constructor
- `WorkerPool::submit()` now dispatches on routing policy

**Cache-aware algorithm (WORK-03 specification):**
1. Chunk `prompt_tokens` into `block_size`-token blocks
2. Hash each block with FNV-1a chaining (via existing `BlockHash::from_tokens`)
3. For each worker, call `block_hashes()` and count overlapping hashes
4. Route to highest-match worker; fall back to round-robin on zero match

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| RED | Add failing tests for RoutingPolicy and cache helpers | 2de4b51 | worker_pool.rs |
| GREEN | Implement RoutingPolicy, new_with_policy, helpers, submit() | 1c8fc4e | worker_pool.rs, mod.rs |

## Test Results

11/11 worker_pool tests pass:
- `test_prefix_match_count_no_overlap` — disjoint hash sets -> 0
- `test_prefix_match_count_full_overlap` — identical sets -> 3
- `test_prefix_match_count_partial` — overlapping sets -> correct count
- `test_compute_prompt_hashes_empty` — empty tokens -> empty hashes
- `test_compute_prompt_hashes_single_block` — 8 tokens, block_size 16 -> 1 hash
- `test_compute_prompt_hashes_multiple_blocks` — 12 tokens, block_size 4 -> 3 chained hashes
- 5 existing round-robin tests unchanged

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check

Files created/modified:
- `crabinfer-core/src/serving/worker_pool.rs` — updated
- `crabinfer-core/src/serving/mod.rs` — updated

Commits:
- 2de4b51 — test(05-02): add failing tests
- 1c8fc4e — feat(05-02): implement RoutingPolicy and cache-aware routing
