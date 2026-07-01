---
phase: 09-cache-salt-engine-bridge
plan: 01
subsystem: serving
tags: [kv-cache, cache-salt, prefix-cache, tenant-isolation, scheduler, block-hash, pcch-01]

# Dependency graph
requires:
  - phase: 07-server-wiring-last-mile
    provides: cache_salt field in SamplingParams (from ChatCompletionRequest via Phase 7)
  - phase: 04
    provides: BlockHash::from_tokens_salted primitive in block.rs (salt-aware FNV-1a chaining)
provides:
  - Scheduler::register_completed_blocks(SeqId) — salted block hash registration in KVCacheManager
  - Production call site in engine_loop::sample_and_distribute after each forward pass
  - 4 unit tests proving isolation, shared-hit, backward compat, and chunked-prefill correctness
affects:
  - 09-02 (subsequent cache-salt plans)
  - verify-work phase 9 (end-to-end HTTP tenant isolation scenario)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "register_completed_blocks uses clone-before-loop borrow pattern: clone prompt_tokens, salt, block_ids before acquiring &mut kv_cache"
    - "Decode-phase early return: already_registered >= floor(min(num_computed, prompt_len) / block_size)"
    - "Approach (b) for engine_loop gating: internal early return in method, unconditional call site"

key-files:
  created: []
  modified:
    - crabinfer-core/src/serving/scheduler.rs
    - crabinfer-core/src/serving/engine_loop.rs

key-decisions:
  - "Task 3 gating approach (b): internal decode-phase early return inside register_completed_blocks keeps engine_loop call unconditional and minimal"
  - "Borrow-checker workaround: clone prompt_tokens + salt + block_ids before the loop, then call cache_block and re-acquire &mut seq via get_mut in each iteration"
  - "seq.block_hashes (Sequence field at sequence.rs:147) is the target — NOT seq.blocks.block_hashes (SequenceBlocks field); avoided dual-field trap"
  - "BlockHash::from_tokens_salted is the only hashing call — no from_tokens regression introduced"
  - "test_cache_salt_shared_hit uses two schedulers (not one) to avoid single-scheduler scheduling-order complexity while still proving hash chain equality and cross-sequence cache hit"

patterns-established:
  - "register_completed_blocks pattern: clone all read data, compute hashes in pure loop, then push to KV cache + Sequence in separate loop"
  - "Engine loop unconditional call + method-internal guard: keeps caller minimal, concentrates gating logic in the method itself"

requirements-completed: [PCCH-01]

# Metrics
duration: 20min
completed: 2026-07-01
---

# Phase 09 Plan 01: Cache-Salt Engine Bridge Summary

**`Scheduler::register_completed_blocks` wires cache_salt into KV block hash computation, closing PCCH-01: salted FNV-1a chains via BlockHash::from_tokens_salted now register in KVCacheManager and Sequence.block_hashes after each prefill forward pass**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-07-01T20:54:00Z
- **Completed:** 2026-07-01T21:14:00Z
- **Tasks:** 3 (TDD: 2 + integration: 1)
- **Files modified:** 2

## Accomplishments
- Implemented `Scheduler::register_completed_blocks(seq_id: SeqId)` — idempotent, skip-partial-blocks, salted block hash registration
- Added `use super::block::BlockHash` import and method adjacent to `update_after_step` in scheduler.rs
- Called unconditionally from `engine_loop.rs::sample_and_distribute` after `update_after_step` (approach b: internal decode guard)
- 4 PCCH-01 unit tests: isolation (disjoint hashes), shared-hit (prefix cache hit), backward compat (None salt), chunked-prefill (3-step idempotent registration)
- Full library suite: 839 passed; 0 failed (up from 835 baseline; +4 new tests)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add 4 failing tests (TDD RED)** - `a67952d` (test)
2. **Task 2: Implement register_completed_blocks (TDD GREEN)** - `ecd19a4` (feat)
3. **Task 3: Wire engine_loop integration** - `8eb7c72` (feat)

## Files Created/Modified
- `crabinfer-core/src/serving/scheduler.rs` - Added `use super::block::BlockHash`, `register_completed_blocks` method (public, ~50 lines), 4 new unit tests, `test_cache_salt_shared_hit` test uses two schedulers
- `crabinfer-core/src/serving/engine_loop.rs` - Added unconditional `self.scheduler.register_completed_blocks(sched.seq_id)` call after `update_after_step` at line ~1631

## Decisions Made

**Task 3 gating approach (b) chosen over approach (a):**
- Approach (a) would use `if let Some(seq) = self.scheduler.get_sequence(sched.seq_id) { if seq.is_prefill() { ... } }` in engine_loop
- Approach (b) adds `already_registered >= max_registerable` early return inside `register_completed_blocks` itself
- Choice: (b) — keeps engine_loop minimal (one unconditional line), concentrates gating logic in the method that owns it
- The guard formula: `max_registerable = min(num_computed, prompt_len) / block_size` — correctly handles the case where num_computed equals prompt_len on the final chunk (avoids premature early return before the last block is registered)

**Borrow-checker approach:**
- Clone `prompt_tokens`, `salt`, `block_ids`, `prev_hash` from `&seq` before dropping the shared borrow
- Then iterate: compute hash (no borrow), call `kv_cache.cache_block(&mut self.kv_cache)`, re-acquire `&mut seq` to push hash
- This avoids holding both `&mut self.kv_cache` and `&seq` simultaneously

**Dual-field trap avoided:**
- `seq.block_hashes` (field on `Sequence` at sequence.rs:147) is populated — verified by `grep seq.block_hashes.push`
- `seq.blocks.block_hashes` (field on `SequenceBlocks`) is NOT touched — verified by zero hits on `grep seq.blocks.block_hashes.push`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `kv_cache()` returns &KVCacheManager but get_computed_blocks takes &mut self**
- **Found during:** Task 2 (implementing test_cache_salt_shared_hit)
- **Issue:** Test used `sched.kv_cache().get_computed_blocks(...)` but the method signature is `&mut self`
- **Fix:** Changed to `sched.kv_cache_mut().get_computed_blocks(...)`
- **Files modified:** crabinfer-core/src/serving/scheduler.rs
- **Committed in:** ecd19a4 (Task 2 commit)

**2. [Rule 1 - Bug] test_cache_salt_shared_hit: single-scheduler seq B had empty block_hashes**
- **Found during:** Task 2 (first test run)
- **Issue:** With one scheduler and two sequences added sequentially, `schedule()` batched both in the same step; `update_after_step(id_b, 0)` did not advance seq B, so `register_completed_blocks(id_b)` found 0 complete blocks
- **Fix:** Refactored test to use two independent schedulers (sched_a and sched_b) — same prompt, same salt, independently prefilled; hash equality proves they would share cache. Cross-sequence hit verified using sched_a's KV cache (which has seq A's blocks registered) queried with seq B's hash chain
- **Files modified:** crabinfer-core/src/serving/scheduler.rs
- **Committed in:** ecd19a4 (Task 2 commit)

**3. [Rule 1 - Bug] Decode-phase early return triggered prematurely on final chunked-prefill step**
- **Found during:** Task 3 (first full test run after engine_loop integration)
- **Issue:** Initial guard `num_computed >= seq.prompt_tokens.len()` fired when num_computed == 48 == prompt_len after 3rd update_after_step, but block 3 (indices 32..48) was not yet registered (block_hashes.len() == 2). This caused test_register_completed_blocks_chunked to assert 2 instead of 3
- **Fix:** Replaced guard with `already_registered >= max_registerable` where `max_registerable = min(num_computed, prompt_len) / block_size`. On the 3rd step: max_registerable = 48/16 = 3, already_registered = 2, so 2 >= 3 is false → proceeds to register the 3rd block
- **Files modified:** crabinfer-core/src/serving/scheduler.rs
- **Committed in:** 8eb7c72 (Task 3 commit)

---

**Total deviations:** 3 auto-fixed (3 Rule 1 bugs)
**Impact on plan:** All fixes were in test setup or the decode-phase guard logic; no API surface or contract changed. The 4 PCCH-01 truths from the plan frontmatter are all verified.

## Issues Encountered
- The plan's verification command `cargo build -p crabinfer-core --tests 2>&1 | grep ... | wc -l` expected count 4 but yielded 8 because test_isolation has 2 call sites and test_chunked has 3 — but all 8 cite the missing method in 4 tests, satisfying the RED state requirement
- Default feature `metal` includes `objc2` which errors on Linux; all tests must be run with `--no-default-features`

## Next Phase Readiness
- PCCH-01 is closed: `cache_salt` now has observable effect on KV cache block hashes and prefix reuse
- Ready for Phase 9 Plan 02 (or the next cache-salt plan)
- End-to-end HTTP verification via `09-VALIDATION.md` "tenant isolation scenario" can proceed

## Self-Check: PASSED

- scheduler.rs: FOUND
- engine_loop.rs: FOUND
- SUMMARY.md: FOUND
- commit a67952d (test): FOUND
- commit ecd19a4 (feat scheduler): FOUND
- commit 8eb7c72 (feat engine_loop): FOUND

---
*Phase: 09-cache-salt-engine-bridge*
*Completed: 2026-07-01*
