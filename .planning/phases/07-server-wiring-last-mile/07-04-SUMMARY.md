---
phase: 07-server-wiring-last-mile
plan: "04"
subsystem: testing
tags: [embeddings, batch, tokenizer, call-chain, contract-testing, TOKN-01]

# Dependency graph
requires:
  - phase: 06-embedding-model-loader
    provides: EngineHandle::embed() calling encode_batch internally (engine_loop.rs:715)
  - phase: 07-server-wiring-last-mile
    provides: POST /v1/embeddings route calling engine.embed(texts)
provides:
  - "Automated tests documenting and pinning the TOKN-01 batch tokenization call chain"
  - "Source-level contract tests verifying routes/embeddings.rs -> engine.embed -> encode_batch path"
affects: [any future refactor of engine_loop.rs::embed or routes/embeddings.rs]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Source-level contract testing using std::fs::read_to_string + CARGO_MANIFEST_DIR for cross-crate call-chain pinning"
    - "Lightweight audit gap closure: source assertions over mock infrastructure when the contract is a single call site"

key-files:
  created: []
  modified:
    - "crabinfer-server/src/routes/embeddings.rs"

key-decisions:
  - "Source-level tests (std::fs::read_to_string) chosen over runtime mock tests: mock would require tokenizer trait refactor in engine_loop.rs — disproportionate scope for a one-line call-site contract"
  - "Tests added to routes/embeddings.rs (not a separate file) because the route module is the natural location for route-side contracts"
  - "TOKN-01 audit gap closed by test evidence, not by code refactor: engine.embed() already calls encode_batch internally, gap was missing test coverage"

patterns-established:
  - "Call-chain pinning: assert source.contains('self.encode_batch(') within embed() window to detect silent regressions"

requirements-completed: [TOKN-01]

# Metrics
duration: 7min
completed: 2026-04-18
---

# Phase 07 Plan 04: TOKN-01 Batch Embedding Call-Chain Contract Tests Summary

**Three source-level contract tests in routes/embeddings.rs pinning the TOKN-01 call chain: EmbeddingInput::Batch -> into_texts() -> engine.embed() -> encode_batch() (rayon parallel), closing the v1.0 audit gap with automated evidence**

## Performance

- **Duration:** 7 min
- **Started:** 2026-04-18T05:23:15Z
- **Completed:** 2026-04-18T05:30:00Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Verified the full TOKN-01 call chain before writing tests: `grep` confirmed `self.encode_batch(` at engine_loop.rs:715 and `engine.embed(` at routes/embeddings.rs:30
- Added `test_batch_input_into_texts_preserves_order`: smoke test confirming `EmbeddingInput::Batch(["a","b","c"]).into_texts()` returns 3 texts in order
- Added `test_embed_call_chain_uses_encode_batch`: source-level assertion that `EngineHandle::embed()` calls `self.encode_batch()` within its function body — pinning the parallel tokenization contract
- Added `test_embeddings_route_calls_engine_embed`: source-level assertion that the HTTP route calls `engine.embed(texts)` — pinning the route dispatch entry point
- No production code changed — verification + test work only, as specified in 07-RESEARCH.md
- Full server test suite: 87 tests pass (84 pre-existing + 3 new)

## Task Commits

1. **Task 1: Add call-chain assertion test for TOKN-01 batch path** - `a7e11aa` (feat)

**Plan metadata:** (pending final commit)

## Files Created/Modified

- `/home/ubuntu/Development/crabinfer/crabinfer-server/src/routes/embeddings.rs` - Added 76-line `#[cfg(test)] mod tests` block with 3 TOKN-01 contract tests

## Decisions Made

- Source-level tests (reading files via `std::fs::read_to_string` with `CARGO_MANIFEST_DIR`) chosen over runtime/mock tests. A runtime test would require plumbing a mock tokenizer through `EngineHandle`, which would need a trait refactor in engine_loop.rs — disproportionate for verifying a single call site.
- Tests placed in `routes/embeddings.rs` rather than a separate file: the route module is the natural location for route-side contracts, and keeps related test context together.
- The audit gap (v1.0-MILESTONE-AUDIT.md: "encode_batch orphaned, batch path unreachable") was already closed by Phase 6 wiring (engine_loop.rs:715). This plan adds the missing test evidence to make closure machine-verifiable.

## Deviations from Plan

None — plan executed exactly as written. The call chain was confirmed intact before writing tests, and all three tests passed on first run.

## Issues Encountered

- Initial `cargo test` run without `--no-default-features` failed with `objc2` compilation error (Apple-only crate, not in scope for Linux env). Resolved by using `--no-default-features` flag, which matches how the Linux/CUDA build targets the server crate. Tests passed cleanly on first attempt.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- TOKN-01 requirement is now closed with automated test evidence
- Future refactors of `engine_loop.rs::embed()` or `routes/embeddings.rs` that break the batch call chain will be caught immediately by CI
- Server test suite at 87 tests, all passing

---
*Phase: 07-server-wiring-last-mile*
*Completed: 2026-04-18*
