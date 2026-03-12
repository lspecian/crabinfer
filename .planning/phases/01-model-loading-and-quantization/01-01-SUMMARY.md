---
phase: 01-model-loading-and-quantization
plan: 01
subsystem: model-loading
tags: [hf-hub, sha256, rust, async, caching, serving-engine]

# Dependency graph
requires: []
provides:
  - HF repo ID detection via is_hf_repo_id()
  - Async model download and caching via ensure_model_cached()
  - SHA-256 file integrity verification via verify_sha256()
  - Server-side transparent HF Hub resolution in load_serving_engine()
affects: [02-quantization, 03-guided-decoding, 04-deployment]

# Tech tracking
tech-stack:
  added:
    - hf-hub 0.5 (tokio feature) — HuggingFace Hub API client
    - dirs 5 — user cache directory (~/.cache/)
    - anyhow 1 — flexible error handling in hub_download
    - sha2 0.10 — SHA-256 hashing (promoted from optional to always-on)
    - tempfile 3 (dev) — temp directories in tests
  patterns:
    - Feature-gated async functions: ensure_model_cached behind #[cfg(feature = "providers")]
    - tokio::runtime::Handle::current().block_on() for calling async from sync context
    - hf-hub ApiBuilder with custom cache_dir for non-default cache location

key-files:
  created:
    - crabinfer-core/src/serving/hub_download.rs
    - crabinfer-core/src/serving/hub_download_tests.rs
  modified:
    - crabinfer-core/Cargo.toml
    - crabinfer-core/src/serving/mod.rs
    - crabinfer-server/src/lib.rs

key-decisions:
  - "sha2 made non-optional (always-on) — SHA-256 is a core security function, not just for providers"
  - "hf-hub and dirs remain optional under providers feature — only needed when downloading"
  - "tokio::runtime::Handle::current().block_on() pattern for sync-to-async bridge in load_serving_engine"
  - "HF repo ID used as model_id instead of directory basename for both safetensors and GGUF branches"
  - "Cache location: ~/.cache/crabinfer/ via dirs::cache_dir() — avoids polluting default HF cache"

patterns-established:
  - "HF repo ID heuristic: exactly one slash, no leading / or ., no file extension, both org and model non-empty"
  - "should_download() whitelist pattern: explicit list of essential files, default deny"

requirements-completed: [MLOAD-01, MLOAD-02, MLOAD-03]

# Metrics
duration: 12min
completed: 2026-03-12
---

# Phase 01 Plan 01: HuggingFace Hub Download Client Summary

**HF Hub download client with repo ID detection, async caching to ~/.cache/crabinfer/, SHA-256 verification, and transparent server-side resolution of --model org/repo-name**

## Performance

- **Duration:** 12 min
- **Started:** 2026-03-12T22:46:13Z
- **Completed:** 2026-03-12T22:58:00Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Implemented `is_hf_repo_id()` with precise heuristic (one slash, no path prefix, no file extension)
- Implemented `ensure_model_cached()` using hf-hub ApiBuilder with custom ~/.cache/crabinfer/ directory
- Implemented `verify_sha256()` for post-download integrity checks (64KB chunk reading)
- Wired HF Hub detection into `load_serving_engine()` transparently — local paths unchanged
- 16 hub_download tests + 1 server integration test all passing

## Task Commits

Each task was committed atomically:

1. **Task 1: Create hub_download.rs with HF Hub download client** - `1f1062a` (feat)
2. **Task 2: Wire HF Hub download into server's load_serving_engine** - `ca18ee6` (feat)

## Files Created/Modified

- `crabinfer-core/src/serving/hub_download.rs` - Main module: is_hf_repo_id, should_download, ensure_model_cached, verify_sha256 with inline tests
- `crabinfer-core/src/serving/hub_download_tests.rs` - External test file with stub replacement tests
- `crabinfer-core/Cargo.toml` - Added hf-hub (optional), dirs (optional), anyhow, sha2 (non-optional), tempfile (dev)
- `crabinfer-core/src/serving/mod.rs` - Registered pub mod hub_download
- `crabinfer-server/src/lib.rs` - HF repo ID detection and resolution in load_serving_engine(), model_id fix for HF models, server test

## Decisions Made

- Made `sha2` non-optional (always compiled in): SHA-256 is security-critical and belongs in all builds, not just provider builds
- Used `anyhow` for error handling in hub_download.rs — simpler than custom thiserror types for a download client
- `tokio::runtime::Handle::current().block_on()` chosen over `Runtime::new()` — the server is already within a tokio runtime, creating a new one would panic
- `hf_repo_id` variable threaded through both safetensors and GGUF branches for consistent model_id reporting

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Promoted sha2 from optional to always-on dependency**
- **Found during:** Task 1 (hub_download.rs creation)
- **Issue:** sha2 was optional behind `providers` feature, but verify_sha256 is a security function that should always be available
- **Fix:** Changed `sha2 = { version = "0.10", optional = true }` to `sha2 = "0.10"` and removed from providers feature gate
- **Files modified:** crabinfer-core/Cargo.toml
- **Verification:** Tests compile without --features providers flag; sha2 always available
- **Committed in:** 1f1062a (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 missing critical — security-critical SHA-256 availability)
**Impact on plan:** No scope creep. The fix improves correctness by making verify_sha256 available in all build configurations.

## Issues Encountered

- Pre-existing stub test file `hub_download_tests.rs` existed with failing stubs — replaced with real tests as part of the GREEN phase
- Linter repeatedly modified hub_download_tests.rs adding `#[cfg(feature = "providers")]` to SHA256 tests — resolved by making sha2 non-optional

## User Setup Required

None - no external service configuration required. HF_TOKEN environment variable is read automatically by hf-hub when set, but is not required for public models.

## Next Phase Readiness

- HF Hub download pipeline complete — server can accept `--model org/repo-name` as model path
- verify_sha256 available for callers to validate downloaded files post-download
- ensure_model_cached returns the snapshot directory path (stable across invocations)
- Ready for Plan 01-02 (GPTQ INT4 quantization loading via safetensors)

---
*Phase: 01-model-loading-and-quantization*
*Completed: 2026-03-12*
