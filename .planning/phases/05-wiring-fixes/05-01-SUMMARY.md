---
phase: 05-wiring-fixes
plan: "01"
subsystem: serving/hub_download
tags: [sha256, integrity, model-download, security, wiring]
dependency_graph:
  requires: []
  provides: [MLOAD-03]
  affects: [crabinfer-core/src/serving/hub_download.rs]
tech_stack:
  added: []
  patterns: [blocking-reqwest-in-async-context, graceful-degradation, pure-function-for-testability]
key_files:
  created: []
  modified:
    - crabinfer-core/src/serving/hub_download.rs
decisions:
  - parse_lfs_sha256_map extracted as pure function for testability without HTTP
  - graceful degradation: empty map on HF API failure, warn and skip verification
  - blocking reqwest wrapped in tokio::task::spawn_blocking for async bridge
metrics:
  duration: 3min
  completed_date: "2026-04-04T16:08:27Z"
  tasks_completed: 1
  files_modified: 1
---

# Phase 05 Plan 01: SHA-256 Verification Wiring Summary

**One-liner:** Wired orphaned `verify_sha256` into `ensure_model_cached` via HF API LFS metadata fetch, closing MLOAD-03 silent-corruption gap.

## What Was Built

`ensure_model_cached` now verifies every downloaded `.safetensors` file against its authoritative LFS SHA-256 from `https://huggingface.co/api/models/{repo_id}` before the server starts serving.

Key additions to `crabinfer-core/src/serving/hub_download.rs`:

1. **`parse_lfs_sha256_map(json: &Value) -> HashMap<String, String>`** — pure function that extracts `rfilename -> lfs.sha256` from the HF API JSON response. Extracted for testability without HTTP.

2. **`fetch_lfs_sha256_map_blocking(repo_id: &str)`** — uses `reqwest::blocking::Client` to call the HF API, adds `Authorization: Bearer` if `HF_TOKEN` is set (gated models), returns `HashMap` or error.

3. **Updated `ensure_model_cached`** — calls `fetch_lfs_sha256_map_blocking` via `tokio::task::spawn_blocking` before the download loop. After each `repo.get()` for a `.safetensors` file, calls `verify_sha256()` if the LFS map contains that filename. Graceful degradation: if the HF API is unreachable, logs a warning and continues with an empty map (no blocking).

4. **4 new unit tests** for `parse_lfs_sha256_map`:
   - `test_parse_lfs_sha256_from_json` — 3-sibling mix (safetensors+LFS, config.json no-LFS, bin+LFS)
   - `test_parse_lfs_sha256_non_lfs_files_empty_map` — all non-LFS files produce empty map
   - `test_parse_lfs_sha256_empty_siblings` — empty array
   - `test_parse_lfs_sha256_missing_siblings_key` — missing key

## Tasks Completed

| # | Task | Commit | Files |
|---|------|--------|-------|
| 1 | Fetch LFS SHA-256 metadata from HF API and wire verify_sha256 into ensure_model_cached | 467c1ac | crabinfer-core/src/serving/hub_download.rs |

## Verification

- `cargo test --package crabinfer-core --no-default-features --features providers hub_download` — 20/20 passed
- `cargo build --package crabinfer-core --no-default-features --features providers` — clean
- `cargo clippy --package crabinfer-core --no-default-features --features providers` — no new warnings

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

- File exists: `/home/ubuntu/Development/crabinfer/crabinfer-core/src/serving/hub_download.rs`
- Commit exists: 467c1ac (feat(05-01): wire verify_sha256 into ensure_model_cached download path)
- All 20 hub_download tests pass
- `verify_sha256` is called from `ensure_model_cached` for every `.safetensors` with LFS SHA-256
- Non-LFS files (config.json, tokenizer.json) skip SHA verification gracefully
- Missing LFS metadata logs a warning but does not block download
