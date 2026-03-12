---
phase: 1
slug: model-loading-and-quantization
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-12
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust `#[test]` + `cargo test` |
| **Config file** | `Cargo.toml` (workspace) |
| **Quick run command** | `cargo test -p crabinfer-core --lib serving` |
| **Full suite command** | `cargo test --workspace` |
| **Estimated runtime** | ~60 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p crabinfer-core --lib serving`
- **After every plan wave:** Run `cargo test --workspace`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Wave 0 Plan

Plan `01-00-PLAN.md` creates 9 failing test stubs before any implementation begins. All stubs compile but fail with clear messages referencing the implementing plan. Wave 1/2 plans make the stubs pass.

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 1-00-01 | 00 | 0 | ALL | stub | `cargo test -p crabinfer-core -- hub_download_tests safetensors_loader_tests quantization_tests` | W0 creates | ⬜ pending |
| 1-01-01 | 01 | 1 | MLOAD-01 | unit | `cargo test -p crabinfer-core hub_download` | W0 stub | ⬜ pending |
| 1-01-02 | 01 | 1 | MLOAD-02 | unit | `cargo test -p crabinfer-core hub_download` | W0 stub | ⬜ pending |
| 1-01-03 | 01 | 1 | MLOAD-03 | unit | `cargo test -p crabinfer-core hub_download` | W0 stub | ⬜ pending |
| 1-01-04 | 01 | 1 | MLOAD-01 | unit | `cargo test -p crabinfer-server hf_repo_id` | W1 creates | ⬜ pending |
| 1-02-01 | 02 | 1 | QLOAD-01 | unit | `cargo test -p crabinfer-core gptq_loading` | W0 stub | ⬜ pending |
| 1-02-02 | 02 | 1 | QLOAD-02 | unit | `cargo test -p crabinfer-core awq_loading` | W0 stub | ⬜ pending |
| 1-03-01 | 03 | 2 | QLOAD-03 | unit+cuda | `cargo test -p crabinfer-core marlin_gemm` | W0 stub | ⬜ pending |
| 1-03-02 | 03 | 2 | QLOAD-03 | unit | `cargo test -p crabinfer-core marlin_reformat` | W0 stub | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] Plan 01-00-PLAN.md created with 9 failing test stubs
- [x] Test stubs for hub download (repo resolution, download, SHA256)
- [x] Test stubs for GPTQ/AWQ HF weight loading (quantize_config parsing, weight loading)
- [x] Test stubs for Marlin kernel (reformat, fused GEMM correctness)

*Existing test infrastructure covers framework setup — only new test files needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| HF Hub download of real model | MLOAD-01 | Requires network + HF Hub access | `crabinfer serve --model meta-llama/Llama-3.1-8B-Instruct-GPTQ` |
| Resume interrupted download | MLOAD-03 | Requires simulating network interruption | Start download, kill process, restart |
| Marlin throughput improvement | QLOAD-03 | Requires GPU benchmarking | Compare tok/s with and without Marlin |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 60s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved (post-revision)
