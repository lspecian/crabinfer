---
phase: 10-multi-arch-marlin-activation
plan: "02"
subsystem: serving/safetensors_loader
tags: [marlin, quantization, safetensors, phi3, mistral, deepseek, cuda]
dependency_graph:
  requires: [10-01]
  provides: [QLOAD-03]
  affects: [crabinfer-core/src/serving/safetensors_loader.rs]
tech_stack:
  added: []
  patterns: [marlin-activation-guard]
key_files:
  created: []
  modified:
    - crabinfer-core/src/serving/safetensors_loader.rs
decisions:
  - "Activation block copied verbatim from Qwen reference (lines 1239-1249) — identical guard structure ensures consistency across all 5 architectures"
metrics:
  duration: "2min"
  completed: "2026-07-01T21:42:33Z"
  tasks_completed: 1
  files_modified: 1
---

# Phase 10 Plan 02: Multi-Arch Marlin Activation — Loader Wiring Summary

Wired `activate_marlin` calls into the Phi3, Mistral, and DeepSeekV2 loader branches in `safetensors_loader.rs`, closing QLOAD-03. All 5 causal LM architectures now invoke Marlin tile reformatting on CUDA load.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Wire activate_marlin into Phi3, Mistral, DeepSeekV2 loader branches | 82a458d | crabinfer-core/src/serving/safetensors_loader.rs |

## What Was Built

Three identical edits were made to `load_model_from_safetensors_with_backend` in `safetensors_loader.rs`:

1. `let model` binding changed to `let mut model` in each branch (required for `&mut self` in `activate_marlin`)
2. Qwen-reference Marlin activation block inserted before each `return Ok(Box::new(model))`

### Final Line Ranges

- **Phi3 branch:** lines 1023–1049 (activation block: 1036–1046)
- **Mistral branch:** lines 1050–1076 (activation block: 1063–1073)
- **DeepSeekV2 branch:** lines 1077–1103 (activation block: 1090–1100)
- **Qwen branch (unchanged):** activation at lines 1278–1288
- **Llama branch (unchanged):** activation at line 1430

### Llama and Qwen Path Verification

Both the Llama and Qwen branches were not touched. Verification:
- Qwen activation block still at lines 1278–1288 (shifted by +39 due to 3x13-line insertions)
- Llama activation at line 1430 (shifted by +39)
- No argument order, comment text, or guard structure differences

### grep -c Verification

```
grep -c "activate_marlin(backend)" crabinfer-core/src/serving/safetensors_loader.rs
5
```

Lines: 1039 (Phi3), 1066 (Mistral), 1093 (DeepSeekV2), 1281 (Qwen), 1430 (Llama)

### Guard Structure

All 5 calls follow the identical pattern:
```rust
// Activate Marlin fused kernels for GPTQ/AWQ layers on CUDA
if device.is_cuda() {
    if let Some(ref backend) = kernel_backend {
        let marlin_count = model.activate_marlin(backend)?;
        if marlin_count > 0 {
            tracing::info!(
                "Activated Marlin fused kernel for {marlin_count} quantized layers"
            );
        }
    }
}
```

## Test Results

```
test result: ok. 851 passed; 0 failed; 0 ignored; 0 measured
```

Pre-change baseline: 851 tests (same count as post-10-01 state). Zero regressions.

New per-model tests from plan 10-01 all pass:
- `serving::models::phi3::tests::test_phi3_activate_marlin_cuda_populates` - ok
- `serving::models::phi3::tests::test_phi3_activate_marlin_cpu_skips` - ok
- `serving::models::mistral::tests::test_mistral_activate_marlin_cuda_populates` - ok
- `serving::models::mistral::tests::test_mistral_activate_marlin_cpu_skips` - ok
- `serving::models::deepseek::tests::test_deepseek_activate_marlin_cuda_dense_mlp` - ok
- `serving::models::deepseek::tests::test_deepseek_activate_marlin_cuda_dense_with_q_a` - ok
- `serving::models::deepseek::tests::test_deepseek_activate_marlin_cuda_moe` - ok
- `serving::models::deepseek::tests::test_deepseek_activate_marlin_cpu_skips` - ok

## Runtime CUDA Verification Note

Runtime verification (actual Marlin kernel dispatch on real GPTQ Phi3/Mistral/DeepSeek checkpoints) is deferred to H100 hardware — VALIDATION.md manual-only section covers this. The code-level contract is verified: the activation methods exist (plan 10-01), the loader calls them with correct guards (this plan), and unit tests exercise the CUDA-simulation path.

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

- Modified file exists: `crabinfer-core/src/serving/safetensors_loader.rs` - FOUND
- Task commit 82a458d - FOUND (`git log --oneline | grep 82a458d`)
- `grep -c "activate_marlin(backend)"` returns 5 - CONFIRMED
- All 851 tests pass - CONFIRMED
