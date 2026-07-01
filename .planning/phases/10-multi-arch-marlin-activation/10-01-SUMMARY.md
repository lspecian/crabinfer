---
phase: 10-multi-arch-marlin-activation
plan: "01"
subsystem: serving/models
tags: [marlin, quantization, gptq, awq, phi3, mistral, deepseek, cuda]
dependency_graph:
  requires: []
  provides: [SafetensorsPhi3Model::activate_marlin, SafetensorsMistralModel::activate_marlin, SafetensorsDeepSeekModel::activate_marlin]
  affects: [crabinfer-core/src/serving/safetensors_loader.rs]
tech_stack:
  added: []
  patterns: [Rust inherent impl block, pub(crate) visibility, Arc<dyn KernelBackend>, MaybeQuantizedLinear match, DeepSeekMlp enum traversal]
key_files:
  created: []
  modified:
    - crabinfer-core/src/serving/models/phi3.rs
    - crabinfer-core/src/serving/models/mistral.rs
    - crabinfer-core/src/serving/models/deepseek.rs
decisions:
  - "activate_marlin placed in per-model files (not safetensors_loader.rs) because DeepSeek MoeLayer fields (gate, experts, shared_expert) and Phi3Layer::qkv_proj are private to their respective modules"
  - "MockCudaBackend duplicated locally in each test module (cannot import from quantization_tests.rs — private module)"
  - "test model construction via direct struct literal with private-field access (test module is pub(crate) scope)"
  - "SwiGluMlp.gate field dims (out=128, in=128 for down) chosen to be Marlin-aligned (multiples of 64 and 128)"
metrics:
  duration: "~8 min"
  completed: "2026-07-01"
  tasks_completed: 3
  files_modified: 3
---

# Phase 10 Plan 01: Multi-Arch Marlin Activation Summary

Added `pub(crate) fn activate_marlin(&mut self, backend: &Arc<dyn KernelBackend>) -> Result<usize>` to Phi3, Mistral, and DeepSeek model structs, with full unit test coverage using local MockCudaBackend stubs — closing QLOAD-03 partial gap for GPTQ/AWQ Marlin reformatting on non-Llama architectures.

## Tasks

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add SafetensorsPhi3Model::activate_marlin | 77ef49e | phi3.rs |
| 2 | Add SafetensorsMistralModel::activate_marlin | d17612a | mistral.rs |
| 3 | Add SafetensorsDeepSeekModel::activate_marlin (Dense + MoE) | b1ce837 | deepseek.rs |

## Imports Added

**phi3.rs:**
```rust
use std::sync::Arc;
use crate::serving::kernels::KernelBackend;
```

**mistral.rs:**
```rust
use std::sync::Arc;
use crate::serving::kernels::KernelBackend;
```

**deepseek.rs:**
```rust
use std::sync::Arc;
use crate::serving::kernels::KernelBackend;
```

## activate_marlin Impl Locations

| File | Line | Notes |
|------|------|-------|
| phi3.rs | 488-543 | After `impl ModelRunner for SafetensorsPhi3Model` block, before `#[cfg(test)] mod tests` |
| mistral.rs | 368-429 | After `impl ModelRunner for SafetensorsMistralModel` block, before `#[cfg(test)] mod tests` |
| deepseek.rs | 833-916 | After `impl ModelRunner for SafetensorsDeepSeekModel` block, before `#[cfg(test)] mod tests` |

## Architecture-Specific Adaptations

**Phi3 (vs Llama reference):**
- Uses fused `qkv_proj` instead of separate `attn_q`/`attn_k`/`attn_v` — traverses only `layer.qkv_proj`
- `mlp.gate`, `mlp.down`, `mlp.up` from `SwiGluMlp` (same as Llama)

**Mistral (vs Llama reference):**
- Structurally identical to Llama — exact port of `SafetensorsLlamaModel::activate_marlin`
- `attn_q`, `attn_k`, `attn_v`, `attn_output` + `mlp.gate`/`down`/`up`

**DeepSeek (vs Llama reference):**
- Optional `q_a_proj`: `if let Some(ref mut q_a) = layer.q_a_proj { process_linear(q_a)?; }`
- MLA projections: `q_b_proj`, `kv_a_proj`, `kv_b_proj`, `attn_output`
- `DeepSeekMlp::Dense` branch: `mlp.gate`/`down`/`up`
- `DeepSeekMlp::Moe` branch: `moe.gate` (router) + per-expert `Vec<SwiGluMlp>` + optional `shared_expert`

## Test Counts per Model

| Model | Tests | Coverage |
|-------|-------|---------|
| Phi3 | 3 | cpu_skips, cuda_populates, dense_noop |
| Mistral | 3 | cpu_skips, cuda_populates (attn_q/k/v verified), dense_noop |
| DeepSeek | 5 | cuda_dense_mlp (7 layers), cuda_dense_with_q_a (8 layers), cuda_moe (20 layers), cpu_skips, dense_noop |
| **Total** | **11** | |

## Pitfalls Encountered

1. **`PagedAttentionLayer` construction**: `PagedAttentionLayer::with_rope()` requires pre-computed cos/sin tensors — used `Tensor::zeros` stubs in test helpers since the layer is never used for actual forward pass in tests.

2. **Marlin alignment for MoE gate**: The router gate in MoE test was sized (64, 256) — 64 is a multiple of 64 and 256 is a multiple of 128, so it correctly counts as reformatted. This meant the MoE test count was `4 + 1 + 4*3 + 3 = 20` (all aligned).

3. **`objc2` metal feature on Linux**: The crate default feature is `metal` which fails on Linux. All tests and builds run with `--no-default-features`.

## Verification

```
cargo test -p crabinfer-core --lib serving::models --no-default-features
→ 76 tests pass (65 pre-existing + 11 new)

cargo build -p crabinfer-core --no-default-features
→ 0 errors, no new warnings introduced

grep -n "pub(crate) fn activate_marlin" phi3.rs mistral.rs deepseek.rs
→ 3 matches, one per file
```

## Deviations from Plan

None — plan executed exactly as written. The three new methods compile and all tests are green.

## Self-Check: PASSED

- [x] phi3.rs modified with `pub(crate) fn activate_marlin` at line 492
- [x] mistral.rs modified with `pub(crate) fn activate_marlin` at line 372
- [x] deepseek.rs modified with `pub(crate) fn activate_marlin` at line 837
- [x] Commit 77ef49e: feat(10-01): add SafetensorsPhi3Model::activate_marlin with tests
- [x] Commit d17612a: feat(10-01): add SafetensorsMistralModel::activate_marlin with tests
- [x] Commit b1ce837: feat(10-01): add SafetensorsDeepSeekModel::activate_marlin (Dense + MoE) with tests
