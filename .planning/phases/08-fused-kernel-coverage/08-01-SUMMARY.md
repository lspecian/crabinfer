---
phase: 08-fused-kernel-coverage
plan: 01
subsystem: serving
tags: [rmsnorm, fused-kernel, cuda, deepseek, mistral, phi3, llama, layernorm, lm_head]

# Dependency graph
requires:
  - phase: 02-fused-kernels
    provides: "RmsNorm::forward_linear_fused API + KernelBackend trait dispatch"
  - phase: 01-model-loading-and-quantization
    provides: "SafetensorsDeepSeekModel, SafetensorsMistralModel, SafetensorsPhi3Model"
provides:
  - "KERN-01 gap closed: all 4 causal LM architectures use fused final norm+lm_head"
  - "Three per-model CPU parity tests pinning forward_linear_fused call chain"
affects: [09-inference-benchmarks, v1.0-milestone-audit]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "forward_linear_fused pattern: all causal LM ModelRunner::forward() end with self.norm.forward_linear_fused(&hidden_states, &self.lm_head, ctx.backend)"

key-files:
  created: []
  modified:
    - crabinfer-core/src/serving/models/deepseek.rs
    - crabinfer-core/src/serving/models/mistral.rs
    - crabinfer-core/src/serving/models/phi3.rs
    - crabinfer-cli/src/cmd_serve.rs

key-decisions:
  - "No per-model #[cfg(feature = 'cuda')] guards — KernelBackend trait handles dispatch internally"
  - "Quantized lm_head handled automatically by forward_linear_fused None-branch (no caller changes)"

patterns-established:
  - "fused-final-layer: all causal LM forward() methods end with self.norm.forward_linear_fused(&hidden_states, &self.lm_head, ctx.backend)"

requirements-completed: [KERN-01]

# Metrics
duration: 9min
completed: 2026-04-18
---

# Phase 8 Plan 01: Fused Kernel Coverage Summary

**RmsNorm::forward_linear_fused wired into DeepSeek, Mistral, and Phi3 ModelRunner::forward() — closing KERN-01 partial gap so all four causal LM architectures use the CUDA fused layernorm+linear kernel on dense lm_head**

## Performance

- **Duration:** 9 min
- **Started:** 2026-04-18T08:06:29Z
- **Completed:** 2026-04-18T08:15:32Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments
- Ported the `forward_linear_fused` call-site pattern from llama.rs into deepseek.rs, mistral.rs, and phi3.rs (three one-line replacements)
- Added three per-model CPU parity tests (`test_deepseek_fused_linear_matches_unfused`, `test_mistral_fused_linear_matches_unfused`, `test_phi3_fused_linear_matches_unfused`) inside existing `mod tests` blocks
- Full workspace regression (873 + 91 + 3 = 967 tests) passes green with no regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: Add parity tests (RED)** - `bfc95db` (test)
2. **Task 2: Wire forward_linear_fused (GREEN)** - `d9be080` (feat)
3. **Task 3: Regression + audit (no code changes)** - verified inline; CLI fix committed separately

**Pre-existing bug fix (Rule 1):** `6800416` — fix(08-01): add missing routing_policy field in CliOverrides initializer

## Files Created/Modified
- `crabinfer-core/src/serving/models/deepseek.rs` - Replaced unfused final norm+lm_head with `forward_linear_fused`; added parity test
- `crabinfer-core/src/serving/models/mistral.rs` - Same change
- `crabinfer-core/src/serving/models/phi3.rs` - Same change
- `crabinfer-cli/src/cmd_serve.rs` - Added missing `routing_policy: None` field to CliOverrides initializer (pre-existing issue)

## Decisions Made
- No `#[cfg(feature = "cuda")]` guards added — `forward_linear_fused` handles backend dispatch internally via the `KernelBackend` trait; CPU/Metal fall through to sequential norm+matmul automatically
- Quantized lm_head needs no special handling at the call site — the existing `weight_tensor().is_none()` branch in `forward_linear_fused` covers it

## Source-Grep Audit Evidence

### After (forward_linear_fused call sites in forward() methods):
```
deepseek.rs:788:  self.norm.forward_linear_fused(&hidden_states, &self.lm_head, ctx.backend)
mistral.rs:323:   self.norm.forward_linear_fused(&hidden_states, &self.lm_head, ctx.backend)
phi3.rs:444:      self.norm.forward_linear_fused(&hidden_states, &self.lm_head, ctx.backend)
llama.rs:364:     self.norm.forward_linear_fused(&hidden_states, &self.lm_head, ctx.backend)
```

### Before pattern (old unfused final step — 0 hits remaining):
```
grep -n "norm.forward_fused(&hidden_states, ctx.backend)" deepseek.rs mistral.rs phi3.rs
(no hits)
```

## Full-Suite Evidence

```
cargo test --workspace --no-default-features
test result: ok. 873 passed; 0 failed; 0 ignored (crabinfer-core)
test result: ok. 91 passed; 0 failed; 0 ignored (crabinfer-server)
test result: ok. 3 passed; 0 failed; 2 ignored (doc-tests)
```

## Audit Impact Statement

**KERN-01 gap closed** — fused LayerNorm+linear is now wired into all four supported causal LM architectures (Llama + DeepSeek + Mistral + Phi3). CUDA backend uses the fused kernel for dense lm_head; CPU/Metal use the default `KernelBackend` fallback; quantized lm_head uses the existing `forward_linear_fused` None-branch fallback. Ready to flip KERN-01 from `partial` to `satisfied` in the next milestone audit.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed missing routing_policy field in CliOverrides initializer**
- **Found during:** Task 3 (full workspace regression run)
- **Issue:** `crabinfer-cli/src/cmd_serve.rs` initialized `CliOverrides` without the `routing_policy` field added in Phase 05, causing compile error in full workspace build
- **Fix:** Added `routing_policy: None` to the struct initializer (CLI flag not yet exposed; server reads it from config)
- **Files modified:** `crabinfer-cli/src/cmd_serve.rs`
- **Verification:** `cargo test --workspace` passes with no failures
- **Committed in:** `6800416` (separate fix commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - pre-existing bug blocking regression verification)
**Impact on plan:** Fix was necessary to complete Task 3 regression verification. Out-of-scope for plan objective but blocked the workspace test. No scope creep to plan artifacts.

## Deferred (Tech Debt)

Per 08-VALIDATION.md:
- **H100 runtime validation**: CUDA fused kernel dispatch on H100 hardware (requires Vast.ai session with real model weights)
- **End-to-end generation parity**: Output comparison between fused and unfused on real DeepSeek/Mistral/Phi3 checkpoints (requires multi-GB weight download)

## Issues Encountered
None in planned task execution. Pre-existing CLI compile error discovered during Task 3 regression and auto-fixed per Rule 1.

## Next Phase Readiness
- KERN-01 is satisfied — all four causal LM models use the fused norm+lm_head pattern
- Source-grep evidence is ready to cite in the v1.0 milestone audit re-run
- H100 runtime validation remains deferred (tracked in 08-VALIDATION.md)

---
*Phase: 08-fused-kernel-coverage*
*Completed: 2026-04-18*
