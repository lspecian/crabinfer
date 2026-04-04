---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: planning
stopped_at: Completed 03-guided-decoding 03-03-PLAN.md
last_updated: "2026-04-04T15:07:33.532Z"
last_activity: 2026-03-12 — Roadmap created, phases derived from 27 v1 requirements
progress:
  total_phases: 6
  completed_phases: 6
  total_plans: 21
  completed_plans: 23
  percent: 25
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-12)

**Core value:** Ship every CUDA/Linux feature buildable on RTX 3060, closing gap with vLLM for single-GPU deployments
**Current focus:** Phase 1 — Model Loading and Quantization

## Current Position

Phase: 1 of 4 (Model Loading and Quantization)
Plan: 0 of TBD in current phase
Status: Ready to plan
Last activity: 2026-03-12 — Roadmap created, phases derived from 27 v1 requirements

Progress: [███░░░░░░░] 25%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: -
- Trend: -

*Updated after each plan completion*
| Phase 01 P02 | 8 | 2 tasks | 4 files |
| Phase 01-model-loading-and-quantization P00 | 15 | 1 tasks | 5 files |
| Phase 01-model-loading-and-quantization P01 | 12min | 2 tasks | 5 files |
| Phase 04 P02 | 6 | 2 tasks | 6 files |
| Phase 01.1 P00 | 2min | 2 tasks | 2 files |
| Phase 01.1 P02 | 33 | 2 tasks | 3 files |
| Phase 01.2 P00 | 2min | 2 tasks | 3 files |
| Phase 01.2 P03 | 8 | 1 tasks | 2 files |
| Phase 03-guided-decoding P03 | 7 | 2 tasks | 5 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Setup: AWQ reuses GPTQ INT4 infrastructure (same W4A16 packed format)
- Setup: KernelBackend trait for fused ops with unfused defaults (CPU/Metal unaffected)
- Setup: Safetensors loader delegates to candle MmapedSafetensors
- [Phase 01]: Norm weights and embeddings always loaded as FP in quantized models
- [Phase 01]: GptqLinear qweight_marlin and backend fields default to None — Plan 03 populates
- [Phase 01]: reformat_for_marlin() stub returns Ok(false) — Plan 03 implements real Marlin reformatting
- [Phase 01-model-loading-and-quantization]: Plan 00: Test stubs upgraded to concrete tests because implementations existed from prior session commits (b03b5ab, b8e95fe, 1f1062a)
- [Phase 01-model-loading-and-quantization]: sha2 and anyhow made non-optional dependencies since verify_sha256 is core functionality (not cloud-only)
- [Phase 01-model-loading-and-quantization]: sha2 made non-optional (always-on) — SHA-256 is core security, not just for providers
- [Phase 01-model-loading-and-quantization]: tokio::runtime::Handle::current().block_on() for sync-to-async bridge in load_serving_engine
- [Phase 01-model-loading-and-quantization]: HF repo ID used as model_id instead of directory basename for consistent reporting
- [Phase 04]: Salt bytes mixed into FNV-1a initial state on first block only; chaining propagates salt
- [Phase 04]: Arc<Mutex<Vec<BlockHash>>> snapshot pattern for non-blocking block_hashes() reads
- [Phase 01.1]: Tests gated with cfg(feature=cuda) so they compile only when CUDA toolchain available
- [Phase 01.1]: I32 as default metadata_dtype for CUDA graph buffers (matches vLLM, halves metadata memory)
- [Phase 01.1]: Safetensors loader uses direct GPU load with CPU fallback for I32/I16 tensors
- [Phase 01.2]: Wave 0 stubs use #[should_panic(expected = 'not yet implemented')] so they compile but fail at runtime — Plan 01 will replace todo!() with real assertions
- [Phase 01.2]: DTYPE-06 (forward pass dtype) has no dedicated test — candle matmul uses tensor native dtype automatically; covered by GPU integration testing
- [Phase 01.2]: weight_dtype_bytes added as final parameter to profile_gpu_memory (after kv_dtype_bytes) for API consistency
- [Phase 01.2]: VRAM savings log guarded by serving_dtype != DType::F32 to avoid noisy F32-only startup logs
- [Phase 03-guided-decoding]: IndexCache::new(vocab, max_entries) + new_default(vocab) API: explicit capacity; default 256
- [Phase 03-guided-decoding]: store() not fetch_add() for guided cache counters — IndexCache snapshots are absolute cumulative values
- [Phase 03-guided-decoding]: guided_cache_stats() reads from EngineHandle shared atomics directly — no engine thread round-trip needed at /metrics read path

### Pending Todos

None yet.

### Roadmap Evolution

- Phase 01.1 inserted after Phase 1: CUDA Graph Capture and Candle Fork Patches (URGENT) — H100 testing revealed CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED is the #1 performance gap vs vLLM (30-40%)
- Phase 01.2 inserted after Phase 1: BF16/FP16 Serving Support (URGENT) — H100 testing showed FP32-only serving wastes 2x VRAM (40GB vs needed 20GB)

### Blockers/Concerns

- Phase 3 (Guided Decoding) depends only on Phase 1 (not Phase 2) — can be parallelized if needed
- Marlin kernel (QLOAD-03) is the highest-risk item: PTX-level optimization targeting sm_86
- CUDA graph capture fix (Phase 1.1) needs GPU hardware to test — requires another Vast.ai session
- BF16 support (Phase 1.2) can be developed and partially tested locally

## Session Continuity

Last session: 2026-04-04T15:07:33.530Z
Stopped at: Completed 03-guided-decoding 03-03-PLAN.md
Resume file: None
