---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: planning
stopped_at: Completed 10-01-PLAN.md (multi-arch Marlin activation — SafetensorsPhi3Model, SafetensorsMistralModel, SafetensorsDeepSeekModel activate_marlin methods)
last_updated: "2026-07-01T21:40:01.481Z"
last_activity: 2026-03-12 — Roadmap created, phases derived from 27 v1 requirements
progress:
  total_phases: 12
  completed_phases: 11
  total_plans: 34
  completed_plans: 36
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
| Phase 03-guided-decoding P04 | 7 | 2 tasks | 8 files |
| Phase 05-wiring-fixes P02 | 3min | 1 tasks | 2 files |
| Phase 05-wiring-fixes P01 | 3min | 1 tasks | 1 files |
| Phase 06-embedding-model-loader P01 | 6 | 2 tasks | 3 files |
| Phase 06-embedding-model-loader P02 | 8 | 1 tasks | 3 files |
| Phase 07-server-wiring-last-mile P01 | 12 | 2 tasks | 3 files |
| Phase 07-server-wiring-last-mile P04 | 7 | 1 tasks | 1 files |
| Phase 07-server-wiring-last-mile P02 | 4min | 2 tasks | 4 files |
| Phase 07-server-wiring-last-mile P03 | 3min | 2 tasks | 4 files |
| Phase 08-fused-kernel-coverage P01 | 9min | 3 tasks | 4 files |
| Phase 09-cache-salt-engine-bridge P01 | 20min | 3 tasks | 2 files |
| Phase 09-cache-salt-engine-bridge P02 | 5min | 2 tasks | 1 files |
| Phase 10-multi-arch-marlin-activation P01 | 8min | 3 tasks | 3 files |

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
- [Phase 03-guided-decoding]: validate_constraint_via_tempvocab: WorkerPool::validate_constraint() builds temporary vocabulary per call — no caching, but correct and avoids architectural changes
- [Phase 03-guided-decoding]: stop_token_suppression: implemented inline in sample_and_distribute() using guided_states.contains_key() — no new message types needed
- [Phase 05-wiring-fixes]: RoutingPolicy::RoundRobin remains the default for backward compatibility; new_with_policy() opt-in to CacheAware
- [Phase 05-wiring-fixes]: block_size field on WorkerPool (default 16) mirrors KV cache block size for hash chunking
- [Phase 05-wiring-fixes]: parse_lfs_sha256_map extracted as pure function for testability without HTTP
- [Phase 05-wiring-fixes]: graceful degradation on HF API failure: empty map + warn, never block download
- [Phase 06-embedding-model-loader]: BertEmbeddingRunner wraps candle-transformers BertModel; NomicBertRunner implements custom encoder inline for non-standard nomic-bert architecture
- [Phase 06-embedding-model-loader]: clone_model() reloads from disk for embedding models (not cheaply clonable)
- [Phase 06-embedding-model-loader]: broadcast_matmul via unsqueeze(0) for 3D x 2D tensor matmul (candle requires same rank)
- [Phase 06-embedding-model-loader]: embedding_model field in EngineHandle as Arc<Mutex<Box<dyn ModelRunner>>> for encoder-only bypass; new_embedding_only() uses dropped receiver so submit() returns Err(Shutdown) safely
- [Phase 06-embedding-model-loader]: HfArchitectureProbe and ModelArchitecture made pub (not pub(crate)) so server crate can detect embedding-only architecture without re-reading config.json in a separate function
- [Phase 07-server-wiring-last-mile]: parse_routing_policy() as private helper near WorkerPool construction site — pure function, easily unit-testable; unknown strings degrade to RoundRobin with warning, never panic
- [Phase 07-server-wiring-last-mile]: TOKN-01: Source-level contract tests chosen over runtime mocks to pin batch embed call chain — avoids tokenizer trait refactor; std::fs::read_to_string + CARGO_MANIFEST_DIR pattern established
- [Phase 07-server-wiring-last-mile]: cache_salt placed adjacent to priority in ChatCompletionRequest for consistent CrabInfer extension grouping; serde(default) preserves backward compatibility
- [Phase 07-server-wiring-last-mile]: 503 returned for /v1/completions when serving engine missing — no legacy fallback; logprobs serializes as null (not omitted) per OpenAI client expectations
- [Phase 08-fused-kernel-coverage]: No per-model #[cfg(feature = 'cuda')] guards — KernelBackend trait handles dispatch internally
- [Phase 08-fused-kernel-coverage]: Quantized lm_head handled automatically by forward_linear_fused None-branch (no caller changes)
- [Phase 09-cache-salt-engine-bridge]: Task 3 gating approach (b): internal decode-phase early return inside register_completed_blocks keeps engine_loop call unconditional and minimal
- [Phase 09-cache-salt-engine-bridge]: seq.block_hashes (Sequence field) is the target for PCCH-01, not seq.blocks.block_hashes (SequenceBlocks) — dual-field trap avoided
- [Phase 09-cache-salt-engine-bridge]: BlockHash::from_tokens_salted is the only hashing call in register_completed_blocks; None salt preserves backward compat via block.rs:55
- [Phase 09-cache-salt-engine-bridge]: compute_prompt_hashes and best_prefix_worker accept Option<&str> salt; None preserves pre-PCCH-02 behavior
- [Phase 09-cache-salt-engine-bridge]: submit() extracts sampling_params.cache_salt.as_deref() at CacheAware branch only; RoundRobin path unchanged
- [Phase 09-cache-salt-engine-bridge]: PCCH-01 (storage) and PCCH-02 (routing) use same from_tokens_salted with same cache_salt — both hash domains align
- [Phase 10-multi-arch-marlin-activation]: activate_marlin placed in per-model files (not safetensors_loader.rs) because DeepSeek MoeLayer fields and Phi3 qkv_proj are private to their modules

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

Last session: 2026-07-01T21:40:01.477Z
Stopped at: Completed 10-01-PLAN.md (multi-arch Marlin activation — SafetensorsPhi3Model, SafetensorsMistralModel, SafetensorsDeepSeekModel activate_marlin methods)
Resume file: None
