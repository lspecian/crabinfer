# Roadmap: CrabInfer ROADMAP-D CUDA/Linux Track

## Overview

Complete the remaining CUDA/Linux features for CrabInfer's PagedAttention serving engine, targeting RTX 3060 (6GB VRAM). Starting from a solid foundation of paged attention, CUDA graphs, fused kernels, and quantization infrastructure, this roadmap delivers HuggingFace model loading, Marlin kernel performance, guided decoding, memory/tokenization optimization, and production infrastructure — closing the gap with vLLM for single-GPU deployments.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Model Loading and Quantization** - Load GPTQ/AWQ models from HuggingFace Hub with Marlin fused kernel
- [ ] **Phase 1.1: CUDA Graph Capture and Candle Fork Patches** - Fix CUDA graph capture failure and patch candle fork for missing I32/I16 kernels (INSERTED)
- [ ] **Phase 1.2: BF16/FP16 Serving Support** - Auto-downcast FP32 models to BF16/FP16, halving VRAM usage (INSERTED)
- [ ] **Phase 2: Performance Optimization** - Fused LayerNorm+linear kernel, memory pool, and fast tokenization
- [ ] **Phase 3: Guided Decoding** - Token-level constrained generation via JSON Schema and regex DFA
- [ ] **Phase 4: Production Infrastructure** - Multi-worker serving, TOML config, prefix cache salting, and embeddings endpoint
- [ ] **Phase 5: Wiring Fixes** - SHA-256 verification in download path + cache-aware worker routing
- [x] **Phase 6: Embedding Model Loader** - BERT/encoder model support for dedicated embedding models (completed 2026-04-04)
- [ ] **Phase 7: Server Wiring Last-Mile** - Wire CacheAware routing, cache_salt, batch tokenization, and /v1/completions through to API surface (gap closure)
- [ ] **Phase 8: Fused Kernel Coverage** - Wire fused LayerNorm+linear into DeepSeek/Mistral/Phi3 final layers (gap closure)

## Phase Details

### Phase 1: Model Loading and Quantization
**Goal**: Users can point the server at any HuggingFace GPTQ or AWQ repo ID and have it download, cache, and serve the model with Marlin-accelerated inference
**Depends on**: Nothing (first phase)
**Requirements**: QLOAD-01, QLOAD-02, QLOAD-03, MLOAD-01, MLOAD-02, MLOAD-03
**Success Criteria** (what must be TRUE):
  1. `crabinfer serve --model meta-llama/Llama-3.1-8B-Instruct-GPTQ` downloads the model to `~/.cache/crabinfer/` and starts serving without any manual file management
  2. An interrupted download resumes from where it left off and the server rejects a corrupted file (failed SHA256) with a clear error
  3. AWQ models load using the same CLI flag and produce correct outputs (QLOAD-02 reuses GPTQ INT4 infrastructure)
  4. Marlin fused dequant+GEMM kernel is used for GPTQ/AWQ inference and delivers measurable throughput improvement over the naive dequant-then-matmul path (target 1.5-2x)
**Plans:** 3/4 plans executed

Plans:
- [ ] 01-00-PLAN.md — Wave 0: failing test stubs for all 9 Phase 1 behaviors (Nyquist compliance)
- [ ] 01-01-PLAN.md — HuggingFace Hub download client with caching and SHA256 verification
- [ ] 01-02-PLAN.md — GPTQ/AWQ weight loading from HuggingFace safetensors format
- [ ] 01-03-PLAN.md — Marlin fused dequant+GEMM CUDA kernel with model-load activation

### Phase 01.1: CUDA Graph Capture and Candle Fork Patches (INSERTED)

**Goal:** Fix CUDA graph capture (`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`) which is the single biggest performance gap vs vLLM (30-40%). Patch the candle fork to add missing I32/I16 CUDA kernels (`const_set_i32`, `const_set_i16`, `cast_u32_i32`, I32 strided copy) to eliminate CPU-first workarounds.
**Requirements**: CUDA-GRAPH-01, CUDA-GRAPH-02, CUDA-GRAPH-03, CUDA-GRAPH-04, KERNEL-I32-01, KERNEL-I32-02, KERNEL-I16-01, BINARY-I32-01, LOADER-01
**Depends on:** Phase 1
**Plans:** 3/3 plans complete

Plans:
- [x] 01.1-00-PLAN.md — Wave 0: failing test stubs for CUDA kernel and graph capture requirements (Nyquist compliance)
- [x] 01.1-01-PLAN.md — Candle fork kernel patches: I32/I16 fill, cast, unary, binary CUDA kernels + Rust dispatch
- [x] 01.1-02-PLAN.md — CUDA graph config, partial capture, and engine loop CPU-workaround elimination

### Phase 01.2: BF16/FP16 Serving Support (INSERTED)

**Goal:** Add automatic dtype downcasting so FP32 safetensors models can be served in BF16/FP16, halving VRAM usage (40GB -> 20GB for Qwen3-8B) and improving throughput via reduced memory bandwidth and H100 FP16 tensor cores.
**Requirements**: DTYPE-01, DTYPE-02, DTYPE-03, DTYPE-04, DTYPE-05, DTYPE-06
**Depends on:** Phase 1
**Plans:** 4/4 plans complete

Plans:
- [ ] 01.2-00-PLAN.md — Wave 0: failing test stubs for all 6 DTYPE behaviors (Nyquist compliance)
- [ ] 01.2-01-PLAN.md — ServingDType enum, GPU auto-detection, --dtype CLI flag, config pipeline, KV cache coupling
- [ ] 01.2-02-PLAN.md — Dtype-parameterized weight loader, mixed precision norm/embed preservation, VRAM estimation
- [ ] 01.2-03-PLAN.md — Gap closure: weight_dtype_bytes in VRAM profiling + savings INFO log

### Phase 2: Performance Optimization
**Goal**: The inference hot path runs with zero per-request GPU allocations, fused LayerNorm+linear, and parallel tokenization — maximizing tokens-per-second on RTX 3060
**Depends on**: Phase 1
**Requirements**: KERN-01, KERN-02, MOPT-01, MOPT-02, MOPT-03, TOKN-01, TOKN-02
**Success Criteria** (what must be TRUE):
  1. Fused LayerNorm+linear kernel is active for CUDA backend via `KernelBackend` trait; CPU and Metal backends fall back to unfused path without code changes
  2. The engine makes zero CUDA allocation calls during a steady-state forward pass (pre-allocated arena and buffer pool cover all temporary tensors)
  3. Concurrent requests tokenize in parallel — throughput scales with batch size without a serialization bottleneck
  4. Tokenizer is compiled/cached at startup; repeated identical strings hit the cache rather than re-running tokenization
**Plans:** 4 plans

Plans:
- [ ] 02-00-PLAN.md — Wave 0: failing test stubs for all Phase 2 behaviors (Nyquist compliance)
- [ ] 02-01-PLAN.md — Fused LayerNorm+linear CUDA kernel with KernelBackend trait wiring
- [ ] 02-02-PLAN.md — Arena allocator and tensor buffer pool for zero-alloc inference
- [ ] 02-03-PLAN.md — Cached tokenizer with LRU cache and parallel batch encoding

### Phase 3: Guided Decoding
**Goal**: Clients can request constrained generation via JSON Schema or regex and receive outputs that conform to the constraint with less than 5% overhead vs unconstrained
**Depends on**: Phase 1
**Requirements**: GDEC-01, GDEC-02, GDEC-03, GDEC-04
**Success Criteria** (what must be TRUE):
  1. A `response_format: {type: "json_schema", json_schema: {...}}` request produces output that validates against the schema — every token is constrained, not just post-processed
  2. A regex pattern submitted via the API compiles to a DFA and every generated token is a valid continuation of the DFA state
  3. Constrained generation adds less than 5% latency overhead compared to the same model running unconstrained on identical prompts
**Plans:** 5 plans

Plans:
- [x] 03-00-PLAN.md — Wave 0: outlines-core dependency and failing test stubs for all 4 GDEC behaviors
- [x] 03-01-PLAN.md — Core guided decoding module (GuidedState, IndexCache, apply_guided_mask) and engine loop integration
- [x] 03-02-PLAN.md — Server-side ResponseFormat to GuidedConstraint wiring in OpenAI routes
- [ ] 03-03-PLAN.md — LRU eviction for IndexCache with Prometheus cache metrics
- [ ] 03-04-PLAN.md — Unified guided endpoint, configurable error behavior, and stop token override

### Phase 4: Production Infrastructure
**Goal**: The server runs across multiple workers with shared weights, reads config from TOML, safely isolates cache across tenants, and serves embedding vectors via the standard OpenAI endpoint
**Depends on**: Phase 2
**Requirements**: WORK-01, WORK-02, WORK-03, CONF-01, CONF-02, PCCH-01, PCCH-02, EMBD-01, EMBD-02, EMBD-03
**Success Criteria** (what must be TRUE):
  1. `--workers N` spawns N inference workers sharing memory-mapped model weights; requests are distributed across workers and all workers serve correct outputs
  2. A `crabinfer.toml` file configures the server without any CLI flags; CLI flags override TOML values when both are present
  3. Cache salting is active by default — requests from different tenants (different salt values) cannot share cached KV blocks
  4. `POST /v1/embeddings` returns vectors in OpenAI format for both single and batched inputs, using a loaded embedding model (nomic-embed or gte-small)
**Plans:** 4 plans

Plans:
- [ ] 04-01-PLAN.md — TOML config file support with CLI/env/TOML precedence
- [ ] 04-02-PLAN.md — Prefix cache salting for tenant isolation and cache-aware routing API
- [ ] 04-03-PLAN.md — OpenAI-compatible /v1/embeddings endpoint with batch support
- [ ] 04-04-PLAN.md — Multi-worker serving with shared weights and round-robin routing

### Phase 5: Wiring Fixes (SHA-256 + Cache-Aware Routing)
**Goal**: Close integration gaps: wire verify_sha256 into the serving download path (MLOAD-03) and consume block_hashes in WorkerPool routing (WORK-03)
**Depends on**: Phase 1, Phase 4
**Requirements**: MLOAD-03, WORK-03
**Gap Closure**: Closes gaps from v1.0 milestone audit
**Success Criteria** (what must be TRUE):
  1. `ensure_model_cached` calls `verify_sha256` on every downloaded safetensors file — a corrupted file is rejected with a clear error
  2. `WorkerPool::submit()` uses `block_hashes()` to route requests to the worker with the best prefix match when cache-aware routing is enabled
**Plans:** 1/2 plans executed

Plans:
- [ ] 05-01-PLAN.md — Wire verify_sha256 into ensure_model_cached with LFS SHA-256 from HF API
- [ ] 05-02-PLAN.md — Cache-aware routing policy for WorkerPool using block_hashes()

### Phase 6: Embedding Model Loader
**Goal**: Support loading dedicated embedding models (nomic-embed, gte-small) so POST /v1/embeddings returns real encoder-quality vectors
**Depends on**: Phase 4
**Requirements**: EMBD-02
**Gap Closure**: Closes gaps from v1.0 milestone audit
**Success Criteria** (what must be TRUE):
  1. `crabinfer serve --model nomic-ai/nomic-embed-text-v1.5` downloads and loads the model without errors
  2. `POST /v1/embeddings` with that model returns semantically meaningful 768-dim vectors (cosine similarity between related texts > 0.5)
**Plans:** 2/2 plans complete

Plans:
- [ ] 06-01-PLAN.md — BERT/NomicBert embedding model runners with architecture detection and safetensors dispatch
- [ ] 06-02-PLAN.md — Engine wiring: bypass PagedAttention for encoder-only models, wire embed() through real encoder

### Phase 7: Server Wiring Last-Mile
**Goal**: Make the existing infrastructure for cache-aware routing, cache salting, batch tokenization, and bare text completions reachable from the user-facing API. All four features have working core implementations but lack the final server-side wiring.
**Depends on**: Phase 5, Phase 6
**Requirements**: WORK-03, PCCH-01, TOKN-01
**Gap Closure**: Closes 4 gaps from v1.0 milestone audit (WORK-03 partial, PCCH-01 partial, TOKN-01 partial, missing /v1/completions route)
**Success Criteria** (what must be TRUE):
  1. `crabinfer serve --routing-policy cache-aware --workers 4` activates `WorkerPool::new_with_policy(CacheAware)` and routes requests by KV-cache prefix match
  2. A request body containing `"cache_salt": "tenant-foo"` results in a `BlockHash` namespaced to that tenant — different salts cannot share cache blocks via the API
  3. `POST /v1/embeddings` with a multi-input request (`"input": ["a", "b", "c"]`) calls `WorkerPool::encode_batch()` once instead of three serial `encode()` calls
  4. `POST /v1/completions` accepts `{"model": "...", "prompt": "..."}` (OpenAI legacy format) and returns a completion — handler exists and route is registered

**Plans:** 4 plans

Plans:
- [ ] 07-01-PLAN.md — Wire routing_policy field + WorkerPool::new_with_policy (WORK-03)
- [ ] 07-02-PLAN.md — Propagate cache_salt from ChatCompletionRequest/MessagesRequest to SamplingParams (PCCH-01)
- [ ] 07-03-PLAN.md — Add /v1/completions handler, types, and route registration (missing flow + PCCH-01)
- [ ] 07-04-PLAN.md — TOKN-01 call-chain verification tests (audit was stale; test + document existing wiring)

### Phase 8: Fused Kernel Coverage
**Goal**: Make the fused LayerNorm+linear CUDA kernel benefit DeepSeek, Mistral, and Phi3 models by wiring `forward_linear_fused()` into their final output paths (currently only Llama uses it).
**Depends on**: Phase 2
**Requirements**: KERN-01
**Gap Closure**: Closes KERN-01 partial gap from v1.0 milestone audit
**Success Criteria** (what must be TRUE):
  1. DeepSeek, Mistral, and Phi3 model `forward()` methods call `RmsNorm::forward_linear_fused()` for their final norm+lm_head step (matching Llama's pattern at `llama.rs:364`)
  2. CUDA backend uses the fused kernel for these models; CPU/Metal backends fall back to the unfused path via the existing default `KernelBackend` implementation
  3. Output token IDs match the unfused path within numerical tolerance (no quality regression)
**Plans:** 1 plan

Plans:
- [ ] 08-01-PLAN.md — Port forward_linear_fused into DeepSeek/Mistral/Phi3 final norm+lm_head step with per-model parity tests

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 1.1 -> 1.2 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Model Loading and Quantization | 4/4 | Complete | 2026-03-13 |
| 1.1 CUDA Graph Capture + Candle Patches | 3/3 | Complete (needs H100 validation) | 2026-04-02 |
| 1.2 BF16/FP16 Serving Support | 4/4 | Complete | 2026-04-04 |
| 2. Performance Optimization | 4/4 | Complete | 2026-04-04 |
| 3. Guided Decoding | 5/5 | Complete | 2026-04-04 |
| 4. Production Infrastructure | 4/4 | Complete | 2026-04-04 |
| 5. Wiring Fixes | 2/2 | Complete | 2026-04-04 |
| 6. Embedding Model Loader | 2/2 | Complete | 2026-04-04 |
| 7. Server Wiring Last-Mile | 0/4 | Planned | - |
| 8. Fused Kernel Coverage | 0/1 | Planned | - |
