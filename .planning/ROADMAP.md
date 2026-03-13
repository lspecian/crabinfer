# Roadmap: CrabInfer ROADMAP-D CUDA/Linux Track

## Overview

Complete the remaining CUDA/Linux features for CrabInfer's PagedAttention serving engine, targeting RTX 3060 (6GB VRAM). Starting from a solid foundation of paged attention, CUDA graphs, fused kernels, and quantization infrastructure, this roadmap delivers HuggingFace model loading, Marlin kernel performance, guided decoding, memory/tokenization optimization, and production infrastructure — closing the gap with vLLM for single-GPU deployments.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Model Loading and Quantization** - Load GPTQ/AWQ models from HuggingFace Hub with Marlin fused kernel
- [ ] **Phase 2: Performance Optimization** - Fused LayerNorm+linear kernel, memory pool, and fast tokenization
- [ ] **Phase 3: Guided Decoding** - Token-level constrained generation via JSON Schema and regex DFA
- [ ] **Phase 4: Production Infrastructure** - Multi-worker serving, TOML config, prefix cache salting, and embeddings endpoint

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
**Plans**: TBD

### Phase 4: Production Infrastructure
**Goal**: The server runs across multiple workers with shared weights, reads config from TOML, safely isolates cache across tenants, and serves embedding vectors via the standard OpenAI endpoint
**Depends on**: Phase 2
**Requirements**: WORK-01, WORK-02, WORK-03, CONF-01, CONF-02, PCCH-01, PCCH-02, EMBD-01, EMBD-02, EMBD-03
**Success Criteria** (what must be TRUE):
  1. `--workers N` spawns N inference workers sharing memory-mapped model weights; requests are distributed across workers and all workers serve correct outputs
  2. A `crabinfer.toml` file configures the server without any CLI flags; CLI flags override TOML values when both are present
  3. Cache salting is active by default — requests from different tenants (different salt values) cannot share cached KV blocks
  4. `POST /v1/embeddings` returns vectors in OpenAI format for both single and batched inputs, using a loaded embedding model (nomic-embed or gte-small)
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Model Loading and Quantization | 3/4 | In Progress|  |
| 2. Performance Optimization | 0/4 | Planned | - |
| 3. Guided Decoding | 0/TBD | Not started | - |
| 4. Production Infrastructure | 0/TBD | Not started | - |
