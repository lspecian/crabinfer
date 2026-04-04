# Requirements: CrabInfer ROADMAP-D CUDA/Linux Track

**Defined:** 2026-03-12
**Core Value:** Ship every CUDA/Linux feature buildable on RTX 3060, closing gap with vLLM for single-GPU deployments

## v1 Requirements

### Quantization Loading

- [x] **QLOAD-01**: Server loads GPTQ models from HuggingFace format (reads `quantize_config.json`, loads packed INT4 qweight/qzeros/scales from safetensors)
- [x] **QLOAD-02**: Server loads AWQ models from AutoAWQ HuggingFace format (reads `quantize_config.json`, reuses GPTQ INT4 infrastructure)
- [x] **QLOAD-03**: Marlin-style fused dequant+GEMM CUDA kernel for GPTQ/AWQ (target 1.5-2x speedup over naive dequant→matmul)

### Model Loading

- [x] **MLOAD-01**: `--model` flag accepts HuggingFace repo ID (e.g., `meta-llama/Llama-3.1-8B-Instruct`)
- [x] **MLOAD-02**: Auto-download from HuggingFace Hub with local caching (~/.cache/crabinfer/)
- [ ] **MLOAD-03**: Resume interrupted downloads, verify file integrity (SHA256)

### Guided Decoding

- [x] **GDEC-01**: JSON Schema constrained generation — generate CFG from JSON Schema, mask invalid tokens during sampling
- [x] **GDEC-02**: Regex constrained generation — compile regex to DFA, mask invalid tokens at each step
- [x] **GDEC-03**: `response_format` with `json_schema` uses token-level constraints (not just prompt-based)
- [x] **GDEC-04**: Constrained generation adds <5% overhead vs unconstrained

### CUDA Kernel Fusion

- [ ] **KERN-01**: Fused LayerNorm+linear CUDA kernel (combined norm + matmul in single kernel pass)
- [ ] **KERN-02**: Kernel wired into model forward via `KernelBackend` trait with unfused default fallback

### Memory Optimization

- [ ] **MOPT-01**: Arena allocator for per-request temporary tensor allocations
- [ ] **MOPT-02**: Tensor buffer pool — reuse GPU memory buffers across forward passes (pre-allocate at init)
- [ ] **MOPT-03**: Minimize CUDA allocation calls during inference (zero allocs in hot path)

### Tokenization

- [ ] **TOKN-01**: Parallel tokenization across requests using Rust async (no GIL bottleneck)
- [ ] **TOKN-02**: Tokenizer compilation/caching — compile tokenizer once at startup, reuse across requests

### Multi-Worker

- [ ] **WORK-01**: Spawn N inference workers behind internal load balancer
- [ ] **WORK-02**: Shared model weights via memory mapping (mmap) across workers
- [x] **WORK-03**: Request routing: round-robin or cache-aware (route to worker with best prefix match)

### Configuration

- [ ] **CONF-01**: TOML config file support (`crabinfer.toml`) as alternative to CLI flags
- [ ] **CONF-02**: Config file precedence: CLI flags > env vars > TOML file > defaults

### Prefix Cache

- [x] **PCCH-01**: Cache salting — per-request salt to isolate tenants (prevent cross-user cache hits)
- [x] **PCCH-02**: Cache-aware request routing — expose block hashes via API for multi-instance deployments

### Embeddings

- [ ] **EMBD-01**: `POST /v1/embeddings` endpoint returning vectors in OpenAI format
- [ ] **EMBD-02**: Load embedding models (nomic-embed, gte-small) in serving engine
- [ ] **EMBD-03**: Batch embedding support (multiple inputs in single request)

## v2 Requirements

### Quantization (Hopper+)
- **FP8-01**: FP8 (E4M3) weight storage and computation (requires H100/H200)
- **FP8-02**: FP8 KV cache option (4x more context vs FP16)

### Distributed (Multi-GPU)
- **DIST-01**: Tensor parallelism via NCCL
- **DIST-02**: Pipeline parallelism
- **DIST-03**: Expert parallelism (MoE)
- **DIST-04**: Disaggregated prefill/decode

### Advanced Features
- **VIS-01**: Multimodal/vision support
- **SPEC-01**: Eagle-3 / Medusa speculative heads
- **LORA-01**: LoRA adapter serving

## Out of Scope

| Feature | Reason |
|---------|--------|
| FP8 quantization | Requires Hopper+ GPU (H100/H200), not testable on RTX 3060 |
| Tensor/pipeline/expert parallelism | Requires multi-GPU, only have single RTX 3060 |
| Disaggregated prefill/decode | Requires multi-GPU pools |
| Multimodal/vision | Separate large effort, not CUDA-specific |
| Eagle-3/Medusa speculative heads | Separate large effort |
| LoRA serving | Separate large effort |
| Metal optimizations (D11.1) | Mac track, not Linux |
| Ray integration | Rust tokio handles coordination natively |
| Python API | Users hit HTTP API |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| QLOAD-01 | Phase 1 | Complete |
| QLOAD-02 | Phase 1 | Complete |
| QLOAD-03 | Phase 1 | Complete |
| MLOAD-01 | Phase 1 | Complete |
| MLOAD-02 | Phase 1 | Complete |
| MLOAD-03 | Phase 5 | Pending |
| GDEC-01 | Phase 3 | Complete |
| GDEC-02 | Phase 3 | Complete |
| GDEC-03 | Phase 3 | Complete |
| GDEC-04 | Phase 3 | Complete |
| KERN-01 | Phase 2 | Complete |
| KERN-02 | Phase 2 | Complete |
| MOPT-01 | Phase 2 | Complete |
| MOPT-02 | Phase 2 | Complete |
| MOPT-03 | Phase 2 | Complete |
| TOKN-01 | Phase 2 | Complete |
| TOKN-02 | Phase 2 | Complete |
| WORK-01 | Phase 4 | Complete |
| WORK-02 | Phase 4 | Complete |
| WORK-03 | Phase 5 | Complete |
| CONF-01 | Phase 4 | Complete |
| CONF-02 | Phase 4 | Complete |
| PCCH-01 | Phase 4 | Complete |
| PCCH-02 | Phase 4 | Complete |
| EMBD-01 | Phase 4 | Complete |
| EMBD-02 | Phase 6 | Pending |
| EMBD-03 | Phase 4 | Complete |

**Coverage:**
- v1 requirements: 27 total
- Mapped to phases: 27
- Complete: 24
- Pending (gap closure): 3 (MLOAD-03, WORK-03, EMBD-02)

---
*Requirements defined: 2026-03-12*
*Last updated: 2026-04-04 — gap closure phases 5-6 added for 3 remaining requirements*
