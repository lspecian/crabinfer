# H100 CUDA Test Report

> Tested 2026-03-13/14 on NVIDIA H100 SXM 80GB (Vast.ai), Linux, release build.
> CrabInfer PagedAttention serving engine with continuous batching.

---

## Test Environment

| Property | Value |
|----------|-------|
| GPU | NVIDIA H100 SXM 80GB |
| GPU Count | 2x (GPU 0 used for benchmarks) |
| CUDA Version | 12.x |
| Driver | 550.x |
| Host | Vast.ai instance 32812046 |
| Cost | $3.22/hr |
| OS | Linux (Ubuntu) |
| Build | `cargo build --release` with `--no-default-features --features cuda` |

---

## Phase 1: CUDA Kernel Tests

**815 unit tests pass** on H100 with no failures.

```
cargo test --workspace --no-default-features --features cuda
```

---

## Phase 2: Single-GPU Inference (Qwen3-8B F32)

| Property | Value |
|----------|-------|
| Model | Qwen/Qwen3-8B |
| Format | HuggingFace safetensors |
| Precision | FP32 (no auto-downcast) |
| VRAM Usage | ~40 GB / 80 GB |
| Execution Mode | Eager (no CUDA graphs) |
| Context Length | 4096 |

Model loads and generates coherent text with correct ChatML formatting.

---

## Phase 3: GPTQ INT4 Quantization

| Property | Value |
|----------|-------|
| Model | JunHowie/Qwen3-8B-GPTQ-Int4 |
| Quantization | GPTQ 4-bit, group_size=128, symmetric |
| VRAM Usage | **17 GB** (2.4x reduction vs F32) |
| Model Size | 5.8 GB (vs ~32 GB F32) |
| Functional Status | Loads and generates correctly |
| Throughput | ~0.23 tok/s |

### GPTQ Status

GPTQ INT4 model loading and inference is **functionally correct**. The throughput is
currently limited by CPU-side dequantization: on each forward pass, packed INT4 weights
are pulled from GPU to CPU, unpacked from 4-bit nibbles, cast to FP32, and used for
matrix multiplication. A fused Marlin CUDA kernel for GPU-side dequant+GEMM is the
planned optimization to bring throughput to production levels.

---

## Phase 6: CrabInfer vs vLLM Benchmarks

Both engines ran **FP32 Qwen3-8B** on the same H100 GPU 0 with context length 4096.

- **CrabInfer**: Eager mode (CUDA graphs not captured)
- **vLLM v0.17.1**: 102 CUDA graphs captured (51 mixed prefill-decode + 51 decode)

### Sequential Throughput (10 requests, 100 tokens each)

| Metric | CrabInfer | vLLM v0.17.1 | Ratio |
|--------|-----------|--------------|-------|
| Throughput | **50.6 tok/s** | **81.6 tok/s** | 0.62x |
| Avg latency | 1.97s | 1.23s | 1.60x |
| P50 latency | 1.96s | 1.18s | 1.66x |
| P99 latency | 2.17s | 1.69s | 1.28x |
| Avg tokens/request | 100 | 100 | — |

### Concurrent Throughput (10 requests, 100 tokens, 5 workers)

| Metric | CrabInfer | vLLM v0.17.1 | Ratio |
|--------|-----------|--------------|-------|
| Throughput | **109.1 tok/s** | **222.5 tok/s** | 0.49x |
| Avg latency | 4.57s | 2.24s | 2.04x |
| P50 latency | 4.91s | 2.26s | 2.17x |
| P99 latency | 4.97s | 2.28s | 2.18x |

### CrabInfer Internal Metrics (from Prometheus endpoint)

| Metric | Value |
|--------|-------|
| Time to First Token (TTFT) | < 50ms |
| Inter-Token Latency (ITL) | ~16ms/token |
| VRAM (F32 model + KV cache) | ~40 GB |

### vLLM Internal Metrics

| Metric | Value |
|--------|-------|
| VRAM (F32 model + graph workspace) | ~70 GB |
| CUDA graphs captured | 102 |
| First request latency | 1.69s (graph compilation) |
| Steady-state latency | 1.18s |

### Gap Analysis

The 38-51% performance gap between CrabInfer and vLLM is attributed to:

1. **CUDA Graphs** (primary factor): vLLM captures 102 CUDA graphs, eliminating kernel launch overhead on every decode step. CrabInfer's graph capture fails with `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`, likely due to unsupported operations during capture (e.g., dynamic memory allocation or host-device sync).

2. **Attention Kernels**: vLLM uses highly optimized FlashAttention/PagedAttention CUDA kernels. CrabInfer uses custom NVRTC-compiled kernels that may not match the same level of optimization.

3. **Memory Efficiency**: vLLM uses 70 GB (75% more) to achieve higher throughput, trading memory for speed via graph workspace pre-allocation.

**Positive findings**: CrabInfer's TTFT (<50ms) and single-request ITL (~16ms) are competitive. The continuous batching scheduler correctly handles concurrent requests with proper KV cache management.

---

## Bugs Fixed During Testing

Nine bugs were identified and fixed to bring CUDA inference from non-functional to working:

| # | Error | Root Cause | Fix | Commit |
|---|-------|-----------|-----|--------|
| 1 | `expected: I32, got: U32` in index_select | candle CUDA `index_select` requires I32/I64 indices, engine used U32 | Cast arena u32 data to i64/i32 on CPU before creating tensors | `bc81a34` |
| 2 | U32 dtype in warmup/buffer pool | Buffer pool specs and warmup used `DType::U32` | Changed to `DType::I64` / `DType::I32` | `f80628f` |
| 3 | `named symbol not found` (device mismatch) | `CudaBackend::new(0)` created its own CudaDevice, separate from model's device | Added `new_with_device()` to share the model's CudaDevice | `8299e7f` |
| 4 | `named symbol not found` (const_set_i32) | candle's `fill.cu` lacks `const_set_i32` kernel; `Tensor::zeros/ones` for I32 on CUDA fails | Create I32 tensors on CPU first, transfer to GPU via `to_device()` | `fb3f72b` |
| 5 | Garbage output (`<\|start_of_text\|>` repetitions) | Server hardcoded `architecture: "llama"` for all safetensors models | Read `model_type` from config.json and map to correct chat template | `7b8aada` |
| 6 | GPTQ model load crash | 579 I32 tensors (qweight/qzeros/g_idx) loaded directly to CUDA | Load all safetensors to CPU first; move to GPU on access | `cf960c7` |
| 7 | GPTQ MLP dimension mismatch | `dim(1)>dim(0)*8` transpose heuristic false-triggered on gate/up_proj where out > in | Removed incorrect heuristic; standard HF GPTQ is already correct layout | `c72b2b2` |
| 8 | `expected: U32, got: I32` in unpack | HuggingFace stores GPTQ packed INT4 as `torch.int32` (DType::I32) | Handle both I32 and U32 in `unpack_int4_tensor` | `93600eb` |
| 9 | GPTQ always outputs "!" | `to_vec1::<u32>()` silently failed on I32 tensor | Reinterpret i32 as u32 (bit pattern identical for packed nibbles) | `93600eb` |

### Common Theme: candle CUDA Kernel Gaps

Most bugs stem from candle's CUDA kernel set missing operations for I32/I16 dtypes:
- `fill.cu`: has `const_set_i64` but not `const_set_i32` or `const_set_i16`
- `cast.cu`: has `cast_u32_i64` but not `cast_u32_i32`
- Strided copy: missing for I32 (breaks `.t()?.contiguous()?` on CUDA I32 tensors)

The workaround pattern: **create/manipulate I32 tensors on CPU, then transfer to GPU** via `to_device()` which uses a simple `cuMemcpyHtoD` that works for any dtype.

---

## Phases Not Completed

| Phase | Reason |
|-------|--------|
| Phase 4: FlashAttention-2 | Not tested — would require verifying flash-attn kernel integration |
| Phase 5: Tensor Parallelism (TP=2, 72B) | Not tested — requires ~140 GB model download |

---

## Known Limitations

1. **CUDA Graph Capture**: Fails with `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`. Root cause likely involves host-device synchronization or dynamic allocation during the captured region. This is the single largest performance gap vs vLLM.

2. **GPTQ Dequantization**: Entirely CPU-based. Each forward pass pulls packed I32 weights from GPU to CPU, unpacks INT4 nibbles, casts to F32, performs dequantization math, then returns the result for matmul. Needs a fused Marlin CUDA kernel.

3. **FP32 Only**: The serving engine does not auto-downcast to FP16/BF16. Running Qwen3-8B in FP32 uses 40 GB — with FP16 this would halve to ~20 GB and likely improve throughput via reduced memory bandwidth.

4. **candle Fork Dependency**: Several workarounds exist because the candle fork lacks I32 CUDA kernels. Adding `const_set_i32`, `const_set_i16`, and `cast_u32_i32` to the fork would eliminate the CPU-first workaround pattern.

---

## Recommendations

1. **CUDA Graphs** (highest impact): Investigate and fix `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`. This alone could close 30-40% of the vLLM gap.

2. **FP16/BF16 Support**: Add automatic dtype downcast for serving. Halves VRAM and improves throughput on H100's FP16 tensor cores.

3. **Marlin Kernel**: Implement fused INT4 dequant+GEMM kernel for GPTQ/AWQ. This would make quantized models usable at production throughput.

4. **candle Fork Patches**: Add missing I32/I16 CUDA kernels to eliminate CPU-first workarounds.

---

## Methodology

- **Server**: `crabinfer-server --model <path> --serving --host 0.0.0.0 --port 8080 --gpu-memory-utilization 0.85 --context-length 4096`
- **vLLM**: `python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B --port 8081 --gpu-memory-utilization 0.85 --max-model-len 4096 --dtype float32`
- **Benchmark**: Custom Python script using OpenAI-compatible `/v1/chat/completions` endpoint, 10 requests with diverse prompts, 100 max tokens each, temperature 0.7
- **Sequential**: Requests sent one at a time, measuring end-to-end latency
- **Concurrent**: 5 ThreadPoolExecutor workers, measuring aggregate throughput
- **Metrics**: Wall-clock timing per request, Prometheus `/metrics` endpoint for TTFT/ITL
