// CrabInfer CUDA kernels for paged attention operations.
//
// Three kernels:
//   1. reshape_and_cache — write K/V tokens into paged cache
//   2. copy_blocks — copy KV cache blocks for prefix sharing
//   3. paged_attention_v1 — single-partition decode attention
//
// Layout conventions match vLLM:
//   Key cache:   [num_blocks, num_kv_heads, head_size/x, block_size, x]
//   Value cache: [num_blocks, num_kv_heads, head_size, block_size]
//   where x = 16 / sizeof(scalar_t)

#include <cuda_fp16.h>

// NVRTC doesn't bundle standard C headers, so define FLT_MAX directly.
#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38F
#endif

// ─── reshape_and_cache ────────────────────────────────────────────────────

// Each threadgroup handles one token. Threads within the group handle
// different (head, dim) elements in parallel.

template <typename scalar_t>
__global__ void reshape_and_cache_kernel(
    const scalar_t* __restrict__ key,       // [num_tokens, num_heads * head_size]
    const scalar_t* __restrict__ value,     // [num_tokens, num_heads * head_size]
    scalar_t* __restrict__ key_cache,       // [num_blocks, num_heads, head_size/x, block_size, x]
    scalar_t* __restrict__ value_cache,     // [num_blocks, num_heads, head_size, block_size]
    const int* __restrict__ slot_mapping,   // [num_tokens]
    int key_stride,
    int value_stride,
    int num_heads,
    int head_size,
    int block_size,
    int x
) {
    const int token_idx = blockIdx.x;
    const int slot = slot_mapping[token_idx];
    const int block_idx = slot / block_size;
    const int block_offset = slot % block_size;

    const int n = num_heads * head_size;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int head = i / head_size;
        const int d = i % head_size;
        const int d_outer = d / x;
        const int d_inner = d % x;

        const int src_idx = token_idx * key_stride + i;

        // Key cache: [block, head, d_outer, block_offset, d_inner]
        const int kc_head_stride = (head_size / x) * block_size * x;
        const int kc_block_stride = num_heads * kc_head_stride;
        const int kc_idx = block_idx * kc_block_stride
                         + head * kc_head_stride
                         + d_outer * block_size * x
                         + block_offset * x
                         + d_inner;
        key_cache[kc_idx] = key[src_idx];

        // Value cache: [block, head, d, block_offset]
        const int vc_head_stride = head_size * block_size;
        const int vc_block_stride = num_heads * vc_head_stride;
        const int vc_idx = block_idx * vc_block_stride
                         + head * vc_head_stride
                         + d * block_size
                         + block_offset;
        value_cache[vc_idx] = value[src_idx];
    }
}

extern "C" __global__ void reshape_and_cache_f32(
    const float* key, const float* value,
    float* key_cache, float* value_cache,
    const int* slot_mapping,
    int key_stride, int value_stride,
    int num_heads, int head_size, int block_size, int x
) {
    reshape_and_cache_kernel<float>(
        key, value, key_cache, value_cache, slot_mapping,
        key_stride, value_stride, num_heads, head_size, block_size, x
    );
}

extern "C" __global__ void reshape_and_cache_f16(
    const __half* key, const __half* value,
    __half* key_cache, __half* value_cache,
    const int* slot_mapping,
    int key_stride, int value_stride,
    int num_heads, int head_size, int block_size, int x
) {
    reshape_and_cache_kernel<__half>(
        key, value, key_cache, value_cache, slot_mapping,
        key_stride, value_stride, num_heads, head_size, block_size, x
    );
}

// ─── copy_blocks ──────────────────────────────────────────────────────────

template <typename scalar_t>
__global__ void copy_blocks_kernel(
    scalar_t* __restrict__ key_cache,
    scalar_t* __restrict__ value_cache,
    const int* __restrict__ block_mapping,  // [num_pairs * 2]: src0, dst0, src1, dst1, ...
    int numel_per_block
) {
    const int pair_idx = blockIdx.x;
    const int src_block = block_mapping[pair_idx * 2];
    const int dst_block = block_mapping[pair_idx * 2 + 1];

    const int src_start = src_block * numel_per_block;
    const int dst_start = dst_block * numel_per_block;

    for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
        key_cache[dst_start + i] = key_cache[src_start + i];
        value_cache[dst_start + i] = value_cache[src_start + i];
    }
}

extern "C" __global__ void copy_blocks_f32(
    float* key_cache, float* value_cache,
    const int* block_mapping, int numel_per_block
) {
    copy_blocks_kernel<float>(key_cache, value_cache, block_mapping, numel_per_block);
}

extern "C" __global__ void copy_blocks_f16(
    __half* key_cache, __half* value_cache,
    const int* block_mapping, int numel_per_block
) {
    copy_blocks_kernel<__half>(key_cache, value_cache, block_mapping, numel_per_block);
}

// ─── paged_attention_v1 ───────────────────────────────────────────────────
//
// Single-partition paged attention for decode (1 query token per sequence).
//
// Grid: (num_heads, num_seqs, 1)
// Block: (256, 1, 1)
//
// Each threadblock handles one (sequence, head) pair:
//   1. Load query vector for this head
//   2. Iterate over all cached tokens via block table
//   3. Compute dot products (attention scores)
//   4. Online softmax reduction
//   5. Weighted sum of values → output

template <typename scalar_t>
__global__ void paged_attention_v1_kernel(
    scalar_t* __restrict__ output,          // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ query,     // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ key_cache,
    const scalar_t* __restrict__ value_cache,
    const int* __restrict__ block_tables,   // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ context_lens,   // [num_seqs]
    int num_kv_heads,
    float scale,
    int max_blocks_per_seq,
    int q_stride,                           // num_heads * head_size
    int kv_block_stride,                    // num_kv_heads * head_size * block_size
    int kv_head_stride,                     // head_size * block_size
    int head_size,
    int block_size
) {
    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int num_heads = gridDim.x;
    const int tid = threadIdx.x;
    const int warp_size = 32;

    const int ctx_len = context_lens[seq_idx];
    if (ctx_len == 0) return;

    const int gqa_ratio = num_heads / num_kv_heads;
    const int kv_head = head_idx / gqa_ratio;

    // Load query vector into registers (each thread handles a subset of dims)
    // For head_size <= 256, each thread handles ceil(head_size/256) dims.
    float q_vals[4];  // max 4 dims per thread (head_size up to 1024)
    const int dims_per_thread = (head_size + blockDim.x - 1) / blockDim.x;

    const int q_base = seq_idx * q_stride + head_idx * head_size;
    for (int i = 0; i < dims_per_thread; i++) {
        const int d = tid * dims_per_thread + i;
        if (d < head_size) {
            q_vals[i] = (float)query[q_base + d];
        } else {
            q_vals[i] = 0.0f;
        }
    }

    // Compute attention scores for all cached tokens.
    // Use online softmax: track running max and exp sum.
    float max_score = -FLT_MAX;
    float exp_sum = 0.0f;
    float acc[4] = {0.0f};  // accumulated output per dim

    const int x = 16 / sizeof(scalar_t);

    for (int tok = 0; tok < ctx_len; tok++) {
        const int block_idx = tok / block_size;
        const int block_offset = tok % block_size;
        const int physical_block = block_tables[seq_idx * max_blocks_per_seq + block_idx];

        // Compute dot product Q · K for this token
        float dot = 0.0f;
        for (int i = 0; i < dims_per_thread; i++) {
            const int d = tid * dims_per_thread + i;
            if (d < head_size) {
                const int d_outer = d / x;
                const int d_inner = d % x;
                const int k_idx = physical_block * kv_block_stride
                                + kv_head * kv_head_stride
                                + d_outer * block_size * x
                                + block_offset * x
                                + d_inner;
                dot += q_vals[i] * (float)key_cache[k_idx];
            }
        }

        // Warp-level reduction of dot product
        for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);
        }

        // Cross-warp reduction via shared memory
        __shared__ float warp_dots[8];  // max 256/32 = 8 warps
        const int warp_id = tid / warp_size;
        const int lane_id = tid % warp_size;

        if (lane_id == 0) {
            warp_dots[warp_id] = dot;
        }
        __syncthreads();

        // Thread 0 reduces across warps
        float score = 0.0f;
        if (tid == 0) {
            const int num_warps = (blockDim.x + warp_size - 1) / warp_size;
            for (int w = 0; w < num_warps; w++) {
                score += warp_dots[w];
            }
            score *= scale;
            warp_dots[0] = score;
        }
        __syncthreads();
        score = warp_dots[0];

        // Online softmax update
        float old_max = max_score;
        if (score > max_score) max_score = score;
        float exp_score = expf(score - max_score);
        float correction = expf(old_max - max_score);
        exp_sum = exp_sum * correction + exp_score;

        // Correct accumulated output and add weighted value
        for (int i = 0; i < dims_per_thread; i++) {
            const int d = tid * dims_per_thread + i;
            if (d < head_size) {
                acc[i] *= correction;
                const int v_idx = physical_block * kv_block_stride
                                + kv_head * kv_head_stride  // reuse strides (same total size)
                                + d * block_size
                                + block_offset;
                // Wait — value cache has different layout. Let me fix:
                // Value cache: [num_blocks, num_kv_heads, head_size, block_size]
                // Stride: num_kv_heads * head_size * block_size per block
                const int vc_block_stride = num_kv_heads * head_size * block_size;
                const int vc_head_stride = head_size * block_size;
                const int vc_idx = physical_block * vc_block_stride
                                 + kv_head * vc_head_stride
                                 + d * block_size
                                 + block_offset;
                acc[i] += exp_score * (float)value_cache[vc_idx];
            }
        }
    }

    // Normalize by exp_sum and write output
    if (exp_sum > 0.0f) {
        const int o_base = seq_idx * q_stride + head_idx * head_size;
        for (int i = 0; i < dims_per_thread; i++) {
            const int d = tid * dims_per_thread + i;
            if (d < head_size) {
                output[o_base + d] = (scalar_t)(acc[i] / exp_sum);
            }
        }
    }
}

extern "C" __global__ void paged_attention_v1_f32(
    float* output, const float* query,
    const float* key_cache, const float* value_cache,
    const int* block_tables, const int* context_lens,
    int num_kv_heads, float scale, int max_blocks_per_seq,
    int q_stride, int kv_block_stride, int kv_head_stride,
    int head_size, int block_size
) {
    paged_attention_v1_kernel<float>(
        output, query, key_cache, value_cache,
        block_tables, context_lens,
        num_kv_heads, scale, max_blocks_per_seq,
        q_stride, kv_block_stride, kv_head_stride,
        head_size, block_size
    );
}

extern "C" __global__ void paged_attention_v1_f16(
    __half* output, const __half* query,
    const __half* key_cache, const __half* value_cache,
    const int* block_tables, const int* context_lens,
    int num_kv_heads, float scale, int max_blocks_per_seq,
    int q_stride, int kv_block_stride, int kv_head_stride,
    int head_size, int block_size
) {
    paged_attention_v1_kernel<__half>(
        output, query, key_cache, value_cache,
        block_tables, context_lens,
        num_kv_heads, scale, max_blocks_per_seq,
        q_stride, kv_block_stride, kv_head_stride,
        head_size, block_size
    );
}

// ─── GPTQ INT4 Dequantization Kernel ────────────────────────────────────
//
// Dequantizes 4-bit packed weights to half-precision (F16) output.
//
// GPTQ packing: 8 INT4 values per u32, bits [3:0] = w0, bits [7:4] = w1, etc.
// Asymmetric group-wise: w_float = (w_int4 - zero) * scale
//
// Input layout:
//   qweight:  [in_features / 8, out_features] as u32
//   scales:   [num_groups, out_features] as half
//   qzeros:   [num_groups, out_features / 8] as u32
//
// Output layout:
//   output:   [out_features, in_features] as half  (row-major, standard linear weight format)
//
// Each thread dequantizes one element (out_j, in_i) of the weight matrix.

extern "C" __global__ void gptq_dequant_f16(
    __half* __restrict__ output,              // [out_features, in_features]
    const unsigned int* __restrict__ qweight, // [in_features / 8, out_features]
    const __half* __restrict__ scales,        // [num_groups, out_features]
    const unsigned int* __restrict__ qzeros,  // [num_groups, out_features / 8]
    int in_features,
    int out_features,
    int group_size
) {
    // Grid: (in_features, out_features)
    const int in_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (in_idx >= in_features || out_idx >= out_features) return;

    // Which group does this input feature belong to?
    const int group_idx = in_idx / group_size;

    // Unpack the 4-bit weight value
    const int pack_row = in_idx / 8;
    const int pack_bit = in_idx % 8;
    const unsigned int packed = qweight[pack_row * out_features + out_idx];
    const int w_int4 = (packed >> (pack_bit * 4)) & 0xF;

    // Unpack the 4-bit zero point
    const int zero_pack_col = out_idx / 8;
    const int zero_bit = out_idx % 8;
    const unsigned int zero_packed = qzeros[group_idx * (out_features / 8) + zero_pack_col];
    const int zero = (zero_packed >> (zero_bit * 4)) & 0xF;

    // Get the scale for this group + output channel
    const __half scale = scales[group_idx * out_features + out_idx];

    // Dequantize: w_float = (w_int4 - zero) * scale
    const float w_float = (float)(w_int4 - zero) * __half2float(scale);

    // Write to output: [out_features, in_features] row-major
    output[out_idx * in_features + in_idx] = __float2half(w_float);
}

// ─── Fused SiLU + Element-wise Multiply Kernel ──────────────────────────
//
// Computes: output[i] = silu(gate[i]) * up[i]
// where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// This fuses three operations into one kernel:
//   1. SiLU activation on the gate projection
//   2. Element-wise multiply with the up projection
//   3. Writing the result to output
//
// Eliminates 2 intermediate tensor allocations vs the unfused path.
//
// Grid: 1D, one thread per element.

template <typename scalar_t>
__global__ void fused_silu_mul_kernel(
    scalar_t* __restrict__ output,        // [num_tokens, hidden_size]
    const scalar_t* __restrict__ gate,    // [num_tokens, hidden_size]
    const scalar_t* __restrict__ up,      // [num_tokens, hidden_size]
    int total_elements
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float g = (float)gate[idx];
    float u = (float)up[idx];

    // silu(g) = g * sigmoid(g) = g / (1 + exp(-g))
    float silu_g = g / (1.0f + expf(-g));

    output[idx] = (scalar_t)(silu_g * u);
}

extern "C" __global__ void fused_silu_mul_f32(
    float* output, const float* gate, const float* up, int total_elements
) {
    fused_silu_mul_kernel<float>(output, gate, up, total_elements);
}

extern "C" __global__ void fused_silu_mul_f16(
    __half* output, const __half* gate, const __half* up, int total_elements
) {
    fused_silu_mul_kernel<__half>(output, gate, up, total_elements);
}

// ─── Fused RoPE Kernel ──────────────────────────────────────────────────
//
// Applies rotary positional embeddings in-place on Q or K tensors
// with layout [total_tokens, num_heads, head_size].
//
// The standard RoPE formula for pairs (x[2i], x[2i+1]):
//   y[2i]   = x[2i]   * cos[pos, i] - x[2i+1] * sin[pos, i]
//   y[2i+1] = x[2i+1] * cos[pos, i] + x[2i]   * sin[pos, i]
//
// This avoids the transpose→unsqueeze→contiguous→rope→squeeze→transpose
// overhead of the unfused candle_nn::rotary_emb::rope path.
//
// Grid: (num_pairs_per_head, num_heads, total_tokens)
// where num_pairs_per_head = rope_dim / 2
//
// cos/sin tables: [max_seq_len, rope_dim/2] — indexed by position

template <typename scalar_t>
__global__ void fused_rope_kernel(
    scalar_t* __restrict__ x,              // [total_tokens, num_heads, head_size] — modified in-place
    const unsigned int* __restrict__ positions, // [total_tokens] — position index per token
    const float* __restrict__ cos_table,   // [max_seq_len, rope_dim/2]
    const float* __restrict__ sin_table,   // [max_seq_len, rope_dim/2]
    int num_heads,
    int head_size,
    int rope_dim                           // typically == head_size, but can be smaller
) {
    const int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;  // which pair within head (0..rope_dim/2)
    const int head_idx = blockIdx.y;                               // which head
    const int token_idx = blockIdx.z;                              // which token

    const int half_rope = rope_dim / 2;
    if (pair_idx >= half_rope) return;

    const int pos = positions[token_idx];

    // cos/sin for this position and pair
    const float c = cos_table[pos * half_rope + pair_idx];
    const float s = sin_table[pos * half_rope + pair_idx];

    // Index into x: [token_idx, head_idx, 2*pair_idx] and [token_idx, head_idx, 2*pair_idx+1]
    const int base = token_idx * (num_heads * head_size) + head_idx * head_size;
    const int idx0 = base + 2 * pair_idx;
    const int idx1 = base + 2 * pair_idx + 1;

    const float x0 = (float)x[idx0];
    const float x1 = (float)x[idx1];

    x[idx0] = (scalar_t)(x0 * c - x1 * s);
    x[idx1] = (scalar_t)(x1 * c + x0 * s);
}

extern "C" __global__ void fused_rope_f32(
    float* x, const unsigned int* positions,
    const float* cos_table, const float* sin_table,
    int num_heads, int head_size, int rope_dim
) {
    fused_rope_kernel<float>(x, positions, cos_table, sin_table, num_heads, head_size, rope_dim);
}

extern "C" __global__ void fused_rope_f16(
    __half* x, const unsigned int* positions,
    const float* cos_table, const float* sin_table,
    int num_heads, int head_size, int rope_dim
) {
    fused_rope_kernel<__half>(x, positions, cos_table, sin_table, num_heads, head_size, rope_dim);
}

// ─── Fused RMSNorm Kernel ────────────────────────────────────────────────
//
// Computes RMS Layer Normalization in a single kernel:
//   rms = sqrt(mean(x^2) + eps)
//   output[i] = (x[i] / rms) * weight[i]
//
// Grid: (num_rows, 1, 1) — one block per row (token)
// Block: (blockDim.x, 1, 1) — threads cooperate on the reduction
//
// Each thread handles ceil(hidden_size / blockDim.x) elements, accumulates
// partial sum-of-squares, then shared-memory reduction computes the RMS.
// Finally each thread normalizes and scales its elements.

template <typename scalar_t>
__global__ void fused_rmsnorm_kernel(
    scalar_t* __restrict__ output,         // [num_rows, hidden_size]
    const scalar_t* __restrict__ input,    // [num_rows, hidden_size]
    const float* __restrict__ weight,      // [hidden_size]
    int hidden_size,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int base = row * hidden_size;

    // Each thread accumulates partial sum of squares
    float partial_ss = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float v = (float)input[base + i];
        partial_ss += v * v;
    }

    // Shared memory reduction to compute total sum of squares
    extern __shared__ float shmem[];
    shmem[tid] = partial_ss;
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        __syncthreads();
    }

    // Broadcast the inverse RMS
    float inv_rms = rsqrtf(shmem[0] / (float)hidden_size + eps);

    // Normalize and scale
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float v = (float)input[base + i];
        output[base + i] = (scalar_t)(v * inv_rms * weight[i]);
    }
}

extern "C" __global__ void fused_rmsnorm_f32(
    float* output, const float* input, const float* weight,
    int hidden_size, float eps
) {
    fused_rmsnorm_kernel<float>(output, input, weight, hidden_size, eps);
}

extern "C" __global__ void fused_rmsnorm_f16(
    __half* output, const __half* input, const float* weight,
    int hidden_size, float eps
) {
    fused_rmsnorm_kernel<__half>(output, input, weight, hidden_size, eps);
}

// ─── Fused residual add + RMSNorm ──────────────────────────────────────
//
// Combines: x = x + residual; output = rmsnorm(x, weight, eps)
// Eliminates one intermediate tensor write vs separate add + rmsnorm.
// Also writes the updated x (with residual added) back in-place for the
// next layer's residual connection.
//
// Inputs:
//   output      — [num_rows, hidden_size] normalized result
//   x           — [num_rows, hidden_size] input (modified in-place: x += residual)
//   residual    — [num_rows, hidden_size] residual to add
//   weight      — [hidden_size] RMSNorm scale (always F32)
//   hidden_size — number of elements per row
//   eps         — normalization epsilon

template <typename scalar_t>
__global__ void fused_add_rmsnorm_kernel(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ x,
    const scalar_t* __restrict__ residual,
    const float* __restrict__ weight,
    int hidden_size,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int base = row * hidden_size;

    // Phase 1: add residual and compute partial sum of squares
    float partial_ss = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = (float)x[base + i] + (float)residual[base + i];
        x[base + i] = (scalar_t)val;  // write back x += residual
        partial_ss += val * val;
    }

    // Shared memory reduction
    extern __shared__ float shmem[];
    shmem[tid] = partial_ss;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        __syncthreads();
    }

    float inv_rms = rsqrtf(shmem[0] / (float)hidden_size + eps);

    // Phase 2: normalize and scale
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float v = (float)x[base + i];
        output[base + i] = (scalar_t)(v * inv_rms * weight[i]);
    }
}

extern "C" __global__ void fused_add_rmsnorm_f32(
    float* output, float* x, const float* residual, const float* weight,
    int hidden_size, float eps
) {
    fused_add_rmsnorm_kernel<float>(output, x, residual, weight, hidden_size, eps);
}

extern "C" __global__ void fused_add_rmsnorm_f16(
    __half* output, __half* x, const __half* residual, const float* weight,
    int hidden_size, float eps
) {
    fused_add_rmsnorm_kernel<__half>(output, x, residual, weight, hidden_size, eps);
}

// ─── Marlin-style fused dequant+GEMM kernel ──────────────────────────────
//
// Simplified Marlin-style fused dequant+GEMM kernel for sm_86 (Ampere).
//
// Each thread block handles a [1, 64] output tile (one batch row, 64 output columns).
// Weights are pre-reformatted into [K/16, N/64, 128] tile layout where each 128-u32
// block encodes a 16x64 sub-tile of INT4 weights.
//
// The kernel loads weight tiles, dequantizes INT4 to FP16, and accumulates
// via FP16 FMA (not full Tensor Core MMA — that's a Phase 2 optimization).
//
// This is simpler than the full Marlin kernel but still fused: no intermediate
// dequantized weight tensor is materialized in global memory.
//
// Grid:  (N/64, M, 1)  where M = batch size, N = out_features
// Block: (256, 1, 1) — 8 warps per block

extern "C" __global__ void marlin_gemm_f16(
    __half* __restrict__ output,            // [M, N]
    const __half* __restrict__ input,       // [M, K]
    const unsigned int* __restrict__ qw,    // [K/16, N/64, 128] Marlin tile layout
    const __half* __restrict__ scales,      // [K/group_size, N]
    const unsigned int* __restrict__ qzeros, // [K/group_size, N/8]
    int M, int N, int K, int group_size
) {
    // Which 64-column tile and which batch row
    const int tile_col = blockIdx.x;   // 0..N/64-1
    const int row = blockIdx.y;        // 0..M-1
    const int tid = threadIdx.x;       // 0..255

    // Each thread accumulates 64/256 = partial set of output columns
    // But with 256 threads and 64 columns, we assign threads to columns
    // and have them iterate over K dimension.
    // Assign: each of 64 columns gets 4 threads (256/64 = 4 threads per column).
    // Those 4 threads split the K-reduction and then reduce.
    const int col_in_tile = tid % 64;        // which column within the 64-col tile
    const int k_worker = tid / 64;           // 0..3 — which K-reduction worker
    const int num_k_workers = 4;

    const int out_col = tile_col * 64 + col_in_tile;
    if (out_col >= N) return;

    float acc = 0.0f;

    // Number of K-tiles
    const int num_k_tiles = K / 16;

    // Each K-worker handles a subset of K-tiles
    for (int kt = k_worker; kt < num_k_tiles; kt += num_k_workers) {
        const int k_base = kt * 16;  // starting input feature index

        // qw index: [kt, tile_col, 128]
        // 128 u32 values encode 16x64 = 1024 INT4 values
        // Layout within the 128-u32 tile:
        //   For row r (0..15) and col c (0..63) within the tile:
        //     linear_idx = r * 64 + c  (0..1023)
        //     u32_idx = linear_idx / 8
        //     bit_pos = linear_idx % 8
        //     value = (qw[u32_idx] >> (bit_pos * 4)) & 0xF
        const int tile_offset = (kt * (N / 64) + tile_col) * 128;

        for (int r = 0; r < 16; r++) {
            const int k_idx = k_base + r;

            // Extract INT4 weight from tile
            const int linear_idx = r * 64 + col_in_tile;
            const int u32_idx = linear_idx / 8;
            const int bit_pos = linear_idx % 8;
            const unsigned int packed = qw[tile_offset + u32_idx];
            const int w_int4 = (packed >> (bit_pos * 4)) & 0xF;

            // Get scale and zero for this group
            const int group_idx = k_idx / group_size;
            const __half scale_val = scales[group_idx * N + out_col];
            const int zero_pack_col = out_col / 8;
            const int zero_bit = out_col % 8;
            const unsigned int zero_packed = qzeros[group_idx * (N / 8) + zero_pack_col];
            const int zero = (zero_packed >> (zero_bit * 4)) & 0xF;

            // Dequantize and FMA
            const float w_float = (float)(w_int4 - zero) * __half2float(scale_val);
            const float in_val = __half2float(input[row * K + k_idx]);
            acc += w_float * in_val;
        }
    }

    // Reduce across K-workers using shared memory
    __shared__ float smem[256];
    smem[tid] = acc;
    __syncthreads();

    // Each column has workers at tid = col_in_tile + {0,1,2,3}*64
    // Reduce: worker 0 accumulates from workers 1,2,3
    if (k_worker == 0) {
        float total = smem[col_in_tile];
        for (int w = 1; w < num_k_workers; w++) {
            total += smem[col_in_tile + w * 64];
        }
        output[row * N + out_col] = __float2half(total);
    }
}

// ─── Fused LayerNorm + Linear Kernel ────────────────────────────────────
//
// Computes: output = rmsnorm(x, norm_weight, eps) @ linear_weight^T
// in a single kernel pass, eliminating the intermediate normalized tensor.
//
// Each block processes one row of x:
//   1. Compute RMS norm (sum of squares reduction in shared memory)
//   2. For each output column, compute dot product of normalized row with
//      the corresponding column of linear_weight
//
// Grid: (num_rows, 1, 1)
// Block: (256, 1, 1)
//
// This kernel targets the common case of F16 activations and F16 weights.
// For hidden_size > 256*4 (1024), fall back to unfused path on the host side.

template <typename scalar_t>
__global__ void fused_layernorm_linear_kernel(
    scalar_t* __restrict__ output,            // [num_rows, out_features]
    const scalar_t* __restrict__ input,       // [num_rows, hidden_size]
    const float* __restrict__ norm_weight,    // [hidden_size]
    const scalar_t* __restrict__ linear_weight, // [out_features, hidden_size]
    int hidden_size,
    int out_features,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int in_base = row * hidden_size;

    // Phase 1: Compute RMS norm — sum of squares reduction
    float partial_ss = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float v = (float)input[in_base + i];
        partial_ss += v * v;
    }

    extern __shared__ float shmem[];
    shmem[tid] = partial_ss;
    __syncthreads();

    // Tree reduction for sum of squares
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        __syncthreads();
    }

    float inv_rms = rsqrtf(shmem[0] / (float)hidden_size + eps);

    // Phase 2: For each output column, compute dot(normalized_row, linear_weight_col)
    // Each thread handles a subset of output columns
    const int out_base = row * out_features;
    for (int oc = tid; oc < out_features; oc += blockDim.x) {
        float dot = 0.0f;
        const int w_base = oc * hidden_size;
        for (int i = 0; i < hidden_size; i++) {
            float x_normed = (float)input[in_base + i] * inv_rms * norm_weight[i];
            dot += x_normed * (float)linear_weight[w_base + i];
        }
        output[out_base + oc] = (scalar_t)dot;
    }
}

extern "C" __global__ void fused_layernorm_linear_f32(
    float* output, const float* input, const float* norm_weight,
    const float* linear_weight, int hidden_size, int out_features, float eps
) {
    fused_layernorm_linear_kernel<float>(
        output, input, norm_weight, linear_weight,
        hidden_size, out_features, eps
    );
}

extern "C" __global__ void fused_layernorm_linear_f16(
    __half* output, const __half* input, const float* norm_weight,
    const __half* linear_weight, int hidden_size, int out_features, float eps
) {
    fused_layernorm_linear_kernel<__half>(
        output, input, norm_weight, linear_weight,
        hidden_size, out_features, eps
    );
}

// ─── FP8 E4M3 KV Cache Quantize/Dequantize ──────────────────────────────
//
// FP8 E4M3 format: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits.
// Max representable value: 240.0, no infinity.
//
// These kernels convert between FP16 (compute dtype) and FP8 E4M3 (storage dtype)
// with per-head scaling factors. Each head's values are scaled independently to
// maximize precision within the FP8 range.
//
// Quantize: runs after attention K/V computation, before writing to cache.
// Dequantize: runs after reading from cache, before attention computation.

#define FP8_E4M3_MAX 240.0f

// Convert f32 to FP8 E4M3 (as unsigned char)
__device__ __forceinline__ unsigned char f32_to_fp8_e4m3(float val) {
    if (val != val) return 0x7F;  // NaN check

    unsigned char sign = (val < 0.0f) ? 1 : 0;
    float abs_val = fabsf(val);

    if (abs_val == 0.0f) return sign << 7;

    // Clamp to max FP8 range
    if (abs_val > FP8_E4M3_MAX) abs_val = FP8_E4M3_MAX;

    // Flush tiny values to zero
    if (abs_val < 1.953125e-3f) return sign << 7;  // 2^-9

    // Extract f32 components
    unsigned int bits = __float_as_uint(abs_val);
    int f32_exp = ((bits >> 23) & 0xFF) - 127;
    unsigned int f32_mant = bits & 0x7FFFFF;

    // Subnormal in FP8
    if (f32_exp < -6) {
        int shift = -6 - f32_exp;
        unsigned int full_mant = (1u << 3) | ((f32_mant >> 20) & 0x7);
        unsigned char mant = (shift < 4) ? ((full_mant >> shift) & 0x7) : 0;
        if (mant == 0) return sign << 7;
        return (sign << 7) | mant;
    }

    // Normal
    unsigned char biased_exp = (unsigned char)(f32_exp + 7);
    if (biased_exp >= 15) return (sign << 7) | 0x7E;  // max normal = 240

    unsigned char mant = (unsigned char)((f32_mant >> 20) & 0x7);
    unsigned int round_bit = (f32_mant >> 19) & 1;
    unsigned int sticky = f32_mant & 0x7FFFF;

    if (round_bit == 1 && (sticky != 0 || (mant & 1) != 0)) {
        mant++;
    }
    if (mant >= 8) {
        biased_exp++;
        mant = 0;
        if (biased_exp >= 15) return (sign << 7) | 0x7E;
    }

    return (sign << 7) | (biased_exp << 3) | mant;
}

// Convert FP8 E4M3 (as unsigned char) back to f32
__device__ __forceinline__ float fp8_e4m3_to_f32(unsigned char val) {
    unsigned char sign = (val >> 7) & 1;
    unsigned char exp = (val >> 3) & 0xF;
    unsigned char mant = val & 0x7;

    if (exp == 0xF && mant == 0x7) return __uint_as_float(0x7FC00000u);  // NaN

    float abs_val;
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        abs_val = exp2f(-6.0f) * ((float)mant / 8.0f);  // subnormal
    } else {
        abs_val = exp2f((float)exp - 7.0f) * (1.0f + (float)mant / 8.0f);
    }

    return sign ? -abs_val : abs_val;
}

// Quantize KV to FP8 E4M3 for cache storage (runs after attention compute).
//
// Input:  [num_tokens, num_heads, head_dim] as __half
// Output: [num_tokens, num_heads, head_dim] as unsigned char (FP8 E4M3)
// Scales: [num_tokens, num_heads] as float (per-head scale factors)
//
// Grid:  (num_tokens, num_heads, 1)
// Block: (min(head_dim, 256), 1, 1)
//
// Phase 1: Compute per-head absmax via shared memory reduction
// Phase 2: Scale and quantize each element
extern "C" __global__ void kv_cache_quantize_fp8(
    unsigned char* __restrict__ output,
    float* __restrict__ scales_out,
    const __half* __restrict__ input,
    int num_tokens,
    int num_heads,
    int head_dim
) {
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;

    if (token_idx >= num_tokens || head_idx >= num_heads) return;

    const int head_offset = (token_idx * num_heads + head_idx) * head_dim;

    // Phase 1: Find per-head absmax
    extern __shared__ float shmem[];
    float local_max = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float v = fabsf(__half2float(input[head_offset + d]));
        if (v > local_max) local_max = v;
    }
    shmem[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shmem[tid + stride] > shmem[tid])
                shmem[tid] = shmem[tid + stride];
        }
        __syncthreads();
    }

    float absmax = shmem[0];
    float scale = (absmax > 0.0f) ? (absmax / FP8_E4M3_MAX) : 1.0f;
    float inv_scale = 1.0f / scale;

    // Thread 0 writes the scale
    if (tid == 0) {
        scales_out[token_idx * num_heads + head_idx] = scale;
    }

    // Phase 2: Quantize
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float v = __half2float(input[head_offset + d]) * inv_scale;
        output[head_offset + d] = f32_to_fp8_e4m3(v);
    }
}

// Dequantize FP8 E4M3 KV from cache (runs before attention).
//
// Input:  [num_tokens, num_heads, head_dim] as unsigned char (FP8 E4M3)
// Scales: [num_tokens, num_heads] as float
// Output: [num_tokens, num_heads, head_dim] as __half
//
// Grid:  (num_tokens, num_heads, 1)
// Block: (min(head_dim, 256), 1, 1)
extern "C" __global__ void kv_cache_dequantize_fp8(
    __half* __restrict__ output,
    const unsigned char* __restrict__ input,
    const float* __restrict__ scales,
    int num_tokens,
    int num_heads,
    int head_dim
) {
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;

    if (token_idx >= num_tokens || head_idx >= num_heads) return;

    const int head_offset = (token_idx * num_heads + head_idx) * head_dim;
    const float scale = scales[token_idx * num_heads + head_idx];

    for (int d = tid; d < head_dim; d += blockDim.x) {
        float v = fp8_e4m3_to_f32(input[head_offset + d]) * scale;
        output[head_offset + d] = __float2half(v);
    }
}

// ─── FP8 E4M3 Dequantization Kernel ──────────────────────────────────
//
// Dequantizes FP8 E4M3 weights (stored as u8) to half-precision (F16) output.
//
// FP8 E4M3 format: sign(1) + exponent(4) + mantissa(3), bias=7
//   - NaN: exp=15, man=7 (0x7F / 0xFF)
//   - Max: 448.0 (exp=15, man=6)
//   - Subnormals: exp=0, man!=0 -> value = man * 2^(-9)
//   - Normals: value = 2^(exp-7) * (1 + man/8)
//
// Each thread dequantizes one element:
//   output[idx] = fp8_to_half(input[idx]) * scale
//
// Scale is either per-tensor (per_channel=0, scale[0]) or
// per-channel (per_channel=1, scale[row]).
//
// Grid: 1D over total elements.

__device__ __half fp8_e4m3_to_half(unsigned char bits) {
    unsigned char sign_bit = (bits >> 7) & 1;
    unsigned char exp_bits = (bits >> 3) & 0xF;
    unsigned char man_bits = bits & 0x7;

    // NaN
    if (exp_bits == 15 && man_bits == 7) {
        return __float2half(0.0f / 0.0f);
    }

    float sign_f = sign_bit ? -1.0f : 1.0f;

    if (exp_bits == 0) {
        if (man_bits == 0) {
            return __float2half(0.0f);
        }
        // Subnormal: value = man * 2^(-9)
        float val = sign_f * (float)man_bits * (1.0f / 512.0f);
        return __float2half(val);
    }

    // Normal: value = 2^(exp-7) * (1 + man/8)
    int exp_unbiased = (int)exp_bits - 7;
    float significand = 1.0f + (float)man_bits / 8.0f;
    float val = sign_f * ldexpf(significand, exp_unbiased);
    return __float2half(val);
}

extern "C" __global__ void fp8_e4m3_dequant_f16(
    __half* __restrict__ output,
    const unsigned char* __restrict__ input,
    const __half* __restrict__ scale,
    int rows,
    int cols,
    int per_channel
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx >= total) return;

    const int row = idx / cols;
    const __half s = per_channel ? scale[row] : scale[0];

    __half dequant = fp8_e4m3_to_half(input[idx]);
    output[idx] = __hmul(dequant, s);
}

// F32 variant of FP8 E4M3 dequantization
extern "C" __global__ void fp8_e4m3_dequant_f32(
    float* __restrict__ output,
    const unsigned char* __restrict__ input,
    const float* __restrict__ scale,
    int rows,
    int cols,
    int per_channel
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx >= total) return;

    const int row = idx / cols;
    const float s = per_channel ? scale[row] : scale[0];

    // Inline FP8 E4M3 to float conversion
    unsigned char bits = input[idx];
    unsigned char sign_bit = (bits >> 7) & 1;
    unsigned char exp_bits = (bits >> 3) & 0xF;
    unsigned char man_bits = bits & 0x7;

    float val;
    if (exp_bits == 15 && man_bits == 7) {
        val = 0.0f / 0.0f; // NaN
    } else if (exp_bits == 0) {
        if (man_bits == 0) {
            val = 0.0f;
        } else {
            val = (sign_bit ? -1.0f : 1.0f) * (float)man_bits * (1.0f / 512.0f);
        }
    } else {
        int exp_unbiased = (int)exp_bits - 7;
        float significand = 1.0f + (float)man_bits / 8.0f;
        val = (sign_bit ? -1.0f : 1.0f) * ldexpf(significand, exp_unbiased);
    }

    output[idx] = val * s;
}

// F32 variant of GPTQ dequantization
extern "C" __global__ void gptq_dequant_f32(
    float* __restrict__ output,               // [out_features, in_features]
    const unsigned int* __restrict__ qweight,  // [in_features / 8, out_features]
    const float* __restrict__ scales,          // [num_groups, out_features]
    const unsigned int* __restrict__ qzeros,   // [num_groups, out_features / 8]
    int in_features,
    int out_features,
    int group_size
) {
    const int in_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (in_idx >= in_features || out_idx >= out_features) return;

    const int group_idx = in_idx / group_size;

    const int pack_row = in_idx / 8;
    const int pack_bit = in_idx % 8;
    const unsigned int packed = qweight[pack_row * out_features + out_idx];
    const int w_int4 = (packed >> (pack_bit * 4)) & 0xF;

    const int zero_pack_col = out_idx / 8;
    const int zero_bit = out_idx % 8;
    const unsigned int zero_packed = qzeros[group_idx * (out_features / 8) + zero_pack_col];
    const int zero = (zero_packed >> (zero_bit * 4)) & 0xF;

    const float scale = scales[group_idx * out_features + out_idx];
    const float w_float = (float)(w_int4 - zero) * scale;

    output[out_idx * in_features + in_idx] = w_float;
}
