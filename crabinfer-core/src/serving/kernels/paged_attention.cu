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
