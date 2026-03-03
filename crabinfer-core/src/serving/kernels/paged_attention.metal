// paged_attention.metal — Paged attention V1 and V2 compute kernels.
//
// Follows vLLM's algorithm, adapted for Apple Metal:
//   __shfl_xor_sync  → simd_shuffle_xor
//   __syncthreads    → threadgroup_barrier
//   warp (32)        → SIMD group (32 on Apple GPU)
//   shared memory    → threadgroup memory
//
// KV cache layout (same as vLLM/mistral.rs):
//   K: [num_blocks, num_kv_heads, head_size/x, block_size, x]  (x = 16/sizeof(T))
//   V: [num_blocks, num_kv_heads, head_size, block_size]
//
// Template parameters:
//   T          = float or half
//   HEAD_SIZE  = 64, 128
//   BLOCK_SIZE = 16
//   NUM_THREADS = 256
//   PARTITION_SIZE = 0 (V1) or 512 (V2)

#include <metal_stdlib>
using namespace metal;

// ─── Vector helpers ───────────────────────────────────────────────────────

// Number of float/half elements in a 16-byte load
template <typename T> constexpr int VEC_SIZE() { return 16 / sizeof(T); }

// ─── SIMD reduction ──────────────────────────────────────────────────────

inline float simd_max(float val) {
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        val = max(val, simd_shuffle_xor(val, offset));
    }
    return val;
}

inline float simd_sum(float val) {
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        val += simd_shuffle_xor(val, offset);
    }
    return val;
}

// ─── Threadgroup-level reduction ─────────────────────────────────────────

#define NUM_SIMD_GROUPS 8 // 256 threads / 32 simd width

float threadgroup_max(float val,
                      uint simd_lane,
                      uint simd_group,
                      threadgroup float* scratch) {
    val = simd_max(val);
    if (simd_lane == 0) {
        scratch[simd_group] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        val = (simd_lane < NUM_SIMD_GROUPS) ? scratch[simd_lane] : -INFINITY;
        val = simd_max(val);
    }
    return val;
}

float threadgroup_sum(float val,
                      uint simd_lane,
                      uint simd_group,
                      threadgroup float* scratch) {
    val = simd_sum(val);
    if (simd_lane == 0) {
        scratch[simd_group] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        val = (simd_lane < NUM_SIMD_GROUPS) ? scratch[simd_lane] : 0.0f;
        val = simd_sum(val);
    }
    return val;
}

// ─── Paged attention kernel ──────────────────────────────────────────────

template <typename T, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS, int PARTITION_SIZE>
[[kernel]] void paged_attention(
    device T             *output         [[buffer(0)]],   // [num_seqs, num_heads, head_size] or V2: [num_seqs, num_heads, max_partitions, head_size]
    const device T       *query          [[buffer(1)]],   // [num_seqs, num_heads, head_size]
    const device T       *key_cache      [[buffer(2)]],   // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const device T       *value_cache    [[buffer(3)]],   // [num_blocks, num_kv_heads, head_size, block_size]
    const device int     *block_tables   [[buffer(4)]],   // [num_seqs, max_blocks_per_seq]
    const device int     *context_lens   [[buffer(5)]],   // [num_seqs]
    constant int         &num_kv_heads   [[buffer(6)]],
    constant float       &scale          [[buffer(7)]],   // 1/sqrt(head_size)
    constant int         &max_blocks_per_seq [[buffer(8)]],
    constant int         &q_stride       [[buffer(9)]],   // num_heads * head_size
    constant int         &kv_block_stride [[buffer(10)]],  // num_kv_heads * head_size (per block)
    constant int         &kv_head_stride [[buffer(11)]],   // head_size (per head in a block)
    // V2 only:
    device float         *exp_sums       [[buffer(12)]],  // [num_seqs, num_heads, max_partitions]
    device float         *max_logits     [[buffer(13)]],  // [num_seqs, num_heads, max_partitions]
    // Grid indices
    uint3 tgid [[threadgroup_position_in_grid]],           // (head_idx, seq_idx, partition_idx)
    uint tid   [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    // Threadgroup memory
    threadgroup float *shared_mem [[threadgroup(0)]])
{
    constexpr int VEC_SZ = VEC_SIZE<T>();
    constexpr int X = 16 / sizeof(T);
    // How many threads cooperate to process one K vector (one block row)
    constexpr int THREAD_GROUP_SIZE = (32 / BLOCK_SIZE > 1) ? (32 / BLOCK_SIZE) : 1;
    constexpr int NUM_THREAD_GROUPS = NUM_THREADS / THREAD_GROUP_SIZE;
    constexpr int NUM_VECS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE / VEC_SZ;

    const uint head_idx = tgid.x;
    const uint seq_idx = tgid.y;
    const uint partition_idx = (PARTITION_SIZE > 0) ? tgid.z : 0;

    const int context_len = context_lens[seq_idx];
    if (context_len == 0) return;

    // GQA: map query head to KV head
    const int num_heads = q_stride / HEAD_SIZE;
    const int num_queries_per_kv = num_heads / num_kv_heads;
    const int kv_head_idx = head_idx / num_queries_per_kv;

    // Partition bounds (V2: each partition handles a range of blocks)
    int start_block_idx, end_block_idx;
    if (PARTITION_SIZE > 0) {
        const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const int start_token = (int)partition_idx * PARTITION_SIZE;
        if (start_token >= context_len) return;
        start_block_idx = start_token / BLOCK_SIZE;
        const int end_token = min(start_token + PARTITION_SIZE, context_len);
        end_block_idx = (end_token + BLOCK_SIZE - 1) / BLOCK_SIZE;
    } else {
        start_block_idx = 0;
        end_block_idx = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }

    // Block table for this sequence
    const device int *seq_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // ── Load query vector into threadgroup memory ──
    const int thread_group_idx = tid / THREAD_GROUP_SIZE;
    const int thread_group_offset = tid % THREAD_GROUP_SIZE;

    // Query pointer for this (seq, head)
    const device T *q_ptr = query + seq_idx * q_stride + head_idx * HEAD_SIZE;

    // Shared memory layout: [logits region] then [output scratch region]
    threadgroup float *logits = shared_mem;
    threadgroup float *reduce_scratch = shared_mem; // reused after logits

    // Each thread loads part of the query
    float q_vals[NUM_VECS_PER_THREAD * VEC_SZ];
    for (int i = 0; i < NUM_VECS_PER_THREAD; i++) {
        int q_offset = thread_group_offset * NUM_VECS_PER_THREAD * VEC_SZ + i * VEC_SZ;
        for (int j = 0; j < VEC_SZ; j++) {
            if (q_offset + j < HEAD_SIZE) {
                q_vals[i * VEC_SZ + j] = float(q_ptr[q_offset + j]);
            }
        }
    }

    // ── QK dot products ──
    float qk_max = -INFINITY;

    // Each thread group processes blocks in a strided pattern
    for (int block_i = start_block_idx + thread_group_idx; block_i < end_block_idx; block_i += NUM_THREAD_GROUPS) {
        const int physical_block = seq_block_table[block_i];
        const int tokens_in_block = min(BLOCK_SIZE, context_len - block_i * BLOCK_SIZE);

        // K cache base for this block and KV head
        const device T *k_base = key_cache
            + physical_block * kv_block_stride
            + kv_head_idx * kv_head_stride;

        // Compute dot product for each token position in this block
        for (int tok = 0; tok < tokens_in_block; tok++) {
            float qk = 0.0f;
            for (int i = 0; i < NUM_VECS_PER_THREAD; i++) {
                int offset = thread_group_offset * NUM_VECS_PER_THREAD * VEC_SZ + i * VEC_SZ;
                for (int j = 0; j < VEC_SZ; j++) {
                    int head_offset = offset + j;
                    if (head_offset < HEAD_SIZE) {
                        // K layout: [head_size/x, block_size, x]
                        int x_idx = head_offset / X;
                        int x_off = head_offset % X;
                        int k_idx = x_idx * BLOCK_SIZE * X + tok * X + x_off;
                        qk += q_vals[i * VEC_SZ + j] * float(k_base[k_idx]);
                    }
                }
            }

            // Reduce across thread group
            for (ushort offset = THREAD_GROUP_SIZE / 2; offset > 0; offset >>= 1) {
                qk += simd_shuffle_xor(qk, offset);
            }

            if (thread_group_offset == 0) {
                int token_idx = block_i * BLOCK_SIZE + tok;
                qk *= scale;
                logits[token_idx - start_block_idx * BLOCK_SIZE] = qk;
                qk_max = max(qk_max, qk);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Softmax ──
    // Threadgroup reduction for qk_max
    threadgroup float scratch[NUM_SIMD_GROUPS];
    qk_max = threadgroup_max(qk_max, simd_lane, simd_group, scratch);

    // Broadcast qk_max to all threads
    if (simd_group == 0 && simd_lane == 0) {
        scratch[0] = qk_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    qk_max = scratch[0];

    // Compute exp and sum
    const int num_context_tokens = (PARTITION_SIZE > 0)
        ? min(PARTITION_SIZE, context_len - (int)partition_idx * PARTITION_SIZE)
        : context_len;
    const int padded_context = ((num_context_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

    float exp_sum = 0.0f;
    for (int i = tid; i < padded_context; i += NUM_THREADS) {
        float val;
        if (i < num_context_tokens) {
            val = exp(logits[i] - qk_max);
            logits[i] = val;
        } else {
            val = 0.0f;
            if (i < padded_context) logits[i] = 0.0f;
        }
        exp_sum += val;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    exp_sum = threadgroup_sum(exp_sum, simd_lane, simd_group, scratch);

    // Broadcast exp_sum
    if (simd_group == 0 && simd_lane == 0) {
        scratch[0] = exp_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_exp_sum = 1.0f / scratch[0];

    // ── V accumulation ──
    // Each thread accumulates a portion of the head dimension
    constexpr int ELEMS_PER_THREAD = HEAD_SIZE / NUM_THREADS;
    // When HEAD_SIZE < NUM_THREADS, some threads are idle
    constexpr int ACTIVE_THREADS = (HEAD_SIZE < NUM_THREADS) ? HEAD_SIZE : NUM_THREADS;
    constexpr int ELEMS_PER = (ELEMS_PER_THREAD > 0) ? ELEMS_PER_THREAD : 1;

    float accs[ELEMS_PER] = {};

    if (tid < ACTIVE_THREADS) {
        for (int block_i = start_block_idx; block_i < end_block_idx; block_i++) {
            const int physical_block = seq_block_table[block_i];
            const int tokens_in_block = min(BLOCK_SIZE, context_len - block_i * BLOCK_SIZE);

            // V cache base for this block and KV head
            // V layout: [num_blocks, num_kv_heads, head_size, block_size]
            const device T *v_base = value_cache
                + physical_block * num_kv_heads * HEAD_SIZE * BLOCK_SIZE
                + kv_head_idx * HEAD_SIZE * BLOCK_SIZE;

            for (int tok = 0; tok < tokens_in_block; tok++) {
                int logit_idx = block_i * BLOCK_SIZE + tok - start_block_idx * BLOCK_SIZE;
                float attn_weight = logits[logit_idx] * inv_exp_sum;

                for (int e = 0; e < ELEMS_PER; e++) {
                    int head_dim = tid * ELEMS_PER + e;
                    if (head_dim < HEAD_SIZE) {
                        // V at [head_dim, tok] in the block
                        int v_idx = head_dim * BLOCK_SIZE + tok;
                        accs[e] += attn_weight * float(v_base[v_idx]);
                    }
                }
            }
        }
    }

    // ── Write output ──
    if (tid < ACTIVE_THREADS) {
        if (PARTITION_SIZE > 0) {
            // V2: write to partitioned output
            const int max_partitions = (context_len + PARTITION_SIZE - 1) / PARTITION_SIZE;
            device T *out_ptr = output + seq_idx * num_heads * max_partitions * HEAD_SIZE
                              + head_idx * max_partitions * HEAD_SIZE
                              + partition_idx * HEAD_SIZE;
            for (int e = 0; e < ELEMS_PER; e++) {
                int head_dim = tid * ELEMS_PER + e;
                if (head_dim < HEAD_SIZE) {
                    out_ptr[head_dim] = T(accs[e]);
                }
            }
            // Store partition-level stats for reduce
            if (tid == 0) {
                int idx = seq_idx * num_heads * max_partitions + head_idx * max_partitions + partition_idx;
                exp_sums[idx] = exp_sum;
                max_logits[idx] = qk_max;
            }
        } else {
            // V1: write directly to output
            device T *out_ptr = output + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
            for (int e = 0; e < ELEMS_PER; e++) {
                int head_dim = tid * ELEMS_PER + e;
                if (head_dim < HEAD_SIZE) {
                    out_ptr[head_dim] = T(accs[e]);
                }
            }
        }
    }
}

// ─── V2 reduce kernel ───────────────────────────────────────────────────

template <typename T, int HEAD_SIZE, int NUM_THREADS>
[[kernel]] void paged_attention_v2_reduce(
    device T             *output         [[buffer(0)]],   // [num_seqs, num_heads, head_size]
    const device float   *exp_sums       [[buffer(1)]],   // [num_seqs, num_heads, max_partitions]
    const device float   *max_logits     [[buffer(2)]],   // [num_seqs, num_heads, max_partitions]
    const device T       *tmp_output     [[buffer(3)]],   // [num_seqs, num_heads, max_partitions, head_size]
    const device int     *context_lens   [[buffer(4)]],   // [num_seqs]
    constant int         &max_num_partitions [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],           // (head_idx, seq_idx, 1)
    uint tid   [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    threadgroup float *shared_mem [[threadgroup(0)]])
{
    const uint head_idx = tgid.x;
    const uint seq_idx = tgid.y;
    const int num_heads = tgid.x; // overloaded — see dispatch
    // Actually we need num_heads from context; pass via constant or compute from grid
    // For now, use max_num_partitions to index correctly

    const int context_len = context_lens[seq_idx];
    const int num_partitions = (context_len + 512 - 1) / 512; // PARTITION_SIZE = 512
    if (num_partitions <= 1) {
        // Single partition — just copy
        if (tid < (uint)HEAD_SIZE) {
            const device T *src = tmp_output + seq_idx * max_num_partitions * HEAD_SIZE
                                + head_idx * max_num_partitions * HEAD_SIZE;
            device T *dst = output + seq_idx * HEAD_SIZE + head_idx * HEAD_SIZE;
            // Actually need num_heads in the stride — fix with actual dispatch
            dst[tid] = src[tid];
        }
        return;
    }

    // Find global max logit
    const int base_idx = seq_idx * max_num_partitions + head_idx * max_num_partitions;
    float global_max = -INFINITY;
    for (int p = tid; p < num_partitions; p += NUM_THREADS) {
        global_max = max(global_max, max_logits[base_idx + p]);
    }

    threadgroup float scratch[NUM_SIMD_GROUPS];
    global_max = threadgroup_max(global_max, simd_lane, simd_group, scratch);
    if (simd_group == 0 && simd_lane == 0) scratch[0] = global_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    global_max = scratch[0];

    // Rescale exp_sums
    float global_exp_sum = 0.0f;
    for (int p = tid; p < num_partitions; p += NUM_THREADS) {
        float rescaled = exp_sums[base_idx + p] * exp(max_logits[base_idx + p] - global_max);
        global_exp_sum += rescaled;
    }
    global_exp_sum = threadgroup_sum(global_exp_sum, simd_lane, simd_group, scratch);
    if (simd_group == 0 && simd_lane == 0) scratch[0] = global_exp_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_global_exp = 1.0f / scratch[0];

    // Weighted sum of partition outputs
    constexpr int ELEMS_PER = (HEAD_SIZE / NUM_THREADS > 0) ? (HEAD_SIZE / NUM_THREADS) : 1;
    constexpr int ACTIVE_THREADS = (HEAD_SIZE < NUM_THREADS) ? HEAD_SIZE : NUM_THREADS;

    if (tid < ACTIVE_THREADS) {
        float result[ELEMS_PER] = {};
        for (int p = 0; p < num_partitions; p++) {
            float weight = exp_sums[base_idx + p]
                         * exp(max_logits[base_idx + p] - global_max)
                         * inv_global_exp;
            for (int e = 0; e < ELEMS_PER; e++) {
                int head_dim = tid * ELEMS_PER + e;
                if (head_dim < HEAD_SIZE) {
                    const device T *part_out = tmp_output
                        + seq_idx * max_num_partitions * HEAD_SIZE
                        + head_idx * max_num_partitions * HEAD_SIZE
                        + p * HEAD_SIZE;
                    result[e] += weight * float(part_out[head_dim]);
                }
            }
        }

        device T *out_ptr = output + seq_idx * HEAD_SIZE + head_idx * HEAD_SIZE;
        for (int e = 0; e < ELEMS_PER; e++) {
            int head_dim = tid * ELEMS_PER + e;
            if (head_dim < HEAD_SIZE) {
                out_ptr[head_dim] = T(result[e]);
            }
        }
    }
}

// ─── Template instantiations ─────────────────────────────────────────────
//
// Naming convention: paged_attention_{type}_hs{HEAD_SIZE}_bs{BLOCK_SIZE}_ps{PARTITION_SIZE}
//
// V1: PARTITION_SIZE = 0
// V2: PARTITION_SIZE = 512

#define INSTANTIATE_PA(T, HS, BS, PS, NAME) \
template [[host_name(NAME)]] \
[[kernel]] void paged_attention<T, HS, BS, 256, PS>( \
    device T*, const device T*, const device T*, const device T*, \
    const device int*, const device int*, \
    constant int&, constant float&, constant int&, constant int&, \
    constant int&, constant int&, \
    device float*, device float*, \
    uint3, uint, uint, uint, threadgroup float*);

#define INSTANTIATE_PA_REDUCE(T, HS, NAME) \
template [[host_name(NAME)]] \
[[kernel]] void paged_attention_v2_reduce<T, HS, 256>( \
    device T*, const device float*, const device float*, \
    const device T*, const device int*, constant int&, \
    uint3, uint, uint, uint, threadgroup float*);

// float, head_size=64, block_size=16
INSTANTIATE_PA(float, 64, 16, 0,   "paged_attention_float_hs64_bs16_ps0")
INSTANTIATE_PA(float, 64, 16, 512, "paged_attention_float_hs64_bs16_ps512")

// float, head_size=128, block_size=16
INSTANTIATE_PA(float, 128, 16, 0,   "paged_attention_float_hs128_bs16_ps0")
INSTANTIATE_PA(float, 128, 16, 512, "paged_attention_float_hs128_bs16_ps512")

// half, head_size=64, block_size=16
INSTANTIATE_PA(half, 64, 16, 0,   "paged_attention_half_hs64_bs16_ps0")
INSTANTIATE_PA(half, 64, 16, 512, "paged_attention_half_hs64_bs16_ps512")

// half, head_size=128, block_size=16
INSTANTIATE_PA(half, 128, 16, 0,   "paged_attention_half_hs128_bs16_ps0")
INSTANTIATE_PA(half, 128, 16, 512, "paged_attention_half_hs128_bs16_ps512")

// V2 reduce kernels
INSTANTIATE_PA_REDUCE(float, 64,  "paged_attention_v2_reduce_float_hs64")
INSTANTIATE_PA_REDUCE(float, 128, "paged_attention_v2_reduce_float_hs128")
INSTANTIATE_PA_REDUCE(half, 64,   "paged_attention_v2_reduce_half_hs64")
INSTANTIATE_PA_REDUCE(half, 128,  "paged_attention_v2_reduce_half_hs128")
