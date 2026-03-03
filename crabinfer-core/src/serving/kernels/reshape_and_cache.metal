// reshape_and_cache.metal — Write K/V tokens into paged KV cache blocks.
//
// K cache layout: [num_blocks, num_kv_heads, head_size/x, block_size, x]
//   where x = 16 / sizeof(T) — enables coalesced 16-byte reads during attention
// V cache layout: [num_blocks, num_kv_heads, head_size, block_size]
//
// slot_mapping[token_idx] maps each input token to a global slot:
//   block_idx = slot / block_size, block_offset = slot % block_size
// A slot of -1 means padding (skip).

#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void reshape_and_cache(
    const device T       *key           [[buffer(0)]],   // [num_tokens, num_kv_heads, head_size]
    const device T       *value         [[buffer(1)]],   // [num_tokens, num_kv_heads, head_size]
    device T             *key_cache     [[buffer(2)]],   // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    device T             *value_cache   [[buffer(3)]],   // [num_blocks, num_kv_heads, head_size, block_size]
    const device int     *slot_mapping  [[buffer(4)]],   // [num_tokens]
    constant int         &key_stride    [[buffer(5)]],   // num_kv_heads * head_size
    constant int         &value_stride  [[buffer(6)]],   // num_kv_heads * head_size
    constant int         &num_heads     [[buffer(7)]],
    constant int         &head_size     [[buffer(8)]],
    constant int         &block_size    [[buffer(9)]],
    constant int         &x             [[buffer(10)]],  // 16 / sizeof(T)
    uint gid  [[threadgroup_position_in_grid]],           // one threadgroup per token
    uint tid  [[thread_position_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]])
{
    const int token_idx = gid;
    const int slot_idx = slot_mapping[token_idx];

    // Skip padding tokens
    if (slot_idx < 0) return;

    const int block_idx = slot_idx / block_size;
    const int block_offset = slot_idx % block_size;

    const int num_elems = num_heads * head_size;
    const int head_size_div_x = head_size / x;

    // Stride loop: each thread handles multiple elements
    for (int i = tid; i < num_elems; i += tpg) {
        const int head_idx = i / head_size;
        const int head_offset = i % head_size;

        const int src_idx = token_idx * key_stride + i;

        // K target: interleaved x layout for coalesced reads
        const int x_idx = head_offset / x;
        const int x_offset = head_offset % x;
        const int k_tgt = block_idx * num_heads * head_size_div_x * block_size * x
                        + head_idx * head_size_div_x * block_size * x
                        + x_idx * block_size * x
                        + block_offset * x
                        + x_offset;

        key_cache[k_tgt] = key[src_idx];

        // V target: simple layout [block, head, head_dim, block_offset]
        const int v_tgt = block_idx * num_heads * head_size * block_size
                        + head_idx * head_size * block_size
                        + head_offset * block_size
                        + block_offset;

        value_cache[v_tgt] = value[token_idx * value_stride + i];
    }
}

// Instantiations
template [[host_name("reshape_and_cache_float")]]
[[kernel]] void reshape_and_cache<float>(
    const device float*, const device float*, device float*, device float*,
    const device int*, constant int&, constant int&, constant int&,
    constant int&, constant int&, constant int&,
    uint, uint, uint);

template [[host_name("reshape_and_cache_half")]]
[[kernel]] void reshape_and_cache<half>(
    const device half*, const device half*, device half*, device half*,
    const device int*, constant int&, constant int&, constant int&,
    constant int&, constant int&, constant int&,
    uint, uint, uint);
