// copy_blocks.metal — Copy KV cache blocks for prefix sharing.
//
// block_mapping is a flat array of [src, dst] pairs:
//   [src0, dst0, src1, dst1, ...]
// Each threadgroup handles one (src, dst) pair.

#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void copy_blocks(
    device T             *key_cache       [[buffer(0)]],
    device T             *value_cache     [[buffer(1)]],
    const device int     *block_mapping   [[buffer(2)]],  // [num_pairs * 2]
    constant int         &numel_per_block [[buffer(3)]],  // elements per block in K/V cache
    uint gid  [[threadgroup_position_in_grid]],            // one per src-dst pair
    uint tid  [[thread_position_in_threadgroup]],
    uint tpg  [[threads_per_threadgroup]])
{
    const int src_block = block_mapping[gid * 2];
    const int dst_block = block_mapping[gid * 2 + 1];

    const int src_offset = src_block * numel_per_block;
    const int dst_offset = dst_block * numel_per_block;

    // Copy key cache block
    for (int i = tid; i < numel_per_block; i += tpg) {
        key_cache[dst_offset + i] = key_cache[src_offset + i];
    }

    // Copy value cache block
    for (int i = tid; i < numel_per_block; i += tpg) {
        value_cache[dst_offset + i] = value_cache[src_offset + i];
    }
}

// Instantiations
template [[host_name("copy_blocks_float")]]
[[kernel]] void copy_blocks<float>(
    device float*, device float*, const device int*,
    constant int&, uint, uint, uint);

template [[host_name("copy_blocks_half")]]
[[kernel]] void copy_blocks<half>(
    device half*, device half*, const device int*,
    constant int&, uint, uint, uint);
