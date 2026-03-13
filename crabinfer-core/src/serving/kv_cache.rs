//! KV cache manager: high-level block management for the scheduler.
//!
//! Translates between sequence-level operations (allocate slots for N new tokens,
//! free a sequence's blocks) and the block pool's physical block operations.
//!
//! Also provides FP8 E4M3 quantization/dequantization for KV cache compression.
//! FP8 E4M3 uses 1 byte per element (4-bit exponent, 3-bit mantissa) with per-head
//! scaling factors, giving 2x memory savings over FP16 at minimal accuracy cost.

use super::block::{BlockHash, BlockId};
use super::block_pool::{BlockPool, BlockPoolConfig};
use candle_core::{DType, Device, Result, Tensor};

/// Configuration for the KV cache manager.
#[derive(Debug, Clone)]
pub struct KVCacheConfig {
    /// Tokens per block.
    pub block_size: usize,
    /// Total number of physical blocks.
    pub num_blocks: usize,
    /// Number of KV heads (determines per-block memory).
    pub num_kv_heads: usize,
    /// Size of each attention head.
    pub head_size: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Whether to enable prefix caching.
    pub enable_prefix_cache: bool,
    /// Bytes per KV cache element (4 for F32, 2 for F16).
    pub dtype_bytes: usize,
}

impl KVCacheConfig {
    /// Calculate the total KV cache memory in bytes.
    ///
    /// Each block stores key and value tensors for all layers.
    /// Key layout: [num_kv_heads, head_size/x, block_size, x] per block per layer
    /// Value layout: [num_kv_heads, head_size, block_size] per block per layer
    pub fn total_memory_bytes(&self) -> usize {
        let kv_per_block_per_layer =
            2 * self.num_kv_heads * self.head_size * self.block_size * self.dtype_bytes;
        self.num_blocks * self.num_layers * kv_per_block_per_layer
    }

    /// Calculate the number of blocks that fit in a given memory budget.
    pub fn blocks_for_memory(
        memory_bytes: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_size: usize,
        num_layers: usize,
        dtype_bytes: usize,
    ) -> usize {
        let kv_per_block_per_layer =
            2 * num_kv_heads * head_size * block_size * dtype_bytes;
        let bytes_per_block = num_layers * kv_per_block_per_layer;
        if bytes_per_block == 0 {
            return 0;
        }
        memory_bytes / bytes_per_block
    }
}

/// Per-sequence block tracking.
#[derive(Debug)]
pub struct SequenceBlocks {
    /// Physical block IDs allocated for this sequence, in order.
    pub block_ids: Vec<BlockId>,
    /// Content hashes for each block (for prefix caching).
    pub block_hashes: Vec<BlockHash>,
    /// Total number of tokens computed (prefix cached + generated).
    pub num_computed_tokens: usize,
}

impl SequenceBlocks {
    pub fn new() -> Self {
        Self {
            block_ids: Vec::new(),
            block_hashes: Vec::new(),
            num_computed_tokens: 0,
        }
    }

    /// Number of blocks currently allocated.
    pub fn num_blocks(&self) -> usize {
        self.block_ids.len()
    }
}

/// High-level KV cache manager used by the scheduler.
///
/// Manages the mapping between sequences and physical KV cache blocks,
/// handling allocation, freeing, and prefix cache coordination.
pub struct KVCacheManager {
    pool: BlockPool,
    block_size: usize,
}

impl KVCacheManager {
    /// Create a new KV cache manager.
    pub fn new(config: &KVCacheConfig) -> Self {
        let pool_config = BlockPoolConfig {
            block_size: config.block_size,
            num_blocks: config.num_blocks,
            enable_prefix_cache: config.enable_prefix_cache,
        };
        Self {
            pool: BlockPool::new(pool_config),
            block_size: config.block_size,
        }
    }

    /// Block size (tokens per block).
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Number of free blocks available.
    pub fn num_free_blocks(&self) -> usize {
        self.pool.num_free_blocks()
    }

    /// Total number of blocks in the pool.
    pub fn num_total_blocks(&self) -> usize {
        self.pool.num_blocks()
    }

    /// Look up prefix cache hits for a sequence's token content.
    ///
    /// Returns the block IDs that can be reused and the number of
    /// tokens they cover. The caller should update `seq_blocks.num_computed_tokens`
    /// accordingly.
    pub fn get_computed_blocks(
        &mut self,
        block_hashes: &[BlockHash],
    ) -> (Vec<BlockId>, usize) {
        self.pool.find_prefix_cache_hit(block_hashes)
    }

    /// Prefix cache hit rate (0.0–1.0). Returns 0.0 if no lookups yet.
    pub fn prefix_cache_hit_rate(&self) -> f64 {
        self.pool.prefix_cache_hit_rate()
    }

    /// Number of prefix cache hits.
    pub fn prefix_cache_hits(&self) -> u64 {
        self.pool.prefix_cache_hits()
    }

    /// Number of prefix cache misses.
    pub fn prefix_cache_misses(&self) -> u64 {
        self.pool.prefix_cache_misses()
    }

    /// Allocate KV cache slots for `num_new_tokens` new tokens.
    ///
    /// This handles the common pattern:
    /// 1. Check if the last existing block has room for more tokens
    /// 2. Allocate new blocks as needed for the remaining tokens
    /// 3. Touch prefix-cached blocks to prevent eviction
    ///
    /// Returns the newly allocated block IDs, or None if insufficient blocks.
    pub fn allocate_slots(
        &mut self,
        seq_blocks: &mut SequenceBlocks,
        num_new_tokens: usize,
        prefix_cache_blocks: Option<&[BlockId]>,
    ) -> Option<Vec<BlockId>> {
        // Touch prefix-cached blocks first (rescue from free queue)
        if let Some(cached_blocks) = prefix_cache_blocks {
            self.pool.touch(cached_blocks);
            // Add cached blocks to the sequence if not already there
            for &block_id in cached_blocks {
                if !seq_blocks.block_ids.contains(&block_id) {
                    seq_blocks.block_ids.push(block_id);
                }
            }
        }

        // Calculate how many new blocks we need
        let total_tokens = seq_blocks.num_computed_tokens + num_new_tokens;
        let total_blocks_needed = (total_tokens + self.block_size - 1) / self.block_size;
        let existing_blocks = seq_blocks.block_ids.len();

        if total_blocks_needed <= existing_blocks {
            // Existing blocks have room — no allocation needed
            return Some(Vec::new());
        }

        let new_blocks_needed = total_blocks_needed - existing_blocks;

        // Allocate new blocks
        let new_block_ids = self.pool.allocate(new_blocks_needed)?;
        seq_blocks.block_ids.extend_from_slice(&new_block_ids);

        Some(new_block_ids)
    }

    /// Free all blocks for a sequence.
    ///
    /// Decrements ref counts. Blocks with ref_count=0 return to the free queue
    /// but retain their content hash for prefix caching.
    pub fn free(&mut self, seq_blocks: &mut SequenceBlocks) {
        self.pool.free(&seq_blocks.block_ids);
        seq_blocks.block_ids.clear();
        seq_blocks.block_hashes.clear();
        seq_blocks.num_computed_tokens = 0;
    }

    /// Register a full block's content hash for prefix caching.
    pub fn cache_block(&mut self, block_id: BlockId, hash: BlockHash, num_tokens: usize) {
        self.pool.set_block_tokens(block_id, num_tokens);
        self.pool.cache_block(block_id, hash);
    }

    /// Check if `n` new blocks can be allocated.
    pub fn can_allocate(&self, n: usize) -> bool {
        self.pool.can_allocate(n)
    }

    /// Compute the slot index for a token at a given position.
    ///
    /// slot = block_table[position / block_size] * block_size + (position % block_size)
    pub fn compute_slot(
        &self,
        seq_blocks: &SequenceBlocks,
        position: usize,
    ) -> Option<usize> {
        let block_idx = position / self.block_size;
        let block_offset = position % self.block_size;

        seq_blocks.block_ids.get(block_idx).map(|&block_id| {
            block_id * self.block_size + block_offset
        })
    }

    /// Compute slot mappings for a range of token positions.
    ///
    /// Returns a vec of physical slot indices, one per token.
    pub fn compute_slot_mapping(
        &self,
        seq_blocks: &SequenceBlocks,
        start_pos: usize,
        num_tokens: usize,
    ) -> Vec<usize> {
        (start_pos..start_pos + num_tokens)
            .map(|pos| {
                self.compute_slot(seq_blocks, pos)
                    .expect("position exceeds allocated blocks")
            })
            .collect()
    }

    /// Build the block table for a sequence (for the paged attention kernel).
    ///
    /// Returns a copy of the sequence's block ID list, which maps
    /// logical block index → physical block ID.
    pub fn block_table(&self, seq_blocks: &SequenceBlocks) -> Vec<u32> {
        seq_blocks
            .block_ids
            .iter()
            .map(|&id| id as u32)
            .collect()
    }

    /// Fork a sequence's blocks for beam search (Copy-on-Write).
    ///
    /// Creates a new `SequenceBlocks` that shares the parent's physical blocks
    /// by incrementing reference counts. No data is copied — the actual KV data
    /// is shared until a write triggers CoW via `cow_block_if_needed`.
    pub fn fork_blocks(&mut self, parent: &SequenceBlocks) -> SequenceBlocks {
        for &block_id in &parent.block_ids {
            self.pool.increment_ref(block_id);
        }
        SequenceBlocks {
            block_ids: parent.block_ids.clone(),
            block_hashes: parent.block_hashes.clone(),
            num_computed_tokens: parent.num_computed_tokens,
        }
    }

    /// Copy-on-Write: ensure the last block in a sequence is exclusively owned.
    ///
    /// If the last block has refcount > 1 (shared with sibling beams), a new
    /// block is allocated and the old block's refcount is decremented. The
    /// caller must copy the KV data from `old_block_id` to `new_block_id`
    /// via the kernel backend's `copy_blocks` method.
    ///
    /// Returns `Some((old_block_id, new_block_id))` if a copy is needed,
    /// or `None` if the block is already exclusively owned.
    pub fn cow_block_if_needed(
        &mut self,
        seq_blocks: &mut SequenceBlocks,
    ) -> Option<(BlockId, BlockId)> {
        let last_idx = seq_blocks.block_ids.len().checked_sub(1)?;
        let old_block_id = seq_blocks.block_ids[last_idx];

        let new_block_id = self.pool.cow_allocate(old_block_id)?;

        // Update the sequence's block table to point to the new block
        seq_blocks.block_ids[last_idx] = new_block_id;

        Some((old_block_id, new_block_id))
    }

    /// Get the reference count of a block.
    pub fn block_ref_count(&self, block_id: BlockId) -> u32 {
        self.pool.ref_count(block_id)
    }

    /// Return the content hashes of all actively allocated blocks.
    ///
    /// Useful for cache-aware load balancing: a router can compare a request's
    /// prefix hashes against each worker's active hashes to route to the
    /// worker with the best prefix match.
    pub fn active_block_hashes(&self) -> Vec<BlockHash> {
        self.pool.active_block_hashes()
    }

    /// Reset prefix cache (invalidate all cached blocks).
    pub fn reset_prefix_cache(&mut self) {
        self.pool.reset_prefix_cache();
    }
}

// ─── FP8 E4M3 KV Cache Quantization ─────────────────────────────────────────

/// Maximum representable value in FP8 E4M3 format.
/// E4M3: 4-bit exponent (bias=7), 3-bit mantissa, no inf (NaN reserved).
/// Max = 2^(14-7) * (1 + 7/8) = 128 * 1.875 = 240.0
const FP8_E4M3_MAX: f32 = 240.0;

/// Convert an f32 value to FP8 E4M3 representation (as u8).
///
/// FP8 E4M3 format: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits.
/// Special values: 0x7F = NaN, no infinity representation.
/// Range: [-240, 240], min subnormal: 2^-9 = 0.001953125
#[inline]
fn f32_to_fp8_e4m3(val: f32) -> u8 {
    if val.is_nan() {
        return 0x7F; // NaN
    }

    let sign = if val < 0.0 { 1u8 } else { 0u8 };
    let abs_val = val.abs();

    if abs_val == 0.0 {
        return sign << 7; // signed zero
    }

    // Clamp to FP8 E4M3 range
    let abs_val = abs_val.min(FP8_E4M3_MAX);

    // Minimum subnormal: 2^(-6) * (1/8) = 2^-9
    let min_subnormal = 2.0f32.powi(-9);
    if abs_val < min_subnormal {
        return sign << 7; // flush to zero
    }

    let bits = abs_val.to_bits();
    let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127; // unbiased exponent
    let f32_mant = bits & 0x7FFFFF; // 23-bit mantissa

    // Check if subnormal in FP8 (exp < -6 after bias)
    if f32_exp < -6 {
        // Subnormal FP8: value = 2^(-6) * (mantissa / 8)
        // mantissa bits represent 0.xxx in binary
        let shift = (-6 - f32_exp) as u32;
        // The implicit 1.mantissa in f32 becomes 0.001... in fp8 subnormal
        let full_mant = (1u32 << 3) | ((f32_mant >> 20) & 0x7);
        let mant = if shift < 4 {
            (full_mant >> shift) & 0x7
        } else {
            0
        };
        if mant == 0 {
            return sign << 7;
        }
        return (sign << 7) | (mant as u8);
    }

    // Normal FP8 value
    let biased_exp = (f32_exp + 7) as u8;
    if biased_exp >= 15 {
        // Overflow: clamp to max value (no inf in E4M3)
        return (sign << 7) | 0x7E; // max normal: exp=14, mant=7 -> 240
    }

    // Round mantissa from 23 bits to 3 bits (round to nearest even)
    let mant = ((f32_mant >> 20) & 0x7) as u8;
    let round_bit = (f32_mant >> 19) & 1;
    let sticky = f32_mant & 0x7FFFF;

    let mant = if round_bit == 1 && (sticky != 0 || (mant & 1) != 0) {
        mant + 1
    } else {
        mant
    };

    // Handle mantissa overflow from rounding
    if mant >= 8 {
        let new_exp = biased_exp + 1;
        if new_exp >= 15 {
            return (sign << 7) | 0x7E; // clamp to max
        }
        return (sign << 7) | (new_exp << 3);
    }

    (sign << 7) | (biased_exp << 3) | mant
}

/// Convert an FP8 E4M3 value (as u8) back to f32.
#[inline]
fn fp8_e4m3_to_f32(val: u8) -> f32 {
    let sign = (val >> 7) & 1;
    let exp = (val >> 3) & 0xF;
    let mant = val & 0x7;

    if exp == 0xF && mant == 0x7 {
        return f32::NAN; // 0x7F or 0xFF = NaN
    }

    let abs_val = if exp == 0 {
        if mant == 0 {
            0.0f32
        } else {
            // Subnormal: 2^(-6) * (mant / 8)
            2.0f32.powi(-6) * (mant as f32 / 8.0)
        }
    } else {
        // Normal: 2^(exp - 7) * (1 + mant/8)
        2.0f32.powi(exp as i32 - 7) * (1.0 + mant as f32 / 8.0)
    };

    if sign == 1 {
        -abs_val
    } else {
        abs_val
    }
}

/// Quantize a KV tensor to FP8 E4M3 with per-head scaling factors.
///
/// Input: `[total_tokens, num_heads, head_size]` in F16 or F32
/// Returns: `(fp8_data, scales)` where:
///   - `fp8_data`: `[total_tokens, num_heads, head_size]` as U8 (FP8 E4M3 encoded)
///   - `scales`: `[total_tokens, num_heads]` as F32 (one scale factor per head per token)
///
/// The scale maps the head's value range to the FP8 representable range.
/// To dequantize: `value_f32 = fp8_to_f32(fp8_data) * scale`
pub fn quantize_kv_to_fp8(kv: &Tensor) -> Result<(Tensor, Tensor)> {
    let dims = kv.dims();
    if dims.len() != 3 {
        return Err(candle_core::Error::Msg(format!(
            "quantize_kv_to_fp8: expected 3D tensor [tokens, heads, head_size], got {:?}",
            dims
        )));
    }
    let (num_tokens, num_heads, head_size) = (dims[0], dims[1], dims[2]);

    // Convert to F32 for quantization
    let kv_f32 = kv.to_dtype(DType::F32)?;
    let kv_data: Vec<f32> = kv_f32.flatten_all()?.to_vec1()?;

    let mut fp8_data = vec![0u8; num_tokens * num_heads * head_size];
    let mut scales = vec![0.0f32; num_tokens * num_heads];

    for t in 0..num_tokens {
        for h in 0..num_heads {
            let head_offset = (t * num_heads + h) * head_size;
            let head_slice = &kv_data[head_offset..head_offset + head_size];

            // Compute per-head absmax
            let absmax = head_slice.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));

            // Scale: maps absmax to FP8_E4M3_MAX
            let scale = if absmax > 0.0 {
                absmax / FP8_E4M3_MAX
            } else {
                1.0
            };
            let inv_scale = 1.0 / scale;

            scales[t * num_heads + h] = scale;

            // Quantize each element
            for d in 0..head_size {
                let scaled_val = head_slice[d] * inv_scale;
                fp8_data[head_offset + d] = f32_to_fp8_e4m3(scaled_val);
            }
        }
    }

    let fp8_tensor = Tensor::from_vec(fp8_data, (num_tokens, num_heads, head_size), kv.device())?;
    let scale_tensor = Tensor::from_vec(scales, (num_tokens, num_heads), kv.device())?;

    Ok((fp8_tensor, scale_tensor))
}

/// Dequantize FP8 E4M3 KV data back to the target compute dtype.
///
/// Input:
///   - `fp8_data`: `[total_tokens, num_heads, head_size]` as U8 (FP8 E4M3 encoded)
///   - `scale`: `[total_tokens, num_heads]` as F32
///   - `target_dtype`: the dtype to dequantize to (F16, BF16, or F32)
///
/// Returns: `[total_tokens, num_heads, head_size]` in `target_dtype`
pub fn dequantize_fp8_kv(fp8_data: &Tensor, scale: &Tensor, target_dtype: DType) -> Result<Tensor> {
    let dims = fp8_data.dims();
    if dims.len() != 3 {
        return Err(candle_core::Error::Msg(format!(
            "dequantize_fp8_kv: expected 3D tensor [tokens, heads, head_size], got {:?}",
            dims
        )));
    }
    let (num_tokens, num_heads, head_size) = (dims[0], dims[1], dims[2]);

    let fp8_bytes: Vec<u8> = fp8_data.flatten_all()?.to_vec1()?;
    let scale_data: Vec<f32> = scale.flatten_all()?.to_vec1()?;

    let mut output = vec![0.0f32; num_tokens * num_heads * head_size];

    for t in 0..num_tokens {
        for h in 0..num_heads {
            let head_offset = (t * num_heads + h) * head_size;
            let s = scale_data[t * num_heads + h];

            for d in 0..head_size {
                let fp8_val = fp8_bytes[head_offset + d];
                output[head_offset + d] = fp8_e4m3_to_f32(fp8_val) * s;
            }
        }
    }

    let result = Tensor::from_vec(output, (num_tokens, num_heads, head_size), fp8_data.device())?;
    result.to_dtype(target_dtype)
}

/// Compute the memory overhead ratio for FP8 scale storage.
///
/// For each block of `block_size` tokens with `num_kv_heads` heads,
/// we store `block_size * num_kv_heads` f32 scale factors (4 bytes each)
/// in addition to the FP8 data (1 byte per element).
///
/// Returns the effective bytes per element including scale overhead.
pub fn fp8_effective_bytes_per_element(num_kv_heads: usize, head_size: usize, block_size: usize) -> f64 {
    let data_bytes = num_kv_heads * head_size * block_size; // 1 byte each
    let scale_bytes = num_kv_heads * block_size * 4; // f32 scales
    (data_bytes + scale_bytes) as f64 / (num_kv_heads * head_size * block_size) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> KVCacheConfig {
        KVCacheConfig {
            block_size: 16,
            num_blocks: 8,
            num_kv_heads: 8,
            head_size: 128,
            num_layers: 32,
            enable_prefix_cache: true,
            dtype_bytes: 2,
        }
    }

    #[test]
    fn test_memory_calculation() {
        let config = test_config();
        let bytes = config.total_memory_bytes();
        // 8 blocks * 32 layers * 2 (k+v) * 8 heads * 128 dim * 16 tokens * 2 bytes
        let expected = 8 * 32 * 2 * 8 * 128 * 16 * 2;
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_blocks_for_memory() {
        let blocks = KVCacheConfig::blocks_for_memory(
            1024 * 1024 * 1024, // 1 GB
            16,                  // block_size
            8,                   // num_kv_heads
            128,                 // head_size
            32,                  // num_layers
            2,                   // dtype_bytes (F16)
        );
        // Each block: 32 layers * 2(kv) * 8 heads * 128 dim * 16 tokens * 2 bytes
        //           = 32 * 2 * 8 * 128 * 16 * 2 = 2,097,152 bytes = 2 MB
        // 1 GB / 2 MB = 512 blocks
        assert_eq!(blocks, 512);
    }

    #[test]
    fn test_allocate_slots_new_sequence() {
        let config = test_config();
        let mut mgr = KVCacheManager::new(&config);
        let mut seq = SequenceBlocks::new();

        // Allocate for 48 tokens (3 blocks of 16)
        let new_blocks = mgr.allocate_slots(&mut seq, 48, None).unwrap();
        assert_eq!(new_blocks.len(), 3);
        assert_eq!(seq.block_ids.len(), 3);
        assert_eq!(mgr.num_free_blocks(), 5);
    }

    #[test]
    fn test_allocate_slots_incremental() {
        let config = test_config();
        let mut mgr = KVCacheManager::new(&config);
        let mut seq = SequenceBlocks::new();

        // Allocate for 10 tokens (1 block, partially filled)
        mgr.allocate_slots(&mut seq, 10, None).unwrap();
        assert_eq!(seq.block_ids.len(), 1);

        // Allocate 5 more tokens (still fits in the same block)
        seq.num_computed_tokens = 10;
        let new = mgr.allocate_slots(&mut seq, 5, None).unwrap();
        assert_eq!(new.len(), 0); // No new blocks needed
        assert_eq!(seq.block_ids.len(), 1);

        // Allocate 5 more tokens (spills into a second block)
        seq.num_computed_tokens = 15;
        let new = mgr.allocate_slots(&mut seq, 5, None).unwrap();
        assert_eq!(new.len(), 1);
        assert_eq!(seq.block_ids.len(), 2);
    }

    #[test]
    fn test_allocate_slots_insufficient() {
        let config = KVCacheConfig {
            num_blocks: 2,
            ..test_config()
        };
        let mut mgr = KVCacheManager::new(&config);
        let mut seq = SequenceBlocks::new();

        // Try to allocate 48 tokens = 3 blocks, but only 2 available
        assert!(mgr.allocate_slots(&mut seq, 48, None).is_none());
    }

    #[test]
    fn test_free_sequence() {
        let config = test_config();
        let mut mgr = KVCacheManager::new(&config);
        let mut seq = SequenceBlocks::new();

        mgr.allocate_slots(&mut seq, 32, None).unwrap();
        assert_eq!(mgr.num_free_blocks(), 6);

        mgr.free(&mut seq);
        assert_eq!(mgr.num_free_blocks(), 8);
        assert!(seq.block_ids.is_empty());
        assert_eq!(seq.num_computed_tokens, 0);
    }

    #[test]
    fn test_compute_slot() {
        let config = test_config();
        let mut mgr = KVCacheManager::new(&config);
        let mut seq = SequenceBlocks::new();

        mgr.allocate_slots(&mut seq, 32, None).unwrap();
        // Blocks are 0 and 1 (first two allocated from pool)

        // Token at position 0 → block 0, offset 0 → slot 0*16+0 = 0
        assert_eq!(mgr.compute_slot(&seq, 0), Some(seq.block_ids[0] * 16));

        // Token at position 15 → block 0, offset 15
        assert_eq!(mgr.compute_slot(&seq, 15), Some(seq.block_ids[0] * 16 + 15));

        // Token at position 16 → block 1, offset 0
        assert_eq!(mgr.compute_slot(&seq, 16), Some(seq.block_ids[1] * 16));
    }

    #[test]
    fn test_compute_slot_mapping() {
        let config = test_config();
        let mut mgr = KVCacheManager::new(&config);
        let mut seq = SequenceBlocks::new();

        mgr.allocate_slots(&mut seq, 32, None).unwrap();

        let slots = mgr.compute_slot_mapping(&seq, 14, 4);
        assert_eq!(slots.len(), 4);
        // Positions 14, 15, 16, 17 — crosses a block boundary
        assert_eq!(slots[0], seq.block_ids[0] * 16 + 14);
        assert_eq!(slots[1], seq.block_ids[0] * 16 + 15);
        assert_eq!(slots[2], seq.block_ids[1] * 16 + 0);
        assert_eq!(slots[3], seq.block_ids[1] * 16 + 1);
    }

    #[test]
    fn test_block_table() {
        let config = test_config();
        let mut mgr = KVCacheManager::new(&config);
        let mut seq = SequenceBlocks::new();

        mgr.allocate_slots(&mut seq, 48, None).unwrap();

        let table = mgr.block_table(&seq);
        assert_eq!(table.len(), 3);
        // Block IDs should be the first 3 allocated (0, 1, 2)
        assert_eq!(table, vec![0, 1, 2]);
    }

    #[test]
    fn test_fork_blocks_shares_refcounts() {
        let config = test_config();
        let mut mgr = KVCacheManager::new(&config);
        let mut parent = SequenceBlocks::new();

        // Allocate 2 blocks for parent
        mgr.allocate_slots(&mut parent, 32, None).unwrap();
        parent.num_computed_tokens = 32;
        assert_eq!(mgr.block_ref_count(parent.block_ids[0]), 1);

        // Fork: child shares parent's blocks
        let child = mgr.fork_blocks(&parent);
        assert_eq!(child.block_ids, parent.block_ids);
        assert_eq!(child.num_computed_tokens, 32);
        assert_eq!(mgr.block_ref_count(parent.block_ids[0]), 2);
        assert_eq!(mgr.block_ref_count(parent.block_ids[1]), 2);

        // Freeing parent's blocks decrements refcount but blocks stay allocated
        mgr.free(&mut parent);
        assert_eq!(mgr.block_ref_count(child.block_ids[0]), 1);
    }

    #[test]
    fn test_cow_block_if_needed_shared() {
        let config = KVCacheConfig {
            num_blocks: 8,
            ..test_config()
        };
        let mut mgr = KVCacheManager::new(&config);
        let mut parent = SequenceBlocks::new();

        // Allocate 2 blocks for parent
        mgr.allocate_slots(&mut parent, 32, None).unwrap();
        parent.num_computed_tokens = 32;

        // Fork
        let mut child = mgr.fork_blocks(&parent);

        // CoW on child's last block (shared, refcount=2)
        let result = mgr.cow_block_if_needed(&mut child);
        assert!(result.is_some());
        let (old_id, new_id) = result.unwrap();
        assert_eq!(old_id, parent.block_ids[1]);
        assert_ne!(new_id, old_id);

        // Child's block table updated
        assert_eq!(child.block_ids[1], new_id);
        // Parent's block table unchanged
        assert_eq!(mgr.block_ref_count(parent.block_ids[1]), 1);
        assert_eq!(mgr.block_ref_count(new_id), 1);
    }

    #[test]
    fn test_cow_block_if_needed_exclusive() {
        let config = test_config();
        let mut mgr = KVCacheManager::new(&config);
        let mut seq = SequenceBlocks::new();

        mgr.allocate_slots(&mut seq, 32, None).unwrap();
        seq.num_computed_tokens = 32;

        // No fork, so refcount=1 — CoW should return None
        assert!(mgr.cow_block_if_needed(&mut seq).is_none());
    }

    #[test]
    fn test_prefix_cache_through_manager() {
        let config = test_config();
        let mut mgr = KVCacheManager::new(&config);
        let mut seq1 = SequenceBlocks::new();

        // Sequence 1: allocate and fill 2 blocks
        mgr.allocate_slots(&mut seq1, 32, None).unwrap();
        let hash0 = BlockHash::from_tokens(&[1, 2, 3], None);
        let hash1 = BlockHash::from_tokens(&[4, 5, 6], Some(hash0));
        mgr.cache_block(seq1.block_ids[0], hash0, 16);
        mgr.cache_block(seq1.block_ids[1], hash1, 16);

        // Free sequence 1
        let old_block_ids = seq1.block_ids.clone();
        mgr.free(&mut seq1);

        // Sequence 2: look up prefix cache
        let (cached_blocks, cached_tokens) = mgr.get_computed_blocks(&[hash0, hash1]);
        assert_eq!(cached_blocks.len(), 2);
        assert_eq!(cached_tokens, 32);
        assert_eq!(cached_blocks, old_block_ids);

        // Allocate slots using cached blocks
        let mut seq2 = SequenceBlocks::new();
        seq2.num_computed_tokens = cached_tokens;
        let new_blocks = mgr
            .allocate_slots(&mut seq2, 16, Some(&cached_blocks))
            .unwrap();
        // Should have rescued the 2 cached blocks + allocated 1 new
        assert_eq!(seq2.block_ids.len(), 3);
        assert_eq!(new_blocks.len(), 1);
    }

    #[test]
    fn test_different_cache_salt_produces_different_hashes() {
        // Two sequences with same tokens but different cache_salt should get different block hashes
        let tokens = &[1u32, 2, 3, 4, 5];
        let hash_a = BlockHash::from_tokens_salted(tokens, None, Some("tenant-a"));
        let hash_b = BlockHash::from_tokens_salted(tokens, None, Some("tenant-b"));
        let hash_none = BlockHash::from_tokens_salted(tokens, None, None);
        let hash_old = BlockHash::from_tokens(tokens, None);

        assert_ne!(hash_a, hash_b, "different salts must produce different hashes");
        assert_ne!(hash_a, hash_none, "salted must differ from unsalted");
        assert_eq!(hash_none, hash_old, "salt=None must match legacy from_tokens");
    }

    #[test]
    fn test_salted_prefix_cache_isolation() {
        let config = test_config();
        let mut mgr = KVCacheManager::new(&config);

        // Sequence 1: tenant-a caches blocks
        let mut seq1 = SequenceBlocks::new();
        mgr.allocate_slots(&mut seq1, 32, None).unwrap();
        let hash0_a = BlockHash::from_tokens_salted(&[1, 2, 3], None, Some("tenant-a"));
        let hash1_a = BlockHash::from_tokens_salted(&[4, 5, 6], Some(hash0_a), Some("tenant-a"));
        mgr.cache_block(seq1.block_ids[0], hash0_a, 16);
        mgr.cache_block(seq1.block_ids[1], hash1_a, 16);
        mgr.free(&mut seq1);

        // Sequence 2: tenant-b looks up with different salt -- should miss
        let hash0_b = BlockHash::from_tokens_salted(&[1, 2, 3], None, Some("tenant-b"));
        let (hit_blocks, _) = mgr.get_computed_blocks(&[hash0_b]);
        assert!(hit_blocks.is_empty(), "different salt must not find cached blocks");

        // Sequence 3: tenant-a looks up with same salt -- should hit
        let hash0_a2 = BlockHash::from_tokens_salted(&[1, 2, 3], None, Some("tenant-a"));
        let (hit_blocks, num_tokens) = mgr.get_computed_blocks(&[hash0_a2]);
        assert_eq!(hit_blocks.len(), 1, "same salt should find cached blocks");
        assert_eq!(num_tokens, 16);
    }

    #[test]
    fn test_prefix_cache_hit_rate_through_manager() {
        let config = test_config();
        let mut mgr = KVCacheManager::new(&config);
        assert_eq!(mgr.prefix_cache_hit_rate(), 0.0);

        // Cache some blocks
        let mut seq = SequenceBlocks::new();
        mgr.allocate_slots(&mut seq, 32, None).unwrap();
        let hash0 = BlockHash::from_tokens(&[1, 2, 3], None);
        let hash1 = BlockHash::from_tokens(&[4, 5, 6], Some(hash0));
        mgr.cache_block(seq.block_ids[0], hash0, 16);
        mgr.cache_block(seq.block_ids[1], hash1, 16);
        mgr.free(&mut seq);

        // Hit
        let (blocks, _) = mgr.get_computed_blocks(&[hash0, hash1]);
        assert_eq!(blocks.len(), 2);
        assert_eq!(mgr.prefix_cache_hits(), 1);
        assert_eq!(mgr.prefix_cache_hit_rate(), 1.0);

        // Miss
        let unknown = BlockHash::from_tokens(&[99], None);
        let (blocks, _) = mgr.get_computed_blocks(&[unknown]);
        assert!(blocks.is_empty());
        assert_eq!(mgr.prefix_cache_misses(), 1);
        assert_eq!(mgr.prefix_cache_hit_rate(), 0.5);
    }

    // ── FP8 E4M3 quantization tests ──────────────────────────────────────

    #[test]
    fn test_fp8_e4m3_roundtrip_basic() {
        // Test that basic values survive roundtrip through FP8
        let test_cases: &[(f32, f32)] = &[
            (0.0, 0.0),
            (1.0, 1.0),
            (-1.0, -1.0),
            (0.5, 0.5),
            (2.0, 2.0),
            (240.0, 240.0),  // max value
            (-240.0, -240.0),
        ];
        for &(input, expected) in test_cases {
            let fp8 = super::f32_to_fp8_e4m3(input);
            let output = super::fp8_e4m3_to_f32(fp8);
            assert!(
                (output - expected).abs() < 0.01,
                "roundtrip failed for {input}: got {output}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_fp8_e4m3_nan() {
        let fp8 = super::f32_to_fp8_e4m3(f32::NAN);
        assert_eq!(fp8, 0x7F);
        assert!(super::fp8_e4m3_to_f32(0x7F).is_nan());
    }

    #[test]
    fn test_fp8_e4m3_clamps_overflow() {
        // Values beyond 240 should clamp
        let fp8 = super::f32_to_fp8_e4m3(500.0);
        let result = super::fp8_e4m3_to_f32(fp8);
        assert!((result - 240.0).abs() < 1.0, "expected ~240, got {result}");
    }

    #[test]
    fn test_fp8_e4m3_small_values() {
        // Very small values should either survive as subnormals or flush to zero
        let tiny = 0.001;
        let fp8 = super::f32_to_fp8_e4m3(tiny);
        let result = super::fp8_e4m3_to_f32(fp8);
        // Should be close-ish (FP8 subnormals have limited precision)
        assert!(result.abs() < 0.01, "expected near zero, got {result}");
    }

    #[test]
    fn test_quantize_dequantize_kv_roundtrip() {
        let num_tokens = 4;
        let num_heads = 8;
        let head_size = 16;
        let device = Device::Cpu;

        // Create a random KV tensor with values in a reasonable range
        let kv = Tensor::randn(
            0f32,
            1.0,
            (num_tokens, num_heads, head_size),
            &device,
        )
        .unwrap();

        let (fp8_data, scales) = super::quantize_kv_to_fp8(&kv).unwrap();

        // Check shapes
        assert_eq!(fp8_data.dims(), &[num_tokens, num_heads, head_size]);
        assert_eq!(fp8_data.dtype(), DType::U8);
        assert_eq!(scales.dims(), &[num_tokens, num_heads]);
        assert_eq!(scales.dtype(), DType::F32);

        // Dequantize back to F32
        let restored = super::dequantize_fp8_kv(&fp8_data, &scales, DType::F32).unwrap();
        assert_eq!(restored.dims(), &[num_tokens, num_heads, head_size]);

        // Check that values are close (FP8 has ~0.1% relative error for normal range)
        let orig: Vec<f32> = kv.flatten_all().unwrap().to_vec1().unwrap();
        let rest: Vec<f32> = restored.flatten_all().unwrap().to_vec1().unwrap();

        let mut max_rel_err = 0.0f32;
        for (o, r) in orig.iter().zip(rest.iter()) {
            if o.abs() > 0.01 {
                let rel_err = (o - r).abs() / o.abs();
                max_rel_err = max_rel_err.max(rel_err);
            }
        }
        // FP8 E4M3 has 3 mantissa bits = ~12.5% relative precision,
        // but with per-head scaling it should be much better
        assert!(
            max_rel_err < 0.15,
            "FP8 roundtrip max relative error too high: {max_rel_err}"
        );
    }

    #[test]
    fn test_quantize_dequantize_kv_to_f16() {
        let device = Device::Cpu;
        let kv = Tensor::randn(0f32, 1.0, (2, 4, 8), &device).unwrap();

        let (fp8_data, scales) = super::quantize_kv_to_fp8(&kv).unwrap();
        let restored = super::dequantize_fp8_kv(&fp8_data, &scales, DType::F16).unwrap();

        assert_eq!(restored.dtype(), DType::F16);
        assert_eq!(restored.dims(), &[2, 4, 8]);
    }

    #[test]
    fn test_fp8_memory_savings_vs_fp16() {
        // FP8 should use half the memory of FP16 for the data portion
        let num_kv_heads = 8;
        let head_size = 128;
        let block_size = 16;

        let fp16_bytes_per_block = 2 * num_kv_heads * head_size * block_size * 2; // 2 bytes (FP16) * 2 (K+V)
        let fp8_bytes_per_block = 2 * num_kv_heads * head_size * block_size * 1; // 1 byte (FP8) * 2 (K+V)
        let fp8_scale_overhead = 2 * num_kv_heads * block_size * 4; // f32 scales for K and V

        let total_fp8 = fp8_bytes_per_block + fp8_scale_overhead;
        let ratio = total_fp8 as f64 / fp16_bytes_per_block as f64;

        // FP8 should be significantly smaller even with scale overhead
        // For head_size=128: overhead = 4/128 = 3.125%, so ratio ~ 0.53
        assert!(
            ratio < 0.6,
            "FP8 should use <60% of FP16 memory, got {:.1}%",
            ratio * 100.0
        );
    }

    #[test]
    fn test_fp8_effective_bytes_per_element() {
        let eff = super::fp8_effective_bytes_per_element(8, 128, 16);
        // 1 byte data + 4/128 bytes scale per element = 1.03125
        let expected = 1.0 + 4.0 / 128.0;
        assert!(
            (eff - expected).abs() < 0.001,
            "expected {expected}, got {eff}"
        );
    }

    #[test]
    fn test_fp8_block_size_calculation() {
        // Verify that KVCacheConfig with FP8 dtype_bytes=1 gives correct memory
        let config_fp16 = KVCacheConfig {
            dtype_bytes: 2,
            ..test_config()
        };
        let config_fp8 = KVCacheConfig {
            dtype_bytes: 1,
            ..test_config()
        };

        let mem_fp16 = config_fp16.total_memory_bytes();
        let mem_fp8 = config_fp8.total_memory_bytes();

        // FP8 data should be exactly half of FP16 (not counting scale overhead)
        assert_eq!(mem_fp8 * 2, mem_fp16);
    }

    #[test]
    fn test_fp8_blocks_for_memory_double_capacity() {
        // With FP8 (1 byte) vs FP16 (2 bytes), we should get 2x the blocks
        let memory = 1024 * 1024 * 1024; // 1 GB
        let blocks_fp16 = KVCacheConfig::blocks_for_memory(memory, 16, 8, 128, 32, 2);
        let blocks_fp8 = KVCacheConfig::blocks_for_memory(memory, 16, 8, 128, 32, 1);

        assert_eq!(blocks_fp8, blocks_fp16 * 2);
    }
}
