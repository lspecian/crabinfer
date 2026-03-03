//! KV cache manager: high-level block management for the scheduler.
//!
//! Translates between sequence-level operations (allocate slots for N new tokens,
//! free a sequence's blocks) and the block pool's physical block operations.

use super::block::{BlockHash, BlockId};
use super::block_pool::{BlockPool, BlockPoolConfig};

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
}

impl KVCacheConfig {
    /// Calculate the total KV cache memory in bytes.
    ///
    /// Each block stores key and value tensors for all layers.
    /// Key layout: [num_kv_heads, head_size/x, block_size, x] per block per layer
    /// Value layout: [num_kv_heads, head_size, block_size] per block per layer
    /// Both use 2 bytes per element (float16).
    pub fn total_memory_bytes(&self) -> usize {
        let bytes_per_element = 2; // float16
        let kv_per_block_per_layer =
            2 * self.num_kv_heads * self.head_size * self.block_size * bytes_per_element;
        self.num_blocks * self.num_layers * kv_per_block_per_layer
    }

    /// Calculate the number of blocks that fit in a given memory budget.
    pub fn blocks_for_memory(
        memory_bytes: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_size: usize,
        num_layers: usize,
    ) -> usize {
        let bytes_per_element = 2;
        let kv_per_block_per_layer =
            2 * num_kv_heads * head_size * block_size * bytes_per_element;
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
        &self,
        block_hashes: &[BlockHash],
    ) -> (Vec<BlockId>, usize) {
        self.pool.find_prefix_cache_hit(block_hashes)
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

    /// Reset prefix cache (invalidate all cached blocks).
    pub fn reset_prefix_cache(&mut self) {
        self.pool.reset_prefix_cache();
    }
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
}
