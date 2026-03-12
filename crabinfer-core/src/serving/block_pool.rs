//! Block pool: physical KV cache block allocation with prefix caching.
//!
//! Manages a fixed-size pool of physical blocks, providing O(1) allocation,
//! O(1) freeing, and content-hash-based prefix caching with LRU eviction.
//!
//! Design follows vLLM V1's BlockPool, simplified for unified memory
//! (no CPU/GPU swap — Apple Silicon shares a single memory pool).

use super::block::{BlockHash, BlockHashMap, BlockId, FreeBlockQueue, KVCacheBlock};

/// Configuration for the block pool.
#[derive(Debug, Clone)]
pub struct BlockPoolConfig {
    /// Number of tokens per block.
    pub block_size: usize,
    /// Total number of physical blocks to allocate.
    pub num_blocks: usize,
    /// Whether to enable prefix caching.
    pub enable_prefix_cache: bool,
}

/// Manages a pool of physical KV cache blocks.
///
/// Provides allocation, freeing, and prefix cache operations.
/// All operations are O(1) amortized.
pub struct BlockPool {
    config: BlockPoolConfig,
    /// Free block queue with LRU eviction ordering.
    free_queue: FreeBlockQueue,
    /// Content hash → block ID mapping for prefix cache.
    hash_map: BlockHashMap,
    /// Number of prefix cache lookups that found at least one cached block.
    prefix_cache_hits: u64,
    /// Number of prefix cache lookups that found zero cached blocks.
    prefix_cache_misses: u64,
}

impl BlockPool {
    /// Create a new block pool.
    pub fn new(config: BlockPoolConfig) -> Self {
        let free_queue = FreeBlockQueue::new(config.num_blocks);
        Self {
            config,
            free_queue,
            hash_map: BlockHashMap::new(),
            prefix_cache_hits: 0,
            prefix_cache_misses: 0,
        }
    }

    /// Number of tokens per block.
    pub fn block_size(&self) -> usize {
        self.config.block_size
    }

    /// Total number of physical blocks in the pool.
    pub fn num_blocks(&self) -> usize {
        self.config.num_blocks
    }

    /// Number of free blocks available for allocation.
    pub fn num_free_blocks(&self) -> usize {
        self.free_queue.len()
    }

    /// Number of allocated (in-use) blocks.
    pub fn num_allocated_blocks(&self) -> usize {
        self.config.num_blocks - self.free_queue.len()
    }

    /// Whether we can allocate `n` blocks (possibly by evicting cached blocks).
    pub fn can_allocate(&self, n: usize) -> bool {
        self.free_queue.len() >= n
    }

    /// Allocate `n` new blocks from the free pool.
    ///
    /// Blocks are taken from the LRU end of the free queue. If a block has
    /// a cached hash (prefix cache entry), the hash is evicted before reuse.
    ///
    /// Returns None if not enough free blocks are available.
    pub fn allocate(&mut self, n: usize) -> Option<Vec<BlockId>> {
        if self.free_queue.len() < n {
            return None;
        }

        let block_ids = self.free_queue.popleft_n(n);
        for &block_id in &block_ids {
            self.evict_cached_block(block_id);
            let block = self.free_queue.get_mut(block_id);
            block.ref_count = 1;
            block.num_tokens = 0;
        }
        Some(block_ids)
    }

    /// Free blocks, decrementing their reference count.
    ///
    /// Blocks with ref_count dropping to 0 are returned to the free queue.
    /// If prefix caching is enabled, freed blocks retain their hash for
    /// potential future reuse.
    ///
    /// Blocks are freed in reverse order so that tail blocks (most recently
    /// filled) are evicted first — head blocks (shared prefixes) survive longer.
    pub fn free(&mut self, block_ids: &[BlockId]) {
        for &block_id in block_ids.iter().rev() {
            let block = self.free_queue.get_mut(block_id);
            debug_assert!(block.ref_count > 0, "double-free of block {}", block_id);
            block.ref_count -= 1;

            if block.ref_count == 0 {
                // Return to free queue (at tail = most recently freed = last to evict)
                self.free_queue.append(block_id);
            }
        }
    }

    /// Increment reference count on blocks (e.g., when a prefix cache hit
    /// rescues blocks from the free queue for a new request).
    pub fn touch(&mut self, block_ids: &[BlockId]) {
        for &block_id in block_ids {
            let needs_rescue = self.free_queue.get(block_id).ref_count == 0;
            if needs_rescue {
                // Rescue from free queue
                self.free_queue.remove(block_id);
            }
            self.free_queue.get_mut(block_id).ref_count += 1;
        }
    }

    /// Register a full block's content hash for prefix caching.
    ///
    /// Called when a block has been completely filled with tokens and its
    /// content hash is known. The block can then be found by hash if freed.
    pub fn cache_block(&mut self, block_id: BlockId, hash: BlockHash) {
        if !self.config.enable_prefix_cache {
            return;
        }

        let block = self.free_queue.get_mut(block_id);
        // Remove old hash if present
        if let Some(old_hash) = block.block_hash {
            self.hash_map.remove(&old_hash, block_id);
        }
        block.block_hash = Some(hash);
        self.hash_map.insert(hash, block_id);
    }

    /// Look up the longest prefix cache hit for a sequence of block hashes.
    ///
    /// Returns the block IDs for the longest consecutive prefix of cache hits,
    /// and the number of tokens those blocks cover.
    ///
    /// The returned blocks are NOT yet touched (ref_count not incremented).
    /// The caller must call `touch()` before using them.
    pub fn find_prefix_cache_hit(&mut self, block_hashes: &[BlockHash]) -> (Vec<BlockId>, usize) {
        if !self.config.enable_prefix_cache {
            self.prefix_cache_misses += 1;
            return (Vec::new(), 0);
        }

        let mut hit_blocks = Vec::new();
        let mut num_tokens = 0;

        for hash in block_hashes {
            if let Some(block_ids) = self.hash_map.get(hash) {
                // Find a block that still has its content (not been overwritten)
                let valid_block = block_ids.iter().find(|&&bid| {
                    let block = self.free_queue.get(bid);
                    block.block_hash.as_ref() == Some(hash)
                });

                if let Some(&block_id) = valid_block {
                    let block = self.free_queue.get(block_id);
                    hit_blocks.push(block_id);
                    num_tokens += block.num_tokens;
                } else {
                    break; // Chain broken — prefix cache hit ends here
                }
            } else {
                break; // No cache hit for this block
            }
        }

        if hit_blocks.is_empty() {
            self.prefix_cache_misses += 1;
        } else {
            self.prefix_cache_hits += 1;
        }

        (hit_blocks, num_tokens)
    }

    /// Number of prefix cache lookups that found cached blocks.
    pub fn prefix_cache_hits(&self) -> u64 {
        self.prefix_cache_hits
    }

    /// Number of prefix cache lookups that found no cached blocks.
    pub fn prefix_cache_misses(&self) -> u64 {
        self.prefix_cache_misses
    }

    /// Prefix cache hit rate (0.0–1.0). Returns 0.0 if no lookups yet.
    pub fn prefix_cache_hit_rate(&self) -> f64 {
        let total = self.prefix_cache_hits + self.prefix_cache_misses;
        if total == 0 {
            return 0.0;
        }
        self.prefix_cache_hits as f64 / total as f64
    }

    /// Clear all prefix cache entries without freeing blocks.
    ///
    /// Used when model weights change (e.g., RLHF) and all cached KV values
    /// become invalid.
    pub fn reset_prefix_cache(&mut self) {
        for block in self.free_queue.iter_all() {
            if let Some(hash) = block.block_hash {
                self.hash_map.remove(&hash, block.block_id);
            }
        }
        // Clear hashes on all blocks
        for i in 0..self.config.num_blocks {
            self.free_queue.get_mut(i).block_hash = None;
        }
        self.hash_map.clear();
    }

    /// Increment the reference count on a block (for CoW sharing).
    ///
    /// Unlike `touch`, this does NOT rescue the block from the free queue —
    /// it is only valid for blocks that are already allocated (ref_count > 0).
    pub fn increment_ref(&mut self, block_id: BlockId) {
        let block = self.free_queue.get_mut(block_id);
        debug_assert!(block.ref_count > 0, "increment_ref on free block {}", block_id);
        block.ref_count += 1;
    }

    /// Get the reference count of a block.
    pub fn ref_count(&self, block_id: BlockId) -> u32 {
        self.free_queue.get(block_id).ref_count
    }

    /// Copy-on-write: if the block has refcount > 1, allocate a fresh block
    /// and decrement the old block's refcount.
    ///
    /// Returns `Some(new_block_id)` if a copy was needed (the caller must
    /// copy the KV data from `block_id` to the returned block via the kernel
    /// backend). Returns `None` if the block is already exclusively owned
    /// (refcount == 1), meaning no copy is needed.
    pub fn cow_allocate(&mut self, block_id: BlockId) -> Option<BlockId> {
        let ref_count = self.free_queue.get(block_id).ref_count;
        if ref_count <= 1 {
            return None; // Already exclusively owned
        }

        // Allocate a fresh block
        let new_block_id = self.allocate(1)?.into_iter().next()?;

        // Decrement refcount on the shared block
        let old_block = self.free_queue.get_mut(block_id);
        old_block.ref_count -= 1;
        if old_block.ref_count == 0 {
            self.free_queue.append(block_id);
        }

        // Copy metadata to the new block
        let num_tokens = self.free_queue.get(block_id).num_tokens;
        self.free_queue.get_mut(new_block_id).num_tokens = num_tokens;

        Some(new_block_id)
    }

    /// Update the number of tokens stored in a block.
    pub fn set_block_tokens(&mut self, block_id: BlockId, num_tokens: usize) {
        self.free_queue.get_mut(block_id).num_tokens = num_tokens;
    }

    /// Get a block by ID (read-only).
    pub fn get_block(&self, block_id: BlockId) -> &KVCacheBlock {
        self.free_queue.get(block_id)
    }

    /// Evict a cached block's hash entry before reusing the physical block.
    fn evict_cached_block(&mut self, block_id: BlockId) {
        let block = self.free_queue.get_mut(block_id);
        if let Some(hash) = block.block_hash.take() {
            self.hash_map.remove(&hash, block_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(num_blocks: usize) -> BlockPoolConfig {
        BlockPoolConfig {
            block_size: 16,
            num_blocks,
            enable_prefix_cache: true,
        }
    }

    #[test]
    fn test_basic_allocate_free() {
        let mut pool = BlockPool::new(test_config(4));
        assert_eq!(pool.num_free_blocks(), 4);

        let blocks = pool.allocate(2).unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(pool.num_free_blocks(), 2);
        assert_eq!(pool.num_allocated_blocks(), 2);

        pool.free(&blocks);
        assert_eq!(pool.num_free_blocks(), 4);
        assert_eq!(pool.num_allocated_blocks(), 0);
    }

    #[test]
    fn test_allocate_insufficient() {
        let mut pool = BlockPool::new(test_config(2));
        assert!(pool.allocate(3).is_none());
        assert_eq!(pool.num_free_blocks(), 2);
    }

    #[test]
    fn test_allocate_all_then_free() {
        let mut pool = BlockPool::new(test_config(3));
        let all = pool.allocate(3).unwrap();
        assert_eq!(pool.num_free_blocks(), 0);
        assert!(pool.allocate(1).is_none());

        pool.free(&all);
        assert_eq!(pool.num_free_blocks(), 3);
    }

    #[test]
    fn test_ref_count_shared_blocks() {
        let mut pool = BlockPool::new(test_config(4));
        let blocks = pool.allocate(2).unwrap();

        // Simulate two sequences sharing prefix blocks
        pool.touch(&blocks); // ref_count: 1 → 2

        // Free from one sequence
        pool.free(&blocks);
        // Still allocated (ref_count: 2 → 1)
        assert_eq!(pool.num_free_blocks(), 2); // Only the 2 unallocated blocks

        // Free from second sequence
        pool.free(&blocks);
        assert_eq!(pool.num_free_blocks(), 4);
    }

    #[test]
    fn test_prefix_cache_hit() {
        let mut pool = BlockPool::new(test_config(4));

        // Allocate 2 blocks and fill them
        let blocks = pool.allocate(2).unwrap();
        let hash0 = BlockHash::from_tokens(&[1, 2, 3], None);
        let hash1 = BlockHash::from_tokens(&[4, 5, 6], Some(hash0));
        pool.set_block_tokens(blocks[0], 16);
        pool.set_block_tokens(blocks[1], 16);
        pool.cache_block(blocks[0], hash0);
        pool.cache_block(blocks[1], hash1);

        // Free the blocks
        pool.free(&blocks);
        assert_eq!(pool.num_free_blocks(), 4);

        // Look up prefix — should find both blocks
        let (hit_blocks, num_tokens) = pool.find_prefix_cache_hit(&[hash0, hash1]);
        assert_eq!(hit_blocks.len(), 2);
        assert_eq!(num_tokens, 32);
        assert_eq!(hit_blocks[0], blocks[0]);
        assert_eq!(hit_blocks[1], blocks[1]);
    }

    #[test]
    fn test_prefix_cache_partial_hit() {
        let mut pool = BlockPool::new(test_config(4));

        let blocks = pool.allocate(2).unwrap();
        let hash0 = BlockHash::from_tokens(&[1, 2, 3], None);
        let hash1 = BlockHash::from_tokens(&[4, 5, 6], Some(hash0));
        pool.set_block_tokens(blocks[0], 16);
        pool.set_block_tokens(blocks[1], 16);
        pool.cache_block(blocks[0], hash0);
        pool.cache_block(blocks[1], hash1);
        pool.free(&blocks);

        // Look up with a different second block hash — only first matches
        let hash1_different = BlockHash::from_tokens(&[7, 8, 9], Some(hash0));
        let (hit_blocks, num_tokens) = pool.find_prefix_cache_hit(&[hash0, hash1_different]);
        assert_eq!(hit_blocks.len(), 1);
        assert_eq!(num_tokens, 16);
    }

    #[test]
    fn test_prefix_cache_eviction() {
        let mut pool = BlockPool::new(test_config(2));

        // Fill both blocks
        let blocks = pool.allocate(2).unwrap();
        let hash0 = BlockHash::from_tokens(&[1, 2, 3], None);
        pool.set_block_tokens(blocks[0], 16);
        pool.cache_block(blocks[0], hash0);
        pool.free(&blocks);

        // Allocate all blocks again — evicts the cached hash
        let _new_blocks = pool.allocate(2).unwrap();

        // Cache hit should fail now
        let (hit_blocks, _) = pool.find_prefix_cache_hit(&[hash0]);
        assert!(hit_blocks.is_empty());
    }

    #[test]
    fn test_prefix_cache_touch_rescues_from_free() {
        let mut pool = BlockPool::new(test_config(4));

        let blocks = pool.allocate(2).unwrap();
        let hash0 = BlockHash::from_tokens(&[1, 2, 3], None);
        pool.set_block_tokens(blocks[0], 16);
        pool.cache_block(blocks[0], hash0);
        pool.free(&blocks);

        assert_eq!(pool.num_free_blocks(), 4);

        // Touch the cached block — rescues it from the free queue
        let (hit_blocks, _) = pool.find_prefix_cache_hit(&[hash0]);
        pool.touch(&hit_blocks);

        // One block is now allocated (rescued), so only 3 free
        assert_eq!(pool.num_free_blocks(), 3);
    }

    #[test]
    fn test_reset_prefix_cache() {
        let mut pool = BlockPool::new(test_config(4));

        let blocks = pool.allocate(2).unwrap();
        let hash0 = BlockHash::from_tokens(&[1, 2, 3], None);
        pool.cache_block(blocks[0], hash0);
        pool.free(&blocks);

        pool.reset_prefix_cache();

        let (hit_blocks, _) = pool.find_prefix_cache_hit(&[hash0]);
        assert!(hit_blocks.is_empty());
    }

    #[test]
    fn test_can_allocate() {
        let mut pool = BlockPool::new(test_config(3));
        assert!(pool.can_allocate(3));
        assert!(!pool.can_allocate(4));

        let _blocks = pool.allocate(2).unwrap();
        assert!(pool.can_allocate(1));
        assert!(!pool.can_allocate(2));
    }

    #[test]
    fn test_increment_ref() {
        let mut pool = BlockPool::new(test_config(4));
        let blocks = pool.allocate(2).unwrap();
        assert_eq!(pool.ref_count(blocks[0]), 1);

        pool.increment_ref(blocks[0]);
        assert_eq!(pool.ref_count(blocks[0]), 2);

        // Freeing once should not return to free queue (refcount 2 → 1)
        pool.free(&[blocks[0]]);
        assert_eq!(pool.ref_count(blocks[0]), 1);
        assert_eq!(pool.num_free_blocks(), 2); // only the 2 unallocated blocks

        // Freeing again returns to free queue (refcount 1 → 0)
        pool.free(&[blocks[0]]);
        assert_eq!(pool.num_free_blocks(), 3);
    }

    #[test]
    fn test_cow_allocate_shared_block() {
        let mut pool = BlockPool::new(test_config(4));
        let blocks = pool.allocate(2).unwrap();
        pool.set_block_tokens(blocks[0], 16);
        pool.set_block_tokens(blocks[1], 10);

        // Share block 1 (refcount 1 → 2)
        pool.increment_ref(blocks[1]);
        assert_eq!(pool.ref_count(blocks[1]), 2);

        // CoW on shared block — should allocate a new block
        let new_id = pool.cow_allocate(blocks[1]).unwrap();
        assert_ne!(new_id, blocks[1]);

        // Old block refcount decremented (2 → 1)
        assert_eq!(pool.ref_count(blocks[1]), 1);
        // New block has refcount 1 and inherited num_tokens
        assert_eq!(pool.ref_count(new_id), 1);
        assert_eq!(pool.get_block(new_id).num_tokens, 10);
    }

    #[test]
    fn test_cow_allocate_exclusive_block() {
        let mut pool = BlockPool::new(test_config(4));
        let blocks = pool.allocate(2).unwrap();

        // CoW on exclusively owned block — no copy needed
        assert!(pool.cow_allocate(blocks[0]).is_none());
    }

    #[test]
    fn test_cow_allocate_no_free_blocks() {
        let mut pool = BlockPool::new(test_config(2));
        let blocks = pool.allocate(2).unwrap();
        pool.increment_ref(blocks[0]);

        // All blocks allocated, CoW should fail
        assert!(pool.cow_allocate(blocks[0]).is_none());
    }

    #[test]
    fn test_prefix_cache_hit_rate_tracking() {
        let mut pool = BlockPool::new(test_config(8));
        assert_eq!(pool.prefix_cache_hits(), 0);
        assert_eq!(pool.prefix_cache_misses(), 0);
        assert_eq!(pool.prefix_cache_hit_rate(), 0.0);

        // Allocate, cache, and free 2 blocks
        let blocks = pool.allocate(2).unwrap();
        let hash0 = BlockHash::from_tokens(&[1, 2, 3], None);
        let hash1 = BlockHash::from_tokens(&[4, 5, 6], Some(hash0));
        pool.set_block_tokens(blocks[0], 16);
        pool.set_block_tokens(blocks[1], 16);
        pool.cache_block(blocks[0], hash0);
        pool.cache_block(blocks[1], hash1);
        pool.free(&blocks);

        // Lookup with matching hashes → hit
        let (hit, _) = pool.find_prefix_cache_hit(&[hash0, hash1]);
        assert_eq!(hit.len(), 2);
        assert_eq!(pool.prefix_cache_hits(), 1);
        assert_eq!(pool.prefix_cache_misses(), 0);
        assert_eq!(pool.prefix_cache_hit_rate(), 1.0);

        // Lookup with non-matching hash → miss
        let unknown = BlockHash::from_tokens(&[99, 98, 97], None);
        let (hit, _) = pool.find_prefix_cache_hit(&[unknown]);
        assert!(hit.is_empty());
        assert_eq!(pool.prefix_cache_hits(), 1);
        assert_eq!(pool.prefix_cache_misses(), 1);
        assert_eq!(pool.prefix_cache_hit_rate(), 0.5);
    }

    #[test]
    fn test_prefix_cache_hit_rate_disabled() {
        let config = BlockPoolConfig {
            block_size: 16,
            num_blocks: 4,
            enable_prefix_cache: false,
        };
        let mut pool = BlockPool::new(config);

        // Lookups on disabled cache always miss
        let hash = BlockHash::from_tokens(&[1, 2, 3], None);
        let (hit, _) = pool.find_prefix_cache_hit(&[hash]);
        assert!(hit.is_empty());
        assert_eq!(pool.prefix_cache_misses(), 1);
        assert_eq!(pool.prefix_cache_hit_rate(), 0.0);
    }

    #[test]
    fn test_no_prefix_cache() {
        let config = BlockPoolConfig {
            block_size: 16,
            num_blocks: 4,
            enable_prefix_cache: false,
        };
        let mut pool = BlockPool::new(config);

        let blocks = pool.allocate(2).unwrap();
        let hash0 = BlockHash::from_tokens(&[1, 2, 3], None);
        pool.cache_block(blocks[0], hash0); // should be a no-op
        pool.free(&blocks);

        let (hit_blocks, _) = pool.find_prefix_cache_hit(&[hash0]);
        assert!(hit_blocks.is_empty());
    }
}
