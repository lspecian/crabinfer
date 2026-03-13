//! KV cache block primitives and free block queue.
//!
//! Follows vLLM V1's design: blocks are fixed-size slots in the KV cache,
//! tracked in a flat array with a doubly-linked free list for O(1) operations.

use std::collections::HashMap;

/// Default number of tokens per KV cache block.
pub const DEFAULT_BLOCK_SIZE: usize = 16;

/// Sentinel value for "no link" in the doubly-linked free list.
const NO_LINK: usize = usize::MAX;

/// Unique identifier for a KV cache block.
pub type BlockId = usize;

/// Content-based hash for prefix caching.
///
/// Two blocks with the same token content will produce the same hash,
/// allowing KV cache reuse across requests with shared prefixes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockHash(pub u64);

impl BlockHash {
    /// Compute block hash from token content using FNV-1a.
    ///
    /// The hash chains: each block's hash includes the previous block's hash,
    /// so the hash of block N encodes the entire prefix up to block N.
    pub fn from_tokens(tokens: &[u32], prev_hash: Option<BlockHash>) -> Self {
        Self::from_tokens_salted(tokens, prev_hash, None)
    }

    /// Compute salt-aware block hash from token content using FNV-1a.
    ///
    /// When `salt` is `Some`, the salt bytes are mixed into the initial hash
    /// state before processing tokens. This ensures that requests with
    /// different salts (e.g., different tenants) cannot share cached KV blocks,
    /// providing tenant isolation in multi-tenant deployments.
    ///
    /// When `salt` is `None`, this behaves identically to `from_tokens`
    /// for backward compatibility.
    pub fn from_tokens_salted(
        tokens: &[u32],
        prev_hash: Option<BlockHash>,
        salt: Option<&str>,
    ) -> Self {
        let mut h: u64 = match prev_hash {
            Some(BlockHash(prev)) => prev,
            None => 0xcbf29ce484222325, // FNV offset basis
        };

        // Mix salt bytes into hash state before tokens (only for the first
        // block in a chain, i.e., when prev_hash is None). When prev_hash is
        // Some, the salt was already mixed into the chain's first block.
        if prev_hash.is_none() {
            if let Some(s) = salt {
                for &byte in s.as_bytes() {
                    h ^= byte as u64;
                    h = h.wrapping_mul(0x100000001b3);
                }
            }
        }

        for &token in tokens {
            h ^= token as u64;
            h = h.wrapping_mul(0x100000001b3); // FNV prime
        }
        BlockHash(h)
    }
}

/// A single KV cache block in the block pool.
#[derive(Debug)]
pub struct KVCacheBlock {
    /// Physical block index (0..num_blocks-1). Immutable after creation.
    pub block_id: BlockId,

    /// Reference count. 0 = free (in the free queue or prefix-cached).
    /// >0 = actively used by one or more sequences.
    pub ref_count: u32,

    /// Content hash for prefix caching. Retained even after freeing
    /// so the block can be found by hash if not yet evicted.
    pub block_hash: Option<BlockHash>,

    /// Number of tokens currently stored in this block (0..block_size).
    pub num_tokens: usize,

    // Doubly-linked free list pointers (indices into the block pool array).
    pub(crate) prev_free: usize,
    pub(crate) next_free: usize,
}

impl KVCacheBlock {
    pub fn new(block_id: BlockId) -> Self {
        Self {
            block_id,
            ref_count: 0,
            block_hash: None,
            num_tokens: 0,
            prev_free: NO_LINK,
            next_free: NO_LINK,
        }
    }

    /// Whether this block is in active use by at least one sequence.
    pub fn is_allocated(&self) -> bool {
        self.ref_count > 0
    }

    /// Whether this block has cached content (hash set) but is free (ref_count == 0).
    pub fn is_cached(&self) -> bool {
        self.ref_count == 0 && self.block_hash.is_some()
    }
}

/// O(1) doubly-linked free block queue with LRU eviction ordering.
///
/// Blocks freed most recently go to the tail (last to be evicted).
/// Blocks freed longest ago are at the head (first to be evicted).
/// Supports O(1) removal of arbitrary elements (for prefix cache rescue).
///
/// Uses sentinel head/tail nodes to eliminate edge-case branching.
pub struct FreeBlockQueue {
    /// All blocks in the pool (including allocated ones).
    /// Indices 0 and 1 are sentinel head/tail nodes.
    blocks: Vec<KVCacheBlock>,

    /// Number of blocks currently in the free queue.
    len: usize,
}

impl FreeBlockQueue {
    /// Create a new free block queue.
    ///
    /// `num_blocks` is the total number of physical KV cache blocks.
    /// All blocks start in the free queue.
    pub fn new(num_blocks: usize) -> Self {
        // Index 0 = sentinel head, index 1 = sentinel tail.
        // User blocks start at index 2.
        let total = num_blocks + 2;
        let mut blocks = Vec::with_capacity(total);

        // Sentinel head (index 0)
        let mut head = KVCacheBlock::new(0);
        head.next_free = if num_blocks > 0 { 2 } else { 1 };
        head.prev_free = NO_LINK;
        blocks.push(head);

        // Sentinel tail (index 1)
        let mut tail = KVCacheBlock::new(0);
        tail.prev_free = if num_blocks > 0 { num_blocks + 1 } else { 0 };
        tail.next_free = NO_LINK;
        blocks.push(tail);

        // User blocks (index 2..num_blocks+2)
        for i in 0..num_blocks {
            let idx = i + 2;
            let mut block = KVCacheBlock::new(i);
            block.prev_free = if i == 0 { 0 } else { idx - 1 };
            block.next_free = if i == num_blocks - 1 { 1 } else { idx + 1 };
            blocks.push(block);
        }

        Self {
            blocks,
            len: num_blocks,
        }
    }

    /// Number of free blocks.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the free queue is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total number of blocks (including allocated ones).
    pub fn total_blocks(&self) -> usize {
        self.blocks.len() - 2 // exclude sentinels
    }

    /// Get a block by its block_id (NOT the internal index).
    pub fn get(&self, block_id: BlockId) -> &KVCacheBlock {
        &self.blocks[block_id + 2]
    }

    /// Get a mutable block by its block_id.
    pub fn get_mut(&mut self, block_id: BlockId) -> &mut KVCacheBlock {
        &mut self.blocks[block_id + 2]
    }

    /// Pop the least-recently-used free block (from the head).
    /// Returns None if the queue is empty.
    pub fn popleft(&mut self) -> Option<BlockId> {
        if self.len == 0 {
            return None;
        }

        let head_next = self.blocks[0].next_free;
        debug_assert!(head_next != 1, "popleft on empty queue");

        let block_id = self.blocks[head_next].block_id;
        self.remove_internal(head_next);
        Some(block_id)
    }

    /// Pop up to `n` blocks from the head (LRU order).
    pub fn popleft_n(&mut self, n: usize) -> Vec<BlockId> {
        let n = n.min(self.len);
        let mut result = Vec::with_capacity(n);
        for _ in 0..n {
            if let Some(id) = self.popleft() {
                result.push(id);
            }
        }
        result
    }

    /// Append a block to the tail (most-recently freed).
    pub fn append(&mut self, block_id: BlockId) {
        let idx = block_id + 2;
        debug_assert!(
            self.blocks[idx].prev_free == NO_LINK && self.blocks[idx].next_free == NO_LINK,
            "block {} is already in the free queue",
            block_id
        );

        let tail_prev = self.blocks[1].prev_free;

        self.blocks[idx].prev_free = tail_prev;
        self.blocks[idx].next_free = 1;
        self.blocks[tail_prev].next_free = idx;
        self.blocks[1].prev_free = idx;

        self.len += 1;
    }

    /// Append multiple blocks to the tail.
    pub fn append_n(&mut self, block_ids: &[BlockId]) {
        for &id in block_ids {
            self.append(id);
        }
    }

    /// Remove a specific block from the free queue (O(1)).
    /// Used when a prefix-cached block is reclaimed by a new request.
    pub fn remove(&mut self, block_id: BlockId) {
        let idx = block_id + 2;
        if self.blocks[idx].prev_free == NO_LINK && self.blocks[idx].next_free == NO_LINK {
            return; // Not in the queue
        }
        self.remove_internal(idx);
    }

    /// Internal removal by array index.
    fn remove_internal(&mut self, idx: usize) {
        let prev = self.blocks[idx].prev_free;
        let next = self.blocks[idx].next_free;

        if prev != NO_LINK {
            self.blocks[prev].next_free = next;
        }
        if next != NO_LINK {
            self.blocks[next].prev_free = prev;
        }

        self.blocks[idx].prev_free = NO_LINK;
        self.blocks[idx].next_free = NO_LINK;
        self.len -= 1;
    }

    /// Iterate over all blocks in the pool (not just free ones).
    pub fn iter_all(&self) -> impl Iterator<Item = &KVCacheBlock> {
        self.blocks[2..].iter()
    }
}

/// Maps block content hashes to physical block IDs for prefix caching.
///
/// When a sequence is freed, its blocks retain their hash. Future requests
/// with the same prefix can find these blocks by hash and reuse them,
/// avoiding redundant prefill computation.
#[derive(Debug, Default)]
pub struct BlockHashMap {
    map: HashMap<BlockHash, Vec<BlockId>>,
}

impl BlockHashMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a block's hash for prefix cache lookup.
    pub fn insert(&mut self, hash: BlockHash, block_id: BlockId) {
        self.map.entry(hash).or_default().push(block_id);
    }

    /// Look up blocks matching a content hash.
    pub fn get(&self, hash: &BlockHash) -> Option<&[BlockId]> {
        self.map.get(hash).map(|v| v.as_slice())
    }

    /// Remove a block from the hash map (when evicted from prefix cache).
    pub fn remove(&mut self, hash: &BlockHash, block_id: BlockId) {
        if let Some(ids) = self.map.get_mut(hash) {
            ids.retain(|&id| id != block_id);
            if ids.is_empty() {
                self.map.remove(hash);
            }
        }
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// Number of cached hashes.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_hash_deterministic() {
        let tokens = vec![1u32, 2, 3, 4];
        let h1 = BlockHash::from_tokens(&tokens, None);
        let h2 = BlockHash::from_tokens(&tokens, None);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_block_hash_chaining() {
        let block1_tokens = vec![100u32, 200, 300, 400, 500];
        let block2_tokens = vec![600u32, 700, 800, 900, 1000];

        let h1 = BlockHash::from_tokens(&block1_tokens, None);
        let h2_chained = BlockHash::from_tokens(&block2_tokens, Some(h1));

        // Chaining should produce a different result than just hashing block2 alone
        let h2_standalone = BlockHash::from_tokens(&block2_tokens, None);
        assert_ne!(h2_chained, h2_standalone);

        // Same chain should be deterministic
        let h1_again = BlockHash::from_tokens(&block1_tokens, None);
        let h2_again = BlockHash::from_tokens(&block2_tokens, Some(h1_again));
        assert_eq!(h2_chained, h2_again);
    }

    #[test]
    fn test_block_hash_different_content() {
        let h1 = BlockHash::from_tokens(&[1, 2, 3], None);
        let h2 = BlockHash::from_tokens(&[4, 5, 6], None);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_free_queue_basic() {
        let mut q = FreeBlockQueue::new(4);
        assert_eq!(q.len(), 4);

        // Pop all blocks (LRU order: 0, 1, 2, 3)
        assert_eq!(q.popleft(), Some(0));
        assert_eq!(q.popleft(), Some(1));
        assert_eq!(q.popleft(), Some(2));
        assert_eq!(q.popleft(), Some(3));
        assert_eq!(q.popleft(), None);
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn test_free_queue_append_after_pop() {
        let mut q = FreeBlockQueue::new(3);

        let b0 = q.popleft().unwrap();
        let b1 = q.popleft().unwrap();
        assert_eq!(q.len(), 1);

        // Free b0 back — goes to tail
        q.append(b0);
        assert_eq!(q.len(), 2);

        // Next pop should be b2 (was at head), not b0 (just freed)
        let next = q.popleft().unwrap();
        assert_eq!(next, 2);

        let next = q.popleft().unwrap();
        assert_eq!(next, b0);
    }

    #[test]
    fn test_free_queue_remove_middle() {
        let mut q = FreeBlockQueue::new(5);

        // Remove block 2 from the middle
        q.remove(2);
        assert_eq!(q.len(), 4);

        // Pop should skip block 2
        assert_eq!(q.popleft(), Some(0));
        assert_eq!(q.popleft(), Some(1));
        assert_eq!(q.popleft(), Some(3));
        assert_eq!(q.popleft(), Some(4));
        assert_eq!(q.popleft(), None);
    }

    #[test]
    fn test_free_queue_remove_head() {
        let mut q = FreeBlockQueue::new(3);
        q.remove(0); // Remove the head element
        assert_eq!(q.popleft(), Some(1));
        assert_eq!(q.popleft(), Some(2));
    }

    #[test]
    fn test_free_queue_remove_tail() {
        let mut q = FreeBlockQueue::new(3);
        q.remove(2); // Remove the tail element
        assert_eq!(q.popleft(), Some(0));
        assert_eq!(q.popleft(), Some(1));
        assert_eq!(q.popleft(), None);
    }

    #[test]
    fn test_free_queue_popleft_n() {
        let mut q = FreeBlockQueue::new(5);
        let popped = q.popleft_n(3);
        assert_eq!(popped, vec![0, 1, 2]);
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_free_queue_popleft_n_more_than_available() {
        let mut q = FreeBlockQueue::new(2);
        let popped = q.popleft_n(5);
        assert_eq!(popped, vec![0, 1]);
        assert!(q.is_empty());
    }

    #[test]
    fn test_free_queue_empty() {
        let q = FreeBlockQueue::new(0);
        assert!(q.is_empty());
        assert_eq!(q.total_blocks(), 0);
    }

    #[test]
    fn test_block_hash_map() {
        let mut map = BlockHashMap::new();
        let hash = BlockHash(12345);

        map.insert(hash, 0);
        map.insert(hash, 1);

        assert_eq!(map.get(&hash), Some(&[0, 1][..]));

        map.remove(&hash, 0);
        assert_eq!(map.get(&hash), Some(&[1][..]));

        map.remove(&hash, 1);
        assert_eq!(map.get(&hash), None);
        assert!(map.is_empty());
    }

    #[test]
    fn test_block_hash_salted_different_from_none() {
        let tokens = vec![1u32, 2, 3, 4];
        let unsalted = BlockHash::from_tokens_salted(&tokens, None, None);
        let salted = BlockHash::from_tokens_salted(&tokens, None, Some("tenant-a"));
        assert_ne!(unsalted, salted, "salt=Some should differ from salt=None");
    }

    #[test]
    fn test_block_hash_salted_different_tenants() {
        let tokens = vec![1u32, 2, 3, 4];
        let a = BlockHash::from_tokens_salted(&tokens, None, Some("tenant-a"));
        let b = BlockHash::from_tokens_salted(&tokens, None, Some("tenant-b"));
        assert_ne!(a, b, "different salts should produce different hashes");
    }

    #[test]
    fn test_block_hash_salted_none_backward_compat() {
        let tokens = vec![1u32, 2, 3, 4];
        let old = BlockHash::from_tokens(&tokens, None);
        let new = BlockHash::from_tokens_salted(&tokens, None, None);
        assert_eq!(old, new, "salt=None must match from_tokens for backward compat");
    }

    #[test]
    fn test_block_hash_salted_chaining() {
        let block1 = vec![1u32, 2, 3];
        let block2 = vec![4u32, 5, 6];
        let h1 = BlockHash::from_tokens_salted(&block1, None, Some("salt"));
        let h2 = BlockHash::from_tokens_salted(&block2, Some(h1), Some("salt"));
        // Chained hash should be deterministic
        let h1b = BlockHash::from_tokens_salted(&block1, None, Some("salt"));
        let h2b = BlockHash::from_tokens_salted(&block2, Some(h1b), Some("salt"));
        assert_eq!(h2, h2b);
    }

    #[test]
    fn test_kv_cache_block_states() {
        let mut block = KVCacheBlock::new(0);
        assert!(!block.is_allocated());
        assert!(!block.is_cached());

        // Allocated
        block.ref_count = 1;
        assert!(block.is_allocated());
        assert!(!block.is_cached());

        // Freed but cached
        block.ref_count = 0;
        block.block_hash = Some(BlockHash(42));
        assert!(!block.is_allocated());
        assert!(block.is_cached());
    }
}
