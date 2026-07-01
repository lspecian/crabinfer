//! Multi-worker pool for distributing inference requests across engine workers.
//!
//! `WorkerPool` wraps one or more `EngineHandle` instances and distributes
//! incoming requests via round-robin or cache-aware routing. All workers share
//! the same model weights (candle tensors are Arc-based internally), but each
//! has its own KV cache, scheduler, and engine thread.

use std::sync::atomic::{AtomicUsize, Ordering};

use tokenizers::Tokenizer;

use super::block::BlockHash;
use super::engine_loop::{EngineError, EngineHandle, GeneratedToken};
use super::sequence::SamplingParams;
use super::tokenizer_cache::CachedTokenizer;

/// Request routing policy for multi-worker pools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RoutingPolicy {
    /// Distribute requests evenly across workers (default).
    #[default]
    RoundRobin,
    /// Route to the worker whose prefix cache best matches the request's prompt.
    ///
    /// Uses `EngineHandle::block_hashes()` to compare the request's prefix
    /// block hashes against each worker's active block hashes, selecting the
    /// worker with the highest number of matching prefix hashes. Falls back to
    /// round-robin when no worker has any matching prefix.
    CacheAware,
}

/// A pool of inference engine workers with configurable request distribution.
///
/// `WorkerPool` provides the same API surface as `EngineHandle`, making it a
/// drop-in replacement. When the pool contains a single worker, behavior is
/// identical to using an `EngineHandle` directly.
pub struct WorkerPool {
    workers: Vec<EngineHandle>,
    next_worker: AtomicUsize,
    /// Routing policy used to select workers for incoming requests.
    routing_policy: RoutingPolicy,
    /// Token block size used when computing prefix block hashes for cache-aware routing.
    block_size: usize,
}

impl WorkerPool {
    /// Create a new worker pool from a non-empty list of engine handles.
    ///
    /// Uses `RoutingPolicy::RoundRobin` and a default block size of 16 tokens.
    ///
    /// # Panics
    /// Panics if `workers` is empty.
    pub fn new(workers: Vec<EngineHandle>) -> Self {
        assert!(!workers.is_empty(), "WorkerPool requires at least one worker");
        Self {
            workers,
            next_worker: AtomicUsize::new(0),
            routing_policy: RoutingPolicy::RoundRobin,
            block_size: super::block::DEFAULT_BLOCK_SIZE,
        }
    }

    /// Create a new worker pool with an explicit routing policy and block size.
    ///
    /// `block_size` must match the KV cache block size of the workers (used to
    /// chunk prompt tokens into prefix blocks for hash computation). Pass
    /// `super::block::DEFAULT_BLOCK_SIZE` (16) when unsure.
    ///
    /// # Panics
    /// Panics if `workers` is empty or `block_size` is zero.
    pub fn new_with_policy(
        workers: Vec<EngineHandle>,
        policy: RoutingPolicy,
        block_size: usize,
    ) -> Self {
        assert!(!workers.is_empty(), "WorkerPool requires at least one worker");
        assert!(block_size > 0, "block_size must be > 0");
        Self {
            workers,
            next_worker: AtomicUsize::new(0),
            routing_policy: policy,
            block_size,
        }
    }

    /// Number of workers in the pool.
    pub fn num_workers(&self) -> usize {
        self.workers.len()
    }

    /// Compute prefix block hashes for the given prompt tokens.
    ///
    /// Tokens are chunked into `block_size`-token blocks and hashed with FNV-1a
    /// chaining so that each block's hash encodes the entire prefix up to that block.
    ///
    /// When `salt` is `Some`, the salt is mixed into the first block's hash state
    /// so that requests with different salts (e.g., different tenants) produce
    /// disjoint hash domains and cannot match each other's cached prefix blocks.
    /// When `salt` is `None`, behavior is identical to the pre-PCCH-02 unsalted
    /// implementation for full backward compatibility.
    fn compute_prompt_hashes(&self, prompt_tokens: &[u32], salt: Option<&str>) -> Vec<BlockHash> {
        let mut hashes = Vec::new();
        let mut prev_hash = None;
        for chunk in prompt_tokens.chunks(self.block_size) {
            let h = BlockHash::from_tokens_salted(chunk, prev_hash, salt);
            hashes.push(h);
            prev_hash = Some(h);
        }
        hashes
    }

    /// Count how many of the prompt's prefix hashes appear in a worker's active block hashes.
    fn prefix_match_count(prompt_hashes: &[BlockHash], worker_hashes: &[BlockHash]) -> usize {
        use std::collections::HashSet;
        let worker_set: HashSet<u64> = worker_hashes.iter().map(|h| h.0).collect();
        prompt_hashes.iter().filter(|h| worker_set.contains(&h.0)).count()
    }

    /// Find the worker index with the best prefix match for the given prompt tokens.
    ///
    /// Falls back to round-robin when no worker has any matching prefix hashes,
    /// or when the prompt produces no block hashes (empty token list).
    ///
    /// `salt` must match the salt used when the engine registered its blocks
    /// (i.e., the request's `sampling_params.cache_salt`). Passing `None` retains
    /// the pre-PCCH-02 unsalted routing behavior.
    fn best_prefix_worker(&self, prompt_tokens: &[u32], salt: Option<&str>) -> usize {
        let prompt_hashes = self.compute_prompt_hashes(prompt_tokens, salt);
        if prompt_hashes.is_empty() {
            return self.next_worker.fetch_add(1, Ordering::Relaxed) % self.workers.len();
        }

        let mut best_idx = 0;
        let mut best_count = 0;
        for (i, worker) in self.workers.iter().enumerate() {
            let count = Self::prefix_match_count(&prompt_hashes, &worker.block_hashes());
            if count > best_count {
                best_count = count;
                best_idx = i;
            }
        }

        if best_count == 0 {
            // No worker has any matching prefix — fall back to round-robin
            self.next_worker.fetch_add(1, Ordering::Relaxed) % self.workers.len()
        } else {
            best_idx
        }
    }

    /// Submit an inference request, routing to a worker according to the pool's `RoutingPolicy`.
    ///
    /// - `RoundRobin`: distributes requests evenly across workers (default).
    /// - `CacheAware`: routes to the worker whose prefix cache best matches the
    ///   request's prompt; falls back to round-robin when no prefix match exists.
    ///   Single-worker pools always route to worker 0.
    pub fn submit(
        &self,
        prompt_tokens: Vec<u32>,
        sampling_params: SamplingParams,
    ) -> Result<tokio::sync::mpsc::Receiver<GeneratedToken>, EngineError> {
        let idx = match self.routing_policy {
            RoutingPolicy::RoundRobin => {
                self.next_worker.fetch_add(1, Ordering::Relaxed) % self.workers.len()
            }
            RoutingPolicy::CacheAware => {
                if self.workers.len() == 1 {
                    0
                } else {
                    self.best_prefix_worker(
                        &prompt_tokens,
                        sampling_params.cache_salt.as_deref(),
                    )
                }
            }
        };
        self.workers[idx].submit(prompt_tokens, sampling_params)
    }

    /// Total number of in-flight requests across all workers.
    pub fn in_flight_count(&self) -> usize {
        self.workers.iter().map(|w| w.in_flight_count()).sum()
    }

    /// Get a reference to the underlying tokenizer.
    ///
    /// All workers share the same tokenizer, so we return from the first.
    pub fn tokenizer(&self) -> &Tokenizer {
        self.workers[0].tokenizer()
    }

    /// Get a reference to the cached tokenizer wrapper.
    pub fn cached_tokenizer(&self) -> &CachedTokenizer {
        self.workers[0].cached_tokenizer()
    }

    /// Get the EOS token ID.
    pub fn eos_token_id(&self) -> u32 {
        self.workers[0].eos_token_id()
    }

    /// Encode a text string into token IDs (cached).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, EngineError> {
        self.workers[0].encode(text)
    }

    /// Encode multiple strings in parallel (cached).
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>, EngineError> {
        self.workers[0].encode_batch(texts)
    }

    /// Decode token IDs into a text string.
    pub fn decode(&self, token_ids: &[u32]) -> Result<String, EngineError> {
        self.workers[0].decode(token_ids)
    }

    /// Total KV cache blocks in use across all workers.
    pub fn kv_blocks_used(&self) -> usize {
        self.workers.iter().map(|w| w.kv_blocks_used()).sum()
    }

    /// Total KV cache blocks allocated across all workers.
    pub fn kv_blocks_total(&self) -> usize {
        self.workers.iter().map(|w| w.kv_blocks_total()).sum()
    }

    /// Aggregate KV cache usage ratio (0.0 = empty, 1.0 = full).
    pub fn kv_cache_usage(&self) -> f64 {
        let total = self.kv_blocks_total();
        if total == 0 {
            return 0.0;
        }
        self.kv_blocks_used() as f64 / total as f64
    }

    /// Average prefix cache hit rate across all workers.
    pub fn prefix_cache_hit_rate(&self) -> f64 {
        let sum: f64 = self.workers.iter().map(|w| w.prefix_cache_hit_rate()).sum();
        sum / self.workers.len() as f64
    }

    /// Total number of sequences waiting across all workers.
    pub fn num_waiting(&self) -> usize {
        self.workers.iter().map(|w| w.num_waiting()).sum()
    }

    /// Aggregated guided decoding cache statistics across all workers.
    ///
    /// Hits, misses, and evictions are summed across workers; size is summed.
    pub fn guided_cache_stats(&self) -> super::guided::CacheStatsSnapshot {
        let mut hits = 0u64;
        let mut misses = 0u64;
        let mut evictions = 0u64;
        let mut size = 0usize;
        for w in &self.workers {
            let s = w.guided_cache_stats();
            hits += s.hits;
            misses += s.misses;
            evictions += s.evictions;
            size += s.size;
        }
        super::guided::CacheStatsSnapshot { hits, misses, evictions, size }
    }

    /// Compute embeddings for the given texts.
    ///
    /// Delegates to the first worker (all workers share the same model/tokenizer).
    pub fn embed(&self, texts: Vec<String>) -> Result<(Vec<Vec<f32>>, Vec<u32>), EngineError> {
        self.workers[0].embed(texts)
    }

    /// Validate a guided decoding constraint without submitting a full request.
    ///
    /// Builds a temporary vocabulary and index to check whether the constraint
    /// compiles successfully. Returns `Ok(())` if valid, or an error message
    /// from outlines-core if not.
    ///
    /// This is used by the `/v1/guided/completions` handler to return HTTP 400
    /// before submitting to the engine when `strict_constraints` is `true`.
    pub fn validate_constraint(
        &self,
        constraint: &super::guided::GuidedConstraint,
    ) -> Result<(), String> {
        let vocab = super::guided::build_vocabulary(
            self.workers[0].tokenizer(),
            self.workers[0].eos_token_id(),
        )?;
        let cache = super::guided::IndexCache::new_default(vocab);
        super::guided::create_guided_state(constraint, &cache)?;
        Ok(())
    }

    /// Signal all workers to shut down gracefully.
    pub fn shutdown(&self) {
        for w in &self.workers {
            w.shutdown();
        }
    }
}

impl Clone for WorkerPool {
    fn clone(&self) -> Self {
        Self {
            workers: self.workers.clone(),
            next_worker: AtomicUsize::new(self.next_worker.load(Ordering::Relaxed)),
            routing_policy: self.routing_policy,
            block_size: self.block_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::block::BlockHash;

    // WorkerPool requires real EngineHandle instances which need a running
    // model. We test the round-robin index distribution logic directly since
    // that is the core algorithmic component of WorkerPool.

    // --- Tests for prefix_match_count ---

    #[test]
    fn test_prefix_match_count_no_overlap() {
        // Two disjoint sets -> 0 matches
        let prompt_hashes = vec![BlockHash(1), BlockHash(2), BlockHash(3)];
        let worker_hashes = vec![BlockHash(4), BlockHash(5), BlockHash(6)];
        assert_eq!(WorkerPool::prefix_match_count(&prompt_hashes, &worker_hashes), 0);
    }

    #[test]
    fn test_prefix_match_count_full_overlap() {
        // Identical sets -> N matches
        let hashes = vec![BlockHash(10), BlockHash(20), BlockHash(30)];
        assert_eq!(WorkerPool::prefix_match_count(&hashes, &hashes), 3);
    }

    #[test]
    fn test_prefix_match_count_partial() {
        // Overlapping sets -> correct count
        let prompt_hashes = vec![BlockHash(1), BlockHash(2), BlockHash(3), BlockHash(4)];
        let worker_hashes = vec![BlockHash(2), BlockHash(4), BlockHash(99)];
        assert_eq!(WorkerPool::prefix_match_count(&prompt_hashes, &worker_hashes), 2);
    }

    // --- Tests for compute_prompt_hashes ---

    #[test]
    fn test_compute_prompt_hashes_empty() {
        let pool = WorkerPool {
            workers: vec![],
            next_worker: AtomicUsize::new(0),
            routing_policy: RoutingPolicy::RoundRobin,
            block_size: 16,
        };
        let hashes = pool.compute_prompt_hashes(&[], None);
        assert!(hashes.is_empty());
    }

    #[test]
    fn test_compute_prompt_hashes_single_block() {
        // Tokens shorter than block_size -> 1 hash
        let pool = WorkerPool {
            workers: vec![],
            next_worker: AtomicUsize::new(0),
            routing_policy: RoutingPolicy::RoundRobin,
            block_size: 16,
        };
        let tokens: Vec<u32> = (0..8).collect();
        let hashes = pool.compute_prompt_hashes(&tokens, None);
        assert_eq!(hashes.len(), 1);
        // Verify the hash matches manual computation
        let expected = BlockHash::from_tokens(&tokens, None);
        assert_eq!(hashes[0], expected);
    }

    #[test]
    fn test_compute_prompt_hashes_multiple_blocks() {
        // Tokens spanning 3 blocks -> 3 hashes with chaining
        let pool = WorkerPool {
            workers: vec![],
            next_worker: AtomicUsize::new(0),
            routing_policy: RoutingPolicy::RoundRobin,
            block_size: 4,
        };
        let tokens: Vec<u32> = (0..12).collect();
        let hashes = pool.compute_prompt_hashes(&tokens, None);
        assert_eq!(hashes.len(), 3);
        // Verify chaining: each hash depends on the previous
        let h0 = BlockHash::from_tokens(&tokens[0..4], None);
        let h1 = BlockHash::from_tokens(&tokens[4..8], Some(h0));
        let h2 = BlockHash::from_tokens(&tokens[8..12], Some(h1));
        assert_eq!(hashes[0], h0);
        assert_eq!(hashes[1], h1);
        assert_eq!(hashes[2], h2);
        // Verify hashes are not equal (chaining produces distinct values)
        assert_ne!(hashes[0], hashes[1]);
        assert_ne!(hashes[1], hashes[2]);
    }

    #[test]
    fn test_round_robin_index_distribution() {
        // Verify the round-robin math distributes evenly across N workers
        let counter = AtomicUsize::new(0);
        let num_workers = 3;

        let mut distribution = vec![0usize; num_workers];
        for _ in 0..9 {
            let idx = counter.fetch_add(1, Ordering::Relaxed) % num_workers;
            distribution[idx] += 1;
        }

        // Each worker should get exactly 3 requests
        assert_eq!(distribution, vec![3, 3, 3]);
    }

    #[test]
    fn test_round_robin_wraps_around() {
        let counter = AtomicUsize::new(0);
        let num_workers = 2;

        let indices: Vec<usize> = (0..6)
            .map(|_| counter.fetch_add(1, Ordering::Relaxed) % num_workers)
            .collect();

        assert_eq!(indices, vec![0, 1, 0, 1, 0, 1]);
    }

    #[test]
    fn test_single_worker_always_routes_to_zero() {
        let counter = AtomicUsize::new(0);
        let num_workers = 1;

        let indices: Vec<usize> = (0..5)
            .map(|_| counter.fetch_add(1, Ordering::Relaxed) % num_workers)
            .collect();

        assert_eq!(indices, vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_round_robin_large_count() {
        // Verify distribution is even for a large number of requests
        let counter = AtomicUsize::new(0);
        let num_workers = 4;

        let mut distribution = vec![0usize; num_workers];
        for _ in 0..1000 {
            let idx = counter.fetch_add(1, Ordering::Relaxed) % num_workers;
            distribution[idx] += 1;
        }

        assert_eq!(distribution, vec![250, 250, 250, 250]);
    }

    #[test]
    #[should_panic(expected = "WorkerPool requires at least one worker")]
    fn test_empty_worker_pool_panics() {
        let _pool: WorkerPool = WorkerPool::new(vec![]);
    }

    // --- Tests for salt-aware compute_prompt_hashes (PCCH-02) ---

    #[test]
    fn test_cache_aware_routing_with_salt() {
        let pool = WorkerPool {
            workers: vec![],
            next_worker: AtomicUsize::new(0),
            routing_policy: RoutingPolicy::RoundRobin,
            block_size: 16,
        };
        let tokens: Vec<u32> = (0..32).collect();

        // Scenario A — same tokens, different salt produces different hashes
        let hashes_a = pool.compute_prompt_hashes(&tokens, Some("tenant-a"));
        let hashes_b = pool.compute_prompt_hashes(&tokens, Some("tenant-b"));
        let hashes_unsalted = pool.compute_prompt_hashes(&tokens, None);

        assert_eq!(hashes_a.len(), 2);
        assert_eq!(hashes_b.len(), 2);
        // Block 0 differs because salt is mixed into block 0's initial state.
        assert_ne!(hashes_a[0], hashes_b[0]);
        assert_ne!(hashes_a[0], hashes_unsalted[0]);
        // Block 1 also differs because the chain propagates the salted block 0 hash forward.
        assert_ne!(hashes_a[1], hashes_b[1]);

        // Scenario B — same tokens, same salt produces identical hashes (deterministic)
        let hashes_a1 = pool.compute_prompt_hashes(&tokens, Some("tenant-a"));
        let hashes_a2 = pool.compute_prompt_hashes(&tokens, Some("tenant-a"));
        assert_eq!(hashes_a1, hashes_a2);

        // Scenario C — backward compat: None salt matches the pre-PCCH-02 unsalted hashes
        let unsalted_via_new = pool.compute_prompt_hashes(&tokens, None);
        // Reproduce what the old unsalted impl did, inline:
        let mut expected = Vec::new();
        let mut prev: Option<BlockHash> = None;
        for chunk in tokens.chunks(16) {
            let h = BlockHash::from_tokens(chunk, prev);
            expected.push(h);
            prev = Some(h);
        }
        assert_eq!(unsalted_via_new, expected);
    }
}
