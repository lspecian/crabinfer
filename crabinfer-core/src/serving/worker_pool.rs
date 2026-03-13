//! Multi-worker pool for distributing inference requests across engine workers.
//!
//! `WorkerPool` wraps one or more `EngineHandle` instances and distributes
//! incoming requests via round-robin. All workers share the same model weights
//! (candle tensors are Arc-based internally), but each has its own KV cache,
//! scheduler, and engine thread.

use std::sync::atomic::{AtomicUsize, Ordering};

use tokenizers::Tokenizer;

use super::engine_loop::{EngineError, EngineHandle, GeneratedToken};
use super::sequence::SamplingParams;
use super::tokenizer_cache::CachedTokenizer;

/// A pool of inference engine workers with round-robin request distribution.
///
/// `WorkerPool` provides the same API surface as `EngineHandle`, making it a
/// drop-in replacement. When the pool contains a single worker, behavior is
/// identical to using an `EngineHandle` directly.
pub struct WorkerPool {
    workers: Vec<EngineHandle>,
    next_worker: AtomicUsize,
}

impl WorkerPool {
    /// Create a new worker pool from a non-empty list of engine handles.
    ///
    /// # Panics
    /// Panics if `workers` is empty.
    pub fn new(workers: Vec<EngineHandle>) -> Self {
        assert!(!workers.is_empty(), "WorkerPool requires at least one worker");
        Self {
            workers,
            next_worker: AtomicUsize::new(0),
        }
    }

    /// Number of workers in the pool.
    pub fn num_workers(&self) -> usize {
        self.workers.len()
    }

    /// Submit an inference request, routing to the next worker via round-robin.
    pub fn submit(
        &self,
        prompt_tokens: Vec<u32>,
        sampling_params: SamplingParams,
    ) -> Result<tokio::sync::mpsc::Receiver<GeneratedToken>, EngineError> {
        let idx = self.next_worker.fetch_add(1, Ordering::Relaxed) % self.workers.len();
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

    /// Compute embeddings for the given texts.
    ///
    /// Delegates to the first worker (all workers share the same model/tokenizer).
    pub fn embed(&self, texts: Vec<String>) -> Result<(Vec<Vec<f32>>, Vec<u32>), EngineError> {
        self.workers[0].embed(texts)
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // WorkerPool requires real EngineHandle instances which need a running
    // model. We test the round-robin index distribution logic directly since
    // that is the core algorithmic component of WorkerPool.

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
}
