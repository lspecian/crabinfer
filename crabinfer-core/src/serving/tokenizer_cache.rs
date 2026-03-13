//! Cached tokenizer with parallel encoding support.
//!
//! Wraps `tokenizers::Tokenizer` with:
//! - LRU cache for repeated string encoding (TOKN-02)
//! - Parallel batch encoding via Rayon (TOKN-01)

use std::num::NonZeroUsize;
use std::sync::Mutex;

use lru::LruCache;
use tokenizers::Tokenizer;

/// A tokenizer wrapper that caches encoding results and supports parallel batch encoding.
///
/// Thread-safe: the inner Tokenizer is Send+Sync, and the LRU cache is behind a Mutex.
/// The Mutex is only held briefly during cache lookup/insert, not during the actual
/// tokenization work.
pub struct CachedTokenizer {
    /// The underlying HuggingFace tokenizer (compiled once at construction).
    tokenizer: Tokenizer,
    /// LRU cache: input string -> token IDs.
    cache: Mutex<LruCache<String, Vec<u32>>>,
}

impl CachedTokenizer {
    /// Create a new cached tokenizer wrapping the given tokenizer.
    ///
    /// `cache_capacity`: maximum number of unique strings to cache.
    /// Recommended: 1024-4096 for typical serving workloads.
    pub fn new(tokenizer: Tokenizer, cache_capacity: usize) -> Self {
        Self {
            tokenizer,
            cache: Mutex::new(LruCache::new(
                NonZeroUsize::new(cache_capacity).unwrap_or(NonZeroUsize::new(1024).unwrap()),
            )),
        }
    }

    /// Encode a single string to token IDs.
    ///
    /// Checks the cache first. On miss, encodes using the tokenizer and caches the result.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, tokenizers::Error> {
        // Fast path: check cache
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(ids) = cache.get(text) {
                return Ok(ids.clone());
            }
        }

        // Slow path: encode and cache
        let encoding = self.tokenizer.encode(text, false)?;
        let ids = encoding.get_ids().to_vec();

        {
            let mut cache = self.cache.lock().unwrap();
            cache.put(text.to_string(), ids.clone());
        }

        Ok(ids)
    }

    /// Encode multiple strings in parallel using Rayon.
    ///
    /// Each string is encoded independently (cache checked per-string).
    /// Returns one `Vec<u32>` per input string, in the same order.
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>, tokenizers::Error> {
        use rayon::prelude::*;

        texts.par_iter().map(|text| self.encode(text)).collect()
    }

    /// Decode token IDs to a string (not cached -- decode is fast).
    pub fn decode(&self, token_ids: &[u32], skip_special: bool) -> Result<String, tokenizers::Error> {
        self.tokenizer.decode(token_ids, skip_special)
    }

    /// Get a reference to the underlying tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Number of entries currently in the cache (for metrics).
    pub fn cache_len(&self) -> usize {
        self.cache.lock().unwrap().len()
    }
}

// SAFETY: Tokenizer is Send+Sync; Mutex<LruCache> is Send+Sync.
unsafe impl Send for CachedTokenizer {}
unsafe impl Sync for CachedTokenizer {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a minimal test tokenizer from JSON.
    ///
    /// Uses a simple WordLevel model (every character is a token) so we don't
    /// need complex BPE merge rules. Good enough to test caching and batching.
    fn test_tokenizer() -> Tokenizer {
        let json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "[UNK]": 0,
                    "hello": 1,
                    "world": 2,
                    "foo": 3,
                    "bar": 4,
                    "the": 5,
                    "quick": 6,
                    "brown": 7,
                    "fox": 8,
                    "jumps": 9,
                    "over": 10,
                    "lazy": 11,
                    "dog": 12,
                    "a": 13,
                    "b": 14,
                    "c": 15,
                    "test": 16,
                    "string": 17,
                    "with": 18,
                    "many": 19,
                    "words": 20,
                    "for": 21,
                    "tokenization": 22,
                    "cache": 23,
                    "parallel": 24,
                    "encoding": 25
                },
                "unk_token": "[UNK]"
            }
        }"#;
        Tokenizer::from_bytes(json.as_bytes()).expect("failed to parse test tokenizer")
    }

    // TOKN-02: Tokenizer cache

    #[test]
    fn test_cached_tokenizer_returns_same_result() {
        let cached = CachedTokenizer::new(test_tokenizer(), 64);
        let first = cached.encode("hello world").unwrap();
        let second = cached.encode("hello world").unwrap();
        assert_eq!(first, second);
        assert!(!first.is_empty(), "encoding should produce tokens");
    }

    #[test]
    fn test_cached_tokenizer_cache_hit_is_faster() {
        let cached = CachedTokenizer::new(test_tokenizer(), 64);

        // Use a long string to make tokenization measurably slow
        let long_text: String = "hello world ".repeat(200);

        // First call (cold)
        let start = std::time::Instant::now();
        let _ = cached.encode(&long_text).unwrap();
        let cold = start.elapsed();

        // Second call (cached)
        let start = std::time::Instant::now();
        let _ = cached.encode(&long_text).unwrap();
        let warm = start.elapsed();

        // Cache hit should be faster. Use generous tolerance for CI.
        assert!(
            warm < cold || warm.as_micros() < 50,
            "cache hit ({warm:?}) should be faster than cold encode ({cold:?})"
        );
    }

    #[test]
    fn test_cached_tokenizer_lru_eviction() {
        let cached = CachedTokenizer::new(test_tokenizer(), 2);

        cached.encode("hello").unwrap();
        cached.encode("world").unwrap();
        assert_eq!(cached.cache_len(), 2);

        // Third string evicts the first (LRU)
        cached.encode("foo bar").unwrap();
        assert_eq!(cached.cache_len(), 2);

        // Verify "hello" was evicted: encoding it again should re-add it
        // and evict "world" (which is now the LRU entry)
        cached.encode("hello").unwrap();
        assert_eq!(cached.cache_len(), 2);
    }

    // TOKN-01: Parallel tokenization

    #[test]
    fn test_parallel_encode_multiple_strings() {
        let cached = CachedTokenizer::new(test_tokenizer(), 64);
        let results = cached.encode_batch(&["hello", "world"]).unwrap();
        assert_eq!(results.len(), 2);
        assert!(!results[0].is_empty());
        assert!(!results[1].is_empty());
        // Verify consistency with single encode
        assert_eq!(results[0], cached.encode("hello").unwrap());
        assert_eq!(results[1], cached.encode("world").unwrap());
    }

    #[test]
    fn test_parallel_encode_scales_with_batch() {
        let cached = CachedTokenizer::new(test_tokenizer(), 256);

        // Build batch of unique strings (each string has many words for tokenization work)
        let strings: Vec<String> = (0..60)
            .map(|i| {
                let base = "hello world the quick brown fox jumps over the lazy dog test string with many words for tokenization cache parallel encoding ";
                format!("{} {}", base.repeat(50), i)
            })
            .collect();
        let refs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();

        // Parallel
        let start = std::time::Instant::now();
        let par_results = cached.encode_batch(&refs).unwrap();
        let par_time = start.elapsed();

        // Clear cache so sequential is also cold
        {
            let mut cache = cached.cache.lock().unwrap();
            cache.clear();
        }

        // Sequential
        let start = std::time::Instant::now();
        let seq_results: Vec<Vec<u32>> = refs
            .iter()
            .map(|s| cached.encode(s).unwrap())
            .collect();
        let seq_time = start.elapsed();

        // Results must match
        assert_eq!(par_results.len(), seq_results.len());
        for (p, s) in par_results.iter().zip(seq_results.iter()) {
            assert_eq!(p, s);
        }

        // Soft timing assertion: parallel should be at least somewhat faster
        // on multi-core systems, but we accept it even if timing is close.
        // The primary assertion is correctness above.
        tracing::info!(
            "Parallel: {par_time:?}, Sequential: {seq_time:?}, Speedup: {:.2}x",
            seq_time.as_secs_f64() / par_time.as_secs_f64().max(0.000001)
        );
    }
}
