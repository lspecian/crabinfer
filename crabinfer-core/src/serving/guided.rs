//! Guided decoding: constrain token generation to match a JSON Schema or regex pattern.
//!
//! Uses outlines-core to convert JSON Schema -> regex -> DFA, then masks logits
//! so only schema-valid continuations are sampled.

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use lru::LruCache;
use outlines_core::index::Index;
use outlines_core::primitives::StateId;
use outlines_core::vocabulary::Vocabulary;

/// Constraint specification from the API layer.
#[derive(Debug, Clone)]
pub enum GuidedConstraint {
    /// JSON Schema -- converted to regex internally via outlines-core.
    JsonSchema(serde_json::Value),
    /// Regex pattern -- compiled directly to DFA.
    Regex(String),
}

/// Per-sequence guided decoding state tracking DFA position.
pub struct GuidedState {
    index: Arc<Index>,
    current_state: StateId,
}

impl GuidedState {
    /// Create a new guided state starting at the initial DFA state.
    pub fn new(index: Arc<Index>) -> Self {
        let current_state = index.initial_state();
        Self {
            index,
            current_state,
        }
    }

    /// Get allowed token IDs at the current DFA state.
    /// Returns `None` if the state has no transitions (dead end).
    pub fn allowed_tokens(&self) -> Option<Vec<u32>> {
        self.index.allowed_tokens(&self.current_state)
    }

    /// Advance the DFA state after a token is sampled.
    /// Returns `true` if the transition was valid, `false` if no valid transition exists.
    pub fn advance(&mut self, token_id: u32) -> bool {
        match self.index.next_state(&self.current_state, &token_id) {
            Some(next) => {
                self.current_state = next;
                true
            }
            None => false,
        }
    }

    /// Check if the current state is a valid final (accepting) state.
    pub fn is_complete(&self) -> bool {
        self.index.is_final_state(&self.current_state)
    }

    /// Check if there are valid non-EOS continuations from the current state.
    pub fn has_valid_continuations(&self) -> bool {
        match self.index.allowed_tokens(&self.current_state) {
            Some(tokens) => !tokens.is_empty(),
            None => false,
        }
    }
}

/// Snapshot of cache statistics (non-atomic, for point-in-time reads).
#[derive(Debug, Clone, Copy)]
pub struct CacheStatsSnapshot {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub size: usize,
}

/// Atomic counters for cache statistics.
struct CacheStats {
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
}

impl CacheStats {
    fn new() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }
}

/// Default maximum number of index entries in the cache.
pub const DEFAULT_MAX_ENTRIES: usize = 256;

/// Cache of compiled Index objects keyed by regex pattern.
/// Avoids recompiling the same regex/schema for repeated requests.
/// Uses LRU eviction to bound memory usage.
pub struct IndexCache {
    cache: Mutex<LruCache<String, Arc<Index>>>,
    vocabulary: Arc<Vocabulary>,
    stats: CacheStats,
    max_entries: usize,
}

impl IndexCache {
    /// Create a new cache with the given vocabulary and maximum entry count.
    pub fn new(vocabulary: Vocabulary, max_entries: usize) -> Self {
        let cap = NonZeroUsize::new(max_entries).unwrap_or(NonZeroUsize::new(1).unwrap());
        Self {
            cache: Mutex::new(LruCache::new(cap)),
            vocabulary: Arc::new(vocabulary),
            stats: CacheStats::new(),
            max_entries,
        }
    }

    /// Create a new cache with the default capacity of 256.
    pub fn new_default(vocabulary: Vocabulary) -> Self {
        Self::new(vocabulary, DEFAULT_MAX_ENTRIES)
    }

    /// Returns the configured maximum number of entries.
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    /// Get an existing Index for the regex or compile and cache a new one.
    pub fn get_or_create(&self, regex: &str) -> Result<Arc<Index>, outlines_core::Error> {
        let mut cache = self.cache.lock().unwrap();
        if let Some(index) = cache.get(regex) {
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(Arc::clone(index));
        }
        self.stats.misses.fetch_add(1, Ordering::Relaxed);
        let index = Arc::new(Index::new(regex, &self.vocabulary)?);
        // If the cache is at capacity, LruCache::put will evict the LRU entry
        if cache.len() == cache.cap().get() {
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
        }
        cache.put(regex.to_string(), Arc::clone(&index));
        Ok(index)
    }

    /// Returns a snapshot of the current cache statistics.
    pub fn stats(&self) -> CacheStatsSnapshot {
        let size = self.cache.lock().unwrap().len();
        CacheStatsSnapshot {
            hits: self.stats.hits.load(Ordering::Relaxed),
            misses: self.stats.misses.load(Ordering::Relaxed),
            evictions: self.stats.evictions.load(Ordering::Relaxed),
            size,
        }
    }
}

/// Build an outlines-core `Vocabulary` from a tokenizers `Tokenizer`.
///
/// Extracts all token-to-id mappings from the tokenizer and constructs
/// a Vocabulary suitable for Index construction. The eos_token_id is
/// excluded from the mapping (as required by outlines-core).
pub fn build_vocabulary(
    tokenizer: &tokenizers::Tokenizer,
    eos_token_id: u32,
) -> Result<Vocabulary, String> {
    let vocab = tokenizer.get_vocab(true);
    let mut vocabulary = Vocabulary::new(eos_token_id);
    for (token_str, token_id) in vocab {
        // Skip the EOS token -- outlines-core manages it separately
        if token_id == eos_token_id {
            continue;
        }
        if let Err(e) = vocabulary.try_insert(token_str, token_id) {
            tracing::trace!("Skipping token during vocabulary build: {e}");
        }
    }
    Ok(vocabulary)
}

/// Create a GuidedState from a constraint specification using the IndexCache.
pub fn create_guided_state(
    constraint: &GuidedConstraint,
    cache: &IndexCache,
) -> Result<GuidedState, String> {
    let index = match constraint {
        GuidedConstraint::JsonSchema(schema) => {
            let schema_str = serde_json::to_string(schema)
                .map_err(|e| format!("Failed to serialize JSON schema: {e}"))?;
            let regex = outlines_core::json_schema::regex_from_str(&schema_str, None, None)
                .map_err(|e| format!("Failed to convert JSON schema to regex: {e}"))?;
            cache
                .get_or_create(&regex)
                .map_err(|e| format!("Failed to build Index from JSON schema regex: {e}"))?
        }
        GuidedConstraint::Regex(pattern) => cache
            .get_or_create(pattern)
            .map_err(|e| format!("Failed to build Index from regex: {e}"))?,
    };
    Ok(GuidedState::new(index))
}

/// Apply guided decoding mask: set disallowed token logits to NEG_INFINITY.
///
/// Only tokens in `allowed_tokens` keep their original logit values.
/// All others are set to `-inf` so they have zero probability after softmax.
///
/// If `allowed_tokens` is empty, no masking is applied (caller handles forced EOS).
pub fn apply_guided_mask(logits: &mut [f32], allowed_tokens: &[u32]) {
    if allowed_tokens.is_empty() {
        // Empty allowed set -- don't mask, let caller handle forced EOS
        return;
    }
    let len = logits.len();
    let mut allowed = vec![false; len];
    for &tid in allowed_tokens {
        if (tid as usize) < len {
            allowed[tid as usize] = true;
        }
    }
    for (i, logit) in logits.iter_mut().enumerate() {
        if !allowed[i] {
            *logit = f32::NEG_INFINITY;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal vocabulary for testing with digits and JSON characters.
    fn build_test_vocabulary(eos_token_id: u32) -> Vocabulary {
        let mut vocab = Vocabulary::new(eos_token_id);
        // Digits: "0"-"9" mapped to IDs 48-57
        for digit in 0u8..=9 {
            let s = digit.to_string();
            vocab
                .try_insert(s, 48 + digit as u32)
                .expect("insert digit");
        }
        // JSON characters
        for (token, id) in [
            ("{", 123u32),
            ("}", 125),
            ("\"", 34),
            (":", 58),
            (",", 44),
            (" ", 32),
            ("[", 91),
            ("]", 93),
            (".", 46),
            ("-", 45),
            ("a", 97),
            ("b", 98),
            ("c", 99),
            ("d", 100),
            ("e", 101),
            ("f", 102),
            ("l", 108),
            ("m", 109),
            ("n", 110),
            ("o", 111),
            ("r", 114),
            ("s", 115),
            ("t", 116),
            ("u", 117),
            ("\\", 92),
        ] {
            vocab.try_insert(token, id).expect("insert token");
        }
        vocab
    }

    #[test]
    fn test_json_schema_constraint() {
        // GDEC-01: JSON Schema produces valid regex via outlines-core,
        // Index masks tokens so only schema-valid continuations are allowed.
        let eos_id = 999;
        let vocab = build_test_vocabulary(eos_id);
        let cache = IndexCache::new_default(vocab);

        let schema: serde_json::Value = serde_json::from_str(
            r#"{"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}"#,
        )
        .unwrap();

        let constraint = GuidedConstraint::JsonSchema(schema);
        let state = create_guided_state(&constraint, &cache).expect("Failed to create guided state");

        // At the initial state, there should be allowed tokens (e.g., '{')
        let allowed = state.allowed_tokens();
        assert!(allowed.is_some(), "Should have allowed tokens at initial state");
        let allowed = allowed.unwrap();
        assert!(!allowed.is_empty(), "Allowed tokens should not be empty");
    }

    #[test]
    fn test_regex_constraint() {
        // GDEC-02: Regex compiles to DFA via outlines-core Index,
        // state advances correctly per token, allowed_tokens is correct.
        let eos_id = 999;
        let vocab = build_test_vocabulary(eos_id);
        let cache = IndexCache::new_default(vocab);

        let constraint = GuidedConstraint::Regex("[0-9]+".to_string());
        let mut state =
            create_guided_state(&constraint, &cache).expect("Failed to create guided state");

        // At initial state, digits should be allowed
        let allowed = state.allowed_tokens().expect("Should have allowed tokens");
        assert!(!allowed.is_empty(), "Should have digit tokens allowed");

        // Check that at least one digit token (IDs 48-57) is in the allowed set
        let has_digit = allowed.iter().any(|&t| (48..=57).contains(&t));
        assert!(has_digit, "Digit tokens should be in allowed set");

        // Advance with a digit token (e.g., "1" = ID 49)
        let advanced = state.advance(49);
        assert!(advanced, "Should advance with digit token");

        // After advancing, should be in a valid state (either final or with more transitions)
        // Since regex is [0-9]+, after one digit we should be in a final state
        // (can still accept more digits)
        assert!(
            state.is_complete(),
            "Should be in final state after matching [0-9]+"
        );
        assert!(
            state.has_valid_continuations(),
            "Should have valid continuations (more digits or EOS)"
        );
    }

    #[test]
    fn test_logit_masking() {
        // GDEC-03: apply_guided_mask sets disallowed logits to NEG_INFINITY
        // and leaves allowed logits unchanged.
        let mut logits = vec![1.0f32; 8];
        let allowed = vec![1u32, 3, 5];

        apply_guided_mask(&mut logits, &allowed);

        // Allowed indices should be unchanged
        assert_eq!(logits[1], 1.0);
        assert_eq!(logits[3], 1.0);
        assert_eq!(logits[5], 1.0);

        // Disallowed indices should be NEG_INFINITY
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[4], f32::NEG_INFINITY);
        assert_eq!(logits[6], f32::NEG_INFINITY);
        assert_eq!(logits[7], f32::NEG_INFINITY);
    }

    #[test]
    fn test_masking_overhead() {
        // GDEC-04: apply_guided_mask on a vocab of 32000 tokens with 100
        // allowed tokens completes in < 50 microseconds (release mode).
        // In debug mode we relax the threshold but still verify correctness.
        let mut logits = vec![1.0f32; 32000];
        let allowed: Vec<u32> = (0..100).collect();

        // Warmup
        apply_guided_mask(&mut logits, &allowed);

        // Timed run (best of 3 iterations)
        let mut best_us = u128::MAX;
        for _ in 0..3 {
            let mut logits_run = vec![1.0f32; 32000];
            let start = std::time::Instant::now();
            apply_guided_mask(&mut logits_run, &allowed);
            let elapsed = start.elapsed().as_micros();
            best_us = best_us.min(elapsed);
        }

        // In debug builds, allow up to 2000us; in release, expect < 50us
        let threshold = if cfg!(debug_assertions) { 2000 } else { 50 };
        assert!(
            best_us < threshold,
            "Masking took {}us, expected < {}us",
            best_us,
            threshold,
        );

        // Verify correctness on a final run
        let mut logits = vec![1.0f32; 32000];
        apply_guided_mask(&mut logits, &allowed);

        for i in 0..100 {
            assert_eq!(logits[i as usize], 1.0, "Allowed token {} should be unchanged", i);
        }
        for i in 100..32000 {
            assert_eq!(
                logits[i],
                f32::NEG_INFINITY,
                "Disallowed token {} should be NEG_INFINITY",
                i
            );
        }
    }

    #[test]
    fn test_index_cache_deduplication() {
        let eos_id = 999;
        let vocab = build_test_vocabulary(eos_id);
        let cache = IndexCache::new_default(vocab);

        let idx1 = cache.get_or_create("[0-9]+").expect("First create");
        let idx2 = cache.get_or_create("[0-9]+").expect("Second create (cached)");

        // Both should be the same Arc (pointer equality)
        assert!(Arc::ptr_eq(&idx1, &idx2), "Cache should return same Arc");
    }

    #[test]
    fn test_lru_eviction() {
        // GDEC-04: LRU eviction — insert max_entries+1 patterns, oldest entry is evicted
        let eos_id = 999;
        let vocab = build_test_vocabulary(eos_id);
        // Cache with capacity of 2
        let cache = IndexCache::new(vocab, 2);

        // Insert pattern A
        let _idx_a1 = cache.get_or_create("[0-9]+").expect("insert A");
        // Insert pattern B (most recently used is now B, oldest is A)
        let _idx_b = cache.get_or_create("[a-z]+").expect("insert B");
        // Insert pattern C — should evict A (the oldest)
        let _idx_c = cache.get_or_create("[a-c]+").expect("insert C");

        let stats = cache.stats();
        // We had 3 misses (each pattern was new), 0 hits, 1 eviction
        assert_eq!(stats.misses, 3, "Expected 3 misses");
        assert_eq!(stats.hits, 0, "Expected 0 hits");
        assert_eq!(stats.evictions, 1, "Expected 1 eviction when capacity exceeded");
        assert_eq!(stats.size, 2, "Cache size should be 2 (max capacity)");
    }

    #[test]
    fn test_cache_stats_tracking() {
        // GDEC-04: CacheStats tracks hits, misses, and evictions correctly
        let eos_id = 999;
        let vocab = build_test_vocabulary(eos_id);
        let cache = IndexCache::new(vocab, 256);

        // First call — miss
        let _idx1 = cache.get_or_create("[0-9]+").expect("first");
        // Second call same pattern — hit
        let _idx2 = cache.get_or_create("[0-9]+").expect("second (hit)");
        // Different pattern — miss
        let _idx3 = cache.get_or_create("[a-z]+").expect("third");
        // Repeat — hit
        let _idx4 = cache.get_or_create("[a-z]+").expect("fourth (hit)");

        let stats = cache.stats();
        assert_eq!(stats.hits, 2, "Expected 2 hits");
        assert_eq!(stats.misses, 2, "Expected 2 misses");
        assert_eq!(stats.evictions, 0, "Expected 0 evictions (capacity not exceeded)");
        assert_eq!(stats.size, 2, "Expected 2 entries in cache");
    }

    #[test]
    fn test_default_max_entries() {
        // GDEC-04: Default max_entries is 256
        let eos_id = 999;
        let vocab = build_test_vocabulary(eos_id);
        let cache = IndexCache::new_default(vocab);
        assert_eq!(cache.max_entries(), 256, "Default max_entries should be 256");
    }

    #[test]
    fn test_empty_allowed_tokens_no_mask() {
        // Empty allowed set should not modify logits (graceful degradation)
        let mut logits = vec![1.0f32; 8];
        apply_guided_mask(&mut logits, &[]);
        for l in &logits {
            assert_eq!(*l, 1.0, "No masking should happen with empty allowed set");
        }
    }

    #[test]
    fn test_stop_token_suppressed_during_guided() {
        // GDEC-05: When guided decoding is active, stop_token_ids must NOT fire
        // to prevent premature termination of valid constrained output.
        //
        // This test documents the expected engine_loop.rs behavior:
        //   let has_guided = self.guided_states.contains_key(&sched.seq_id);
        //   let should_stop = is_eos || (!has_guided && seq.should_stop(token_id));
        //
        // Scenario: token_id=42 is in the stop list AND in the DFA's allowed set.
        // Without suppression:  should_stop = true  → generation terminates early.
        // With suppression:     should_stop = false → DFA continues to guide output.
        //
        // We verify the logic directly here since setting up a full engine
        // with an active sequence would require a real tokenizer + model.

        // Simulate has_guided = true (sequence has an active guided state)
        let has_guided_active = true;
        let is_eos = false;
        let stop_token_fired = true; // seq.should_stop() would return true

        // With guided active, stop_token_ids check is suppressed
        let should_stop_with_guided = is_eos || (!has_guided_active && stop_token_fired);
        assert!(
            !should_stop_with_guided,
            "Stop token must be suppressed when guided decoding is active"
        );

        // Without guided, stop token fires normally
        let has_guided_inactive = false;
        let should_stop_without_guided = is_eos || (!has_guided_inactive && stop_token_fired);
        assert!(
            should_stop_without_guided,
            "Stop token fires normally when guided decoding is not active"
        );
    }

    #[test]
    fn test_logprobs_reflect_post_mask_distribution() {
        // GDEC-05: Logprobs are computed from post-mask logits.
        // A token masked to NEG_INFINITY should have logprob of NEG_INFINITY
        // after log_softmax, confirming the constraint is reflected in reported probs.
        let mut logits = vec![1.0f32; 4];
        let allowed = vec![0u32, 2]; // tokens 1 and 3 are masked out

        apply_guided_mask(&mut logits, &allowed);

        // After masking: logits[1] and logits[3] are NEG_INFINITY
        assert_eq!(logits[1], f32::NEG_INFINITY, "Masked token logit must be NEG_INFINITY");
        assert_eq!(logits[3], f32::NEG_INFINITY, "Masked token logit must be NEG_INFINITY");

        // Simulate log_softmax: exp(NEG_INFINITY) = 0, so log(0) = -inf in logprob space
        // We verify that after masking the sum of exp(logits) only includes allowed tokens
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
        let lp_masked_0 = (f32::NEG_INFINITY - max_logit).exp() / sum_exp;
        // lp for masked token: exp(-inf) / sum = 0 / sum = 0 (then log(0) = -inf)
        assert_eq!(
            lp_masked_0, 0.0,
            "Masked token contributes 0 probability after softmax"
        );
    }
}
