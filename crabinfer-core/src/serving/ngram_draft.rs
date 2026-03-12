//! N-gram suffix decoding: draft token speculation from existing context.
//!
//! Instead of using a separate draft model, this module finds n-gram matches
//! in the sequence's own token history and uses the following tokens as draft
//! candidates. This is particularly effective for:
//! - Repetitive generation (code, structured output, lists)
//! - Agentic workloads (tool calls with recurring patterns)
//! - Multi-turn chat (similar phrasings across turns)
//!
//! Overhead is ~20μs per speculative step (pure CPU hash table lookup),
//! making it essentially free compared to model inference.
//!
//! Reference: "REST: Retrieval-Based Speculative Decoding" (He et al. 2023)

use std::collections::HashMap;

// ─── Configuration ───────────────────────────────────────────────────────

/// Configuration for n-gram suffix decoding.
#[derive(Debug, Clone)]
pub struct NgramDraftConfig {
    /// N-gram size for matching (number of tokens to match).
    /// Larger values produce higher-quality matches but fewer hits.
    /// Default: 3 (trigram matching).
    pub ngram_size: usize,
    /// Maximum number of draft tokens to produce per step.
    /// Default: 5.
    pub max_draft_tokens: usize,
    /// Minimum number of context tokens before n-gram matching activates.
    /// Default: 10.
    pub min_context_tokens: usize,
}

impl Default for NgramDraftConfig {
    fn default() -> Self {
        Self {
            ngram_size: 3,
            max_draft_tokens: 5,
            min_context_tokens: 10,
        }
    }
}

// ─── N-gram index ────────────────────────────────────────────────────────

/// Hash key for an n-gram (sequence of token IDs).
type NgramKey = u64;

/// Compute a hash key for an n-gram slice.
fn hash_ngram(tokens: &[u32]) -> NgramKey {
    // FNV-1a hash for fast, low-collision hashing of short sequences
    let mut hash: u64 = 0xcbf29ce484222325;
    for &t in tokens {
        hash ^= t as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// N-gram index for a single sequence.
///
/// Maps n-gram hashes to the positions where they occur, enabling O(1)
/// lookup of continuation tokens.
pub struct NgramIndex {
    /// Maps n-gram hash → list of positions where the n-gram ends (exclusive).
    /// Multiple positions allow finding matches that aren't the suffix itself.
    index: HashMap<NgramKey, Vec<usize>>,
    /// N-gram size.
    ngram_size: usize,
    /// Number of tokens indexed so far.
    indexed_up_to: usize,
}

impl NgramIndex {
    pub fn new(ngram_size: usize) -> Self {
        Self {
            index: HashMap::new(),
            ngram_size,
            indexed_up_to: 0,
        }
    }

    /// Update the index with new tokens.
    ///
    /// Call this incrementally as tokens are generated. Only indexes new
    /// n-grams since the last update.
    pub fn update(&mut self, tokens: &[u32]) {
        if tokens.len() < self.ngram_size {
            return;
        }

        let start = if self.indexed_up_to > self.ngram_size {
            self.indexed_up_to - self.ngram_size + 1
        } else {
            0
        };

        for i in start..tokens.len().saturating_sub(self.ngram_size - 1) {
            let ngram = &tokens[i..i + self.ngram_size];
            let key = hash_ngram(ngram);
            // Store position after the n-gram (where continuation starts)
            self.index.entry(key).or_default().push(i + self.ngram_size);
        }

        self.indexed_up_to = tokens.len();
    }

    /// Look up draft tokens following a matching n-gram suffix.
    ///
    /// Given the last `ngram_size` tokens of the current context, finds
    /// a matching n-gram earlier in the sequence and returns the tokens
    /// that followed it (up to `max_draft_tokens`).
    ///
    /// Returns an empty vec if no match is found.
    pub fn draft_tokens(&self, tokens: &[u32], max_draft: usize) -> Vec<u32> {
        if tokens.len() < self.ngram_size {
            return Vec::new();
        }

        let suffix_start = tokens.len() - self.ngram_size;
        let suffix = &tokens[suffix_start..];
        let key = hash_ngram(suffix);

        if let Some(positions) = self.index.get(&key) {
            // Try positions from most recent to oldest
            for &continuation_pos in positions.iter().rev() {
                // Must have continuation tokens available
                if continuation_pos >= tokens.len() {
                    continue;
                }

                // Must not be the suffix itself
                let match_start = continuation_pos - self.ngram_size;
                if match_start == suffix_start {
                    continue;
                }

                // Verify actual token match (not just hash collision)
                if tokens[match_start..continuation_pos] != *suffix {
                    continue;
                }

                // Return continuation tokens
                let end = (continuation_pos + max_draft).min(tokens.len());
                return tokens[continuation_pos..end].to_vec();
            }
            Vec::new()
        } else {
            Vec::new()
        }
    }

    /// Number of n-grams indexed.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Clear the index (e.g., on context reset).
    pub fn clear(&mut self) {
        self.index.clear();
        self.indexed_up_to = 0;
    }
}

// ─── N-gram drafter ──────────────────────────────────────────────────────

/// N-gram-based speculative drafter.
///
/// Maintains per-sequence n-gram indices and generates draft tokens
/// without requiring a separate model.
pub struct NgramDrafter {
    config: NgramDraftConfig,
    /// Per-sequence n-gram indices.
    indices: HashMap<u64, NgramIndex>,
    /// Per-sequence token histories.
    histories: HashMap<u64, Vec<u32>>,
    /// Stats: total draft tokens produced.
    pub total_drafted: u64,
    /// Stats: total draft tokens accepted by target model.
    pub total_accepted: u64,
}

impl NgramDrafter {
    pub fn new(config: NgramDraftConfig) -> Self {
        Self {
            config,
            indices: HashMap::new(),
            histories: HashMap::new(),
            total_drafted: 0,
            total_accepted: 0,
        }
    }

    /// Register a new sequence with its initial prompt tokens.
    pub fn add_sequence(&mut self, seq_id: u64, prompt_tokens: &[u32]) {
        let mut index = NgramIndex::new(self.config.ngram_size);
        index.update(prompt_tokens);
        self.indices.insert(seq_id, index);
        self.histories.insert(seq_id, prompt_tokens.to_vec());
    }

    /// Append a generated token to a sequence's history.
    pub fn append_token(&mut self, seq_id: u64, token_id: u32) {
        if let Some(history) = self.histories.get_mut(&seq_id) {
            history.push(token_id);
            if let Some(index) = self.indices.get_mut(&seq_id) {
                index.update(history);
            }
        }
    }

    /// Generate draft tokens for a sequence using n-gram matching.
    ///
    /// Returns candidate draft tokens (may be empty if no match found).
    pub fn draft(&self, seq_id: u64) -> Vec<u32> {
        let history = match self.histories.get(&seq_id) {
            Some(h) => h,
            None => return Vec::new(),
        };

        if history.len() < self.config.min_context_tokens {
            return Vec::new();
        }

        let index = match self.indices.get(&seq_id) {
            Some(i) => i,
            None => return Vec::new(),
        };

        index.draft_tokens(history, self.config.max_draft_tokens)
    }

    /// Record verification results for adaptive statistics.
    pub fn record_verification(&mut self, num_drafted: usize, num_accepted: usize) {
        self.total_drafted += num_drafted as u64;
        self.total_accepted += num_accepted as u64;
    }

    /// Current acceptance rate (0.0–1.0).
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_drafted == 0 {
            0.0
        } else {
            self.total_accepted as f64 / self.total_drafted as f64
        }
    }

    /// Remove a sequence and free its resources.
    pub fn remove_sequence(&mut self, seq_id: u64) {
        self.indices.remove(&seq_id);
        self.histories.remove(&seq_id);
    }

    /// Number of active sequences.
    pub fn num_sequences(&self) -> usize {
        self.indices.len()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_ngram_deterministic() {
        let a = hash_ngram(&[1, 2, 3]);
        let b = hash_ngram(&[1, 2, 3]);
        assert_eq!(a, b);
    }

    #[test]
    fn test_hash_ngram_different_inputs() {
        let a = hash_ngram(&[1, 2, 3]);
        let b = hash_ngram(&[4, 5, 6]);
        assert_ne!(a, b);
    }

    #[test]
    fn test_ngram_index_basic() {
        let mut index = NgramIndex::new(3);
        // Tokens: "the cat sat on the mat"
        let tokens = vec![10, 20, 30, 40, 10, 20];
        index.update(&tokens);

        // Query: last 3 tokens are [40, 10, 20]
        // Match: [10, 20, 30] at position 0, continuation at position 3
        // But [40, 10, 20] doesn't match [10, 20, 30]
        // Actually [10, 20] suffix with ngram_size=3 means we look at [40, 10, 20]
        // We need tokens where [40, 10, 20] appeared earlier — it didn't
        let draft = index.draft_tokens(&tokens, 3);
        // [10, 20, 30] is indexed. Suffix is [40, 10, 20] — no match
        assert!(draft.is_empty());
    }

    #[test]
    fn test_ngram_index_repeated_pattern() {
        let mut index = NgramIndex::new(3);
        // Pattern: A B C X Y A B C ...
        // Tokens [1,2,3] appear at pos 0-2 and pos 5-7
        let tokens = vec![1, 2, 3, 4, 5, 1, 2, 3];
        index.update(&tokens);

        // Last 3 tokens: [2, 3, ?] — wait, last 3 are [1, 2, 3]
        // Looking for [1, 2, 3] earlier → found at pos 0, continuation at pos 3
        // Continuation: [4, 5]... but we need to check bounds
        let draft = index.draft_tokens(&tokens, 5);
        assert_eq!(draft, vec![4, 5, 1, 2, 3]);
    }

    #[test]
    fn test_ngram_index_no_match() {
        let mut index = NgramIndex::new(3);
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        index.update(&tokens);

        // Last 3: [7, 8, 9] — unique, no earlier match
        let draft = index.draft_tokens(&tokens, 3);
        assert!(draft.is_empty());
    }

    #[test]
    fn test_ngram_index_empty() {
        let index = NgramIndex::new(3);
        let tokens = vec![1, 2];
        let draft = index.draft_tokens(&tokens, 3);
        assert!(draft.is_empty());
    }

    #[test]
    fn test_ngram_index_incremental_update() {
        let mut index = NgramIndex::new(2);
        let mut tokens = vec![10, 20, 30];
        index.update(&tokens);
        assert_eq!(index.len(), 2); // [10,20] and [20,30]

        // Add more tokens
        tokens.push(10);
        tokens.push(20);
        index.update(&tokens);

        // Now [10, 20] appears again at pos 3-4, continuation at pos 5 (OOB)
        // But draft_tokens looks for last 2: [10, 20]
        // Match at pos 0, continuation at pos 2 → [30, 10, 20]
        let draft = index.draft_tokens(&tokens, 5);
        assert_eq!(draft, vec![30, 10, 20]);
    }

    #[test]
    fn test_ngram_drafter_basic() {
        let config = NgramDraftConfig {
            ngram_size: 2,
            max_draft_tokens: 3,
            min_context_tokens: 4,
        };
        let mut drafter = NgramDrafter::new(config);

        // Add sequence with repeating pattern
        let prompt = vec![1, 2, 3, 1, 2, 3, 1, 2];
        drafter.add_sequence(0, &prompt);

        let draft = drafter.draft(0);
        // Last 2 tokens: [1, 2] → match at pos 0, continuation at pos 2: [3, 1, 2]
        assert_eq!(draft, vec![3, 1, 2]);
    }

    #[test]
    fn test_ngram_drafter_append_token() {
        let config = NgramDraftConfig {
            ngram_size: 2,
            max_draft_tokens: 3,
            min_context_tokens: 4,
        };
        let mut drafter = NgramDrafter::new(config);

        drafter.add_sequence(0, &[10, 20, 30, 40, 50]);

        // Append token that creates a repeated bigram
        drafter.append_token(0, 10);
        drafter.append_token(0, 20);

        let draft = drafter.draft(0);
        // Last 2: [10, 20] → match at pos 0, continuation at pos 2: [30, 40, 50]
        assert_eq!(draft, vec![30, 40, 50]);
    }

    #[test]
    fn test_ngram_drafter_too_few_tokens() {
        let config = NgramDraftConfig {
            ngram_size: 3,
            max_draft_tokens: 5,
            min_context_tokens: 10,
        };
        let drafter = NgramDrafter::new(config);
        // No sequences added
        let draft = drafter.draft(0);
        assert!(draft.is_empty());
    }

    #[test]
    fn test_ngram_drafter_remove_sequence() {
        let config = NgramDraftConfig::default();
        let mut drafter = NgramDrafter::new(config);

        drafter.add_sequence(42, &[1, 2, 3, 4, 5]);
        assert_eq!(drafter.num_sequences(), 1);

        drafter.remove_sequence(42);
        assert_eq!(drafter.num_sequences(), 0);
        assert!(drafter.draft(42).is_empty());
    }

    #[test]
    fn test_ngram_drafter_acceptance_rate() {
        let config = NgramDraftConfig::default();
        let mut drafter = NgramDrafter::new(config);

        assert_eq!(drafter.acceptance_rate(), 0.0);

        drafter.record_verification(5, 3);
        assert!((drafter.acceptance_rate() - 0.6).abs() < 1e-10);

        drafter.record_verification(5, 5);
        assert!((drafter.acceptance_rate() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_ngram_index_self_match_prevention() {
        let mut index = NgramIndex::new(3);
        // Only 3 tokens — the suffix IS the only trigram, should not match itself
        let tokens = vec![1, 2, 3];
        index.update(&tokens);

        let draft = index.draft_tokens(&tokens, 3);
        assert!(draft.is_empty(), "should not match the suffix itself");
    }

    #[test]
    fn test_ngram_config_default() {
        let config = NgramDraftConfig::default();
        assert_eq!(config.ngram_size, 3);
        assert_eq!(config.max_draft_tokens, 5);
        assert_eq!(config.min_context_tokens, 10);
    }

    #[test]
    fn test_ngram_index_clear() {
        let mut index = NgramIndex::new(2);
        index.update(&[1, 2, 3, 4, 5]);
        assert!(!index.is_empty());

        index.clear();
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }
}
