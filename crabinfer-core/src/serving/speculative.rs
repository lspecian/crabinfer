//! Speculative decoding: draft-then-verify token generation.
//!
//! Uses a small draft model to generate K candidate tokens autoregressively,
//! then verifies them against the target model in a single batched forward pass.
//! Accepted tokens (plus a bonus/resampled token) are emitted in one step,
//! improving per-token latency when the draft model's predictions align well
//! with the target model.
//!
//! Algorithm reference: Leviathan et al. 2023, "Fast Inference from
//! Transformers via Speculative Decoding"

use std::collections::HashMap;

use candle_core::{DType, Device, Result as CandleResult, Tensor};

use super::kv_cache::{KVCacheConfig, KVCacheManager, SequenceBlocks};
use super::models::ModelRunner;
use super::sequence::SeqId;

// ─── Configuration ───────────────────────────────────────────────────────

/// Configuration for speculative decoding.
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of draft tokens to generate per speculative step (K).
    pub num_draft_tokens: usize,
    /// Whether to adaptively tune K based on acceptance rate.
    pub adaptive: bool,
    /// Minimum K for adaptive tuning.
    pub min_draft_tokens: usize,
    /// Maximum K for adaptive tuning.
    pub max_draft_tokens: usize,
    /// Whether to share KV cache between draft and target models.
    ///
    /// When the draft model has the same `num_kv_heads` and `head_size`
    /// as the target model, the draft model can write to the target's
    /// KV cache blocks directly. This eliminates the entire draft model's
    /// KV cache allocation (often 100s of MB).
    ///
    /// After verification:
    /// - Accepted tokens' KV entries are already correct (written by draft)
    /// - Only the correction token needs recomputation by target
    ///
    /// Falls back to separate caches when dimensions don't match.
    pub shared_kv_cache: bool,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            num_draft_tokens: 4,
            adaptive: true,
            min_draft_tokens: 1,
            max_draft_tokens: 8,
            shared_kv_cache: false,
        }
    }
}

// ─── Speculative state ───────────────────────────────────────────────────

/// Per-sequence draft model block tracking and state.
pub struct SpeculativeState {
    /// Configuration.
    pub config: SpeculativeConfig,
    /// The draft model (smaller, faster).
    pub draft_model: Box<dyn ModelRunner>,
    /// Draft model KV caches (separate from target).
    /// Empty when `shared_kv_cache` is true and dimensions match.
    pub draft_kv_caches: Vec<(Tensor, Tensor)>,
    /// Draft model KV cache block manager.
    /// When sharing KV cache, this is unused (target's manager is used instead).
    pub draft_kv_manager: KVCacheManager,
    /// Per-sequence draft block tracking.
    /// When sharing KV cache, this is unused (target's blocks are used).
    pub draft_blocks: HashMap<SeqId, SequenceBlocks>,
    /// Current K (may be adapted over time).
    pub current_k: usize,
    /// Total tokens accepted across all speculative steps (for adaptive K).
    pub total_accepted: u64,
    /// Total tokens drafted across all speculative steps (for adaptive K).
    pub total_drafted: u64,
    /// Whether KV cache is shared with the target model.
    ///
    /// True when `config.shared_kv_cache` was requested AND the draft model's
    /// `num_kv_heads` and `head_size` match the target model's dimensions.
    /// When true, `draft_kv_caches` is empty and `draft_kv_manager`/`draft_blocks`
    /// are unused.
    pub using_shared_kv: bool,
}

impl SpeculativeState {
    /// Create a new speculative state.
    ///
    /// Allocates draft model KV caches on the given device.
    /// If `shared_kv_cache` is enabled and the draft model's KV dimensions
    /// match the target model's, no separate draft KV caches are allocated.
    pub fn new(
        draft_model: Box<dyn ModelRunner>,
        device: &Device,
        num_draft_kv_blocks: usize,
        config: SpeculativeConfig,
    ) -> CandleResult<Self> {
        Self::new_with_target_config(draft_model, device, num_draft_kv_blocks, config, None)
    }

    /// Create a new speculative state with optional target model config for shared KV.
    ///
    /// When `target_config` is provided and `config.shared_kv_cache` is true,
    /// checks if dimensions match. If they do, skips draft KV cache allocation.
    pub fn new_with_target_config(
        draft_model: Box<dyn ModelRunner>,
        device: &Device,
        num_draft_kv_blocks: usize,
        config: SpeculativeConfig,
        target_config: Option<&super::models::ModelConfig>,
    ) -> CandleResult<Self> {
        let draft_config = draft_model.config();

        // Check if shared KV cache is possible
        let can_share = config.shared_kv_cache
            && target_config.map_or(false, |tc| {
                can_share_kv_cache(draft_config, tc)
            });

        if can_share {
            tracing::info!(
                "Shared KV cache enabled: draft and target have matching KV dimensions \
                 (num_kv_heads={}, head_size={})",
                draft_config.num_kv_heads,
                draft_config.head_size,
            );
        } else if config.shared_kv_cache && target_config.is_some() {
            let tc = target_config.unwrap();
            tracing::warn!(
                "Shared KV cache requested but dimensions mismatch: \
                 draft (kv_heads={}, head_size={}) vs target (kv_heads={}, head_size={}). \
                 Falling back to separate caches.",
                draft_config.num_kv_heads,
                draft_config.head_size,
                tc.num_kv_heads,
                tc.head_size,
            );
        }

        let kv_dtype = DType::F32;
        let block_size = super::kernels::BLOCK_SIZE;

        let (draft_kv_caches, draft_kv_manager) = if can_share {
            // Shared mode: no separate draft KV caches needed.
            // Create a minimal manager (won't be used for allocation).
            let minimal_config = KVCacheConfig {
                block_size,
                num_blocks: 0,
                num_kv_heads: draft_config.num_kv_heads,
                head_size: draft_config.head_size,
                num_layers: draft_config.num_layers,
                enable_prefix_cache: false,
                dtype_bytes: 4,
            };
            (Vec::new(), KVCacheManager::new(&minimal_config))
        } else {
            // Separate mode: allocate draft KV caches
            let x = 16 / kv_dtype.size_in_bytes();
            let mut caches = Vec::with_capacity(draft_config.num_layers);
            for _ in 0..draft_config.num_layers {
                let key_cache = Tensor::zeros(
                    (
                        num_draft_kv_blocks,
                        draft_config.num_kv_heads,
                        draft_config.head_size / x,
                        block_size,
                        x,
                    ),
                    kv_dtype,
                    device,
                )?;
                let value_cache = Tensor::zeros(
                    (
                        num_draft_kv_blocks,
                        draft_config.num_kv_heads,
                        draft_config.head_size,
                        block_size,
                    ),
                    kv_dtype,
                    device,
                )?;
                caches.push((key_cache, value_cache));
            }

            let kv_cache_config = KVCacheConfig {
                block_size,
                num_blocks: num_draft_kv_blocks,
                num_kv_heads: draft_config.num_kv_heads,
                head_size: draft_config.head_size,
                num_layers: draft_config.num_layers,
                enable_prefix_cache: false,
                dtype_bytes: 4,
            };
            (caches, KVCacheManager::new(&kv_cache_config))
        };

        let current_k = config.num_draft_tokens;

        Ok(Self {
            config,
            draft_model,
            draft_kv_caches,
            draft_kv_manager,
            draft_blocks: HashMap::new(),
            current_k,
            total_accepted: 0,
            total_drafted: 0,
            using_shared_kv: can_share,
        })
    }

    /// Start tracking a new sequence in the draft model.
    ///
    /// When using shared KV cache, no separate draft blocks are allocated.
    pub fn add_sequence(&mut self, seq_id: SeqId) {
        if !self.using_shared_kv {
            self.draft_blocks.insert(seq_id, SequenceBlocks::new());
        }
        // In shared mode, the draft uses the target's blocks directly
    }

    /// Stop tracking a sequence and free its draft blocks.
    pub fn remove_sequence(&mut self, seq_id: SeqId) {
        if !self.using_shared_kv {
            if let Some(mut blocks) = self.draft_blocks.remove(&seq_id) {
                self.draft_kv_manager.free(&mut blocks);
            }
        }
        // In shared mode, the target's cleanup handles block freeing
    }

    /// Sync draft model state after speculative verification.
    ///
    /// Sets the draft model's `num_computed_tokens` to match the target model.
    /// Stale KV entries beyond this point will be overwritten in the next
    /// draft phase via `reshape_and_cache`.
    ///
    /// In shared KV mode, this is a no-op since draft and target share state.
    pub fn sync_after_verify(&mut self, seq_id: SeqId, target_num_computed: usize) {
        if self.using_shared_kv {
            // Shared mode: draft and target share KV cache, so the target's
            // num_computed_tokens is already correct. The draft just needs to
            // know where to resume, but this is handled by the target's blocks.
            return;
        }
        if let Some(blocks) = self.draft_blocks.get_mut(&seq_id) {
            blocks.num_computed_tokens = target_num_computed;
        }
    }

    /// Get the KV caches for the draft model's forward pass.
    ///
    /// In shared mode, returns None (the caller should use the target's caches).
    /// In separate mode, returns the draft's own KV caches.
    pub fn draft_kv_caches_for_forward(&self) -> Option<&[(Tensor, Tensor)]> {
        if self.using_shared_kv {
            None
        } else {
            Some(&self.draft_kv_caches)
        }
    }

    /// Returns the memory saved by using shared KV cache, in bytes.
    ///
    /// When not in shared mode, returns 0.
    pub fn shared_kv_memory_savings(&self) -> usize {
        if !self.using_shared_kv {
            return 0;
        }
        // Calculate what the draft KV caches would have cost
        let draft_config = self.draft_model.config();
        let block_size = super::kernels::BLOCK_SIZE;
        let dtype_bytes = 4; // F32
        let kv_per_block_per_layer =
            2 * draft_config.num_kv_heads * draft_config.head_size * block_size * dtype_bytes;
        // Use the config's num_draft_tokens as a rough proxy for blocks
        // (actual savings depend on how many blocks would have been allocated)
        draft_config.num_layers * kv_per_block_per_layer
    }

    /// Update the adaptive K based on running acceptance rate.
    ///
    /// Called after each speculative step:
    /// - If acceptance rate > 80%, increase K (more aggressive drafting)
    /// - If acceptance rate < 50%, decrease K (too many rejections)
    pub fn update_adaptive_k(&mut self) {
        if !self.config.adaptive || self.total_drafted == 0 {
            return;
        }

        let acceptance_rate = self.total_accepted as f64 / self.total_drafted as f64;

        if acceptance_rate > 0.8 && self.current_k < self.config.max_draft_tokens {
            self.current_k += 1;
            tracing::debug!(
                "Adaptive K increased to {} (acceptance rate: {:.1}%)",
                self.current_k,
                acceptance_rate * 100.0
            );
        } else if acceptance_rate < 0.5 && self.current_k > self.config.min_draft_tokens {
            self.current_k -= 1;
            tracing::debug!(
                "Adaptive K decreased to {} (acceptance rate: {:.1}%)",
                self.current_k,
                acceptance_rate * 100.0
            );
        }
    }

    /// Get the current acceptance rate (0.0 to 1.0).
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_drafted == 0 {
            0.0
        } else {
            self.total_accepted as f64 / self.total_drafted as f64
        }
    }
}

// ─── Shared KV cache compatibility check ─────────────────────────────────

/// Check whether draft and target models can share a KV cache.
///
/// Returns true if both models have the same `num_kv_heads` and `head_size`,
/// meaning their KV tensors have identical shapes per layer. The number of
/// layers can differ (the draft model simply uses fewer layers of the same
/// shaped cache).
pub fn can_share_kv_cache(
    draft_config: &super::models::ModelConfig,
    target_config: &super::models::ModelConfig,
) -> bool {
    draft_config.num_kv_heads == target_config.num_kv_heads
        && draft_config.head_size == target_config.head_size
}

// ─── Accept/reject algorithm ─────────────────────────────────────────────

/// Result of the speculative accept/reject step for one sequence.
#[derive(Debug)]
pub struct VerifyResult {
    /// Number of draft tokens accepted (0 to K).
    pub num_accepted: usize,
    /// The accepted draft token IDs, in order.
    pub accepted_tokens: Vec<u32>,
    /// The final token (bonus if all accepted, or resampled from adjusted distribution).
    pub final_token: u32,
}

/// Run the speculative decoding accept/reject algorithm.
///
/// Given K draft tokens with their probabilities and the corresponding target
/// model probabilities, determines which draft tokens to accept and produces
/// the final output tokens.
///
/// # Arguments
/// - `target_probs`: Target model probability distributions, one per draft position.
///   Each is a full vocab-size probability vector. Length = K.
/// - `draft_probs`: Draft model probability distributions, one per draft position.
///   Each is a full vocab-size probability vector. Length = K.
/// - `draft_tokens`: The draft token IDs sampled from the draft model. Length = K.
/// - `bonus_probs`: Target model probability distribution at position K+1 (after all
///   draft tokens). Used to sample the bonus token if all drafts are accepted.
///   If None, the last target_probs entry is used for resampling on full acceptance.
///
/// # Returns
/// A `VerifyResult` with accepted tokens and the final bonus/resampled token.
pub fn accept_reject(
    target_probs: &[Vec<f32>],
    draft_probs: &[Vec<f32>],
    draft_tokens: &[u32],
    bonus_probs: Option<&[f32]>,
) -> VerifyResult {
    assert_eq!(target_probs.len(), draft_tokens.len());
    assert_eq!(draft_probs.len(), draft_tokens.len());

    let k = draft_tokens.len();
    let mut accepted_tokens = Vec::with_capacity(k);

    for i in 0..k {
        let token = draft_tokens[i] as usize;
        let p_target = target_probs[i].get(token).copied().unwrap_or(0.0);
        let p_draft = draft_probs[i].get(token).copied().unwrap_or(0.0);

        // Accept with probability min(1, p_target / p_draft)
        if p_draft > 0.0 {
            let acceptance_prob = (p_target / p_draft).min(1.0);
            let r = pseudo_random();

            if r < acceptance_prob {
                accepted_tokens.push(draft_tokens[i]);
                continue;
            }
        } else if p_target > 0.0 {
            // Draft assigned 0 probability but target didn't — always accept
            accepted_tokens.push(draft_tokens[i]);
            continue;
        }

        // Rejected at position i: resample from adjusted distribution
        let adjusted = adjusted_distribution(&target_probs[i], &draft_probs[i]);
        let resampled = sample_from_distribution(&adjusted);

        return VerifyResult {
            num_accepted: i,
            accepted_tokens,
            final_token: resampled as u32,
        };
    }

    // All K tokens accepted — sample bonus token
    let final_token = if let Some(bonus) = bonus_probs {
        sample_from_distribution(bonus) as u32
    } else {
        // No separate bonus probs; shouldn't happen in practice
        // Fall back to sampling from the last target position
        sample_from_distribution(&target_probs[k - 1]) as u32
    };

    VerifyResult {
        num_accepted: k,
        accepted_tokens,
        final_token,
    }
}

/// Compute the adjusted distribution: max(0, p_target - p_draft), normalized.
///
/// Used when a draft token is rejected — we resample from this distribution
/// to maintain the target model's output distribution.
fn adjusted_distribution(target: &[f32], draft: &[f32]) -> Vec<f32> {
    let vocab_size = target.len();
    let mut adjusted = Vec::with_capacity(vocab_size);
    let mut total = 0.0f32;

    for i in 0..vocab_size {
        let p_target = target.get(i).copied().unwrap_or(0.0);
        let p_draft = draft.get(i).copied().unwrap_or(0.0);
        let val = (p_target - p_draft).max(0.0);
        adjusted.push(val);
        total += val;
    }

    // Normalize
    if total > 0.0 {
        for p in &mut adjusted {
            *p /= total;
        }
    } else {
        // Fallback: uniform distribution (shouldn't happen in practice)
        let uniform = 1.0 / vocab_size as f32;
        for p in &mut adjusted {
            *p = uniform;
        }
    }

    adjusted
}

/// Sample a token index from a probability distribution.
fn sample_from_distribution(probs: &[f32]) -> usize {
    let r = pseudo_random();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= r {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

/// Compute softmax probabilities from logits.
pub fn softmax_probs(logits: &[f32]) -> Vec<f32> {
    let max = logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    if sum == 0.0 {
        vec![1.0 / logits.len() as f32; logits.len()]
    } else {
        exp.iter().map(|&e| e / sum).collect()
    }
}

/// Pseudo-random float in [0, 1) using thread-local xorshift64.
fn pseudo_random() -> f32 {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u64> = const { Cell::new(0xABCDEF01_23456789) };
    }
    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        (x >> 11) as f32 / (1u64 << 53) as f32
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a probability distribution with one token having
    /// most of the mass.
    fn peaked_probs(vocab_size: usize, peak_idx: usize, peak_prob: f32) -> Vec<f32> {
        let mut probs = vec![(1.0 - peak_prob) / (vocab_size - 1) as f32; vocab_size];
        probs[peak_idx] = peak_prob;
        probs
    }

    /// Helper: create a uniform probability distribution.
    fn uniform_probs(vocab_size: usize) -> Vec<f32> {
        vec![1.0 / vocab_size as f32; vocab_size]
    }

    #[test]
    fn test_accept_reject_all_accepted() {
        // Target and draft agree perfectly — all tokens should be accepted
        let vocab_size = 100;
        let k = 4;
        let draft_tokens: Vec<u32> = vec![10, 20, 30, 40];

        let mut target_probs = Vec::new();
        let mut draft_probs = Vec::new();
        for &token in &draft_tokens {
            // Both models assign high probability to the same token
            target_probs.push(peaked_probs(vocab_size, token as usize, 0.9));
            draft_probs.push(peaked_probs(vocab_size, token as usize, 0.9));
        }

        let bonus = peaked_probs(vocab_size, 50, 0.9);

        let result = accept_reject(&target_probs, &draft_probs, &draft_tokens, Some(&bonus));

        assert_eq!(result.num_accepted, k);
        assert_eq!(result.accepted_tokens, draft_tokens);
        // Bonus token should be sampled (most likely 50 given peaked probs)
    }

    #[test]
    fn test_accept_reject_first_rejected() {
        // Target assigns zero probability to the draft token → immediate rejection
        let vocab_size = 100;
        let draft_tokens: Vec<u32> = vec![10, 20, 30, 40];

        let mut target_probs = Vec::new();
        let mut draft_probs = Vec::new();

        // Position 0: target assigns 0 probability to token 10
        let mut tp0 = uniform_probs(vocab_size);
        tp0[10] = 0.0;
        // Renormalize
        let sum: f32 = tp0.iter().sum();
        for p in &mut tp0 {
            *p /= sum;
        }
        target_probs.push(tp0);
        draft_probs.push(peaked_probs(vocab_size, 10, 0.5));

        // Remaining positions (won't be reached)
        for &token in &draft_tokens[1..] {
            target_probs.push(peaked_probs(vocab_size, token as usize, 0.9));
            draft_probs.push(peaked_probs(vocab_size, token as usize, 0.9));
        }

        let result = accept_reject(&target_probs, &draft_probs, &draft_tokens, None);

        assert_eq!(result.num_accepted, 0);
        assert!(result.accepted_tokens.is_empty());
        // Resampled token should not be 10 (target has 0 prob for it)
        assert_ne!(result.final_token, 10);
    }

    #[test]
    fn test_accept_reject_partial() {
        // First 2 tokens match well, 3rd has very different probabilities
        let vocab_size = 100;
        let draft_tokens: Vec<u32> = vec![10, 20, 30, 40];

        let mut target_probs = Vec::new();
        let mut draft_probs = Vec::new();

        // Positions 0,1: target agrees with draft
        for &token in &draft_tokens[..2] {
            target_probs.push(peaked_probs(vocab_size, token as usize, 0.9));
            draft_probs.push(peaked_probs(vocab_size, token as usize, 0.9));
        }

        // Position 2: target strongly disagrees (target wants token 99, not 30)
        target_probs.push(peaked_probs(vocab_size, 99, 0.99));
        draft_probs.push(peaked_probs(vocab_size, 30, 0.99));

        // Position 3: (won't be reached)
        target_probs.push(peaked_probs(vocab_size, 40, 0.9));
        draft_probs.push(peaked_probs(vocab_size, 40, 0.9));

        let result = accept_reject(&target_probs, &draft_probs, &draft_tokens, None);

        // First 2 should be accepted (target agrees)
        assert_eq!(result.num_accepted, 2);
        assert_eq!(result.accepted_tokens, vec![10, 20]);
        // The resampled token at position 2 should likely be 99 (target's peak)
    }

    #[test]
    fn test_accept_reject_deterministic_greedy() {
        // When both models assign probability 1.0 to the same token,
        // acceptance is guaranteed
        let vocab_size = 10;
        let draft_tokens: Vec<u32> = vec![5, 3, 7];

        let mut target_probs = Vec::new();
        let mut draft_probs = Vec::new();

        for &token in &draft_tokens {
            let mut tp = vec![0.0f32; vocab_size];
            tp[token as usize] = 1.0;
            let mut dp = vec![0.0f32; vocab_size];
            dp[token as usize] = 1.0;
            target_probs.push(tp);
            draft_probs.push(dp);
        }

        let bonus = {
            let mut bp = vec![0.0f32; vocab_size];
            bp[2] = 1.0;
            bp
        };

        let result = accept_reject(&target_probs, &draft_probs, &draft_tokens, Some(&bonus));

        assert_eq!(result.num_accepted, 3);
        assert_eq!(result.accepted_tokens, vec![5, 3, 7]);
        assert_eq!(result.final_token, 2); // bonus token
    }

    #[test]
    fn test_adjusted_distribution() {
        let target = vec![0.3, 0.5, 0.2];
        let draft = vec![0.1, 0.8, 0.1];

        let adjusted = adjusted_distribution(&target, &draft);

        // max(0, 0.3 - 0.1) = 0.2
        // max(0, 0.5 - 0.8) = 0.0
        // max(0, 0.2 - 0.1) = 0.1
        // Total = 0.3, normalized: [0.667, 0.0, 0.333]
        assert!((adjusted[0] - 2.0 / 3.0).abs() < 1e-5);
        assert!((adjusted[1] - 0.0).abs() < 1e-5);
        assert!((adjusted[2] - 1.0 / 3.0).abs() < 1e-5);

        // Should sum to 1.0
        let sum: f32 = adjusted.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_adjusted_distribution_identical() {
        // When target == draft, adjusted distribution is all zeros → uniform fallback
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let adjusted = adjusted_distribution(&probs, &probs);

        let sum: f32 = adjusted.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_probs() {
        let logits = vec![1.0f32, 2.0, 3.0];
        let probs = softmax_probs(&logits);

        assert_eq!(probs.len(), 3);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs[0] < probs[1]);
        assert!(probs[1] < probs[2]);
    }

    #[test]
    fn test_softmax_probs_large_values() {
        let logits = vec![1000.0f32, 1001.0, 999.0];
        let probs = softmax_probs(&logits);

        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_adaptive_k_increase() {
        let config = SpeculativeConfig {
            num_draft_tokens: 4,
            adaptive: true,
            min_draft_tokens: 1,
            max_draft_tokens: 8,
            shared_kv_cache: false,
        };

        // Use a minimal mock — we only need config, not actual model/caches
        let mut state = SpeculativeState {
            config,
            draft_model: Box::new(MockModelForTest::default()),
            draft_kv_caches: Vec::new(),
            draft_kv_manager: KVCacheManager::new(&KVCacheConfig {
                block_size: 16,
                num_blocks: 4,
                num_kv_heads: 1,
                head_size: 16,
                num_layers: 1,
                enable_prefix_cache: false,
                dtype_bytes: 2,
            }),
            draft_blocks: HashMap::new(),
            current_k: 4,
            total_accepted: 90,
            total_drafted: 100,
            using_shared_kv: false,
        };

        state.update_adaptive_k();
        assert_eq!(state.current_k, 5); // Should increase (90% > 80%)
    }

    #[test]
    fn test_adaptive_k_decrease() {
        let config = SpeculativeConfig {
            num_draft_tokens: 4,
            adaptive: true,
            min_draft_tokens: 1,
            max_draft_tokens: 8,
            shared_kv_cache: false,
        };

        let mut state = SpeculativeState {
            config,
            draft_model: Box::new(MockModelForTest::default()),
            draft_kv_caches: Vec::new(),
            draft_kv_manager: KVCacheManager::new(&KVCacheConfig {
                block_size: 16,
                num_blocks: 4,
                num_kv_heads: 1,
                head_size: 16,
                num_layers: 1,
                enable_prefix_cache: false,
                dtype_bytes: 2,
            }),
            draft_blocks: HashMap::new(),
            current_k: 4,
            total_accepted: 30,
            total_drafted: 100,
            using_shared_kv: false,
        };

        state.update_adaptive_k();
        assert_eq!(state.current_k, 3); // Should decrease (30% < 50%)
    }

    #[test]
    fn test_adaptive_k_clamp_max() {
        let config = SpeculativeConfig {
            num_draft_tokens: 4,
            adaptive: true,
            min_draft_tokens: 1,
            max_draft_tokens: 4, // Already at max
            shared_kv_cache: false,
        };

        let mut state = SpeculativeState {
            config,
            draft_model: Box::new(MockModelForTest::default()),
            draft_kv_caches: Vec::new(),
            draft_kv_manager: KVCacheManager::new(&KVCacheConfig {
                block_size: 16,
                num_blocks: 4,
                num_kv_heads: 1,
                head_size: 16,
                num_layers: 1,
                enable_prefix_cache: false,
                dtype_bytes: 2,
            }),
            draft_blocks: HashMap::new(),
            current_k: 4, // At max
            total_accepted: 95,
            total_drafted: 100,
            using_shared_kv: false,
        };

        state.update_adaptive_k();
        assert_eq!(state.current_k, 4); // Should stay at max
    }

    #[test]
    fn test_adaptive_k_clamp_min() {
        let config = SpeculativeConfig {
            num_draft_tokens: 4,
            adaptive: true,
            min_draft_tokens: 1,
            max_draft_tokens: 8,
            shared_kv_cache: false,
        };

        let mut state = SpeculativeState {
            config,
            draft_model: Box::new(MockModelForTest::default()),
            draft_kv_caches: Vec::new(),
            draft_kv_manager: KVCacheManager::new(&KVCacheConfig {
                block_size: 16,
                num_blocks: 4,
                num_kv_heads: 1,
                head_size: 16,
                num_layers: 1,
                enable_prefix_cache: false,
                dtype_bytes: 2,
            }),
            draft_blocks: HashMap::new(),
            current_k: 1, // At min
            total_accepted: 10,
            total_drafted: 100,
            using_shared_kv: false,
        };

        state.update_adaptive_k();
        assert_eq!(state.current_k, 1); // Should stay at min
    }

    #[test]
    fn test_adaptive_k_disabled() {
        let config = SpeculativeConfig {
            num_draft_tokens: 4,
            adaptive: false,
            min_draft_tokens: 1,
            max_draft_tokens: 8,
            shared_kv_cache: false,
        };

        let mut state = SpeculativeState {
            config,
            draft_model: Box::new(MockModelForTest::default()),
            draft_kv_caches: Vec::new(),
            draft_kv_manager: KVCacheManager::new(&KVCacheConfig {
                block_size: 16,
                num_blocks: 4,
                num_kv_heads: 1,
                head_size: 16,
                num_layers: 1,
                enable_prefix_cache: false,
                dtype_bytes: 2,
            }),
            draft_blocks: HashMap::new(),
            current_k: 4,
            total_accepted: 95,
            total_drafted: 100,
            using_shared_kv: false,
        };

        state.update_adaptive_k();
        assert_eq!(state.current_k, 4); // Should not change
    }

    #[test]
    fn test_add_remove_sequence() {
        let config = SpeculativeConfig::default();
        let mut state = SpeculativeState {
            config,
            draft_model: Box::new(MockModelForTest::default()),
            draft_kv_caches: Vec::new(),
            draft_kv_manager: KVCacheManager::new(&KVCacheConfig {
                block_size: 16,
                num_blocks: 8,
                num_kv_heads: 1,
                head_size: 16,
                num_layers: 1,
                enable_prefix_cache: false,
                dtype_bytes: 2,
            }),
            draft_blocks: HashMap::new(),
            current_k: 4,
            total_accepted: 0,
            total_drafted: 0,
            using_shared_kv: false,
        };

        state.add_sequence(1);
        assert!(state.draft_blocks.contains_key(&1));

        state.add_sequence(2);
        assert_eq!(state.draft_blocks.len(), 2);

        state.remove_sequence(1);
        assert!(!state.draft_blocks.contains_key(&1));
        assert_eq!(state.draft_blocks.len(), 1);
    }

    #[test]
    fn test_sync_after_verify() {
        let config = SpeculativeConfig::default();
        let mut state = SpeculativeState {
            config,
            draft_model: Box::new(MockModelForTest::default()),
            draft_kv_caches: Vec::new(),
            draft_kv_manager: KVCacheManager::new(&KVCacheConfig {
                block_size: 16,
                num_blocks: 8,
                num_kv_heads: 1,
                head_size: 16,
                num_layers: 1,
                enable_prefix_cache: false,
                dtype_bytes: 2,
            }),
            draft_blocks: HashMap::new(),
            current_k: 4,
            total_accepted: 0,
            total_drafted: 0,
            using_shared_kv: false,
        };

        state.add_sequence(1);
        // Simulate draft running ahead to position 50
        state.draft_blocks.get_mut(&1).unwrap().num_computed_tokens = 50;

        // After verification, target accepted up to position 45
        state.sync_after_verify(1, 45);
        assert_eq!(
            state.draft_blocks.get(&1).unwrap().num_computed_tokens,
            45
        );
    }

    #[test]
    fn test_acceptance_rate() {
        let config = SpeculativeConfig::default();
        let state = SpeculativeState {
            config,
            draft_model: Box::new(MockModelForTest::default()),
            draft_kv_caches: Vec::new(),
            draft_kv_manager: KVCacheManager::new(&KVCacheConfig {
                block_size: 16,
                num_blocks: 4,
                num_kv_heads: 1,
                head_size: 16,
                num_layers: 1,
                enable_prefix_cache: false,
                dtype_bytes: 2,
            }),
            draft_blocks: HashMap::new(),
            current_k: 4,
            total_accepted: 75,
            total_drafted: 100,
            using_shared_kv: false,
        };

        assert!((state.acceptance_rate() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_acceptance_rate_empty() {
        let config = SpeculativeConfig::default();
        let state = SpeculativeState {
            config,
            draft_model: Box::new(MockModelForTest::default()),
            draft_kv_caches: Vec::new(),
            draft_kv_manager: KVCacheManager::new(&KVCacheConfig {
                block_size: 16,
                num_blocks: 4,
                num_kv_heads: 1,
                head_size: 16,
                num_layers: 1,
                enable_prefix_cache: false,
                dtype_bytes: 2,
            }),
            draft_blocks: HashMap::new(),
            current_k: 4,
            total_accepted: 0,
            total_drafted: 0,
            using_shared_kv: false,
        };

        assert_eq!(state.acceptance_rate(), 0.0);
    }

    #[test]
    fn test_speculative_config_default() {
        let config = SpeculativeConfig::default();
        assert_eq!(config.num_draft_tokens, 4);
        assert!(config.adaptive);
        assert_eq!(config.min_draft_tokens, 1);
        assert_eq!(config.max_draft_tokens, 8);
        assert!(!config.shared_kv_cache);
    }

    /// Minimal mock model for testing SpeculativeState methods that don't
    /// actually call forward().
    struct MockModelForTest {
        config: super::super::models::ModelConfig,
    }

    impl Default for MockModelForTest {
        fn default() -> Self {
            Self {
                config: super::super::models::ModelConfig {
                    hidden_size: 16,
                    intermediate_size: 64,
                    num_heads: 1,
                    num_kv_heads: 1,
                    num_layers: 1,
                    head_size: 16,
                    vocab_size: 10,
                    rms_norm_eps: 1e-5,
                    rope_theta: 10000.0,
                    rope_dim: 16,
                    max_seq_len: 128,
                },
            }
        }
    }

    impl ModelRunner for MockModelForTest {
        fn forward(
            &self,
            input_ids: &Tensor,
            _ctx: &super::super::models::ForwardContext,
        ) -> CandleResult<Tensor> {
            let total = input_ids.dims()[0];
            Tensor::zeros((total, self.config.vocab_size), DType::F32, input_ids.device())
        }
        fn clone_model(&self) -> Box<dyn ModelRunner> {
            Box::new(MockModelForTest {
                config: self.config.clone(),
            })
        }
        fn num_layers(&self) -> usize {
            self.config.num_layers
        }
        fn num_kv_heads(&self) -> usize {
            self.config.num_kv_heads
        }
        fn head_size(&self) -> usize {
            self.config.head_size
        }
        fn num_heads(&self) -> usize {
            self.config.num_heads
        }
        fn config(&self) -> &super::super::models::ModelConfig {
            &self.config
        }
    }

    fn mock_model_config(num_kv_heads: usize, head_size: usize) -> super::super::models::ModelConfig {
        super::super::models::ModelConfig {
            hidden_size: 64,
            intermediate_size: 256,
            num_heads: 4,
            num_kv_heads,
            num_layers: 2,
            head_size,
            vocab_size: 100,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_dim: head_size,
            max_seq_len: 512,
        }
    }

    #[test]
    fn test_can_share_kv_cache_matching() {
        let draft = mock_model_config(8, 128);
        let target = mock_model_config(8, 128);
        assert!(can_share_kv_cache(&draft, &target));
    }

    #[test]
    fn test_can_share_kv_cache_different_kv_heads() {
        let draft = mock_model_config(4, 128);
        let target = mock_model_config(8, 128);
        assert!(!can_share_kv_cache(&draft, &target));
    }

    #[test]
    fn test_can_share_kv_cache_different_head_size() {
        let draft = mock_model_config(8, 64);
        let target = mock_model_config(8, 128);
        assert!(!can_share_kv_cache(&draft, &target));
    }

    #[test]
    fn test_can_share_kv_cache_both_different() {
        let draft = mock_model_config(4, 64);
        let target = mock_model_config(8, 128);
        assert!(!can_share_kv_cache(&draft, &target));
    }

    #[test]
    fn test_shared_kv_add_remove_sequence() {
        let config = SpeculativeConfig {
            shared_kv_cache: true,
            ..SpeculativeConfig::default()
        };
        let mut state = SpeculativeState {
            config,
            draft_model: Box::new(MockModelForTest::default()),
            draft_kv_caches: Vec::new(),
            draft_kv_manager: KVCacheManager::new(&KVCacheConfig {
                block_size: 16,
                num_blocks: 0,
                num_kv_heads: 1,
                head_size: 16,
                num_layers: 1,
                enable_prefix_cache: false,
                dtype_bytes: 2,
            }),
            draft_blocks: HashMap::new(),
            current_k: 4,
            total_accepted: 0,
            total_drafted: 0,
            using_shared_kv: true,
        };

        // In shared mode, add_sequence should NOT create draft blocks
        state.add_sequence(1);
        assert!(!state.draft_blocks.contains_key(&1));

        // remove_sequence should be a no-op in shared mode
        state.remove_sequence(1);
        assert!(state.draft_blocks.is_empty());
    }

    #[test]
    fn test_shared_kv_sync_after_verify_noop() {
        let config = SpeculativeConfig {
            shared_kv_cache: true,
            ..SpeculativeConfig::default()
        };
        let mut state = SpeculativeState {
            config,
            draft_model: Box::new(MockModelForTest::default()),
            draft_kv_caches: Vec::new(),
            draft_kv_manager: KVCacheManager::new(&KVCacheConfig {
                block_size: 16,
                num_blocks: 0,
                num_kv_heads: 1,
                head_size: 16,
                num_layers: 1,
                enable_prefix_cache: false,
                dtype_bytes: 2,
            }),
            draft_blocks: HashMap::new(),
            current_k: 4,
            total_accepted: 0,
            total_drafted: 0,
            using_shared_kv: true,
        };

        // Should not panic; it's a no-op in shared mode
        state.sync_after_verify(1, 100);
    }

    #[test]
    fn test_draft_kv_caches_for_forward_shared() {
        let config = SpeculativeConfig {
            shared_kv_cache: true,
            ..SpeculativeConfig::default()
        };
        let state = SpeculativeState {
            config,
            draft_model: Box::new(MockModelForTest::default()),
            draft_kv_caches: Vec::new(),
            draft_kv_manager: KVCacheManager::new(&KVCacheConfig {
                block_size: 16,
                num_blocks: 0,
                num_kv_heads: 1,
                head_size: 16,
                num_layers: 1,
                enable_prefix_cache: false,
                dtype_bytes: 2,
            }),
            draft_blocks: HashMap::new(),
            current_k: 4,
            total_accepted: 0,
            total_drafted: 0,
            using_shared_kv: true,
        };

        // Shared mode: returns None (use target's caches)
        assert!(state.draft_kv_caches_for_forward().is_none());
    }

    #[test]
    fn test_draft_kv_caches_for_forward_separate() {
        let config = SpeculativeConfig::default();
        let state = SpeculativeState {
            config,
            draft_model: Box::new(MockModelForTest::default()),
            draft_kv_caches: Vec::new(),
            draft_kv_manager: KVCacheManager::new(&KVCacheConfig {
                block_size: 16,
                num_blocks: 4,
                num_kv_heads: 1,
                head_size: 16,
                num_layers: 1,
                enable_prefix_cache: false,
                dtype_bytes: 2,
            }),
            draft_blocks: HashMap::new(),
            current_k: 4,
            total_accepted: 0,
            total_drafted: 0,
            using_shared_kv: false,
        };

        // Separate mode: returns Some (draft's own caches)
        assert!(state.draft_kv_caches_for_forward().is_some());
    }

    #[test]
    fn test_shared_kv_memory_savings_when_shared() {
        let config = SpeculativeConfig {
            shared_kv_cache: true,
            ..SpeculativeConfig::default()
        };
        let state = SpeculativeState {
            config,
            draft_model: Box::new(MockModelForTest::default()),
            draft_kv_caches: Vec::new(),
            draft_kv_manager: KVCacheManager::new(&KVCacheConfig {
                block_size: 16,
                num_blocks: 0,
                num_kv_heads: 1,
                head_size: 16,
                num_layers: 1,
                enable_prefix_cache: false,
                dtype_bytes: 2,
            }),
            draft_blocks: HashMap::new(),
            current_k: 4,
            total_accepted: 0,
            total_drafted: 0,
            using_shared_kv: true,
        };

        let savings = state.shared_kv_memory_savings();
        assert!(savings > 0, "shared mode should report memory savings");
    }

    #[test]
    fn test_shared_kv_memory_savings_when_separate() {
        let config = SpeculativeConfig::default();
        let state = SpeculativeState {
            config,
            draft_model: Box::new(MockModelForTest::default()),
            draft_kv_caches: Vec::new(),
            draft_kv_manager: KVCacheManager::new(&KVCacheConfig {
                block_size: 16,
                num_blocks: 4,
                num_kv_heads: 1,
                head_size: 16,
                num_layers: 1,
                enable_prefix_cache: false,
                dtype_bytes: 2,
            }),
            draft_blocks: HashMap::new(),
            current_k: 4,
            total_accepted: 0,
            total_drafted: 0,
            using_shared_kv: false,
        };

        assert_eq!(state.shared_kv_memory_savings(), 0);
    }
}
