//! Beam search sampling for multi-candidate generation.
//!
//! When a request specifies `best_of > 1`, the engine uses beam search
//! instead of independent sampling. At each decode step:
//!
//! 1. For each active beam, get top-k log probabilities from logits
//! 2. Expand all beams by their top-k candidates
//! 3. Score candidates with length penalty
//! 4. Prune to `beam_width` best candidates
//! 5. Use CoW `fork_sequence` to share KV cache blocks between beams
//!
//! Length penalty formula: `score = cumulative_log_prob / (length ^ length_penalty)`
//! where length_penalty=1.0 gives no normalization, >1.0 favors longer sequences,
//! and <1.0 favors shorter sequences.

use std::collections::HashMap;

use super::sequence::SeqId;

/// A single beam in a beam search group.
#[derive(Debug, Clone)]
pub struct Beam {
    /// Token IDs generated so far (output tokens only, not prompt).
    pub token_ids: Vec<u32>,
    /// Cumulative log probability of the beam.
    pub cumulative_log_prob: f64,
    /// Length-penalized score.
    pub score: f64,
    /// Sequence ID in the scheduler (for KV cache tracking).
    pub sequence_id: SeqId,
}

impl Beam {
    /// Create a new beam from an existing sequence.
    pub fn new(sequence_id: SeqId) -> Self {
        Self {
            token_ids: Vec::new(),
            cumulative_log_prob: 0.0,
            score: 0.0,
            sequence_id,
        }
    }

    /// Compute the length-penalized score.
    ///
    /// score = cumulative_log_prob / (length ^ length_penalty)
    ///
    /// When length_penalty = 1.0, this is just dividing by length (average log prob).
    /// When length_penalty = 0.0, score equals cumulative_log_prob (no normalization).
    pub fn compute_score(&self, length_penalty: f32) -> f64 {
        let len = (self.token_ids.len().max(1)) as f64;
        if length_penalty.abs() < 1e-7 {
            self.cumulative_log_prob
        } else {
            self.cumulative_log_prob / len.powf(length_penalty as f64)
        }
    }
}

/// State for a beam search group (one per request that uses beam search).
#[derive(Debug)]
pub struct BeamSearchState {
    /// Active (unfinished) beams.
    pub beams: Vec<Beam>,
    /// Beam width (number of beams to keep at each step).
    pub beam_width: usize,
    /// Number of best results to return (n from the request).
    pub n_best: usize,
    /// Length penalty exponent.
    pub length_penalty: f32,
    /// Whether to stop early when top beam can't be beaten.
    pub early_stopping: bool,
    /// Beams that have finished (hit EOS or stop token).
    pub finished_beams: Vec<Beam>,
    /// The original (parent) sequence ID that spawned this beam group.
    pub origin_seq_id: SeqId,
}

impl BeamSearchState {
    /// Create a new beam search state for a request.
    ///
    /// Initially there is one beam (the original sequence).
    pub fn new(
        origin_seq_id: SeqId,
        beam_width: usize,
        n_best: usize,
        length_penalty: f32,
        early_stopping: bool,
    ) -> Self {
        let initial_beam = Beam::new(origin_seq_id);
        Self {
            beams: vec![initial_beam],
            beam_width,
            n_best,
            length_penalty,
            early_stopping,
            finished_beams: Vec::new(),
            origin_seq_id,
        }
    }

    /// Whether beam search is complete.
    ///
    /// Complete when we have enough finished beams and (if early_stopping)
    /// the best finished beam can't be beaten by any active beam.
    pub fn is_complete(&self) -> bool {
        if self.beams.is_empty() {
            return true;
        }

        if self.finished_beams.len() < self.n_best {
            return false;
        }

        if !self.early_stopping {
            return false;
        }

        // Check if any active beam could potentially beat the worst finished beam
        let worst_finished_score = self
            .finished_beams
            .iter()
            .map(|b| b.score)
            .fold(f64::INFINITY, f64::min);

        // Best possible score for active beams (assuming 0.0 log prob for each future token)
        // Since log probs are non-positive, the current cumulative is the best possible
        let best_active_potential = self
            .beams
            .iter()
            .map(|b| b.compute_score(self.length_penalty))
            .fold(f64::NEG_INFINITY, f64::max);

        best_active_potential <= worst_finished_score
    }

    /// Expand beams with new token candidates and prune to beam_width.
    ///
    /// For each active beam, takes the top-k token candidates (with their log probs),
    /// creates expanded candidates, scores them, and keeps the best `beam_width`.
    ///
    /// Returns a list of `BeamExpansion` actions that the engine should execute:
    /// - Fork sequences for new beams
    /// - Drop sequences for pruned beams
    /// - Append tokens to surviving beams
    pub fn expand_and_prune(
        &mut self,
        beam_logprobs: &HashMap<SeqId, Vec<(u32, f64)>>,
        stop_token_ids: &[u32],
        eos_token_id: u32,
    ) -> BeamStepResult {
        let mut candidates: Vec<BeamCandidate> = Vec::new();

        // Expand each active beam
        for beam in &self.beams {
            if let Some(top_tokens) = beam_logprobs.get(&beam.sequence_id) {
                for &(token_id, log_prob) in top_tokens {
                    let new_cum_log_prob = beam.cumulative_log_prob + log_prob;
                    let mut new_token_ids = beam.token_ids.clone();
                    new_token_ids.push(token_id);

                    let is_stop = token_id == eos_token_id
                        || stop_token_ids.contains(&token_id);

                    let candidate = BeamCandidate {
                        parent_seq_id: beam.sequence_id,
                        token_id,
                        token_ids: new_token_ids,
                        cumulative_log_prob: new_cum_log_prob,
                        is_finished: is_stop,
                        score: 0.0, // Computed below
                    };
                    candidates.push(candidate);
                }
            }
        }

        // Score all candidates
        for c in &mut candidates {
            let len = c.token_ids.len().max(1) as f64;
            if self.length_penalty.abs() < 1e-7 {
                c.score = c.cumulative_log_prob;
            } else {
                c.score = c.cumulative_log_prob / len.powf(self.length_penalty as f64);
            }
        }

        // Separate finished and active candidates
        let (finished, mut active): (Vec<_>, Vec<_>) =
            candidates.into_iter().partition(|c| c.is_finished);

        // Add finished candidates to finished_beams
        let mut newly_finished = Vec::new();
        for fc in finished {
            let beam = Beam {
                token_ids: fc.token_ids,
                cumulative_log_prob: fc.cumulative_log_prob,
                score: fc.score,
                sequence_id: fc.parent_seq_id,
            };
            newly_finished.push(beam.clone());
            self.finished_beams.push(beam);
        }

        // Sort active candidates by score (descending)
        active.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Prune to beam_width
        active.truncate(self.beam_width);

        // Determine which parent sequences are still needed
        let old_seq_ids: Vec<SeqId> = self.beams.iter().map(|b| b.sequence_id).collect();
        let new_parent_seq_ids: Vec<SeqId> = active.iter().map(|c| c.parent_seq_id).collect();

        // Sequences to drop (not used by any surviving candidate)
        let seqs_to_drop: Vec<SeqId> = old_seq_ids
            .iter()
            .filter(|id| !new_parent_seq_ids.contains(id))
            .copied()
            .collect();

        // Build token assignments: which token to append to which sequence
        // If multiple candidates share the same parent, we need to fork
        let mut parent_usage_count: HashMap<SeqId, usize> = HashMap::new();
        for c in &active {
            *parent_usage_count.entry(c.parent_seq_id).or_insert(0) += 1;
        }

        let mut token_assignments: Vec<BeamTokenAssignment> = Vec::new();
        let mut forks_needed: Vec<ForkRequest> = Vec::new();
        let mut parent_first_use: HashMap<SeqId, bool> = HashMap::new();

        for c in &active {
            let is_first = !parent_first_use.contains_key(&c.parent_seq_id);
            parent_first_use.insert(c.parent_seq_id, true);

            if is_first {
                // First use of this parent: reuse the existing sequence
                token_assignments.push(BeamTokenAssignment {
                    seq_id: c.parent_seq_id,
                    token_id: c.token_id,
                    cumulative_log_prob: c.cumulative_log_prob,
                    token_ids: c.token_ids.clone(),
                });
            } else {
                // Subsequent use: need to fork from parent
                forks_needed.push(ForkRequest {
                    parent_seq_id: c.parent_seq_id,
                    token_id: c.token_id,
                    cumulative_log_prob: c.cumulative_log_prob,
                    token_ids: c.token_ids.clone(),
                });
            }
        }

        BeamStepResult {
            token_assignments,
            forks_needed,
            seqs_to_drop,
            newly_finished,
            is_complete: self.is_complete(),
        }
    }

    /// Update beam state after the engine has executed forks and token appends.
    ///
    /// `new_beams` maps sequence IDs to their updated beam state.
    pub fn update_beams(&mut self, new_beams: Vec<Beam>) {
        self.beams = new_beams;
        // Recompute scores
        for beam in &mut self.beams {
            beam.score = beam.compute_score(self.length_penalty);
        }
    }

    /// Get the top-n finished beams, sorted by score (best first).
    pub fn best_finished(&self, n: usize) -> Vec<&Beam> {
        let mut sorted: Vec<&Beam> = self.finished_beams.iter().collect();
        sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(n);
        sorted
    }

    /// Get all active sequence IDs (for cleanup on cancellation).
    pub fn active_seq_ids(&self) -> Vec<SeqId> {
        self.beams.iter().map(|b| b.sequence_id).collect()
    }
}

/// A candidate during beam expansion (before pruning).
#[derive(Debug)]
struct BeamCandidate {
    parent_seq_id: SeqId,
    token_id: u32,
    token_ids: Vec<u32>,
    cumulative_log_prob: f64,
    is_finished: bool,
    #[allow(dead_code)]
    score: f64,
}

// This is defined at the struct level but the score field needs to be set during expansion
impl BeamCandidate {
    #[allow(dead_code)]
    fn new() -> Self {
        Self {
            parent_seq_id: 0,
            token_id: 0,
            token_ids: Vec::new(),
            cumulative_log_prob: 0.0,
            is_finished: false,
            score: 0.0,
        }
    }
}

/// Result of a beam search expansion step.
#[derive(Debug)]
pub struct BeamStepResult {
    /// Tokens to append to existing sequences (reusing the parent).
    pub token_assignments: Vec<BeamTokenAssignment>,
    /// Fork requests for candidates that share a parent with another candidate.
    pub forks_needed: Vec<ForkRequest>,
    /// Sequence IDs to drop (no longer in any beam).
    pub seqs_to_drop: Vec<SeqId>,
    /// Beams that finished this step.
    pub newly_finished: Vec<Beam>,
    /// Whether beam search is now complete.
    pub is_complete: bool,
}

/// Token assignment for an existing beam sequence.
#[derive(Debug, Clone)]
pub struct BeamTokenAssignment {
    /// The sequence to append to.
    pub seq_id: SeqId,
    /// Token ID to append.
    pub token_id: u32,
    /// Updated cumulative log probability.
    pub cumulative_log_prob: f64,
    /// Full token history (for beam state update).
    pub token_ids: Vec<u32>,
}

/// Request to fork a sequence for a beam that diverges from its parent.
#[derive(Debug, Clone)]
pub struct ForkRequest {
    /// Parent sequence to fork from.
    pub parent_seq_id: SeqId,
    /// Token ID to append to the forked sequence.
    pub token_id: u32,
    /// Updated cumulative log probability.
    pub cumulative_log_prob: f64,
    /// Full token history.
    pub token_ids: Vec<u32>,
}

/// Extract top-k log probabilities from a logits vector.
///
/// Returns `(token_id, log_prob)` pairs sorted by log_prob descending.
pub fn top_k_logprobs(logits: &[f32], k: usize) -> Vec<(u32, f64)> {
    // Compute log softmax
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp: f32 = logits.iter().map(|&l| (l - max).exp()).sum::<f32>().ln() + max;

    let mut indexed: Vec<(u32, f64)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u32, (l - log_sum_exp) as f64))
        .collect();

    // Partial sort: find top-k without fully sorting
    let k = k.min(indexed.len());
    indexed.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    indexed.truncate(k);
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beam_new() {
        let beam = Beam::new(42);
        assert_eq!(beam.sequence_id, 42);
        assert!(beam.token_ids.is_empty());
        assert_eq!(beam.cumulative_log_prob, 0.0);
        assert_eq!(beam.score, 0.0);
    }

    #[test]
    fn test_beam_compute_score_no_penalty() {
        let mut beam = Beam::new(1);
        beam.token_ids = vec![10, 20, 30];
        beam.cumulative_log_prob = -3.0;

        // length_penalty = 0.0 means no normalization
        let score = beam.compute_score(0.0);
        assert!((score - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn test_beam_compute_score_default_penalty() {
        let mut beam = Beam::new(1);
        beam.token_ids = vec![10, 20, 30]; // length = 3
        beam.cumulative_log_prob = -6.0;

        // length_penalty = 1.0 means divide by length
        let score = beam.compute_score(1.0);
        assert!((score - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_beam_compute_score_high_penalty() {
        let mut beam = Beam::new(1);
        beam.token_ids = vec![10, 20]; // length = 2
        beam.cumulative_log_prob = -4.0;

        // length_penalty = 2.0 means divide by length^2 = 4
        let score = beam.compute_score(2.0);
        assert!((score - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_beam_compute_score_empty_tokens() {
        let beam = Beam::new(1);
        // Empty tokens, length.max(1) = 1
        let score = beam.compute_score(1.0);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_beam_search_state_new() {
        let state = BeamSearchState::new(100, 4, 2, 1.0, false);
        assert_eq!(state.beams.len(), 1);
        assert_eq!(state.beams[0].sequence_id, 100);
        assert_eq!(state.beam_width, 4);
        assert_eq!(state.n_best, 2);
        assert_eq!(state.length_penalty, 1.0);
        assert!(!state.early_stopping);
        assert!(state.finished_beams.is_empty());
    }

    #[test]
    fn test_beam_search_is_complete_no_beams() {
        let mut state = BeamSearchState::new(1, 4, 2, 1.0, false);
        state.beams.clear();
        assert!(state.is_complete());
    }

    #[test]
    fn test_beam_search_is_complete_insufficient_finished() {
        let state = BeamSearchState::new(1, 4, 2, 1.0, true);
        assert!(!state.is_complete()); // 0 finished, need 2
    }

    #[test]
    fn test_beam_search_is_complete_early_stopping() {
        let mut state = BeamSearchState::new(1, 4, 2, 1.0, true);

        // Add 2 finished beams with decent scores
        state.finished_beams.push(Beam {
            token_ids: vec![1, 2],
            cumulative_log_prob: -1.0,
            score: -0.5,
            sequence_id: 10,
        });
        state.finished_beams.push(Beam {
            token_ids: vec![3, 4],
            cumulative_log_prob: -1.5,
            score: -0.75,
            sequence_id: 11,
        });

        // Active beam with worse potential
        state.beams[0].cumulative_log_prob = -10.0;
        state.beams[0].token_ids = vec![5, 6, 7];

        assert!(state.is_complete());
    }

    #[test]
    fn test_beam_search_no_early_stopping() {
        let mut state = BeamSearchState::new(1, 4, 2, 1.0, false);

        // Even with enough finished beams, don't stop without early_stopping
        state.finished_beams.push(Beam {
            token_ids: vec![1, 2],
            cumulative_log_prob: -1.0,
            score: -0.5,
            sequence_id: 10,
        });
        state.finished_beams.push(Beam {
            token_ids: vec![3, 4],
            cumulative_log_prob: -1.5,
            score: -0.75,
            sequence_id: 11,
        });

        assert!(!state.is_complete());
    }

    #[test]
    fn test_expand_and_prune_basic() {
        let mut state = BeamSearchState::new(1, 2, 1, 1.0, false);

        let mut beam_logprobs = HashMap::new();
        beam_logprobs.insert(1, vec![
            (10, -0.5_f64),
            (20, -1.0),
            (30, -2.0),
        ]);

        let result = state.expand_and_prune(&beam_logprobs, &[], 999);

        // Should keep top 2 (beam_width=2)
        assert_eq!(result.token_assignments.len() + result.forks_needed.len(), 2);
        assert!(!result.is_complete);
    }

    #[test]
    fn test_expand_and_prune_with_eos() {
        let mut state = BeamSearchState::new(1, 2, 1, 1.0, false);

        let mut beam_logprobs = HashMap::new();
        // EOS token (999) is in the candidates
        beam_logprobs.insert(1, vec![
            (999, -0.1_f64),  // EOS with high probability
            (10, -0.5),
            (20, -1.0),
        ]);

        let result = state.expand_and_prune(&beam_logprobs, &[], 999);

        // EOS candidate should be in newly_finished
        assert_eq!(result.newly_finished.len(), 1);
        assert_eq!(result.newly_finished[0].token_ids, vec![999]);
    }

    #[test]
    fn test_expand_and_prune_with_stop_tokens() {
        let mut state = BeamSearchState::new(1, 2, 1, 1.0, false);

        let mut beam_logprobs = HashMap::new();
        beam_logprobs.insert(1, vec![
            (50, -0.1_f64),  // stop token
            (10, -0.5),
            (20, -1.0),
        ]);

        let result = state.expand_and_prune(&beam_logprobs, &[50], 999);
        assert_eq!(result.newly_finished.len(), 1);
    }

    #[test]
    fn test_expand_and_prune_multiple_beams() {
        let mut state = BeamSearchState::new(1, 2, 1, 0.0, false);

        // Start with 2 beams
        state.beams = vec![
            Beam {
                token_ids: vec![10],
                cumulative_log_prob: -1.0,
                score: -1.0,
                sequence_id: 1,
            },
            Beam {
                token_ids: vec![20],
                cumulative_log_prob: -2.0,
                score: -2.0,
                sequence_id: 2,
            },
        ];

        let mut beam_logprobs = HashMap::new();
        beam_logprobs.insert(1, vec![
            (11, -0.5_f64),
            (12, -1.0),
        ]);
        beam_logprobs.insert(2, vec![
            (21, -0.3_f64),
            (22, -0.8),
        ]);

        let result = state.expand_and_prune(&beam_logprobs, &[], 999);

        // Total candidates: 4, keep best 2
        let total_kept = result.token_assignments.len() + result.forks_needed.len();
        assert_eq!(total_kept, 2);
    }

    #[test]
    fn test_expand_and_prune_fork_needed() {
        let mut state = BeamSearchState::new(1, 2, 1, 0.0, false);

        // Single beam with 2 top candidates that are both better than any alternative
        let mut beam_logprobs = HashMap::new();
        beam_logprobs.insert(1, vec![
            (10, -0.1_f64),
            (20, -0.2),
        ]);

        let result = state.expand_and_prune(&beam_logprobs, &[], 999);

        // First candidate reuses parent, second needs fork
        assert_eq!(result.token_assignments.len(), 1);
        assert_eq!(result.forks_needed.len(), 1);
    }

    #[test]
    fn test_best_finished() {
        let mut state = BeamSearchState::new(1, 4, 2, 1.0, false);

        state.finished_beams = vec![
            Beam {
                token_ids: vec![1],
                cumulative_log_prob: -3.0,
                score: -3.0,
                sequence_id: 10,
            },
            Beam {
                token_ids: vec![2],
                cumulative_log_prob: -1.0,
                score: -1.0,
                sequence_id: 11,
            },
            Beam {
                token_ids: vec![3],
                cumulative_log_prob: -2.0,
                score: -2.0,
                sequence_id: 12,
            },
        ];

        let best = state.best_finished(2);
        assert_eq!(best.len(), 2);
        assert_eq!(best[0].sequence_id, 11); // score -1.0 (best)
        assert_eq!(best[1].sequence_id, 12); // score -2.0
    }

    #[test]
    fn test_update_beams() {
        let mut state = BeamSearchState::new(1, 2, 1, 1.0, false);

        let new_beams = vec![
            Beam {
                token_ids: vec![10, 20],
                cumulative_log_prob: -2.0,
                score: 0.0, // Will be recomputed
                sequence_id: 5,
            },
            Beam {
                token_ids: vec![30],
                cumulative_log_prob: -1.0,
                score: 0.0,
                sequence_id: 6,
            },
        ];

        state.update_beams(new_beams);
        assert_eq!(state.beams.len(), 2);
        // Scores should be recomputed
        assert!((state.beams[0].score - (-1.0)).abs() < 1e-10); // -2.0 / 2^1.0
        assert!((state.beams[1].score - (-1.0)).abs() < 1e-10); // -1.0 / 1^1.0
    }

    #[test]
    fn test_active_seq_ids() {
        let mut state = BeamSearchState::new(1, 3, 1, 1.0, false);
        state.beams = vec![
            Beam::new(10),
            Beam::new(20),
            Beam::new(30),
        ];

        let ids = state.active_seq_ids();
        assert_eq!(ids, vec![10, 20, 30]);
    }

    #[test]
    fn test_top_k_logprobs() {
        let logits = vec![1.0f32, 5.0, 3.0, 2.0, 4.0];
        let top3 = top_k_logprobs(&logits, 3);

        assert_eq!(top3.len(), 3);
        // Should be sorted by logprob descending
        assert_eq!(top3[0].0, 1); // logit 5.0 -> highest logprob
        assert_eq!(top3[1].0, 4); // logit 4.0
        assert_eq!(top3[2].0, 2); // logit 3.0

        // Log probs should be negative (or zero for degenerate cases)
        assert!(top3[0].1 <= 0.0);
        assert!(top3[0].1 >= top3[1].1);
        assert!(top3[1].1 >= top3[2].1);
    }

    #[test]
    fn test_top_k_logprobs_k_larger_than_vocab() {
        let logits = vec![1.0f32, 2.0];
        let top = top_k_logprobs(&logits, 10);
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn test_top_k_logprobs_single() {
        let logits = vec![5.0f32];
        let top = top_k_logprobs(&logits, 1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].0, 0);
        // Single element log softmax = 0.0
        assert!((top[0].1 - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_length_penalty_favors_shorter() {
        // With length_penalty < 1.0, shorter sequences get relatively better scores
        let mut short_beam = Beam::new(1);
        short_beam.token_ids = vec![1];
        short_beam.cumulative_log_prob = -1.0;

        let mut long_beam = Beam::new(2);
        long_beam.token_ids = vec![1, 2, 3, 4];
        long_beam.cumulative_log_prob = -2.0;

        let lp = 0.5;
        let short_score = short_beam.compute_score(lp);
        let long_score = long_beam.compute_score(lp);

        // short: -1.0 / 1^0.5 = -1.0
        // long: -2.0 / 4^0.5 = -2.0 / 2.0 = -1.0
        // They should be equal in this case
        assert!((short_score - long_score).abs() < 1e-10);
    }

    #[test]
    fn test_seqs_to_drop() {
        let mut state = BeamSearchState::new(1, 1, 1, 0.0, false);
        state.beams = vec![
            Beam {
                token_ids: vec![10],
                cumulative_log_prob: -1.0,
                score: -1.0,
                sequence_id: 1,
            },
            Beam {
                token_ids: vec![20],
                cumulative_log_prob: -5.0,
                score: -5.0,
                sequence_id: 2,
            },
        ];

        let mut beam_logprobs = HashMap::new();
        beam_logprobs.insert(1, vec![(11, -0.5_f64)]);
        beam_logprobs.insert(2, vec![(21, -0.3_f64)]);

        let result = state.expand_and_prune(&beam_logprobs, &[], 999);

        // beam_width=1, so one beam gets dropped
        assert_eq!(result.seqs_to_drop.len(), 1);
    }
}
