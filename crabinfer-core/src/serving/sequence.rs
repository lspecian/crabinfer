//! Sequence: the unit of work in the serving system.
//!
//! Each sequence represents a single request being processed through the
//! inference engine. It tracks tokens, KV cache blocks, generation state,
//! and sampling parameters.

use super::block::BlockHash;
use super::kv_cache::SequenceBlocks;
use std::collections::VecDeque;

/// Unique identifier for a sequence.
pub type SeqId = u64;

/// Lifecycle state of a sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    /// Waiting in the queue, not yet scheduled.
    Waiting,
    /// Actively running (has KV cache blocks allocated).
    Running,
    /// Generation complete (EOS, max tokens, or stop string).
    Finished(FinishReason),
}

/// Why a sequence finished generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Hit the EOS token.
    EndOfSequence,
    /// Reached max_tokens limit.
    MaxTokens,
    /// Hit a stop string/token.
    Stop,
    /// Cancelled by the client (e.g., SSE disconnect).
    Cancelled,
    /// Preempted by the scheduler due to memory pressure.
    Preempted,
}

/// Sampling parameters for token generation.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Temperature for softmax scaling. 0.0 = greedy.
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold.
    pub top_p: f32,
    /// Top-k sampling. 0 = disabled.
    pub top_k: usize,
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Stop token IDs (generation halts when any is produced).
    pub stop_token_ids: Vec<u32>,
    /// Repetition penalty. 1.0 = no penalty.
    pub repetition_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 0,
            max_tokens: 2048,
            stop_token_ids: Vec::new(),
            repetition_penalty: 1.0,
        }
    }
}

/// A single inference sequence (request).
pub struct Sequence {
    /// Unique sequence ID.
    pub id: SeqId,

    /// Current status.
    pub status: SequenceStatus,

    /// Input (prompt) token IDs.
    pub prompt_tokens: Vec<u32>,

    /// Generated output token IDs (appended one at a time during decode).
    pub output_tokens: Vec<u32>,

    /// KV cache block tracking for this sequence.
    pub blocks: SequenceBlocks,

    /// Block content hashes for prefix caching.
    /// Computed from the prompt tokens at scheduling time.
    pub block_hashes: Vec<BlockHash>,

    /// Number of prompt tokens that were prefix-cached (already computed).
    pub num_prefix_cached_tokens: usize,

    /// Sampling parameters.
    pub sampling_params: SamplingParams,

    /// Arrival order (for FCFS scheduling). Lower = earlier.
    pub arrival_order: u64,

    /// Tokens waiting to be sent to the client.
    pub pending_output: VecDeque<u32>,
}

impl Sequence {
    /// Create a new sequence in the Waiting state.
    pub fn new(
        id: SeqId,
        prompt_tokens: Vec<u32>,
        sampling_params: SamplingParams,
        arrival_order: u64,
    ) -> Self {
        Self {
            id,
            status: SequenceStatus::Waiting,
            prompt_tokens,
            output_tokens: Vec::new(),
            blocks: SequenceBlocks::new(),
            block_hashes: Vec::new(),
            num_prefix_cached_tokens: 0,
            sampling_params,
            arrival_order,
            pending_output: VecDeque::new(),
        }
    }

    /// Total number of tokens (prompt + generated).
    pub fn num_tokens(&self) -> usize {
        self.prompt_tokens.len() + self.output_tokens.len()
    }

    /// Number of tokens that have been computed (KV cached).
    pub fn num_computed_tokens(&self) -> usize {
        self.blocks.num_computed_tokens
    }

    /// Number of tokens that still need computation.
    ///
    /// For a new sequence, this is the full prompt length.
    /// For a running sequence that just did prefill, this is 1 (next decode token).
    /// For a prefix-cached sequence, this is prompt_len - cached_tokens.
    pub fn num_uncomputed_tokens(&self) -> usize {
        self.num_tokens().saturating_sub(self.num_computed_tokens())
    }

    /// Whether this sequence is in its first scheduling step (prefill).
    pub fn is_prefill(&self) -> bool {
        self.num_computed_tokens() < self.prompt_tokens.len()
    }

    /// Whether this sequence has finished generating.
    pub fn is_finished(&self) -> bool {
        matches!(self.status, SequenceStatus::Finished(_))
    }

    /// Whether this sequence is waiting to be scheduled.
    pub fn is_waiting(&self) -> bool {
        self.status == SequenceStatus::Waiting
    }

    /// Whether this sequence is currently running.
    pub fn is_running(&self) -> bool {
        self.status == SequenceStatus::Running
    }

    /// Append a generated token.
    pub fn append_token(&mut self, token: u32) {
        self.output_tokens.push(token);
        self.pending_output.push_back(token);
    }

    /// Check if the generated token should stop generation.
    pub fn should_stop(&self, token: u32) -> bool {
        self.sampling_params.stop_token_ids.contains(&token)
    }

    /// Whether we've hit the max output token limit.
    pub fn reached_max_tokens(&self) -> bool {
        self.output_tokens.len() >= self.sampling_params.max_tokens
    }

    /// All token IDs in order (prompt + output).
    pub fn all_tokens(&self) -> Vec<u32> {
        let mut tokens = self.prompt_tokens.clone();
        tokens.extend_from_slice(&self.output_tokens);
        tokens
    }

    /// Get the tokens for the next forward pass (owned).
    pub fn get_input_tokens_owned(&self, num_scheduled_tokens: usize) -> Vec<u32> {
        let start = self.num_computed_tokens();
        let end = start + num_scheduled_tokens;
        let mut tokens = Vec::with_capacity(end - start);
        let prompt_len = self.prompt_tokens.len();

        for i in start..end {
            if i < prompt_len {
                tokens.push(self.prompt_tokens[i]);
            } else {
                tokens.push(self.output_tokens[i - prompt_len]);
            }
        }
        tokens
    }
}

impl std::fmt::Debug for Sequence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sequence")
            .field("id", &self.id)
            .field("status", &self.status)
            .field("prompt_tokens", &self.prompt_tokens.len())
            .field("output_tokens", &self.output_tokens.len())
            .field("computed", &self.num_computed_tokens())
            .field("blocks", &self.blocks.num_blocks())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_seq(id: SeqId, prompt_len: usize) -> Sequence {
        let tokens: Vec<u32> = (0..prompt_len as u32).collect();
        Sequence::new(id, tokens, SamplingParams::default(), id)
    }

    #[test]
    fn test_new_sequence() {
        let seq = make_seq(1, 32);
        assert_eq!(seq.id, 1);
        assert!(seq.is_waiting());
        assert!(!seq.is_running());
        assert!(!seq.is_finished());
        assert_eq!(seq.num_tokens(), 32);
        assert_eq!(seq.num_computed_tokens(), 0);
        assert_eq!(seq.num_uncomputed_tokens(), 32);
        assert!(seq.is_prefill());
    }

    #[test]
    fn test_sequence_after_prefill() {
        let mut seq = make_seq(1, 32);
        seq.status = SequenceStatus::Running;
        seq.blocks.num_computed_tokens = 32;

        assert!(seq.is_running());
        assert!(!seq.is_prefill());
        assert_eq!(seq.num_uncomputed_tokens(), 0);
    }

    #[test]
    fn test_sequence_decode() {
        let mut seq = make_seq(1, 10);
        seq.status = SequenceStatus::Running;
        seq.blocks.num_computed_tokens = 10;

        // Generate 3 tokens
        seq.append_token(100);
        seq.append_token(101);
        seq.append_token(102);

        assert_eq!(seq.num_tokens(), 13);
        assert_eq!(seq.output_tokens.len(), 3);
        assert_eq!(seq.pending_output.len(), 3);

        // After updating computed tokens
        seq.blocks.num_computed_tokens = 11;
        assert_eq!(seq.num_uncomputed_tokens(), 2);
    }

    #[test]
    fn test_sequence_finish_conditions() {
        let mut seq = make_seq(1, 5);
        seq.sampling_params.stop_token_ids = vec![999];
        seq.sampling_params.max_tokens = 3;

        assert!(!seq.should_stop(100));
        assert!(seq.should_stop(999));

        seq.output_tokens = vec![1, 2, 3];
        assert!(seq.reached_max_tokens());

        seq.output_tokens = vec![1, 2];
        assert!(!seq.reached_max_tokens());
    }

    #[test]
    fn test_get_input_tokens_owned_prefill() {
        let seq = make_seq(1, 10);
        // Full prefill: all 10 prompt tokens
        let tokens = seq.get_input_tokens_owned(10);
        assert_eq!(tokens, (0..10u32).collect::<Vec<_>>());

        // Chunked prefill: first 5 tokens
        let tokens = seq.get_input_tokens_owned(5);
        assert_eq!(tokens, (0..5u32).collect::<Vec<_>>());
    }

    #[test]
    fn test_get_input_tokens_owned_decode() {
        let mut seq = make_seq(1, 5);
        seq.blocks.num_computed_tokens = 5;
        seq.output_tokens = vec![100, 101];
        // 5 prompt + 2 output = 7 total, 5 computed, tokens at positions 5 and 6
        // If we schedule 1 token at position 5:
        seq.blocks.num_computed_tokens = 6;
        let tokens = seq.get_input_tokens_owned(1);
        assert_eq!(tokens, vec![101]);
    }

    #[test]
    fn test_all_tokens() {
        let mut seq = make_seq(1, 3);
        seq.output_tokens = vec![10, 11];
        assert_eq!(seq.all_tokens(), vec![0, 1, 2, 10, 11]);
    }

    #[test]
    fn test_sequence_status_transitions() {
        let mut seq = make_seq(1, 10);
        assert_eq!(seq.status, SequenceStatus::Waiting);

        seq.status = SequenceStatus::Running;
        assert!(seq.is_running());

        seq.status = SequenceStatus::Finished(FinishReason::EndOfSequence);
        assert!(seq.is_finished());
        assert!(!seq.is_running());
    }

    #[test]
    fn test_sampling_params_default() {
        let params = SamplingParams::default();
        assert_eq!(params.temperature, 0.7);
        assert_eq!(params.top_p, 0.9);
        assert_eq!(params.top_k, 0);
        assert_eq!(params.max_tokens, 2048);
        assert!(params.stop_token_ids.is_empty());
        assert_eq!(params.repetition_penalty, 1.0);
    }
}
