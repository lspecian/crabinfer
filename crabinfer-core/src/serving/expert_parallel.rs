//! Expert parallelism (EP) for Mixture-of-Experts (MoE) distribution across GPUs.
//!
//! Expert parallelism assigns different experts to different GPUs, enabling
//! MoE models with hundreds of experts (e.g., DeepSeek-V3 with 256 routed
//! experts) to be served across a multi-GPU cluster.
//!
//! Unlike tensor parallelism (which shards individual weight matrices),
//! expert parallelism keeps each expert's weights intact on a single GPU
//! and routes tokens to the GPU hosting their assigned expert.
//!
//! # Components
//!
//! - [`ExpertParallelConfig`]: Configuration for EP (num groups, placement strategy).
//! - [`ExpertPlacement`]: Maps expert IDs to GPU ranks.
//! - [`TokenRouter`]: Routes tokens to experts across GPUs with capacity constraints.
//! - [`AllToAllDispatcher`]: Dispatches tokens to remote experts and collects results.
//! - [`LoadBalancer`]: Tracks expert utilization and computes auxiliary balance loss.
//! - [`ExpertParallelGroup`]: End-to-end EP orchestration combining all components.
//!
//! # CPU fallback
//!
//! On non-CUDA builds, the AllToAll communication is simulated via local
//! memory shuffles (memcpy), enabling the same code path for development
//! and testing on CPU-only machines.

use std::collections::HashMap;

// ─── Configuration ────────────────────────────────────────────────────────

/// Strategy for placing experts across GPU ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlacementStrategy {
    /// Distribute experts uniformly across ranks (round-robin).
    /// Expert `i` goes to rank `i % num_gpus`.
    Uniform,

    /// Distribute experts weighted by GPU memory capacity.
    /// GPUs with more memory get more experts. Falls back to uniform
    /// if no capacity info is provided.
    CapacityAware,
}

/// Configuration for expert parallelism.
#[derive(Debug, Clone)]
pub struct ExpertParallelConfig {
    /// Number of expert parallel groups (typically equals number of GPUs
    /// dedicated to expert parallelism).
    pub num_expert_groups: usize,

    /// Strategy for assigning experts to GPU ranks.
    pub placement_strategy: PlacementStrategy,

    /// Capacity factor controlling the maximum number of tokens each expert
    /// can process per batch. The capacity is:
    ///   `capacity = capacity_factor * (num_tokens * top_k / num_experts)`
    ///
    /// - `1.0`: Strict capacity — expects perfectly balanced routing.
    /// - `> 1.0`: Allows headroom for imbalanced routing (fewer drops).
    /// - `< 1.0`: Aggressive dropping (rarely useful).
    ///
    /// Default: `1.0`.
    pub capacity_factor: f32,
}

impl Default for ExpertParallelConfig {
    fn default() -> Self {
        Self {
            num_expert_groups: 1,
            placement_strategy: PlacementStrategy::Uniform,
            capacity_factor: 1.0,
        }
    }
}

impl ExpertParallelConfig {
    /// Create a config for the given number of GPU groups with uniform placement.
    pub fn uniform(num_expert_groups: usize) -> Self {
        Self {
            num_expert_groups,
            placement_strategy: PlacementStrategy::Uniform,
            capacity_factor: 1.0,
        }
    }

    /// Create a config with capacity-aware placement.
    pub fn capacity_aware(num_expert_groups: usize, capacity_factor: f32) -> Self {
        Self {
            num_expert_groups,
            placement_strategy: PlacementStrategy::CapacityAware,
            capacity_factor,
        }
    }
}

// ─── Expert placement ─────────────────────────────────────────────────────

/// Maps expert IDs to GPU ranks.
///
/// This determines which GPU holds each expert's weights and is responsible
/// for computing that expert's output.
#[derive(Debug, Clone)]
pub struct ExpertPlacement {
    /// expert_id -> gpu_rank mapping.
    expert_to_rank: Vec<usize>,
    /// Total number of GPU ranks.
    num_gpus: usize,
    /// Total number of experts.
    num_experts: usize,
}

impl ExpertPlacement {
    /// Create a uniform placement: expert `i` goes to rank `i % num_gpus`.
    ///
    /// This is the simplest and most common strategy, distributing experts
    /// as evenly as possible across all GPUs.
    pub fn uniform(num_experts: usize, num_gpus: usize) -> Self {
        assert!(num_gpus > 0, "num_gpus must be > 0");
        let expert_to_rank: Vec<usize> = (0..num_experts).map(|i| i % num_gpus).collect();
        Self {
            expert_to_rank,
            num_gpus,
            num_experts,
        }
    }

    /// Create a capacity-aware placement given per-GPU capacity weights.
    ///
    /// GPUs with higher capacity weights receive proportionally more experts.
    /// `capacities` must have length `num_gpus` with positive values.
    pub fn capacity_aware(num_experts: usize, capacities: &[f32]) -> Self {
        let num_gpus = capacities.len();
        assert!(num_gpus > 0, "capacities must be non-empty");

        let total_cap: f32 = capacities.iter().sum();
        assert!(total_cap > 0.0, "total capacity must be positive");

        // Calculate how many experts each GPU should get (proportional to capacity).
        let mut experts_per_gpu: Vec<usize> = capacities
            .iter()
            .map(|c| ((c / total_cap) * num_experts as f32).floor() as usize)
            .collect();

        // Distribute remaining experts to GPUs with highest capacity.
        let assigned: usize = experts_per_gpu.iter().sum();
        let remaining = num_experts.saturating_sub(assigned);
        // Sort GPU indices by capacity descending to assign remainder.
        let mut gpu_order: Vec<usize> = (0..num_gpus).collect();
        gpu_order.sort_by(|&a, &b| capacities[b].partial_cmp(&capacities[a]).unwrap());
        for i in 0..remaining {
            experts_per_gpu[gpu_order[i % num_gpus]] += 1;
        }

        // Build expert -> rank mapping.
        let mut expert_to_rank = vec![0usize; num_experts];
        let mut expert_idx = 0;
        for (rank, &count) in experts_per_gpu.iter().enumerate() {
            for _ in 0..count {
                if expert_idx < num_experts {
                    expert_to_rank[expert_idx] = rank;
                    expert_idx += 1;
                }
            }
        }

        Self {
            expert_to_rank,
            num_gpus,
            num_experts,
        }
    }

    /// Get the GPU rank that hosts the given expert.
    pub fn get_rank(&self, expert_id: usize) -> usize {
        self.expert_to_rank[expert_id]
    }

    /// Get the list of expert IDs hosted on the given rank.
    pub fn local_experts(&self, rank: usize) -> Vec<usize> {
        self.expert_to_rank
            .iter()
            .enumerate()
            .filter(|&(_, &r)| r == rank)
            .map(|(expert_id, _)| expert_id)
            .collect()
    }

    /// Check whether an expert is local to the given rank.
    pub fn is_local(&self, expert_id: usize, rank: usize) -> bool {
        self.expert_to_rank[expert_id] == rank
    }

    /// Total number of experts.
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// Total number of GPU ranks.
    pub fn num_gpus(&self) -> usize {
        self.num_gpus
    }
}

// ─── Token routing ────────────────────────────────────────────────────────

/// A single expert assignment for a token.
#[derive(Debug, Clone, Copy)]
pub struct ExpertAssignment {
    /// Index of the token in the batch.
    pub token_idx: usize,
    /// Expert ID this token is routed to.
    pub expert_id: usize,
    /// Routing weight for this expert (after normalization).
    pub weight: f32,
}

/// Result of routing tokens to experts.
#[derive(Debug, Clone)]
pub struct RoutingResult {
    /// Per-token expert assignments (may have multiple entries per token for top-k > 1).
    pub assignments: Vec<ExpertAssignment>,
    /// Number of tokens that were dropped due to expert capacity overflow.
    pub dropped_count: usize,
    /// Total number of tokens in the batch.
    pub num_tokens: usize,
    /// Number of experts.
    pub num_experts: usize,
    /// Top-k value used.
    pub top_k: usize,
    /// Per-expert token counts (how many tokens were routed to each expert).
    pub expert_counts: Vec<usize>,
}

/// Routes tokens to experts based on gating logits.
///
/// Applies top-k selection, capacity constraints, and weight normalization
/// to determine which experts process each token.
pub struct TokenRouter {
    /// Maximum tokens per expert = capacity_factor * (num_tokens * top_k / num_experts).
    capacity_factor: f32,
}

impl TokenRouter {
    /// Create a new token router with the given capacity factor.
    pub fn new(capacity_factor: f32) -> Self {
        Self { capacity_factor }
    }

    /// Route tokens to experts based on gate logits.
    ///
    /// # Arguments
    /// - `gate_logits`: `[num_tokens, num_experts]` — raw gating logits.
    /// - `top_k`: Number of experts per token.
    ///
    /// # Returns
    /// A `RoutingResult` containing expert assignments, dropped token count, and stats.
    pub fn route(&self, gate_logits: &[Vec<f32>], top_k: usize) -> RoutingResult {
        let num_tokens = gate_logits.len();
        if num_tokens == 0 {
            return RoutingResult {
                assignments: Vec::new(),
                dropped_count: 0,
                num_tokens: 0,
                num_experts: 0,
                top_k,
                expert_counts: Vec::new(),
            };
        }

        let num_experts = gate_logits[0].len();

        // Compute softmax probabilities for each token.
        let probs: Vec<Vec<f32>> = gate_logits.iter().map(|logits| softmax(logits)).collect();

        // Compute per-expert capacity.
        let capacity = if num_experts > 0 {
            let raw = self.capacity_factor * (num_tokens * top_k) as f32 / num_experts as f32;
            raw.ceil() as usize
        } else {
            0
        };

        // Track per-expert token counts for capacity enforcement.
        let mut expert_counts = vec![0usize; num_experts];
        let mut assignments = Vec::with_capacity(num_tokens * top_k);
        let mut dropped_count = 0usize;

        for token_idx in 0..num_tokens {
            let token_probs = &probs[token_idx];

            // Find top-k experts for this token.
            let mut indexed: Vec<(usize, f32)> =
                token_probs.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let topk: Vec<(usize, f32)> = indexed.into_iter().take(top_k).collect();

            // Normalize top-k weights so they sum to 1.
            let weight_sum: f32 = topk.iter().map(|(_, w)| w).sum();
            let norm = if weight_sum > 0.0 {
                1.0 / weight_sum
            } else {
                0.0
            };

            for (expert_id, weight) in topk {
                if expert_counts[expert_id] < capacity {
                    expert_counts[expert_id] += 1;
                    assignments.push(ExpertAssignment {
                        token_idx,
                        expert_id,
                        weight: weight * norm,
                    });
                } else {
                    dropped_count += 1;
                }
            }
        }

        RoutingResult {
            assignments,
            dropped_count,
            num_tokens,
            num_experts,
            top_k,
            expert_counts,
        }
    }
}

/// Compute softmax over a slice of logits.
fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        vec![0.0; logits.len()]
    } else {
        exps.iter().map(|&e| e / sum).collect()
    }
}

// ─── AllToAll dispatcher ──────────────────────────────────────────────────

/// Tokens dispatched to a specific GPU rank, ready for expert computation.
#[derive(Debug, Clone)]
pub struct DispatchedTokens {
    /// For each destination rank: list of (token_index, expert_id, weight, hidden_state_index).
    /// `hidden_state_index` is the position in the original token tensor.
    pub per_rank: Vec<Vec<DispatchEntry>>,
    /// Total number of GPU ranks.
    pub num_ranks: usize,
}

/// A single dispatched token entry.
#[derive(Debug, Clone, Copy)]
pub struct DispatchEntry {
    /// Index of the token in the original batch.
    pub token_idx: usize,
    /// Expert ID to process this token.
    pub expert_id: usize,
    /// Routing weight for weighted combination of results.
    pub weight: f32,
}

/// Dispatches tokens to remote expert GPUs and collects results.
///
/// On CUDA builds with world_size > 1, this uses NCCL AllToAll communication.
/// On CPU or single-GPU, this uses local memory shuffles that simulate the
/// same data movement pattern.
pub struct AllToAllDispatcher {
    placement: ExpertPlacement,
}

impl AllToAllDispatcher {
    /// Create a new dispatcher with the given expert placement.
    pub fn new(placement: ExpertPlacement) -> Self {
        Self { placement }
    }

    /// Dispatch tokens to the appropriate GPU ranks based on routing assignments.
    ///
    /// Groups tokens by their destination rank (determined by expert placement)
    /// and prepares them for transfer.
    pub fn dispatch(&self, routing: &RoutingResult) -> DispatchedTokens {
        let num_ranks = self.placement.num_gpus();
        let mut per_rank: Vec<Vec<DispatchEntry>> = vec![Vec::new(); num_ranks];

        for assignment in &routing.assignments {
            let rank = self.placement.get_rank(assignment.expert_id);
            per_rank[rank].push(DispatchEntry {
                token_idx: assignment.token_idx,
                expert_id: assignment.expert_id,
                weight: assignment.weight,
            });
        }

        DispatchedTokens { per_rank, num_ranks }
    }

    /// Collect expert outputs and combine them into the final output.
    ///
    /// Takes per-expert results (keyed by `(token_idx, expert_id)`) and performs
    /// weighted summation to produce the final output for each token.
    ///
    /// # Arguments
    /// - `expert_outputs`: Map from `(token_idx, expert_id)` to the expert's output vector.
    /// - `routing`: The routing result used to determine weights.
    /// - `num_tokens`: Number of tokens in the batch.
    /// - `hidden_dim`: Dimension of each token's hidden state.
    ///
    /// # Returns
    /// A flat `Vec<f32>` of shape `[num_tokens, hidden_dim]` with the combined outputs.
    pub fn collect(
        &self,
        expert_outputs: &HashMap<(usize, usize), Vec<f32>>,
        routing: &RoutingResult,
        num_tokens: usize,
        hidden_dim: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; num_tokens * hidden_dim];

        for assignment in &routing.assignments {
            if let Some(expert_out) = expert_outputs.get(&(assignment.token_idx, assignment.expert_id))
            {
                let offset = assignment.token_idx * hidden_dim;
                for (j, &val) in expert_out.iter().enumerate() {
                    if offset + j < output.len() {
                        output[offset + j] += val * assignment.weight;
                    }
                }
            }
        }

        output
    }

    /// Get the underlying expert placement.
    pub fn placement(&self) -> &ExpertPlacement {
        &self.placement
    }
}

// ─── Load balancer ────────────────────────────────────────────────────────

/// Tracks expert utilization across batches and computes auxiliary loss
/// for load balancing during training (or monitoring during inference).
///
/// The auxiliary loss penalizes uneven expert utilization, encouraging
/// the gating network to distribute tokens more uniformly.
#[derive(Debug, Clone)]
pub struct LoadBalancer {
    /// Cumulative token counts per expert.
    cumulative_counts: Vec<f64>,
    /// Total tokens routed across all updates.
    total_tokens_routed: f64,
    /// Number of experts.
    num_experts: usize,
    /// Number of update calls.
    num_updates: usize,
}

impl LoadBalancer {
    /// Create a new load balancer for the given number of experts.
    pub fn new(num_experts: usize) -> Self {
        Self {
            cumulative_counts: vec![0.0; num_experts],
            total_tokens_routed: 0.0,
            num_experts,
            num_updates: 0,
        }
    }

    /// Update statistics from a routing result.
    pub fn update(&mut self, routing: &RoutingResult) {
        for (expert_id, &count) in routing.expert_counts.iter().enumerate() {
            if expert_id < self.num_experts {
                self.cumulative_counts[expert_id] += count as f64;
            }
        }
        // Total tokens routed = sum of all expert counts (each token may be
        // counted top_k times, once per assigned expert).
        let batch_total: usize = routing.expert_counts.iter().sum();
        self.total_tokens_routed += batch_total as f64;
        self.num_updates += 1;
    }

    /// Compute the auxiliary load balance loss.
    ///
    /// This is the variance of per-expert utilization fractions. A perfectly
    /// balanced router produces 0 loss; a completely skewed one produces high
    /// loss.
    ///
    /// Formula: `num_experts * sum_i(f_i^2)` where `f_i` is the fraction of
    /// tokens routed to expert `i`. For perfectly uniform routing, each
    /// `f_i = 1/num_experts` and the loss is 1.0. Deviation increases loss.
    /// We subtract 1.0 so that perfect balance yields 0.0.
    pub fn auxiliary_loss(&self) -> f32 {
        if self.total_tokens_routed == 0.0 || self.num_experts == 0 {
            return 0.0;
        }

        let fractions: Vec<f64> = self
            .cumulative_counts
            .iter()
            .map(|&c| c / self.total_tokens_routed)
            .collect();

        let sum_sq: f64 = fractions.iter().map(|f| f * f).sum();
        let loss = (self.num_experts as f64) * sum_sq - 1.0;

        // Clamp to non-negative (floating point rounding may produce tiny negatives).
        loss.max(0.0) as f32
    }

    /// Get per-expert utilization fractions in [0, 1].
    ///
    /// Each value represents the fraction of total routed tokens that went
    /// to that expert. For perfect balance, each value is `1/num_experts`.
    pub fn utilization(&self) -> Vec<f32> {
        if self.total_tokens_routed == 0.0 {
            return vec![0.0; self.num_experts];
        }
        self.cumulative_counts
            .iter()
            .map(|&c| (c / self.total_tokens_routed) as f32)
            .collect()
    }

    /// Reset all accumulated statistics.
    pub fn reset(&mut self) {
        self.cumulative_counts.fill(0.0);
        self.total_tokens_routed = 0.0;
        self.num_updates = 0;
    }

    /// Number of experts being tracked.
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// Number of update calls since last reset.
    pub fn num_updates(&self) -> usize {
        self.num_updates
    }
}

// ─── Expert parallel group ────────────────────────────────────────────────

/// End-to-end expert parallelism orchestration.
///
/// Combines expert placement, token routing, AllToAll dispatch, and load
/// balancing into a single cohesive unit for MoE parallel forward passes.
pub struct ExpertParallelGroup {
    /// Configuration.
    pub config: ExpertParallelConfig,
    /// Expert placement mapping.
    pub placement: ExpertPlacement,
    /// Token router with capacity constraints.
    pub router: TokenRouter,
    /// AllToAll dispatcher for cross-GPU token exchange.
    pub dispatcher: AllToAllDispatcher,
    /// Load balance tracker.
    pub balancer: LoadBalancer,
    /// This GPU's rank.
    pub rank: usize,
}

impl ExpertParallelGroup {
    /// Create a new expert parallel group.
    ///
    /// # Arguments
    /// - `config`: Expert parallelism configuration.
    /// - `num_experts`: Total number of routed experts in the model.
    /// - `rank`: This GPU's rank (0..num_expert_groups-1).
    pub fn new(config: ExpertParallelConfig, num_experts: usize, rank: usize) -> Self {
        let placement = match config.placement_strategy {
            PlacementStrategy::Uniform => {
                ExpertPlacement::uniform(num_experts, config.num_expert_groups)
            }
            PlacementStrategy::CapacityAware => {
                // Fall back to uniform if no capacity info provided at construction.
                // Use `with_capacities` for actual capacity-aware placement.
                ExpertPlacement::uniform(num_experts, config.num_expert_groups)
            }
        };

        let router = TokenRouter::new(config.capacity_factor);
        let dispatcher = AllToAllDispatcher::new(placement.clone());
        let balancer = LoadBalancer::new(num_experts);

        Self {
            config,
            placement,
            router,
            dispatcher,
            balancer,
            rank,
        }
    }

    /// Create a new expert parallel group with capacity-aware placement.
    pub fn with_capacities(
        config: ExpertParallelConfig,
        num_experts: usize,
        rank: usize,
        capacities: &[f32],
    ) -> Self {
        let placement = ExpertPlacement::capacity_aware(num_experts, capacities);
        let router = TokenRouter::new(config.capacity_factor);
        let dispatcher = AllToAllDispatcher::new(placement.clone());
        let balancer = LoadBalancer::new(num_experts);

        Self {
            config,
            placement,
            router,
            dispatcher,
            balancer,
            rank,
        }
    }

    /// Route tokens and dispatch to experts (full pipeline).
    ///
    /// # Arguments
    /// - `gate_logits`: `[num_tokens, num_experts]` — raw gating logits.
    /// - `top_k`: Number of experts per token.
    ///
    /// # Returns
    /// `(RoutingResult, DispatchedTokens)` for downstream expert computation.
    pub fn route_and_dispatch(
        &mut self,
        gate_logits: &[Vec<f32>],
        top_k: usize,
    ) -> (RoutingResult, DispatchedTokens) {
        let routing = self.router.route(gate_logits, top_k);
        self.balancer.update(&routing);
        let dispatched = self.dispatcher.dispatch(&routing);
        (routing, dispatched)
    }

    /// Get experts that are local to this rank.
    pub fn local_experts(&self) -> Vec<usize> {
        self.placement.local_experts(self.rank)
    }

    /// Check if an expert is local to this rank.
    pub fn is_local(&self, expert_id: usize) -> bool {
        self.placement.is_local(expert_id, self.rank)
    }

    /// Get the current auxiliary load balance loss.
    pub fn auxiliary_loss(&self) -> f32 {
        self.balancer.auxiliary_loss()
    }

    /// Get per-expert utilization.
    pub fn utilization(&self) -> Vec<f32> {
        self.balancer.utilization()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Expert placement: uniform distribution ──

    #[test]
    fn test_uniform_placement_4_experts_2_gpus() {
        let p = ExpertPlacement::uniform(4, 2);
        assert_eq!(p.get_rank(0), 0);
        assert_eq!(p.get_rank(1), 1);
        assert_eq!(p.get_rank(2), 0);
        assert_eq!(p.get_rank(3), 1);
    }

    #[test]
    fn test_uniform_placement_8_experts_4_gpus() {
        let p = ExpertPlacement::uniform(8, 4);
        for i in 0..8 {
            assert_eq!(p.get_rank(i), i % 4);
        }
    }

    #[test]
    fn test_uniform_placement_local_experts() {
        let p = ExpertPlacement::uniform(6, 3);
        assert_eq!(p.local_experts(0), vec![0, 3]);
        assert_eq!(p.local_experts(1), vec![1, 4]);
        assert_eq!(p.local_experts(2), vec![2, 5]);
    }

    #[test]
    fn test_uniform_placement_is_local() {
        let p = ExpertPlacement::uniform(4, 2);
        assert!(p.is_local(0, 0));
        assert!(!p.is_local(0, 1));
        assert!(p.is_local(1, 1));
        assert!(!p.is_local(1, 0));
    }

    #[test]
    fn test_uniform_placement_single_gpu_all_local() {
        let p = ExpertPlacement::uniform(8, 1);
        for i in 0..8 {
            assert_eq!(p.get_rank(i), 0);
            assert!(p.is_local(i, 0));
        }
        assert_eq!(p.local_experts(0), vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_uniform_placement_more_gpus_than_experts() {
        let p = ExpertPlacement::uniform(3, 8);
        assert_eq!(p.get_rank(0), 0);
        assert_eq!(p.get_rank(1), 1);
        assert_eq!(p.get_rank(2), 2);
        // GPUs 3-7 have no experts
        assert_eq!(p.local_experts(3), Vec::<usize>::new());
        assert_eq!(p.local_experts(7), Vec::<usize>::new());
    }

    // ── Expert placement: capacity-aware ──

    #[test]
    fn test_capacity_aware_placement() {
        // GPU 0 has 3x the capacity of GPU 1, so should get ~3x experts
        let p = ExpertPlacement::capacity_aware(8, &[3.0, 1.0]);
        let gpu0_count = p.local_experts(0).len();
        let gpu1_count = p.local_experts(1).len();
        assert_eq!(gpu0_count + gpu1_count, 8);
        // GPU 0 should have 6 experts (3/4 * 8 = 6), GPU 1 should have 2
        assert_eq!(gpu0_count, 6);
        assert_eq!(gpu1_count, 2);
    }

    // ── Token routing: correct top-k selection ──

    #[test]
    fn test_routing_top1_selects_highest() {
        let router = TokenRouter::new(10.0); // large capacity to avoid drops
        let gate_logits = vec![
            vec![0.1, 0.9, 0.5], // expert 1 is highest
            vec![0.8, 0.1, 0.3], // expert 0 is highest
        ];
        let result = router.route(&gate_logits, 1);
        assert_eq!(result.assignments.len(), 2);
        assert_eq!(result.assignments[0].expert_id, 1);
        assert_eq!(result.assignments[1].expert_id, 0);
        assert_eq!(result.dropped_count, 0);
    }

    #[test]
    fn test_routing_top2_selects_two_highest() {
        let router = TokenRouter::new(10.0);
        let gate_logits = vec![
            vec![0.1, 0.9, 0.5, 0.2], // experts 1, 2 are top-2
        ];
        let result = router.route(&gate_logits, 2);
        assert_eq!(result.assignments.len(), 2);
        let expert_ids: Vec<usize> = result.assignments.iter().map(|a| a.expert_id).collect();
        assert!(expert_ids.contains(&1));
        assert!(expert_ids.contains(&2));
    }

    #[test]
    fn test_routing_weight_normalization() {
        let router = TokenRouter::new(10.0);
        // Make logits such that softmax gives known probabilities
        let gate_logits = vec![
            vec![10.0, 0.0, 0.0, 0.0], // expert 0 dominates
        ];
        let result = router.route(&gate_logits, 2);
        // Top-2 weights should sum to 1.0 (normalized)
        let weight_sum: f32 = result
            .assignments
            .iter()
            .filter(|a| a.token_idx == 0)
            .map(|a| a.weight)
            .sum();
        assert!((weight_sum - 1.0).abs() < 1e-5, "weight_sum={weight_sum}");
    }

    // ── Token routing: capacity overflow drops ──

    #[test]
    fn test_routing_capacity_drops_tokens() {
        let router = TokenRouter::new(1.0);
        // 4 tokens, 2 experts, top-1
        // capacity = 1.0 * (4 * 1) / 2 = 2 tokens per expert
        // Route all tokens to expert 0 -> only 2 can fit, 2 dropped
        let gate_logits = vec![
            vec![10.0, 0.0],
            vec![10.0, 0.0],
            vec![10.0, 0.0],
            vec![10.0, 0.0],
        ];
        let result = router.route(&gate_logits, 1);
        assert_eq!(result.expert_counts[0], 2);
        assert_eq!(result.dropped_count, 2);
    }

    #[test]
    fn test_routing_capacity_factor_2_accepts_more() {
        let router = TokenRouter::new(2.0);
        // 4 tokens, 2 experts, top-1
        // capacity = 2.0 * (4 * 1) / 2 = 4 tokens per expert
        let gate_logits = vec![
            vec![10.0, 0.0],
            vec![10.0, 0.0],
            vec![10.0, 0.0],
            vec![10.0, 0.0],
        ];
        let result = router.route(&gate_logits, 1);
        assert_eq!(result.expert_counts[0], 4);
        assert_eq!(result.dropped_count, 0);
    }

    #[test]
    fn test_routing_no_drops_when_balanced() {
        let router = TokenRouter::new(1.0);
        // 4 tokens, 4 experts, top-1, each token goes to a different expert
        let gate_logits = vec![
            vec![10.0, 0.0, 0.0, 0.0],
            vec![0.0, 10.0, 0.0, 0.0],
            vec![0.0, 0.0, 10.0, 0.0],
            vec![0.0, 0.0, 0.0, 10.0],
        ];
        let result = router.route(&gate_logits, 1);
        assert_eq!(result.dropped_count, 0);
        for &c in &result.expert_counts {
            assert_eq!(c, 1);
        }
    }

    // ── Token routing: empty and edge cases ──

    #[test]
    fn test_routing_empty_batch() {
        let router = TokenRouter::new(1.0);
        let result = router.route(&[], 2);
        assert_eq!(result.assignments.len(), 0);
        assert_eq!(result.dropped_count, 0);
        assert_eq!(result.num_tokens, 0);
    }

    #[test]
    fn test_routing_single_token_single_expert() {
        let router = TokenRouter::new(1.0);
        let gate_logits = vec![vec![1.0]];
        let result = router.route(&gate_logits, 1);
        assert_eq!(result.assignments.len(), 1);
        assert_eq!(result.assignments[0].expert_id, 0);
        assert!((result.assignments[0].weight - 1.0).abs() < 1e-5);
    }

    // ── AllToAll dispatch: token grouping ──

    #[test]
    fn test_dispatch_groups_by_rank() {
        let placement = ExpertPlacement::uniform(4, 2);
        let dispatcher = AllToAllDispatcher::new(placement);

        let router = TokenRouter::new(10.0);
        let gate_logits = vec![
            vec![10.0, 0.0, 0.0, 0.0], // expert 0 -> rank 0
            vec![0.0, 10.0, 0.0, 0.0], // expert 1 -> rank 1
            vec![0.0, 0.0, 10.0, 0.0], // expert 2 -> rank 0
            vec![0.0, 0.0, 0.0, 10.0], // expert 3 -> rank 1
        ];
        let routing = router.route(&gate_logits, 1);
        let dispatched = dispatcher.dispatch(&routing);

        assert_eq!(dispatched.per_rank.len(), 2);
        assert_eq!(dispatched.per_rank[0].len(), 2); // tokens 0, 2 -> rank 0
        assert_eq!(dispatched.per_rank[1].len(), 2); // tokens 1, 3 -> rank 1
    }

    #[test]
    fn test_dispatch_single_gpu_all_local() {
        let placement = ExpertPlacement::uniform(4, 1);
        let dispatcher = AllToAllDispatcher::new(placement);

        let router = TokenRouter::new(10.0);
        let gate_logits = vec![
            vec![10.0, 0.0, 0.0, 0.0],
            vec![0.0, 10.0, 0.0, 0.0],
        ];
        let routing = router.route(&gate_logits, 1);
        let dispatched = dispatcher.dispatch(&routing);

        assert_eq!(dispatched.per_rank.len(), 1);
        assert_eq!(dispatched.per_rank[0].len(), 2); // all tokens local
    }

    // ── AllToAll collect: reassembly and round-trip ──

    #[test]
    fn test_collect_weighted_sum() {
        let placement = ExpertPlacement::uniform(2, 1);
        let dispatcher = AllToAllDispatcher::new(placement);

        let router = TokenRouter::new(10.0);
        // 1 token, 2 experts, top-2
        let gate_logits = vec![vec![1.0, 1.0]]; // equal routing
        let routing = router.route(&gate_logits, 2);

        let hidden_dim = 3;
        let mut expert_outputs = HashMap::new();
        // Expert 0 output: [1, 2, 3]
        expert_outputs.insert((0, 0), vec![1.0f32, 2.0, 3.0]);
        // Expert 1 output: [4, 5, 6]
        expert_outputs.insert((0, 1), vec![4.0f32, 5.0, 6.0]);

        let result = dispatcher.collect(&expert_outputs, &routing, 1, hidden_dim);

        // Each weight should be 0.5 (normalized equally), so output = 0.5*[1,2,3] + 0.5*[4,5,6] = [2.5, 3.5, 4.5]
        assert_eq!(result.len(), 3);
        assert!((result[0] - 2.5).abs() < 1e-5);
        assert!((result[1] - 3.5).abs() < 1e-5);
        assert!((result[2] - 4.5).abs() < 1e-5);
    }

    #[test]
    fn test_collect_round_trip_preserves_order() {
        let placement = ExpertPlacement::uniform(4, 2);
        let dispatcher = AllToAllDispatcher::new(placement);

        let router = TokenRouter::new(10.0);
        let gate_logits = vec![
            vec![10.0, 0.0, 0.0, 0.0], // token 0 -> expert 0
            vec![0.0, 10.0, 0.0, 0.0], // token 1 -> expert 1
            vec![0.0, 0.0, 10.0, 0.0], // token 2 -> expert 2
        ];
        let routing = router.route(&gate_logits, 1);

        // Simulate expert outputs (identity-like: each expert returns the token_idx as values)
        let hidden_dim = 2;
        let mut expert_outputs = HashMap::new();
        expert_outputs.insert((0, 0), vec![10.0f32, 11.0]);
        expert_outputs.insert((1, 1), vec![20.0f32, 21.0]);
        expert_outputs.insert((2, 2), vec![30.0f32, 31.0]);

        let result = dispatcher.collect(&expert_outputs, &routing, 3, hidden_dim);

        // Token 0's output
        assert!((result[0] - 10.0).abs() < 1e-5);
        assert!((result[1] - 11.0).abs() < 1e-5);
        // Token 1's output
        assert!((result[2] - 20.0).abs() < 1e-5);
        assert!((result[3] - 21.0).abs() < 1e-5);
        // Token 2's output
        assert!((result[4] - 30.0).abs() < 1e-5);
        assert!((result[5] - 31.0).abs() < 1e-5);
    }

    // ── Load balancer: uniform routing → low aux loss ──

    #[test]
    fn test_load_balancer_uniform_routing_low_loss() {
        let mut balancer = LoadBalancer::new(4);
        // Perfectly balanced: each expert gets 10 tokens
        let routing = RoutingResult {
            assignments: Vec::new(), // not used by balancer
            dropped_count: 0,
            num_tokens: 40,
            num_experts: 4,
            top_k: 1,
            expert_counts: vec![10, 10, 10, 10],
        };
        balancer.update(&routing);
        let loss = balancer.auxiliary_loss();
        assert!(loss < 1e-5, "uniform routing should have ~0 loss, got {loss}");
    }

    #[test]
    fn test_load_balancer_skewed_routing_high_loss() {
        let mut balancer = LoadBalancer::new(4);
        // All tokens go to expert 0
        let routing = RoutingResult {
            assignments: Vec::new(),
            dropped_count: 0,
            num_tokens: 40,
            num_experts: 4,
            top_k: 1,
            expert_counts: vec![40, 0, 0, 0],
        };
        balancer.update(&routing);
        let loss = balancer.auxiliary_loss();
        // For completely skewed: f_0=1.0, f_1=f_2=f_3=0.0
        // loss = 4 * (1^2 + 0 + 0 + 0) - 1 = 3.0
        assert!((loss - 3.0).abs() < 1e-5, "skewed loss should be 3.0, got {loss}");
    }

    #[test]
    fn test_load_balancer_utilization_uniform() {
        let mut balancer = LoadBalancer::new(4);
        let routing = RoutingResult {
            assignments: Vec::new(),
            dropped_count: 0,
            num_tokens: 100,
            num_experts: 4,
            top_k: 1,
            expert_counts: vec![25, 25, 25, 25],
        };
        balancer.update(&routing);
        let util = balancer.utilization();
        assert_eq!(util.len(), 4);
        for &u in &util {
            assert!((u - 0.25).abs() < 1e-5);
        }
    }

    #[test]
    fn test_load_balancer_reset() {
        let mut balancer = LoadBalancer::new(2);
        let routing = RoutingResult {
            assignments: Vec::new(),
            dropped_count: 0,
            num_tokens: 10,
            num_experts: 2,
            top_k: 1,
            expert_counts: vec![10, 0],
        };
        balancer.update(&routing);
        assert!(balancer.auxiliary_loss() > 0.0);

        balancer.reset();
        assert_eq!(balancer.auxiliary_loss(), 0.0);
        assert_eq!(balancer.num_updates(), 0);
    }

    #[test]
    fn test_load_balancer_multiple_updates() {
        let mut balancer = LoadBalancer::new(2);
        // First batch: skewed
        let r1 = RoutingResult {
            assignments: Vec::new(),
            dropped_count: 0,
            num_tokens: 10,
            num_experts: 2,
            top_k: 1,
            expert_counts: vec![10, 0],
        };
        balancer.update(&r1);
        // Second batch: opposite skew -> overall balanced
        let r2 = RoutingResult {
            assignments: Vec::new(),
            dropped_count: 0,
            num_tokens: 10,
            num_experts: 2,
            top_k: 1,
            expert_counts: vec![0, 10],
        };
        balancer.update(&r2);
        // Overall: each expert got 10 out of 20
        let loss = balancer.auxiliary_loss();
        assert!(loss < 1e-5, "balanced overall should have ~0 loss, got {loss}");
        assert_eq!(balancer.num_updates(), 2);
    }

    // ── Capacity factor tests ──

    #[test]
    fn test_capacity_factor_1_drops_overflow() {
        let router = TokenRouter::new(1.0);
        // 6 tokens, 3 experts, top-1
        // capacity = 1.0 * 6/3 = 2 per expert
        // All 6 tokens want expert 0 -> only 2 accepted, 4 dropped
        let gate_logits = vec![
            vec![10.0, 0.0, 0.0],
            vec![10.0, 0.0, 0.0],
            vec![10.0, 0.0, 0.0],
            vec![10.0, 0.0, 0.0],
            vec![10.0, 0.0, 0.0],
            vec![10.0, 0.0, 0.0],
        ];
        let result = router.route(&gate_logits, 1);
        assert_eq!(result.expert_counts[0], 2);
        assert_eq!(result.dropped_count, 4);
    }

    #[test]
    fn test_capacity_factor_2_allows_double() {
        let router = TokenRouter::new(2.0);
        // 6 tokens, 3 experts, top-1
        // capacity = 2.0 * 6/3 = 4 per expert
        // All 6 tokens want expert 0 -> 4 accepted, 2 dropped
        let gate_logits = vec![
            vec![10.0, 0.0, 0.0],
            vec![10.0, 0.0, 0.0],
            vec![10.0, 0.0, 0.0],
            vec![10.0, 0.0, 0.0],
            vec![10.0, 0.0, 0.0],
            vec![10.0, 0.0, 0.0],
        ];
        let result = router.route(&gate_logits, 1);
        assert_eq!(result.expert_counts[0], 4);
        assert_eq!(result.dropped_count, 2);
    }

    #[test]
    fn test_capacity_factor_large_no_drops() {
        let router = TokenRouter::new(100.0);
        // Even heavily skewed routing drops nothing with high capacity
        let gate_logits = vec![
            vec![10.0, 0.0],
            vec![10.0, 0.0],
            vec![10.0, 0.0],
            vec![10.0, 0.0],
        ];
        let result = router.route(&gate_logits, 1);
        assert_eq!(result.dropped_count, 0);
    }

    // ── ExpertParallelGroup end-to-end ──

    #[test]
    fn test_group_route_and_dispatch() {
        let config = ExpertParallelConfig::uniform(2);
        let mut group = ExpertParallelGroup::new(config, 4, 0);

        let gate_logits = vec![
            vec![10.0, 0.0, 0.0, 0.0],
            vec![0.0, 10.0, 0.0, 0.0],
            vec![0.0, 0.0, 10.0, 0.0],
            vec![0.0, 0.0, 0.0, 10.0],
        ];
        let (routing, dispatched) = group.route_and_dispatch(&gate_logits, 1);

        assert_eq!(routing.assignments.len(), 4);
        assert_eq!(routing.dropped_count, 0);
        assert_eq!(dispatched.num_ranks, 2);
        // Experts 0,2 -> rank 0; experts 1,3 -> rank 1
        assert_eq!(dispatched.per_rank[0].len(), 2);
        assert_eq!(dispatched.per_rank[1].len(), 2);
    }

    #[test]
    fn test_group_local_experts() {
        let config = ExpertParallelConfig::uniform(4);
        let group = ExpertParallelGroup::new(config, 16, 2);
        let local = group.local_experts();
        // With 16 experts and 4 GPUs, rank 2 gets experts 2, 6, 10, 14
        assert_eq!(local, vec![2, 6, 10, 14]);
    }

    #[test]
    fn test_group_is_local() {
        let config = ExpertParallelConfig::uniform(2);
        let group = ExpertParallelGroup::new(config, 4, 0);
        assert!(group.is_local(0));
        assert!(!group.is_local(1));
        assert!(group.is_local(2));
        assert!(!group.is_local(3));
    }

    #[test]
    fn test_group_auxiliary_loss_tracks() {
        let config = ExpertParallelConfig::uniform(1);
        let mut group = ExpertParallelGroup::new(config, 4, 0);

        // Uniform routing -> low loss
        let gate_logits = vec![
            vec![10.0, 0.0, 0.0, 0.0],
            vec![0.0, 10.0, 0.0, 0.0],
            vec![0.0, 0.0, 10.0, 0.0],
            vec![0.0, 0.0, 0.0, 10.0],
        ];
        group.route_and_dispatch(&gate_logits, 1);
        let loss = group.auxiliary_loss();
        assert!(loss < 1e-5, "uniform routing loss should be ~0, got {loss}");
    }

    // ── Softmax utility ──

    #[test]
    fn test_softmax_basic() {
        let result = softmax(&[0.0, 0.0, 0.0]);
        for &v in &result {
            assert!((v - 1.0 / 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax_empty() {
        let result = softmax(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let result = softmax(&[1.0, 2.0, 3.0, 4.0]);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    // ── Config tests ──

    #[test]
    fn test_config_default() {
        let config = ExpertParallelConfig::default();
        assert_eq!(config.num_expert_groups, 1);
        assert_eq!(config.placement_strategy, PlacementStrategy::Uniform);
        assert!((config.capacity_factor - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_config_uniform_constructor() {
        let config = ExpertParallelConfig::uniform(8);
        assert_eq!(config.num_expert_groups, 8);
        assert_eq!(config.placement_strategy, PlacementStrategy::Uniform);
    }

    #[test]
    fn test_config_capacity_aware_constructor() {
        let config = ExpertParallelConfig::capacity_aware(4, 1.5);
        assert_eq!(config.num_expert_groups, 4);
        assert_eq!(config.placement_strategy, PlacementStrategy::CapacityAware);
        assert!((config.capacity_factor - 1.5).abs() < 1e-5);
    }
}
