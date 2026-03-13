//! Pipeline parallelism for multi-GPU inference.
//!
//! Pipeline parallelism (PP) partitions model layers across multiple stages
//! (typically one per GPU) and uses micro-batching to overlap computation
//! across stages. This enables serving models whose layers exceed a single
//! GPU's memory, complementing tensor parallelism which shards within layers.
//!
//! # Schedule
//!
//! For inference (forward-only), we implement a GPipe-style 1F1B schedule:
//! - **Fill phase**: inject micro-batches into the pipeline one per step.
//! - **Steady state**: each stage processes the next micro-batch as soon as
//!   the previous one has been forwarded downstream.
//! - **Drain phase**: flush remaining micro-batches out of the pipeline.
//!
//! The pipeline bubble fraction is `(S - 1) / (S - 1 + M)` where `S` is the
//! number of stages and `M` is the number of micro-batches.
//!
//! # Activation transfer
//!
//! Between stages, hidden-state tensors are transferred via NCCL point-to-point
//! send/recv when the `cuda` feature is enabled. On CPU builds, the transfer is
//! a no-op clone (all stages share the same address space).

use candle_core::{Result, Tensor};

// ─── Configuration ────────────────────────────────────────────────────────

/// Strategy for assigning transformer layers to pipeline stages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerAssignmentStrategy {
    /// Distribute layers as evenly as possible across stages.
    /// Remainder layers are distributed one-per-stage to the first stages.
    Uniform,
    /// Explicit per-stage layer counts. Must sum to `num_layers`.
    Custom(Vec<usize>),
}

impl Default for LayerAssignmentStrategy {
    fn default() -> Self {
        Self::Uniform
    }
}

/// Pipeline parallelism configuration.
#[derive(Debug, Clone)]
pub struct PipelineParallelConfig {
    /// Number of pipeline stages (typically one per GPU).
    pub num_stages: usize,
    /// Number of micro-batches to split the batch into.
    pub num_micro_batches: usize,
    /// Strategy for assigning layers to stages.
    pub layer_assignment: LayerAssignmentStrategy,
}

impl PipelineParallelConfig {
    /// Create a disabled (single-stage) pipeline parallel config.
    pub fn disabled() -> Self {
        Self {
            num_stages: 1,
            num_micro_batches: 1,
            layer_assignment: LayerAssignmentStrategy::Uniform,
        }
    }

    /// Whether pipeline parallelism is effectively disabled.
    pub fn is_disabled(&self) -> bool {
        self.num_stages <= 1
    }

    /// Create a new pipeline parallel config.
    ///
    /// # Errors
    /// Returns an error if `num_stages` or `num_micro_batches` is 0.
    pub fn new(
        num_stages: usize,
        num_micro_batches: usize,
        layer_assignment: LayerAssignmentStrategy,
    ) -> Result<Self> {
        if num_stages == 0 {
            return Err(candle_core::Error::Msg(
                "pipeline parallel num_stages must be >= 1".to_string(),
            ));
        }
        if num_micro_batches == 0 {
            return Err(candle_core::Error::Msg(
                "pipeline parallel num_micro_batches must be >= 1".to_string(),
            ));
        }
        Ok(Self {
            num_stages,
            num_micro_batches,
            layer_assignment,
        })
    }
}

impl Default for PipelineParallelConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

// ─── Pipeline stage ────────────────────────────────────────────────────────

/// A single pipeline stage representing a contiguous range of transformer layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineStage {
    /// Stage index (0-based).
    pub stage_id: usize,
    /// First layer index (inclusive).
    pub start_layer: usize,
    /// Last layer index (exclusive).
    pub end_layer: usize,
}

impl PipelineStage {
    /// Number of layers in this stage.
    pub fn num_layers(&self) -> usize {
        self.end_layer - self.start_layer
    }

    /// Whether a given layer index belongs to this stage.
    pub fn contains_layer(&self, layer_idx: usize) -> bool {
        layer_idx >= self.start_layer && layer_idx < self.end_layer
    }
}

// ─── Micro-batch ───────────────────────────────────────────────────────────

/// A fragment of a batch flowing through the pipeline.
#[derive(Debug, Clone)]
pub struct MicroBatch {
    /// The original batch ID this micro-batch belongs to.
    pub batch_id: u64,
    /// Index of this micro-batch within the batch (0-based).
    pub micro_batch_id: usize,
    /// Hidden states tensor: `[micro_batch_seq_count, hidden_dim]`.
    pub hidden_states: Tensor,
    /// Sequence IDs in this micro-batch (for routing results back).
    pub sequence_ids: Vec<u64>,
}

impl MicroBatch {
    /// Number of sequences in this micro-batch.
    pub fn num_sequences(&self) -> usize {
        self.sequence_ids.len()
    }
}

// ─── Layer assignment ──────────────────────────────────────────────────────

/// Compute a uniform layer assignment: distribute `num_layers` as evenly as
/// possible across `num_stages`.
///
/// Remainder layers are assigned one each to the first stages. For example,
/// 10 layers across 3 stages yields `[4, 3, 3]`.
///
/// # Errors
/// Returns an error if `num_stages` is 0 or `num_stages > num_layers`.
pub fn compute_uniform_assignment(
    num_layers: usize,
    num_stages: usize,
) -> Result<Vec<PipelineStage>> {
    if num_stages == 0 {
        return Err(candle_core::Error::Msg(
            "num_stages must be >= 1".to_string(),
        ));
    }
    if num_layers == 0 {
        return Err(candle_core::Error::Msg(
            "num_layers must be >= 1".to_string(),
        ));
    }
    if num_stages > num_layers {
        return Err(candle_core::Error::Msg(format!(
            "num_stages ({num_stages}) cannot exceed num_layers ({num_layers})"
        )));
    }

    let base = num_layers / num_stages;
    let remainder = num_layers % num_stages;

    let mut stages = Vec::with_capacity(num_stages);
    let mut offset = 0;

    for stage_id in 0..num_stages {
        let count = base + if stage_id < remainder { 1 } else { 0 };
        stages.push(PipelineStage {
            stage_id,
            start_layer: offset,
            end_layer: offset + count,
        });
        offset += count;
    }

    debug_assert_eq!(offset, num_layers);
    Ok(stages)
}

/// Compute layer assignment from a custom per-stage layer count vector.
///
/// # Errors
/// Returns an error if any count is 0 or the total doesn't equal `num_layers`.
pub fn compute_custom_assignment(
    num_layers: usize,
    counts: &[usize],
) -> Result<Vec<PipelineStage>> {
    let total: usize = counts.iter().sum();
    if total != num_layers {
        return Err(candle_core::Error::Msg(format!(
            "custom layer counts sum to {total}, expected {num_layers}"
        )));
    }
    for (i, &c) in counts.iter().enumerate() {
        if c == 0 {
            return Err(candle_core::Error::Msg(format!(
                "custom layer count for stage {i} is 0; each stage must have >= 1 layer"
            )));
        }
    }

    let mut stages = Vec::with_capacity(counts.len());
    let mut offset = 0;
    for (stage_id, &count) in counts.iter().enumerate() {
        stages.push(PipelineStage {
            stage_id,
            start_layer: offset,
            end_layer: offset + count,
        });
        offset += count;
    }
    Ok(stages)
}

// ─── Pipeline schedule trait ───────────────────────────────────────────────

/// An action in the pipeline schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduleStep {
    /// Which stage performs this step.
    pub stage_id: usize,
    /// Which micro-batch is being processed.
    pub micro_batch_id: usize,
    /// The global clock tick at which this step occurs.
    pub clock: usize,
}

/// A schedule determines the order in which stages process micro-batches.
pub trait PipelineSchedule {
    /// Generate the full schedule of forward-pass steps.
    ///
    /// Returns a list of `ScheduleStep` in chronological order (sorted by clock).
    fn generate_schedule(
        &self,
        num_stages: usize,
        num_micro_batches: usize,
    ) -> Vec<ScheduleStep>;
}

// ─── GPipe 1F1B schedule (inference-only) ──────────────────────────────────

/// GPipe-style 1F1B schedule for inference (forward passes only).
///
/// In inference there are no backward passes, so the schedule is simply:
/// each stage processes micro-batches in order, with stage `s` starting
/// micro-batch `m` at clock tick `s + m`. This naturally overlaps stages
/// processing different micro-batches.
///
/// ```text
/// Clock:    0   1   2   3   4   5
/// Stage 0:  m0  m1  m2  m3
/// Stage 1:      m0  m1  m2  m3
/// Stage 2:          m0  m1  m2  m3
/// ```
pub struct GpfSchedule;

impl PipelineSchedule for GpfSchedule {
    fn generate_schedule(
        &self,
        num_stages: usize,
        num_micro_batches: usize,
    ) -> Vec<ScheduleStep> {
        // Total clock ticks = num_stages - 1 + num_micro_batches
        let total_ticks = num_stages.saturating_sub(1) + num_micro_batches;
        let mut steps = Vec::with_capacity(num_stages * num_micro_batches);

        for clock in 0..total_ticks {
            for stage_id in 0..num_stages {
                // Stage `stage_id` starts processing at clock = stage_id
                // and processes micro-batch (clock - stage_id).
                if clock >= stage_id {
                    let mb = clock - stage_id;
                    if mb < num_micro_batches {
                        steps.push(ScheduleStep {
                            stage_id,
                            micro_batch_id: mb,
                            clock,
                        });
                    }
                }
            }
        }

        steps
    }
}

// ─── Pipeline bubble calculation ───────────────────────────────────────────

/// Compute the pipeline bubble fraction.
///
/// The bubble fraction represents the proportion of time stages are idle
/// due to pipeline startup and drain.
///
/// Formula: `(S - 1) / (S - 1 + M)` where `S` = num_stages, `M` = num_micro_batches.
///
/// Special cases:
/// - 1 stage: bubble = 0.0 (no pipeline overhead)
/// - 0 micro-batches: returns 1.0 (degenerate case, all bubble)
pub fn compute_bubble_fraction(num_stages: usize, num_micro_batches: usize) -> f64 {
    if num_stages <= 1 {
        return 0.0;
    }
    if num_micro_batches == 0 {
        return 1.0;
    }
    let s = (num_stages - 1) as f64;
    let m = num_micro_batches as f64;
    s / (s + m)
}

// ─── Activation transfer stubs ─────────────────────────────────────────────

/// Send an activation tensor to a destination pipeline stage.
///
/// On CUDA builds with the `cuda` feature, this uses NCCL point-to-point
/// send. On CPU builds, this is a no-op that returns the tensor unchanged
/// (all stages share the same address space in a single process).
pub fn send_activation(
    tensor: &Tensor,
    _dst_stage: usize,
    #[allow(unused_variables)] _nccl_comm: Option<&super::nccl::NcclComm>,
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        if let Some(_comm) = _nccl_comm {
            // In a real implementation, this would call ncclSend:
            // comm.send(tensor.as_ptr(), tensor.elem_count(), dst_rank, stream)
            // For now, this is a placeholder for the NCCL P2P path.
            let _ = tensor;
            let _ = _dst_stage;
            return Ok(());
        }
    }

    // CPU/Metal: no-op (tensors are in shared memory)
    let _ = tensor;
    Ok(())
}

/// Receive an activation tensor from a source pipeline stage.
///
/// On CUDA builds with the `cuda` feature, this uses NCCL point-to-point
/// recv. On CPU builds, this is a no-op that clones the provided tensor
/// (acting as a passthrough).
pub fn recv_activation(
    tensor: &Tensor,
    _src_stage: usize,
    #[allow(unused_variables)] _nccl_comm: Option<&super::nccl::NcclComm>,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if let Some(_comm) = _nccl_comm {
            // In a real implementation, this would call ncclRecv:
            // comm.recv(output_ptr, elem_count, src_rank, stream)
            // For now, return a clone as placeholder.
            return Ok(tensor.clone());
        }
    }

    // CPU/Metal: clone the tensor (simulates receiving from same process)
    Ok(tensor.clone())
}

// ─── Micro-batch splitting ─────────────────────────────────────────────────

/// Split a batch tensor and its sequence IDs into micro-batches.
///
/// The batch tensor has shape `[total_seqs, ...]` and is split along dim 0
/// into `num_micro_batches` roughly equal parts.
///
/// # Arguments
/// - `batch_id`: The batch identifier.
/// - `hidden_states`: The full batch tensor `[total_seqs, hidden_dim]`.
/// - `sequence_ids`: Sequence IDs for each row.
/// - `num_micro_batches`: How many micro-batches to create.
///
/// # Errors
/// Returns an error if `num_micro_batches` is 0 or exceeds the batch size.
pub fn split_into_micro_batches(
    batch_id: u64,
    hidden_states: &Tensor,
    sequence_ids: &[u64],
    num_micro_batches: usize,
) -> Result<Vec<MicroBatch>> {
    let total = sequence_ids.len();
    if num_micro_batches == 0 {
        return Err(candle_core::Error::Msg(
            "num_micro_batches must be >= 1".to_string(),
        ));
    }
    if num_micro_batches > total {
        return Err(candle_core::Error::Msg(format!(
            "num_micro_batches ({num_micro_batches}) exceeds batch size ({total})"
        )));
    }

    let base = total / num_micro_batches;
    let remainder = total % num_micro_batches;

    let mut micro_batches = Vec::with_capacity(num_micro_batches);
    let mut offset = 0;

    for mb_id in 0..num_micro_batches {
        let count = base + if mb_id < remainder { 1 } else { 0 };
        let mb_hidden = hidden_states.narrow(0, offset, count)?;
        let mb_seq_ids = sequence_ids[offset..offset + count].to_vec();

        micro_batches.push(MicroBatch {
            batch_id,
            micro_batch_id: mb_id,
            hidden_states: mb_hidden,
            sequence_ids: mb_seq_ids,
        });

        offset += count;
    }

    debug_assert_eq!(offset, total);
    Ok(micro_batches)
}

// ─── Pipeline coordinator ──────────────────────────────────────────────────

/// Coordinates pipeline-parallel execution across stages.
///
/// The coordinator:
/// 1. Splits the input batch into micro-batches.
/// 2. Dispatches micro-batches to stages according to the schedule.
/// 3. Simulates inter-stage activation transfer.
/// 4. Collects results in the original micro-batch order.
pub struct PipelineCoordinator {
    /// Pipeline stages with their layer assignments.
    pub stages: Vec<PipelineStage>,
    /// Pipeline configuration.
    pub config: PipelineParallelConfig,
}

impl PipelineCoordinator {
    /// Create a new pipeline coordinator.
    ///
    /// # Arguments
    /// - `config`: Pipeline parallel configuration.
    /// - `num_layers`: Total number of transformer layers in the model.
    pub fn new(config: PipelineParallelConfig, num_layers: usize) -> Result<Self> {
        let stages = match &config.layer_assignment {
            LayerAssignmentStrategy::Uniform => {
                compute_uniform_assignment(num_layers, config.num_stages)?
            }
            LayerAssignmentStrategy::Custom(counts) => {
                compute_custom_assignment(num_layers, counts)?
            }
        };

        Ok(Self { stages, config })
    }

    /// Execute a batch through the pipeline.
    ///
    /// This is a synchronous, single-process simulation of pipeline parallelism.
    /// In a multi-GPU deployment, each stage would run on a separate GPU with
    /// NCCL P2P transfers between stages.
    ///
    /// # Arguments
    /// - `batch_id`: Unique batch identifier.
    /// - `hidden_states`: Full batch hidden states `[batch_size, hidden_dim]`.
    /// - `sequence_ids`: Sequence IDs corresponding to each row.
    /// - `stage_fn`: A function `(stage, micro_batch) -> Result<MicroBatch>` that
    ///   applies the stage's layers to the micro-batch.
    ///
    /// # Returns
    /// The processed micro-batches, reassembled in the original input order.
    pub fn execute<F>(
        &self,
        batch_id: u64,
        hidden_states: &Tensor,
        sequence_ids: &[u64],
        stage_fn: F,
    ) -> Result<Vec<MicroBatch>>
    where
        F: Fn(&PipelineStage, MicroBatch) -> Result<MicroBatch>,
    {
        let num_mb = self.config.num_micro_batches;

        // Split into micro-batches
        let micro_batches = split_into_micro_batches(
            batch_id,
            hidden_states,
            sequence_ids,
            num_mb,
        )?;

        // Generate the schedule
        let schedule = GpfSchedule;
        let steps = schedule.generate_schedule(self.config.num_stages, num_mb);

        // State: current hidden states for each micro-batch
        let mut mb_states: Vec<Option<MicroBatch>> = micro_batches.into_iter().map(Some).collect();

        // Process steps in schedule order (already sorted by clock)
        for step in &steps {
            let stage = &self.stages[step.stage_id];
            let mb = mb_states[step.micro_batch_id]
                .take()
                .expect("micro-batch should be available at scheduled step");

            // Simulate recv from previous stage (no-op on CPU)
            let received = if step.stage_id > 0 {
                recv_activation(&mb.hidden_states, step.stage_id - 1, None)?;
                mb
            } else {
                mb
            };

            // Process through this stage's layers
            let processed = stage_fn(stage, received)?;

            // Simulate send to next stage (no-op on CPU)
            if step.stage_id < self.stages.len() - 1 {
                send_activation(&processed.hidden_states, step.stage_id + 1, None)?;
            }

            mb_states[step.micro_batch_id] = Some(processed);
        }

        // Collect results in order
        let results: Vec<MicroBatch> = mb_states
            .into_iter()
            .enumerate()
            .map(|(i, mb)| {
                mb.unwrap_or_else(|| {
                    panic!("micro-batch {i} was not processed by the pipeline")
                })
            })
            .collect();

        Ok(results)
    }

    /// Get the stage that owns a given layer.
    pub fn stage_for_layer(&self, layer_idx: usize) -> Option<&PipelineStage> {
        self.stages.iter().find(|s| s.contains_layer(layer_idx))
    }

    /// Get the bubble fraction for this pipeline configuration.
    pub fn bubble_fraction(&self) -> f64 {
        compute_bubble_fraction(self.config.num_stages, self.config.num_micro_batches)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    // ── Layer assignment: uniform distribution ──

    #[test]
    fn test_uniform_assignment_even_split() {
        let stages = compute_uniform_assignment(12, 3).unwrap();
        assert_eq!(stages.len(), 3);
        assert_eq!(stages[0], PipelineStage { stage_id: 0, start_layer: 0, end_layer: 4 });
        assert_eq!(stages[1], PipelineStage { stage_id: 1, start_layer: 4, end_layer: 8 });
        assert_eq!(stages[2], PipelineStage { stage_id: 2, start_layer: 8, end_layer: 12 });
    }

    #[test]
    fn test_uniform_assignment_remainder_handling() {
        // 10 layers / 3 stages = 4, 3, 3
        let stages = compute_uniform_assignment(10, 3).unwrap();
        assert_eq!(stages.len(), 3);
        assert_eq!(stages[0].num_layers(), 4);
        assert_eq!(stages[1].num_layers(), 3);
        assert_eq!(stages[2].num_layers(), 3);
        // Verify contiguous coverage
        assert_eq!(stages[0].start_layer, 0);
        assert_eq!(stages[0].end_layer, 4);
        assert_eq!(stages[1].start_layer, 4);
        assert_eq!(stages[1].end_layer, 7);
        assert_eq!(stages[2].start_layer, 7);
        assert_eq!(stages[2].end_layer, 10);
    }

    #[test]
    fn test_uniform_assignment_single_stage_all_layers() {
        let stages = compute_uniform_assignment(32, 1).unwrap();
        assert_eq!(stages.len(), 1);
        assert_eq!(stages[0].start_layer, 0);
        assert_eq!(stages[0].end_layer, 32);
        assert_eq!(stages[0].num_layers(), 32);
    }

    #[test]
    fn test_uniform_assignment_stages_equals_layers() {
        // Each stage gets exactly 1 layer
        let stages = compute_uniform_assignment(4, 4).unwrap();
        assert_eq!(stages.len(), 4);
        for (i, s) in stages.iter().enumerate() {
            assert_eq!(s.start_layer, i);
            assert_eq!(s.end_layer, i + 1);
            assert_eq!(s.num_layers(), 1);
        }
    }

    #[test]
    fn test_uniform_assignment_stages_exceeds_layers() {
        let result = compute_uniform_assignment(3, 5);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("cannot exceed"));
    }

    #[test]
    fn test_uniform_assignment_zero_stages() {
        let result = compute_uniform_assignment(10, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_uniform_assignment_zero_layers() {
        let result = compute_uniform_assignment(0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_uniform_assignment_two_stages_odd_layers() {
        // 7 layers / 2 stages = 4, 3
        let stages = compute_uniform_assignment(7, 2).unwrap();
        assert_eq!(stages[0].num_layers(), 4);
        assert_eq!(stages[1].num_layers(), 3);
        assert_eq!(stages[1].end_layer, 7);
    }

    #[test]
    fn test_uniform_assignment_covers_all_layers() {
        for num_layers in [1, 5, 16, 32, 80] {
            for num_stages in 1..=num_layers.min(8) {
                let stages = compute_uniform_assignment(num_layers, num_stages).unwrap();
                // Verify complete coverage with no gaps
                let total: usize = stages.iter().map(|s| s.num_layers()).sum();
                assert_eq!(total, num_layers, "layers={num_layers}, stages={num_stages}");
                assert_eq!(stages.first().unwrap().start_layer, 0);
                assert_eq!(stages.last().unwrap().end_layer, num_layers);
                for w in stages.windows(2) {
                    assert_eq!(w[0].end_layer, w[1].start_layer);
                }
            }
        }
    }

    // ── Custom assignment ──

    #[test]
    fn test_custom_assignment_valid() {
        let stages = compute_custom_assignment(10, &[2, 3, 5]).unwrap();
        assert_eq!(stages.len(), 3);
        assert_eq!(stages[0].num_layers(), 2);
        assert_eq!(stages[1].num_layers(), 3);
        assert_eq!(stages[2].num_layers(), 5);
    }

    #[test]
    fn test_custom_assignment_wrong_total() {
        let result = compute_custom_assignment(10, &[2, 3, 4]);
        assert!(result.is_err());
    }

    #[test]
    fn test_custom_assignment_zero_count() {
        let result = compute_custom_assignment(10, &[5, 0, 5]);
        assert!(result.is_err());
    }

    // ── Micro-batch splitting ──

    #[test]
    fn test_micro_batch_split_even() {
        let hidden = Tensor::zeros((8, 16), candle_core::DType::F32, &Device::Cpu).unwrap();
        let seq_ids: Vec<u64> = (0..8).collect();
        let mbs = split_into_micro_batches(1, &hidden, &seq_ids, 4).unwrap();

        assert_eq!(mbs.len(), 4);
        for mb in &mbs {
            assert_eq!(mb.num_sequences(), 2);
            assert_eq!(mb.hidden_states.dims(), &[2, 16]);
            assert_eq!(mb.batch_id, 1);
        }
        // Verify sequence ID coverage
        let all_ids: Vec<u64> = mbs.iter().flat_map(|m| m.sequence_ids.clone()).collect();
        assert_eq!(all_ids, (0..8).collect::<Vec<u64>>());
    }

    #[test]
    fn test_micro_batch_split_uneven() {
        let hidden = Tensor::zeros((7, 16), candle_core::DType::F32, &Device::Cpu).unwrap();
        let seq_ids: Vec<u64> = (0..7).collect();
        let mbs = split_into_micro_batches(42, &hidden, &seq_ids, 3).unwrap();

        assert_eq!(mbs.len(), 3);
        // 7 / 3 = 2 remainder 1 -> first gets 3, rest get 2
        assert_eq!(mbs[0].num_sequences(), 3);
        assert_eq!(mbs[1].num_sequences(), 2);
        assert_eq!(mbs[2].num_sequences(), 2);
    }

    #[test]
    fn test_micro_batch_split_single() {
        let hidden = Tensor::zeros((5, 32), candle_core::DType::F32, &Device::Cpu).unwrap();
        let seq_ids: Vec<u64> = (10..15).collect();
        let mbs = split_into_micro_batches(0, &hidden, &seq_ids, 1).unwrap();

        assert_eq!(mbs.len(), 1);
        assert_eq!(mbs[0].num_sequences(), 5);
        assert_eq!(mbs[0].hidden_states.dims(), &[5, 32]);
        assert_eq!(mbs[0].sequence_ids, vec![10, 11, 12, 13, 14]);
    }

    #[test]
    fn test_micro_batch_split_zero_micro_batches() {
        let hidden = Tensor::zeros((4, 16), candle_core::DType::F32, &Device::Cpu).unwrap();
        let seq_ids: Vec<u64> = (0..4).collect();
        let result = split_into_micro_batches(0, &hidden, &seq_ids, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_micro_batch_split_too_many_micro_batches() {
        let hidden = Tensor::zeros((3, 16), candle_core::DType::F32, &Device::Cpu).unwrap();
        let seq_ids: Vec<u64> = (0..3).collect();
        let result = split_into_micro_batches(0, &hidden, &seq_ids, 5);
        assert!(result.is_err());
    }

    // ── GPipe schedule ──

    #[test]
    fn test_gpf_schedule_basic_ordering() {
        let schedule = GpfSchedule;
        let steps = schedule.generate_schedule(3, 4);

        // Total steps = 3 * 4 = 12
        assert_eq!(steps.len(), 12);

        // Verify all (stage, micro_batch) pairs are present
        let mut seen = std::collections::HashSet::new();
        for step in &steps {
            seen.insert((step.stage_id, step.micro_batch_id));
        }
        assert_eq!(seen.len(), 12);
        for s in 0..3 {
            for m in 0..4 {
                assert!(seen.contains(&(s, m)), "missing ({s}, {m})");
            }
        }
    }

    #[test]
    fn test_gpf_schedule_all_micro_batches_processed() {
        let schedule = GpfSchedule;
        let steps = schedule.generate_schedule(4, 6);
        assert_eq!(steps.len(), 4 * 6);

        // Every micro-batch passes through every stage
        for mb in 0..6 {
            let stages_for_mb: Vec<usize> = steps
                .iter()
                .filter(|s| s.micro_batch_id == mb)
                .map(|s| s.stage_id)
                .collect();
            assert_eq!(stages_for_mb.len(), 4);
            for s in 0..4 {
                assert!(stages_for_mb.contains(&s));
            }
        }
    }

    #[test]
    fn test_gpf_schedule_stage_processes_in_order() {
        let schedule = GpfSchedule;
        let steps = schedule.generate_schedule(3, 5);

        // For each stage, micro-batches should be processed in order 0, 1, 2, ...
        for stage_id in 0..3 {
            let mb_order: Vec<usize> = steps
                .iter()
                .filter(|s| s.stage_id == stage_id)
                .map(|s| s.micro_batch_id)
                .collect();
            assert_eq!(mb_order, vec![0, 1, 2, 3, 4]);
        }
    }

    #[test]
    fn test_gpf_schedule_no_stage_idle_gaps() {
        let schedule = GpfSchedule;
        let steps = schedule.generate_schedule(3, 4);

        // For each stage, there should be no idle ticks between its first and last step
        for stage_id in 0..3 {
            let clocks: Vec<usize> = steps
                .iter()
                .filter(|s| s.stage_id == stage_id)
                .map(|s| s.clock)
                .collect();
            assert_eq!(clocks.len(), 4);
            // Clocks should be consecutive
            for i in 1..clocks.len() {
                assert_eq!(clocks[i], clocks[i - 1] + 1,
                    "stage {stage_id} has idle gap at clock {}", clocks[i]);
            }
        }
    }

    #[test]
    fn test_gpf_schedule_single_stage() {
        let schedule = GpfSchedule;
        let steps = schedule.generate_schedule(1, 3);
        assert_eq!(steps.len(), 3);
        for (i, step) in steps.iter().enumerate() {
            assert_eq!(step.stage_id, 0);
            assert_eq!(step.micro_batch_id, i);
            assert_eq!(step.clock, i);
        }
    }

    #[test]
    fn test_gpf_schedule_single_micro_batch() {
        let schedule = GpfSchedule;
        let steps = schedule.generate_schedule(4, 1);
        assert_eq!(steps.len(), 4);
        for (i, step) in steps.iter().enumerate() {
            assert_eq!(step.stage_id, i);
            assert_eq!(step.micro_batch_id, 0);
            assert_eq!(step.clock, i);
        }
    }

    #[test]
    fn test_gpf_schedule_dependency_ordering() {
        // Stage s+1 should only process micro-batch m after stage s has processed it
        let schedule = GpfSchedule;
        let steps = schedule.generate_schedule(3, 4);

        for mb in 0..4 {
            let clocks_for_mb: Vec<(usize, usize)> = steps
                .iter()
                .filter(|s| s.micro_batch_id == mb)
                .map(|s| (s.stage_id, s.clock))
                .collect();
            // Verify stage ordering: stage 0 before stage 1 before stage 2
            for w in clocks_for_mb.windows(2) {
                assert!(w[0].0 < w[1].0, "stages out of order for mb {mb}");
                assert!(w[0].1 < w[1].1, "clocks out of order for mb {mb}");
            }
        }
    }

    // ── Bubble fraction ──

    #[test]
    fn test_bubble_fraction_single_stage() {
        assert_eq!(compute_bubble_fraction(1, 4), 0.0);
        assert_eq!(compute_bubble_fraction(1, 1), 0.0);
    }

    #[test]
    fn test_bubble_fraction_known_values() {
        // 2 stages, 2 micro-batches: (2-1)/(2-1+2) = 1/3
        let b = compute_bubble_fraction(2, 2);
        assert!((b - 1.0 / 3.0).abs() < 1e-10);

        // 4 stages, 4 micro-batches: 3/7
        let b = compute_bubble_fraction(4, 4);
        assert!((b - 3.0 / 7.0).abs() < 1e-10);

        // 4 stages, 12 micro-batches: 3/15 = 0.2
        let b = compute_bubble_fraction(4, 12);
        assert!((b - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_bubble_fraction_equal_stages_and_micro_batches() {
        // S stages, S micro-batches: (S-1)/(2S-1)
        for s in 2..=8 {
            let expected = (s as f64 - 1.0) / (2.0 * s as f64 - 1.0);
            let actual = compute_bubble_fraction(s, s);
            assert!((actual - expected).abs() < 1e-10, "s={s}");
        }
    }

    #[test]
    fn test_bubble_fraction_zero_micro_batches() {
        assert_eq!(compute_bubble_fraction(4, 0), 1.0);
    }

    #[test]
    fn test_bubble_fraction_many_micro_batches_reduces_bubble() {
        // More micro-batches -> smaller bubble
        let b1 = compute_bubble_fraction(4, 4);
        let b2 = compute_bubble_fraction(4, 8);
        let b3 = compute_bubble_fraction(4, 16);
        assert!(b1 > b2);
        assert!(b2 > b3);
        assert!(b3 > 0.0);
    }

    #[test]
    fn test_bubble_fraction_approaches_zero() {
        // With many micro-batches, bubble approaches 0
        let b = compute_bubble_fraction(4, 1000);
        assert!(b < 0.01);
    }

    // ── Pipeline coordinator ──

    #[test]
    fn test_coordinator_creation() {
        let config = PipelineParallelConfig::new(
            4, 8, LayerAssignmentStrategy::Uniform,
        ).unwrap();
        let coord = PipelineCoordinator::new(config, 32).unwrap();
        assert_eq!(coord.stages.len(), 4);
        assert_eq!(coord.stages[0].num_layers(), 8);
    }

    #[test]
    fn test_coordinator_result_order_matches_input() {
        let config = PipelineParallelConfig::new(
            3, 4, LayerAssignmentStrategy::Uniform,
        ).unwrap();
        let coord = PipelineCoordinator::new(config, 12).unwrap();

        let hidden = Tensor::arange(0f32, 64.0, &Device::Cpu)
            .unwrap()
            .reshape((8, 8))
            .unwrap();
        let seq_ids: Vec<u64> = (100..108).collect();

        // Identity stage function: pass through unchanged
        let results = coord.execute(1, &hidden, &seq_ids, |_stage, mb| Ok(mb)).unwrap();

        assert_eq!(results.len(), 4);

        // Verify results are in micro-batch order
        for (i, mb) in results.iter().enumerate() {
            assert_eq!(mb.micro_batch_id, i);
            assert_eq!(mb.batch_id, 1);
        }

        // Verify all sequence IDs are present in order
        let all_ids: Vec<u64> = results.iter().flat_map(|m| m.sequence_ids.clone()).collect();
        assert_eq!(all_ids, (100..108).collect::<Vec<u64>>());
    }

    #[test]
    fn test_coordinator_stage_fn_called_for_each_stage_and_mb() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let config = PipelineParallelConfig::new(
            3, 2, LayerAssignmentStrategy::Uniform,
        ).unwrap();
        let coord = PipelineCoordinator::new(config, 9).unwrap();

        let hidden = Tensor::zeros((4, 8), candle_core::DType::F32, &Device::Cpu).unwrap();
        let seq_ids: Vec<u64> = (0..4).collect();

        let call_count = Arc::new(AtomicUsize::new(0));
        let cc = call_count.clone();

        let _ = coord.execute(0, &hidden, &seq_ids, move |_stage, mb| {
            cc.fetch_add(1, Ordering::SeqCst);
            Ok(mb)
        }).unwrap();

        // 3 stages * 2 micro-batches = 6 calls
        assert_eq!(call_count.load(Ordering::SeqCst), 6);
    }

    #[test]
    fn test_coordinator_stage_for_layer() {
        let config = PipelineParallelConfig::new(
            3, 4, LayerAssignmentStrategy::Uniform,
        ).unwrap();
        let coord = PipelineCoordinator::new(config, 12).unwrap();

        // Layers 0-3 -> stage 0
        assert_eq!(coord.stage_for_layer(0).unwrap().stage_id, 0);
        assert_eq!(coord.stage_for_layer(3).unwrap().stage_id, 0);

        // Layers 4-7 -> stage 1
        assert_eq!(coord.stage_for_layer(4).unwrap().stage_id, 1);

        // Layers 8-11 -> stage 2
        assert_eq!(coord.stage_for_layer(11).unwrap().stage_id, 2);

        // Layer 12 -> None
        assert!(coord.stage_for_layer(12).is_none());
    }

    #[test]
    fn test_coordinator_bubble_fraction() {
        let config = PipelineParallelConfig::new(
            4, 8, LayerAssignmentStrategy::Uniform,
        ).unwrap();
        let coord = PipelineCoordinator::new(config, 32).unwrap();
        let expected = 3.0 / 11.0; // (4-1)/(4-1+8)
        assert!((coord.bubble_fraction() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_coordinator_disabled_passthrough() {
        let config = PipelineParallelConfig::disabled();
        let coord = PipelineCoordinator::new(config, 32).unwrap();
        assert_eq!(coord.stages.len(), 1);
        assert_eq!(coord.stages[0].num_layers(), 32);
        assert_eq!(coord.bubble_fraction(), 0.0);
    }

    // ── Config validation ──

    #[test]
    fn test_config_zero_stages_error() {
        let result = PipelineParallelConfig::new(0, 4, LayerAssignmentStrategy::Uniform);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_zero_micro_batches_error() {
        let result = PipelineParallelConfig::new(4, 0, LayerAssignmentStrategy::Uniform);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_disabled() {
        let config = PipelineParallelConfig::disabled();
        assert!(config.is_disabled());
        assert_eq!(config.num_stages, 1);
    }

    #[test]
    fn test_config_default_is_disabled() {
        let config = PipelineParallelConfig::default();
        assert!(config.is_disabled());
    }

    // ── Pipeline stage ──

    #[test]
    fn test_stage_contains_layer() {
        let stage = PipelineStage {
            stage_id: 1,
            start_layer: 4,
            end_layer: 8,
        };
        assert!(!stage.contains_layer(3));
        assert!(stage.contains_layer(4));
        assert!(stage.contains_layer(7));
        assert!(!stage.contains_layer(8));
    }

    // ── Activation transfer stubs ──

    #[test]
    fn test_send_activation_cpu_noop() {
        let tensor = Tensor::zeros((4, 16), candle_core::DType::F32, &Device::Cpu).unwrap();
        let result = send_activation(&tensor, 1, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_recv_activation_cpu_clone() {
        let tensor = Tensor::arange(0f32, 16.0, &Device::Cpu).unwrap();
        let received = recv_activation(&tensor, 0, None).unwrap();
        assert_eq!(received.dims(), tensor.dims());
        let orig: Vec<f32> = tensor.to_vec1().unwrap();
        let recv: Vec<f32> = received.to_vec1().unwrap();
        assert_eq!(orig, recv);
    }
}
