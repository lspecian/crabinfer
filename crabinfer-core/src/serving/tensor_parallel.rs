//! Tensor parallelism for multi-GPU inference.
//!
//! Tensor parallelism (TP) shards model weights across multiple GPUs and uses
//! NCCL AllReduce for synchronization. This enables serving models too large
//! for a single GPU.
//!
//! # Sharding strategy
//!
//! For each transformer layer:
//! - **Column-parallel** (Q, K, V, gate, up projections): shard output dim (dim=0).
//!   Each GPU computes a slice of the output. No communication needed for
//!   column-parallel followed by row-parallel pairs.
//! - **Row-parallel** (O, down projections): shard input dim (dim=1).
//!   Each GPU computes a partial result, then AllReduce to sum across ranks.
//! - **Replicated** (norms, embeddings): full copy on each rank.
//!
//! AllReduce is placed after O_proj and after down_proj in each layer
//! (2 AllReduce operations per transformer layer).
//!
//! # Single-rank mode
//!
//! When `world_size == 1`, all operations are identity/no-ops. This allows
//! the same code path to handle both single-GPU and multi-GPU inference.

use candle_core::{DType, Result, Tensor};

use super::nccl::NcclComm;

// ─── Configuration ────────────────────────────────────────────────────────

/// Tensor parallel configuration.
#[derive(Debug, Clone)]
pub struct TensorParallelConfig {
    /// Number of GPUs participating in tensor parallelism.
    pub world_size: usize,
    /// This GPU's rank (0..world_size-1).
    pub rank: usize,
    /// NCCL unique ID bytes for communicator initialization.
    /// `None` when world_size == 1 (no communication needed).
    pub nccl_id: Option<Vec<u8>>,
}

impl TensorParallelConfig {
    /// Create a single-rank (no parallelism) configuration.
    pub fn single() -> Self {
        Self {
            world_size: 1,
            rank: 0,
            nccl_id: None,
        }
    }

    /// Create a multi-rank configuration.
    ///
    /// # Errors
    /// Returns an error if:
    /// - `world_size` is 0
    /// - `rank >= world_size`
    /// - `nccl_id` is `None` when `world_size > 1`
    pub fn new(world_size: usize, rank: usize, nccl_id: Option<Vec<u8>>) -> Result<Self> {
        if world_size == 0 {
            return Err(candle_core::Error::Msg(
                "tensor parallel world_size must be >= 1".to_string(),
            ));
        }
        if rank >= world_size {
            return Err(candle_core::Error::Msg(format!(
                "tensor parallel rank ({rank}) must be < world_size ({world_size})"
            )));
        }
        if world_size > 1 && nccl_id.is_none() {
            return Err(candle_core::Error::Msg(
                "NCCL unique ID is required for world_size > 1".to_string(),
            ));
        }
        Ok(Self {
            world_size,
            rank,
            nccl_id,
        })
    }

    /// Whether this config represents single-GPU (no actual parallelism).
    pub fn is_single(&self) -> bool {
        self.world_size == 1
    }
}

impl Default for TensorParallelConfig {
    fn default() -> Self {
        Self::single()
    }
}

// ─── Tensor parallel group ───────────────────────────────────────────────

/// A tensor parallel group managing an NCCL communicator.
///
/// The group is created once at engine startup and shared across all
/// forward passes. It provides AllReduce and AllGather primitives
/// operating on candle `Tensor`s.
pub struct TensorParallelGroup {
    /// Configuration for this group.
    pub config: TensorParallelConfig,
    /// NCCL communicator (only meaningful when world_size > 1 and cuda feature enabled).
    #[allow(dead_code)]
    comm: NcclComm,
}

// TensorParallelGroup is Send + Sync because NcclComm is Send + Sync
// and config is just data.
unsafe impl Send for TensorParallelGroup {}
unsafe impl Sync for TensorParallelGroup {}

impl TensorParallelGroup {
    /// Create a new tensor parallel group.
    ///
    /// For `world_size == 1`, creates a no-op group (no NCCL initialization).
    /// For `world_size > 1`, initializes an NCCL communicator.
    pub fn new(config: TensorParallelConfig) -> Result<Self> {
        let nccl_id = config
            .nccl_id
            .clone()
            .unwrap_or_else(|| NcclComm::get_unique_id().unwrap_or_else(|_| vec![0u8; 128]));

        let comm = NcclComm::init(&nccl_id, config.world_size, config.rank)?;

        Ok(Self { config, comm })
    }

    /// AllReduce sum across all ranks.
    ///
    /// Used after row-parallel layers (O_proj, down_proj) to sum partial
    /// results from each rank.
    ///
    /// When `world_size == 1`, returns the input tensor unchanged (no copy).
    pub fn all_reduce_sum(&self, tensor: &Tensor) -> Result<Tensor> {
        if self.config.is_single() {
            return Ok(tensor.clone());
        }

        // On CUDA: dispatch NCCL AllReduce
        #[cfg(feature = "cuda")]
        {
            self.nccl_all_reduce(tensor)
        }

        // On non-CUDA: world_size > 1 should have been rejected at init time,
        // but handle gracefully just in case.
        #[cfg(not(feature = "cuda"))]
        {
            Err(candle_core::Error::Msg(
                "AllReduce requires CUDA for world_size > 1".to_string(),
            ))
        }
    }

    /// AllGather across all ranks.
    ///
    /// Each rank contributes its local tensor, and the result is the
    /// concatenation of all ranks' tensors along dimension 0.
    /// Used for gathering sharded embeddings.
    ///
    /// When `world_size == 1`, returns the input tensor unchanged.
    pub fn all_gather(&self, tensor: &Tensor) -> Result<Tensor> {
        if self.config.is_single() {
            return Ok(tensor.clone());
        }

        #[cfg(feature = "cuda")]
        {
            self.nccl_all_gather(tensor)
        }

        #[cfg(not(feature = "cuda"))]
        {
            Err(candle_core::Error::Msg(
                "AllGather requires CUDA for world_size > 1".to_string(),
            ))
        }
    }

    /// NCCL AllReduce implementation (CUDA only).
    #[cfg(feature = "cuda")]
    fn nccl_all_reduce(&self, tensor: &Tensor) -> Result<Tensor> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;

        let device = tensor.device();
        let elem_count = tensor.elem_count();

        // Allocate output tensor with same shape/dtype
        let output = Tensor::zeros(tensor.shape(), tensor.dtype(), device)?;

        // Ensure tensor is contiguous
        let tensor = tensor.contiguous()?;

        // Extract raw CUDA pointers from candle's storage
        {
            let (in_storage, _in_layout) = tensor.storage_and_layout();
            let (out_storage, _out_layout) = output.storage_and_layout();

            let in_cuda = match &*in_storage {
                candle_core::Storage::Cuda(s) => s,
                _ => {
                    return Err(candle_core::Error::Msg(
                        "AllReduce: tensor must be on a CUDA device".to_string(),
                    ))
                }
            };
            let out_cuda = match &*out_storage {
                candle_core::Storage::Cuda(s) => s,
                _ => {
                    return Err(candle_core::Error::Msg(
                        "AllReduce: output must be on a CUDA device".to_string(),
                    ))
                }
            };

            let stream = in_cuda.device.cuda_stream();
            let in_slice = in_cuda.as_cuda_slice::<u8>()?;
            let out_slice = out_cuda.as_cuda_slice::<u8>()?;
            let (in_ptr, _in_sync) = in_slice.device_ptr(&stream);
            let (out_ptr, _out_sync) = out_slice.device_ptr(&stream);

            let input_ptr = in_ptr as *const std::ffi::c_void;
            let output_ptr = out_ptr as *mut std::ffi::c_void;

            // Use null stream for default CUDA stream
            let nccl_stream = std::ptr::null_mut();

            self.comm
                .all_reduce_sum_f32(input_ptr, output_ptr, elem_count, nccl_stream)?;
        }

        Ok(output)
    }

    /// NCCL AllGather implementation (CUDA only).
    #[cfg(feature = "cuda")]
    fn nccl_all_gather(&self, tensor: &Tensor) -> Result<Tensor> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;

        let device = tensor.device();
        let send_count = tensor.elem_count();
        let world_size = self.config.world_size;

        // Output shape: first dim scaled by world_size
        let mut output_shape: Vec<usize> = tensor.dims().to_vec();
        if output_shape.is_empty() {
            return Err(candle_core::Error::Msg(
                "AllGather: cannot gather a scalar tensor".to_string(),
            ));
        }
        output_shape[0] *= world_size;

        let output = Tensor::zeros(output_shape.as_slice(), tensor.dtype(), device)?;
        let tensor = tensor.contiguous()?;

        // Extract raw CUDA pointers from candle's storage
        {
            let (in_storage, _in_layout) = tensor.storage_and_layout();
            let (out_storage, _out_layout) = output.storage_and_layout();

            let in_cuda = match &*in_storage {
                candle_core::Storage::Cuda(s) => s,
                _ => {
                    return Err(candle_core::Error::Msg(
                        "AllGather: tensor must be on a CUDA device".to_string(),
                    ))
                }
            };
            let out_cuda = match &*out_storage {
                candle_core::Storage::Cuda(s) => s,
                _ => {
                    return Err(candle_core::Error::Msg(
                        "AllGather: output must be on a CUDA device".to_string(),
                    ))
                }
            };

            let stream = in_cuda.device.cuda_stream();
            let in_slice = in_cuda.as_cuda_slice::<u8>()?;
            let out_slice = out_cuda.as_cuda_slice::<u8>()?;
            let (in_ptr, _in_sync) = in_slice.device_ptr(&stream);
            let (out_ptr, _out_sync) = out_slice.device_ptr(&stream);

            let input_ptr = in_ptr as *const std::ffi::c_void;
            let output_ptr = out_ptr as *mut std::ffi::c_void;

            let nccl_stream = std::ptr::null_mut();

            self.comm
                .all_gather_f32(input_ptr, output_ptr, send_count, nccl_stream)?;
        }

        Ok(output)
    }

    /// This GPU's rank.
    pub fn rank(&self) -> usize {
        self.config.rank
    }

    /// Total number of GPUs.
    pub fn world_size(&self) -> usize {
        self.config.world_size
    }
}

// ─── Weight sharding ──────────────────────────────────────────────────────

/// Weight sharding strategies for tensor parallelism.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardStrategy {
    /// Column-parallel: shard along output dimension (dim=0).
    ///
    /// Used for QKV projections and gate/up projections. Each GPU gets
    /// `output_features / world_size` output features. No communication
    /// is needed after column-parallel layers when paired with row-parallel.
    ColumnParallel,

    /// Row-parallel: shard along input dimension (dim=1).
    ///
    /// Used for O projection and down projection. Each GPU gets
    /// `input_features / world_size` input features and computes a partial
    /// result. AllReduce sum is required after the matmul.
    RowParallel,

    /// Replicated: full copy on each rank.
    ///
    /// Used for normalization weights, biases, and embedding tables.
    /// No sharding or communication needed.
    Replicated,
}

/// Shard a weight tensor for tensor parallelism.
///
/// # Arguments
/// - `weight`: The full (un-sharded) weight tensor. Shape `[out_features, in_features]`
///   for linear layers.
/// - `strategy`: How to shard this weight.
/// - `rank`: This GPU's rank.
/// - `world_size`: Total number of GPUs.
///
/// # Returns
/// The sharded weight tensor for this rank.
///
/// # Sharding details
/// - `ColumnParallel`: Splits dim=0 into `world_size` equal chunks, returns chunk `rank`.
///   Shape: `[out_features/world_size, in_features]`.
/// - `RowParallel`: Splits dim=1 into `world_size` equal chunks, returns chunk `rank`.
///   Shape: `[out_features, in_features/world_size]`.
/// - `Replicated`: Returns the full weight unchanged.
pub fn shard_weight(
    weight: &Tensor,
    strategy: ShardStrategy,
    rank: usize,
    world_size: usize,
) -> Result<Tensor> {
    if world_size <= 1 {
        return Ok(weight.clone());
    }

    match strategy {
        ShardStrategy::Replicated => Ok(weight.clone()),

        ShardStrategy::ColumnParallel => {
            // Shard along dim=0 (output dimension)
            let dims = weight.dims();
            if dims.is_empty() {
                return Err(candle_core::Error::Msg(
                    "Cannot column-shard a scalar weight".to_string(),
                ));
            }
            let out_features = dims[0];
            if out_features % world_size != 0 {
                return Err(candle_core::Error::Msg(format!(
                    "ColumnParallel: output features ({out_features}) must be divisible by world_size ({world_size})"
                )));
            }
            let shard_size = out_features / world_size;
            let start = rank * shard_size;
            weight.narrow(0, start, shard_size)
        }

        ShardStrategy::RowParallel => {
            // Shard along dim=1 (input dimension)
            let dims = weight.dims();
            if dims.len() < 2 {
                return Err(candle_core::Error::Msg(
                    "RowParallel requires at least a 2D weight tensor".to_string(),
                ));
            }
            let in_features = dims[1];
            if in_features % world_size != 0 {
                return Err(candle_core::Error::Msg(format!(
                    "RowParallel: input features ({in_features}) must be divisible by world_size ({world_size})"
                )));
            }
            let shard_size = in_features / world_size;
            let start = rank * shard_size;
            weight.narrow(1, start, shard_size)
        }
    }
}

/// Shard a 1D bias or normalization weight for column-parallel layers.
///
/// Column-parallel layers produce `out_features / world_size` outputs per rank,
/// so biases must be sharded identically along their only dimension.
///
/// For row-parallel layers, biases are replicated (each rank has the full bias,
/// and the AllReduce sums the partial matmul results before adding bias).
pub fn shard_bias(
    bias: &Tensor,
    strategy: ShardStrategy,
    rank: usize,
    world_size: usize,
) -> Result<Tensor> {
    if world_size <= 1 {
        return Ok(bias.clone());
    }

    match strategy {
        ShardStrategy::Replicated | ShardStrategy::RowParallel => Ok(bias.clone()),

        ShardStrategy::ColumnParallel => {
            let size = bias.elem_count();
            if size % world_size != 0 {
                return Err(candle_core::Error::Msg(format!(
                    "ColumnParallel bias: size ({size}) must be divisible by world_size ({world_size})"
                )));
            }
            let shard_size = size / world_size;
            let start = rank * shard_size;
            bias.narrow(0, start, shard_size)
        }
    }
}

/// Determine the sharding strategy for a given layer weight name.
///
/// Maps GGUF/safetensors weight names to their tensor parallel sharding
/// strategy. This covers the standard Llama-family architecture.
///
/// # Returns
/// The appropriate `ShardStrategy` for the given weight name.
pub fn strategy_for_weight(name: &str) -> ShardStrategy {
    // Column-parallel: QKV projections, gate/up MLPs
    if name.contains("attn_q.weight")
        || name.contains("attn_k.weight")
        || name.contains("attn_v.weight")
        || name.contains("q_proj")
        || name.contains("k_proj")
        || name.contains("v_proj")
        || name.contains("ffn_gate.weight")
        || name.contains("ffn_up.weight")
        || name.contains("gate_proj")
        || name.contains("up_proj")
    {
        return ShardStrategy::ColumnParallel;
    }

    // Row-parallel: output projection, down MLP
    if name.contains("attn_output.weight")
        || name.contains("o_proj")
        || name.contains("ffn_down.weight")
        || name.contains("down_proj")
    {
        return ShardStrategy::RowParallel;
    }

    // Everything else: norms, embeddings, lm_head -> replicated
    ShardStrategy::Replicated
}

// ─── Helper: compute per-rank head counts ─────────────────────────────────

/// Compute the number of attention heads assigned to a given rank.
///
/// For GQA models where `num_kv_heads < num_heads`, both Q heads and KV heads
/// are partitioned across ranks. The Q-to-KV head ratio is preserved.
pub fn heads_per_rank(num_heads: usize, world_size: usize) -> Result<usize> {
    if num_heads % world_size != 0 {
        return Err(candle_core::Error::Msg(format!(
            "num_heads ({num_heads}) must be divisible by world_size ({world_size})"
        )));
    }
    Ok(num_heads / world_size)
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    // ── Config tests ──

    #[test]
    fn test_single_config() {
        let config = TensorParallelConfig::single();
        assert_eq!(config.world_size, 1);
        assert_eq!(config.rank, 0);
        assert!(config.is_single());
        assert!(config.nccl_id.is_none());
    }

    #[test]
    fn test_default_config_is_single() {
        let config = TensorParallelConfig::default();
        assert!(config.is_single());
    }

    #[test]
    fn test_config_validation_zero_world_size() {
        let result = TensorParallelConfig::new(0, 0, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_validation_rank_out_of_bounds() {
        let result = TensorParallelConfig::new(2, 2, Some(vec![0u8; 128]));
        assert!(result.is_err());
    }

    #[test]
    fn test_config_validation_missing_nccl_id() {
        let result = TensorParallelConfig::new(2, 0, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_validation_valid_multi_rank() {
        let result = TensorParallelConfig::new(4, 2, Some(vec![0u8; 128]));
        assert!(result.is_ok());
        let config = result.unwrap();
        assert_eq!(config.world_size, 4);
        assert_eq!(config.rank, 2);
        assert!(!config.is_single());
    }

    #[test]
    fn test_config_single_rank_no_nccl_id_ok() {
        let result = TensorParallelConfig::new(1, 0, None);
        assert!(result.is_ok());
    }

    // ── Shard weight tests ──

    #[test]
    fn test_shard_weight_single_rank_identity() {
        let weight = Tensor::randn(0f32, 1.0, (64, 32), &Device::Cpu).unwrap();
        let sharded = shard_weight(&weight, ShardStrategy::ColumnParallel, 0, 1).unwrap();
        assert_eq!(sharded.dims(), weight.dims());
    }

    #[test]
    fn test_shard_weight_column_parallel() {
        let weight = Tensor::arange(0f32, 128.0, &Device::Cpu)
            .unwrap()
            .reshape((8, 16))
            .unwrap();

        // world_size=2: each rank gets 4 rows
        let shard0 = shard_weight(&weight, ShardStrategy::ColumnParallel, 0, 2).unwrap();
        let shard1 = shard_weight(&weight, ShardStrategy::ColumnParallel, 1, 2).unwrap();

        assert_eq!(shard0.dims(), &[4, 16]);
        assert_eq!(shard1.dims(), &[4, 16]);

        // Verify shard0 contains first 4 rows, shard1 contains last 4 rows
        let s0: Vec<f32> = shard0.flatten_all().unwrap().to_vec1().unwrap();
        let s1: Vec<f32> = shard1.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(s0[0], 0.0); // First element of row 0
        assert_eq!(s1[0], 64.0); // First element of row 4
    }

    #[test]
    fn test_shard_weight_row_parallel() {
        let weight = Tensor::arange(0f32, 128.0, &Device::Cpu)
            .unwrap()
            .reshape((8, 16))
            .unwrap();

        // world_size=2: each rank gets 8 columns
        let shard0 = shard_weight(&weight, ShardStrategy::RowParallel, 0, 2).unwrap();
        let shard1 = shard_weight(&weight, ShardStrategy::RowParallel, 1, 2).unwrap();

        assert_eq!(shard0.dims(), &[8, 8]);
        assert_eq!(shard1.dims(), &[8, 8]);

        // Verify shard0 contains first 8 columns, shard1 contains last 8 columns
        let s0: Vec<f32> = shard0.flatten_all().unwrap().to_vec1().unwrap();
        let s1: Vec<f32> = shard1.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(s0[0], 0.0); // row0, col0
        assert_eq!(s0[1], 1.0); // row0, col1
        assert_eq!(s1[0], 8.0); // row0, col8
    }

    #[test]
    fn test_shard_weight_replicated() {
        let weight = Tensor::randn(0f32, 1.0, (64, 32), &Device::Cpu).unwrap();
        let shard0 = shard_weight(&weight, ShardStrategy::Replicated, 0, 4).unwrap();
        let shard1 = shard_weight(&weight, ShardStrategy::Replicated, 1, 4).unwrap();

        // Both ranks get the full weight
        assert_eq!(shard0.dims(), &[64, 32]);
        assert_eq!(shard1.dims(), &[64, 32]);
    }

    #[test]
    fn test_shard_weight_column_not_divisible() {
        let weight = Tensor::randn(0f32, 1.0, (7, 16), &Device::Cpu).unwrap();
        let result = shard_weight(&weight, ShardStrategy::ColumnParallel, 0, 2);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("divisible"));
    }

    #[test]
    fn test_shard_weight_row_not_divisible() {
        let weight = Tensor::randn(0f32, 1.0, (16, 7), &Device::Cpu).unwrap();
        let result = shard_weight(&weight, ShardStrategy::RowParallel, 0, 2);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("divisible"));
    }

    #[test]
    fn test_shard_weight_4_way_column() {
        let weight = Tensor::randn(0f32, 1.0, (128, 64), &Device::Cpu).unwrap();
        for rank in 0..4 {
            let shard = shard_weight(&weight, ShardStrategy::ColumnParallel, rank, 4).unwrap();
            assert_eq!(shard.dims(), &[32, 64]); // 128/4 = 32
        }
    }

    #[test]
    fn test_shard_weight_4_way_row() {
        let weight = Tensor::randn(0f32, 1.0, (128, 64), &Device::Cpu).unwrap();
        for rank in 0..4 {
            let shard = shard_weight(&weight, ShardStrategy::RowParallel, rank, 4).unwrap();
            assert_eq!(shard.dims(), &[128, 16]); // 64/4 = 16
        }
    }

    #[test]
    fn test_column_shards_cover_full_weight() {
        // Verify that concatenating all column shards reproduces the original weight
        let weight = Tensor::arange(0f32, 64.0, &Device::Cpu)
            .unwrap()
            .reshape((4, 16))
            .unwrap();

        let world_size = 2;
        let shard0 = shard_weight(&weight, ShardStrategy::ColumnParallel, 0, world_size).unwrap();
        let shard1 = shard_weight(&weight, ShardStrategy::ColumnParallel, 1, world_size).unwrap();

        let reconstructed = Tensor::cat(&[&shard0, &shard1], 0).unwrap();
        let orig: Vec<f32> = weight.flatten_all().unwrap().to_vec1().unwrap();
        let recon: Vec<f32> = reconstructed.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(orig, recon);
    }

    #[test]
    fn test_row_shards_cover_full_weight() {
        // Verify that concatenating all row shards reproduces the original weight
        let weight = Tensor::arange(0f32, 64.0, &Device::Cpu)
            .unwrap()
            .reshape((4, 16))
            .unwrap();

        let world_size = 2;
        let shard0 = shard_weight(&weight, ShardStrategy::RowParallel, 0, world_size).unwrap();
        let shard1 = shard_weight(&weight, ShardStrategy::RowParallel, 1, world_size).unwrap();

        let reconstructed = Tensor::cat(&[&shard0, &shard1], 1).unwrap();
        let orig: Vec<f32> = weight.flatten_all().unwrap().to_vec1().unwrap();
        let recon: Vec<f32> = reconstructed.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(orig, recon);
    }

    // ── Shard bias tests ──

    #[test]
    fn test_shard_bias_column_parallel() {
        let bias = Tensor::arange(0f32, 32.0, &Device::Cpu).unwrap();
        let shard0 = shard_bias(&bias, ShardStrategy::ColumnParallel, 0, 4).unwrap();
        let shard1 = shard_bias(&bias, ShardStrategy::ColumnParallel, 1, 4).unwrap();

        assert_eq!(shard0.dims(), &[8]); // 32/4
        assert_eq!(shard1.dims(), &[8]);

        let s0: Vec<f32> = shard0.to_vec1().unwrap();
        let s1: Vec<f32> = shard1.to_vec1().unwrap();
        assert_eq!(s0[0], 0.0);
        assert_eq!(s1[0], 8.0);
    }

    #[test]
    fn test_shard_bias_row_parallel_is_replicated() {
        let bias = Tensor::arange(0f32, 16.0, &Device::Cpu).unwrap();
        let shard = shard_bias(&bias, ShardStrategy::RowParallel, 0, 4).unwrap();
        assert_eq!(shard.dims(), &[16]); // Full copy
    }

    #[test]
    fn test_shard_bias_replicated() {
        let bias = Tensor::randn(0f32, 1.0, 64, &Device::Cpu).unwrap();
        let shard = shard_bias(&bias, ShardStrategy::Replicated, 2, 4).unwrap();
        assert_eq!(shard.dims(), &[64]);
    }

    // ── Strategy for weight tests ──

    #[test]
    fn test_strategy_for_qkv_weights() {
        assert_eq!(
            strategy_for_weight("blk.0.attn_q.weight"),
            ShardStrategy::ColumnParallel
        );
        assert_eq!(
            strategy_for_weight("blk.0.attn_k.weight"),
            ShardStrategy::ColumnParallel
        );
        assert_eq!(
            strategy_for_weight("blk.0.attn_v.weight"),
            ShardStrategy::ColumnParallel
        );
        assert_eq!(
            strategy_for_weight("model.layers.0.self_attn.q_proj.weight"),
            ShardStrategy::ColumnParallel
        );
        assert_eq!(
            strategy_for_weight("model.layers.0.self_attn.k_proj.weight"),
            ShardStrategy::ColumnParallel
        );
        assert_eq!(
            strategy_for_weight("model.layers.0.self_attn.v_proj.weight"),
            ShardStrategy::ColumnParallel
        );
    }

    #[test]
    fn test_strategy_for_output_projection() {
        assert_eq!(
            strategy_for_weight("blk.0.attn_output.weight"),
            ShardStrategy::RowParallel
        );
        assert_eq!(
            strategy_for_weight("model.layers.0.self_attn.o_proj.weight"),
            ShardStrategy::RowParallel
        );
    }

    #[test]
    fn test_strategy_for_mlp_weights() {
        assert_eq!(
            strategy_for_weight("blk.0.ffn_gate.weight"),
            ShardStrategy::ColumnParallel
        );
        assert_eq!(
            strategy_for_weight("blk.0.ffn_up.weight"),
            ShardStrategy::ColumnParallel
        );
        assert_eq!(
            strategy_for_weight("blk.0.ffn_down.weight"),
            ShardStrategy::RowParallel
        );
        assert_eq!(
            strategy_for_weight("model.layers.0.mlp.gate_proj.weight"),
            ShardStrategy::ColumnParallel
        );
        assert_eq!(
            strategy_for_weight("model.layers.0.mlp.up_proj.weight"),
            ShardStrategy::ColumnParallel
        );
        assert_eq!(
            strategy_for_weight("model.layers.0.mlp.down_proj.weight"),
            ShardStrategy::RowParallel
        );
    }

    #[test]
    fn test_strategy_for_norms_and_embeddings() {
        assert_eq!(
            strategy_for_weight("blk.0.attn_norm.weight"),
            ShardStrategy::Replicated
        );
        assert_eq!(
            strategy_for_weight("blk.0.ffn_norm.weight"),
            ShardStrategy::Replicated
        );
        assert_eq!(
            strategy_for_weight("output_norm.weight"),
            ShardStrategy::Replicated
        );
        assert_eq!(
            strategy_for_weight("token_embd.weight"),
            ShardStrategy::Replicated
        );
        assert_eq!(
            strategy_for_weight("output.weight"),
            ShardStrategy::Replicated
        );
    }

    // ── Heads per rank tests ──

    #[test]
    fn test_heads_per_rank_basic() {
        assert_eq!(heads_per_rank(32, 1).unwrap(), 32);
        assert_eq!(heads_per_rank(32, 2).unwrap(), 16);
        assert_eq!(heads_per_rank(32, 4).unwrap(), 8);
        assert_eq!(heads_per_rank(32, 8).unwrap(), 4);
    }

    #[test]
    fn test_heads_per_rank_not_divisible() {
        let result = heads_per_rank(32, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_heads_per_rank_gqa() {
        // GQA: 32 Q heads, 8 KV heads, TP=4
        assert_eq!(heads_per_rank(32, 4).unwrap(), 8); // Q heads per rank
        assert_eq!(heads_per_rank(8, 4).unwrap(), 2); // KV heads per rank
    }

    // ── Single-rank TP group (no-op) ──

    #[test]
    fn test_tp_group_single_rank() {
        let config = TensorParallelConfig::single();
        let group = TensorParallelGroup::new(config).unwrap();
        assert_eq!(group.rank(), 0);
        assert_eq!(group.world_size(), 1);
    }

    #[test]
    fn test_tp_group_all_reduce_single_rank_identity() {
        let config = TensorParallelConfig::single();
        let group = TensorParallelGroup::new(config).unwrap();

        let tensor = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &Device::Cpu).unwrap();
        let result = group.all_reduce_sum(&tensor).unwrap();

        let orig: Vec<f32> = tensor.to_vec1().unwrap();
        let res: Vec<f32> = result.to_vec1().unwrap();
        assert_eq!(orig, res);
    }

    #[test]
    fn test_tp_group_all_gather_single_rank_identity() {
        let config = TensorParallelConfig::single();
        let group = TensorParallelGroup::new(config).unwrap();

        let tensor = Tensor::randn(0f32, 1.0, (4, 16), &Device::Cpu).unwrap();
        let result = group.all_gather(&tensor).unwrap();
        assert_eq!(result.dims(), tensor.dims());
    }

    // ── Simulated sharded forward pass (single rank = identity) ──

    #[test]
    fn test_sharded_forward_single_rank_matches_unsharded() {
        // Simulate a simple linear layer: output = input @ weight.T
        // With TP=1 (single rank), sharded should be identical to unsharded.
        let dev = &Device::Cpu;
        let batch = 4;
        let in_features = 64;
        let out_features = 32;

        let input = Tensor::randn(0f32, 1.0, (batch, in_features), dev).unwrap();
        let weight = Tensor::randn(0f32, 1.0, (out_features, in_features), dev).unwrap();

        // Unsharded forward: output = input @ weight.T
        let output_ref = input.matmul(&weight.t().unwrap()).unwrap();

        // Sharded forward (single rank)
        let sharded_weight =
            shard_weight(&weight, ShardStrategy::ColumnParallel, 0, 1).unwrap();
        let output_tp = input.matmul(&sharded_weight.t().unwrap()).unwrap();

        assert_eq!(output_ref.dims(), output_tp.dims());

        let ref_data: Vec<f32> = output_ref.flatten_all().unwrap().to_vec1().unwrap();
        let tp_data: Vec<f32> = output_tp.flatten_all().unwrap().to_vec1().unwrap();
        for (r, t) in ref_data.iter().zip(tp_data.iter()) {
            assert!(
                (r - t).abs() < 1e-5,
                "mismatch: ref={r}, tp={t}"
            );
        }
    }

    #[test]
    fn test_column_parallel_matmul_concat_matches_full() {
        // Verify: column-parallel sharded matmul + concat = full matmul
        // Full: output = input @ W.T  where W = [out, in]
        // TP=2: W_0 = W[0:out/2, :], W_1 = W[out/2:out, :]
        //   output_0 = input @ W_0.T  -> [batch, out/2]
        //   output_1 = input @ W_1.T  -> [batch, out/2]
        //   output = cat(output_0, output_1, dim=1)  -> [batch, out]
        let dev = &Device::Cpu;
        let batch = 4;
        let in_features = 32;
        let out_features = 16;
        let world_size = 2;

        let input = Tensor::randn(0f32, 1.0, (batch, in_features), dev).unwrap();
        let weight = Tensor::randn(0f32, 1.0, (out_features, in_features), dev).unwrap();

        // Full forward
        let output_full = input.matmul(&weight.t().unwrap()).unwrap();

        // Sharded forward
        let w0 = shard_weight(&weight, ShardStrategy::ColumnParallel, 0, world_size).unwrap();
        let w1 = shard_weight(&weight, ShardStrategy::ColumnParallel, 1, world_size).unwrap();
        let out0 = input.matmul(&w0.t().unwrap()).unwrap();
        let out1 = input.matmul(&w1.t().unwrap()).unwrap();
        let output_tp = Tensor::cat(&[&out0, &out1], 1).unwrap();

        assert_eq!(output_full.dims(), output_tp.dims());

        let full_data: Vec<f32> = output_full.flatten_all().unwrap().to_vec1().unwrap();
        let tp_data: Vec<f32> = output_tp.flatten_all().unwrap().to_vec1().unwrap();
        for (f, t) in full_data.iter().zip(tp_data.iter()) {
            assert!((f - t).abs() < 1e-4, "mismatch: full={f}, tp={t}");
        }
    }

    #[test]
    fn test_row_parallel_matmul_sum_matches_full() {
        // Verify: row-parallel sharded matmul + sum = full matmul
        // Full: output = input @ W.T  where W = [out, in]
        // TP=2: W_0 = W[:, 0:in/2], W_1 = W[:, in/2:in]
        //   input_0 = input[:, 0:in/2], input_1 = input[:, in/2:in]
        //   output_0 = input_0 @ W_0.T  -> [batch, out]
        //   output_1 = input_1 @ W_1.T  -> [batch, out]
        //   output = output_0 + output_1  (AllReduce sum)
        let dev = &Device::Cpu;
        let batch = 4;
        let in_features = 32;
        let out_features = 16;
        let world_size = 2;

        let input = Tensor::randn(0f32, 1.0, (batch, in_features), dev).unwrap();
        let weight = Tensor::randn(0f32, 1.0, (out_features, in_features), dev).unwrap();

        // Full forward
        let output_full = input.matmul(&weight.t().unwrap()).unwrap();

        // Sharded forward
        let w0 = shard_weight(&weight, ShardStrategy::RowParallel, 0, world_size).unwrap();
        let w1 = shard_weight(&weight, ShardStrategy::RowParallel, 1, world_size).unwrap();

        // Each rank gets its slice of the input (matching the sharded weight's input dim)
        let input_half = in_features / world_size;
        let in0 = input.narrow(1, 0, input_half).unwrap();
        let in1 = input.narrow(1, input_half, input_half).unwrap();

        let out0 = in0.matmul(&w0.t().unwrap()).unwrap();
        let out1 = in1.matmul(&w1.t().unwrap()).unwrap();

        // AllReduce sum
        let output_tp = (&out0 + &out1).unwrap();

        assert_eq!(output_full.dims(), output_tp.dims());

        let full_data: Vec<f32> = output_full.flatten_all().unwrap().to_vec1().unwrap();
        let tp_data: Vec<f32> = output_tp.flatten_all().unwrap().to_vec1().unwrap();
        for (f, t) in full_data.iter().zip(tp_data.iter()) {
            assert!((f - t).abs() < 1e-4, "mismatch: full={f}, tp={t}");
        }
    }
}
