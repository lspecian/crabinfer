//! Kernel backend abstraction for paged attention operations.
//!
//! The `KernelBackend` trait decouples paged attention operations from a specific
//! GPU API. Implementations exist for Metal (Apple Silicon), CUDA (NVIDIA GPUs),
//! and CPU (fallback for CI / testing).

use candle_core::{DType, Device, Result, Tensor};

/// Abstract interface for the three core paged-attention kernel operations.
///
/// Every GPU backend (Metal, CUDA) and the CPU fallback implement this trait.
/// The `ForwardContext` carries a `&dyn KernelBackend` so attention layers
/// dispatch through the trait without compile-time feature flags.
pub trait KernelBackend: Send + Sync {
    /// Human-readable name for logging (e.g. "metal", "cuda", "cpu").
    fn name(&self) -> &'static str;

    /// Run paged attention over batched queries and a paged KV cache.
    ///
    /// Writes the result into `output` (pre-allocated).
    ///
    /// # Arguments
    /// - `output`:       `[num_seqs, num_heads, head_size]`
    /// - `query`:        `[num_seqs, num_heads, head_size]`
    /// - `key_cache`:    paged key cache (layout is backend-specific)
    /// - `value_cache`:  paged value cache
    /// - `block_tables`: `[num_seqs, max_blocks_per_seq]` i32
    /// - `context_lens`: `[num_seqs]` i32
    /// - `config`:       attention parameters
    fn paged_attention(
        &self,
        output: &Tensor,
        query: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        block_tables: &Tensor,
        context_lens: &Tensor,
        config: &PagedAttentionConfig,
    ) -> Result<()>;

    /// Write new K/V tokens into the paged KV cache.
    ///
    /// # Arguments
    /// - `key`:         `[num_tokens, num_kv_heads, head_size]`
    /// - `value`:       `[num_tokens, num_kv_heads, head_size]`
    /// - `key_cache`:   paged key cache (mutated in place)
    /// - `value_cache`: paged value cache (mutated in place)
    /// - `slot_mapping`: `[num_tokens]` i32 — physical slot index per token
    fn reshape_and_cache(
        &self,
        key: &Tensor,
        value: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        slot_mapping: &Tensor,
    ) -> Result<()>;

    /// Copy KV cache blocks for prefix sharing / CoW.
    ///
    /// # Arguments
    /// - `key_cache`:       paged key cache (mutated in place)
    /// - `value_cache`:     paged value cache (mutated in place)
    /// - `block_mapping`:   flat i32 tensor of `[src, dst]` pairs
    /// - `numel_per_block`: number of elements in each block
    fn copy_blocks(
        &self,
        key_cache: &Tensor,
        value_cache: &Tensor,
        block_mapping: &Tensor,
        numel_per_block: usize,
    ) -> Result<()>;

    /// Allocate KV cache tensors for all layers.
    ///
    /// Returns `(key_caches, value_caches)` — one pair per layer.
    ///
    /// K cache layout: `[num_blocks, num_kv_heads, head_size/x, block_size, x]`
    /// V cache layout: `[num_blocks, num_kv_heads, head_size, block_size]`
    fn allocate_kv_caches(
        &self,
        num_layers: usize,
        num_blocks: usize,
        num_kv_heads: usize,
        head_size: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<(Vec<Tensor>, Vec<Tensor>)>;

    // ─── Fused kernels (optional, with default fallbacks) ────────────────

    /// FP8 E4M3 dequantize + GEMM.
    ///
    /// Default: dequantize FP8 weights to the input dtype, then perform standard matmul.
    /// On Hopper GPUs (H100/H200), the CUDA backend can override this with
    /// `cublasLtMatmul` using native `CUDA_R_8F_E4M3` input types.
    ///
    /// # Arguments
    /// - `input`: `[M, K]` activations in FP16/FP32
    /// - `weight_fp8`: `[N, K]` weights stored as U8 (FP8 E4M3 bit patterns)
    /// - `scale`: `[1]` or `[N]` per-tensor or per-channel scale factors
    /// - `bias`: optional `[N]` bias
    /// - `per_channel`: whether scale is per-channel
    ///
    /// # Returns
    /// `[M, N]` output tensor in input dtype
    fn fp8_gemm(
        &self,
        _input: &Tensor,
        _weight_fp8: &Tensor,
        _scale: &Tensor,
        _bias: Option<&Tensor>,
        _per_channel: bool,
    ) -> Result<Tensor> {
        candle_core::bail!("fp8_gemm not supported on {} backend", self.name())
    }

    /// Fused INT4 dequant + GEMM (Marlin-style).
    ///
    /// Performs a fused dequantization and matrix multiplication for GPTQ/AWQ INT4
    /// weights that have been pre-reformatted into Marlin tile layout. Eliminates
    /// the intermediate FP16 weight materialization in global memory.
    ///
    /// # Arguments
    /// - `input`: `[M, K]` FP16 activations
    /// - `qweight_marlin`: `[K/16, N/64, 128]` u32 Marlin tile layout
    /// - `scales`: `[K/group_size, N]` FP16 per-group scale factors
    /// - `qzeros`: `[K/group_size, N/8]` u32 packed INT4 zero points
    /// - `group_size`: number of input features sharing the same scale/zero
    ///
    /// # Returns
    /// `[M, N]` FP16 output tensor
    fn marlin_gemm(
        &self,
        _input: &Tensor,
        _qweight_marlin: &Tensor,
        _scales: &Tensor,
        _qzeros: &Tensor,
        _group_size: usize,
    ) -> Result<Tensor> {
        candle_core::bail!("marlin_gemm not supported on {} backend", self.name())
    }

    /// Fused SiLU activation + element-wise multiply.
    ///
    /// Computes `output = silu(gate) * up` in a single kernel pass,
    /// eliminating 2 intermediate tensor allocations.
    ///
    /// # Arguments
    /// - `gate`: `[total_tokens, intermediate_size]` — output of gate projection
    /// - `up`: `[total_tokens, intermediate_size]` — output of up projection
    ///
    /// # Returns
    /// `[total_tokens, intermediate_size]`
    fn fused_silu_mul(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        // Default: unfused path using candle ops
        let silu_gate = candle_nn::ops::silu(gate)?;
        silu_gate.mul(up)
    }

    /// Fused RMS Layer Normalization.
    ///
    /// Computes `output[i] = (x[i] / rms) * weight[i]` where `rms = sqrt(mean(x^2) + eps)`
    /// in a single kernel pass, avoiding intermediate tensor allocations.
    ///
    /// # Arguments
    /// - `x`: `[num_rows, hidden_size]` — input tensor
    /// - `weight`: `[hidden_size]` — normalization scale weights
    /// - `eps`: small constant for numerical stability
    ///
    /// # Returns
    /// `[num_rows, hidden_size]`
    fn fused_rmsnorm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        // Default: use candle_nn's rms_norm
        candle_nn::ops::rms_norm(x, weight, eps)
    }

    /// Fused residual add + RMSNorm.
    ///
    /// Computes `x = x + residual` then `output = rmsnorm(x, weight, eps)` in a single
    /// kernel pass, eliminating one intermediate tensor write. Also returns the updated
    /// `x` (with residual added) for use as the next layer's residual.
    ///
    /// # Arguments
    /// - `x`: `[num_rows, hidden_size]` — input tensor (consumed, modified in-place on CUDA)
    /// - `residual`: `[num_rows, hidden_size]` — residual to add
    /// - `weight`: `[hidden_size]` — normalization scale weights
    /// - `eps`: small constant for numerical stability
    ///
    /// # Returns
    /// `(normed, x_with_residual)` — the normalized output and x+residual
    fn fused_add_rmsnorm(
        &self,
        x: &Tensor,
        residual: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor)> {
        // Default unfused path: add then normalize
        let x_plus_res = (x + residual)?;
        let normed = candle_nn::ops::rms_norm(&x_plus_res, weight, eps)?;
        Ok((normed, x_plus_res))
    }

    /// Fused RMSNorm + linear projection.
    ///
    /// Computes `output = rmsnorm(x, norm_weight, eps) @ linear_weight^T` in a single
    /// kernel pass, eliminating the intermediate normalized tensor in global memory.
    ///
    /// # Arguments
    /// - `x`: `[num_rows, hidden_size]` input
    /// - `norm_weight`: `[hidden_size]` RMSNorm scale
    /// - `eps`: normalization epsilon
    /// - `linear_weight`: `[out_features, hidden_size]` projection weight
    ///
    /// # Returns
    /// `[num_rows, out_features]`
    fn fused_layernorm_linear(
        &self,
        x: &Tensor,
        norm_weight: &Tensor,
        eps: f32,
        linear_weight: &Tensor,
    ) -> Result<Tensor> {
        // Default: unfused path
        let normed = candle_nn::ops::rms_norm(x, norm_weight, eps)?;
        normed.matmul(&linear_weight.t()?)
    }

    /// Fused RoPE (Rotary Positional Embedding) applied in-place.
    ///
    /// Applies rotary embeddings directly to `[total_tokens, num_heads, head_size]`
    /// tensors without the transpose/unsqueeze/contiguous overhead of the unfused path.
    ///
    /// # Arguments
    /// - `x`: `[total_tokens, num_heads, head_size]` — Q or K tensor (modified in-place)
    /// - `positions`: `[total_tokens]` u32 — position index per token
    /// - `cos`: `[max_seq_len, rope_dim/2]` f32 — precomputed cosine table
    /// - `sin`: `[max_seq_len, rope_dim/2]` f32 — precomputed sine table
    /// - `rope_dim`: rotary embedding dimension (typically == head_size)
    ///
    /// # Returns
    /// `[total_tokens, num_heads, head_size]` (may be the same allocation or a new tensor)
    fn fused_rope(
        &self,
        x: &Tensor,
        positions: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        _num_heads: usize,
        _head_size: usize,
        _rope_dim: usize,
    ) -> Result<Tensor> {
        // Default: unfused path using candle's rotary_emb::rope
        let cos_gathered = cos.index_select(positions, 0)?;
        let sin_gathered = sin.index_select(positions, 0)?;
        let x = x.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
        let result = candle_nn::rotary_emb::rope(&x, &cos_gathered, &sin_gathered)?;
        result.squeeze(0)?.transpose(0, 1)?.contiguous()
    }
}

/// Configuration for a paged attention dispatch.
#[derive(Debug, Clone)]
pub struct PagedAttentionConfig {
    pub head_size: usize,
    pub num_kv_heads: usize,
    pub scale: f32,
    pub max_context_len: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serving::kernels::cpu_backend::CpuBackend;

    // KERN-01 + KERN-02: Fused LayerNorm+linear

    #[test]
    fn test_fused_layernorm_linear_produces_correct_shape() {
        // backend.fused_layernorm_linear(x, norm_weight, eps, linear_weight) returns [batch, out_features]
        let b = CpuBackend::new();
        let dev = &Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (4, 64), dev).unwrap();
        let norm_weight = Tensor::ones(64, DType::F32, dev).unwrap();
        // linear_weight: [out_features, hidden_size] = [128, 64]
        let linear_weight = Tensor::randn(0f32, 1.0, (128, 64), dev).unwrap();
        let result = b
            .fused_layernorm_linear(&x, &norm_weight, 1e-5, &linear_weight)
            .unwrap();
        assert_eq!(result.dims(), &[4, 128]);
    }

    #[test]
    fn test_fused_layernorm_linear_matches_unfused() {
        // Fused result matches sequential rmsnorm then matmul within tolerance
        let b = CpuBackend::new();
        let dev = &Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (8, 64), dev).unwrap();
        let norm_weight = Tensor::randn(0f32, 1.0, 64, dev).unwrap();
        let linear_weight = Tensor::randn(0f32, 1.0, (128, 64), dev).unwrap();
        let eps = 1e-5f32;

        // Fused path (via default impl)
        let fused = b
            .fused_layernorm_linear(&x, &norm_weight, eps, &linear_weight)
            .unwrap();

        // Manual unfused reference
        let normed = candle_nn::ops::rms_norm(&x, &norm_weight, eps).unwrap();
        let unfused = normed.matmul(&linear_weight.t().unwrap()).unwrap();

        let diff = (&fused - &unfused).unwrap().abs().unwrap();
        let max_diff: f32 = diff
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(
            max_diff < 1e-3,
            "fused vs unfused max_diff={max_diff}"
        );
    }

    #[test]
    fn test_fused_layernorm_linear_default_fallback() {
        // Default trait impl (used by CpuBackend) produces valid output
        let b = CpuBackend::new();
        let dev = &Device::Cpu;
        let x = Tensor::randn(0f32, 1.0, (4, 32), dev).unwrap();
        let norm_weight = Tensor::ones(32, DType::F32, dev).unwrap();
        let linear_weight = Tensor::randn(0f32, 1.0, (16, 32), dev).unwrap();

        let result = b
            .fused_layernorm_linear(&x, &norm_weight, 1e-5, &linear_weight)
            .unwrap();

        // Output should be finite and have correct shape
        assert_eq!(result.dims(), &[4, 16]);
        let data: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        for (i, v) in data.iter().enumerate() {
            assert!(v.is_finite(), "non-finite at index {i}: {v}");
        }
    }
}
