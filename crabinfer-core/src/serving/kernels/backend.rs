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
