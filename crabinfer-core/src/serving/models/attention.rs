//! Paged attention layer shared across all model architectures.
//!
//! Handles both prefill (standard SDPA) and decode (paged attention kernel).
//! During both modes, new K/V tokens are written to the paged cache via
//! `reshape_and_cache` so subsequent decode steps can read them.
//!
//! Dispatch is through the `KernelBackend` trait (Metal, CUDA, or CPU)
//! carried on `ForwardContext::backend`.

use candle_core::{Device, Result, Tensor};

use super::ForwardContext;
use crate::serving::kernels::backend::PagedAttentionConfig;

// ─── PagedAttentionLayer ──────────────────────────────────────────────────

/// Shared attention layer that dispatches to paged attention for decode
/// and standard SDPA for prefill.
///
/// Each transformer layer creates one of these. The RoPE cos/sin tables
/// can be shared across layers by cloning (Tensor clone is O(1) via Arc).
#[derive(Clone)]
pub struct PagedAttentionLayer {
    pub(crate) num_heads: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) head_size: usize,
    scale: f32,
    /// Precomputed RoPE cosines `[max_seq_len, rope_dim/2]`.
    pub(crate) cos: Tensor,
    /// Precomputed RoPE sines `[max_seq_len, rope_dim/2]`.
    pub(crate) sin: Tensor,
}

impl PagedAttentionLayer {
    /// Create a new layer, precomputing RoPE tables.
    pub fn new(
        num_heads: usize,
        num_kv_heads: usize,
        head_size: usize,
        rope_theta: f32,
        rope_dim: usize,
        max_seq_len: usize,
        device: &Device,
    ) -> Result<Self> {
        let (cos, sin) = precompute_rope(rope_dim, rope_theta, max_seq_len, device)?;
        Ok(Self::with_rope(num_heads, num_kv_heads, head_size, cos, sin))
    }

    /// Create a new layer with pre-computed RoPE tables (avoids recomputation).
    pub fn with_rope(
        num_heads: usize,
        num_kv_heads: usize,
        head_size: usize,
        cos: Tensor,
        sin: Tensor,
    ) -> Self {
        let scale = 1.0 / (head_size as f32).sqrt();
        Self {
            num_heads,
            num_kv_heads,
            head_size,
            scale,
            cos,
            sin,
        }
    }

    /// Apply attention with paged KV cache.
    ///
    /// Steps:
    /// 1. Reshape Q/K/V to `[total_tokens, num_heads, head_size]`
    /// 2. Apply RoPE to Q and K using position tensor
    /// 3. Write K, V to paged cache via backend's `reshape_and_cache`
    /// 4. Compute attention:
    ///    - Decode (all seqs have 1 token): backend's paged attention kernel
    ///    - Prefill: standard SDPA per sequence
    ///
    /// # Arguments
    /// - `query`: `[total_tokens, num_heads * head_size]`
    /// - `key`: `[total_tokens, num_kv_heads * head_size]`
    /// - `value`: `[total_tokens, num_kv_heads * head_size]`
    /// - `ctx`: forward context with paged attention state
    /// - `layer_idx`: which layer's KV cache to use
    ///
    /// # Returns
    /// `[total_tokens, num_heads * head_size]`
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        ctx: &ForwardContext,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let total_tokens = query.dims()[0];

        // Reshape to [total_tokens, num_heads, head_size]
        let q = query.reshape((total_tokens, self.num_heads, self.head_size))?;
        let k = key.reshape((total_tokens, self.num_kv_heads, self.head_size))?;
        let v = value.reshape((total_tokens, self.num_kv_heads, self.head_size))?;

        // Apply RoPE using fused kernel (CUDA) or unfused path (CPU/Metal)
        let q = ctx.backend.fused_rope(
            &q, ctx.positions, &self.cos, &self.sin,
            self.num_heads, self.head_size, self.cos.dims()[1] * 2,
        )?;
        let k = ctx.backend.fused_rope(
            &k, ctx.positions, &self.cos, &self.sin,
            self.num_kv_heads, self.head_size, self.cos.dims()[1] * 2,
        )?;

        // Write K, V to paged cache via backend
        let (key_cache, value_cache) = &ctx.kv_caches[layer_idx];
        ctx.backend
            .reshape_and_cache(&k, &v, key_cache, value_cache, ctx.slot_mapping)?;

        // Dispatch attention
        if ctx.is_all_decode {
            self.decode_attention(&q, key_cache, value_cache, ctx)
        } else {
            // TODO: For sequences with prefix cache hits, this only attends
            // to the new tokens, missing cross-attention with cached tokens.
            // Full support requires reading cached K/V from the paged cache
            // or using per-token paged attention for prefill.
            self.prefill_attention(&q, &k, &v, ctx)
        }
    }

    /// Decode attention: dispatch to backend's paged attention kernel.
    ///
    /// All sequences have exactly 1 new token, so query is `[num_seqs, num_heads, head_size]`.
    fn decode_attention(
        &self,
        query: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        ctx: &ForwardContext,
    ) -> Result<Tensor> {
        let num_seqs = ctx.seq_lens.len();

        let output = Tensor::zeros(
            (num_seqs, self.num_heads, self.head_size),
            query.dtype(),
            query.device(),
        )?;

        let config = PagedAttentionConfig {
            head_size: self.head_size,
            num_kv_heads: self.num_kv_heads,
            scale: self.scale,
            max_context_len: ctx.max_context_len,
        };

        ctx.backend.paged_attention(
            &output,
            query,
            key_cache,
            value_cache,
            ctx.block_table,
            ctx.context_lens,
            &config,
        )?;

        // Reshape to [num_seqs, num_heads * head_size]
        output.reshape((num_seqs, self.num_heads * self.head_size))
    }

    /// Prefill attention: dispatches to FlashAttention-2 (if available) or
    /// falls back to standard SDPA per sequence.
    fn prefill_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        ctx: &ForwardContext,
    ) -> Result<Tensor> {
        #[cfg(feature = "flash-attn")]
        {
            // FlashAttention-2 via candle-flash-attn: IO-aware tiled attention,
            // 2-4x faster than standard SDPA for long sequences.
            // Supports GQA natively (num_heads_q must be divisible by num_heads_kv).
            // Only works with f16/bf16 on CUDA sm_80+.
            if matches!(query.device(), Device::Cuda(_))
                && matches!(query.dtype(), candle_core::DType::F16 | candle_core::DType::BF16)
            {
                return self.flash_attn_prefill(query, key, value, ctx);
            }
        }
        self.sdpa_prefill(query, key, value, ctx)
    }

    /// FlashAttention-2 prefill using variable-length batching.
    ///
    /// Uses `flash_attn_varlen` which handles multiple sequences of different
    /// lengths packed into a single tensor — perfect for continuous batching.
    ///
    /// Tensors are `[total_tokens, num_heads, head_size]` with cumulative
    /// sequence lengths to demarcate boundaries.
    #[cfg(feature = "flash-attn")]
    fn flash_attn_prefill(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        ctx: &ForwardContext,
    ) -> Result<Tensor> {
        // Handle GQA: flash_attn_varlen supports it natively when
        // num_heads_q % num_heads_kv == 0, so no need to repeat KV heads.

        // Build cumulative sequence lengths as u32 tensor on the same device.
        // flash_attn_varlen expects seqlens_q and seqlens_k of shape [batch_size + 1].
        let seqlens: Vec<u32> = ctx.query_start_loc.iter().map(|&x| x as u32).collect();
        let seqlens_q = Tensor::new(&seqlens[..], query.device())?;
        let seqlens_k = seqlens_q.clone();

        let max_seqlen = ctx.query_start_loc
            .windows(2)
            .map(|w| w[1] - w[0])
            .max()
            .unwrap_or(0);

        // q, k, v are [total_tokens, num_heads, head_size] — already in the right layout
        // for flash_attn_varlen which expects [total_tokens, num_heads, head_size]
        let q_contig = query.contiguous()?;
        let k_contig = key.contiguous()?;
        let v_contig = value.contiguous()?;

        let attn_output = candle_flash_attn::flash_attn_varlen(
            &q_contig,
            &k_contig,
            &v_contig,
            &seqlens_q,
            &seqlens_k,
            max_seqlen,
            max_seqlen,
            self.scale,
            true, // causal
        )?;

        // Output is [total_tokens, num_heads, head_size]
        // Reshape to [total_tokens, num_heads * head_size]
        let total_tokens = query.dims()[0];
        attn_output.reshape((total_tokens, self.num_heads * self.head_size))
    }

    /// Standard SDPA prefill: per-sequence scaled dot-product attention.
    ///
    /// For each sequence, extracts its Q/K/V slice and computes causally-masked
    /// attention. This handles GQA by repeating KV heads.
    fn sdpa_prefill(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        ctx: &ForwardContext,
    ) -> Result<Tensor> {
        let num_seqs = ctx.query_start_loc.len() - 1;
        let mut outputs = Vec::with_capacity(num_seqs);

        for seq_idx in 0..num_seqs {
            let start = ctx.query_start_loc[seq_idx];
            let end = ctx.query_start_loc[seq_idx + 1];
            let seq_len = end - start;

            if seq_len == 0 {
                continue;
            }

            // Extract this sequence's Q, K, V: [seq_len, num_heads, head_size]
            let q_seq = query.narrow(0, start, seq_len)?;
            let k_seq = key.narrow(0, start, seq_len)?;
            let v_seq = value.narrow(0, start, seq_len)?;

            // Reshape to [1, num_heads, seq_len, head_size] for batched matmul
            // Must call .contiguous() after transpose to satisfy GPU GEMM stride requirements
            let q_seq = q_seq.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
            let k_seq = k_seq.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
            let v_seq = v_seq.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;

            // Handle GQA: repeat KV heads to match Q heads
            let n_rep = self.num_heads / self.num_kv_heads;
            let k_seq = repeat_kv(k_seq, n_rep)?;
            let v_seq = repeat_kv(v_seq, n_rep)?;

            // Scaled dot-product attention with causal mask
            let attn_weights = (q_seq.matmul(&k_seq.t()?)? * (self.scale as f64))?;
            let mask = causal_mask(seq_len, query.device())?;
            let attn_weights = masked_fill(&attn_weights, &mask)?;
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            let attn_output = attn_weights.matmul(&v_seq)?;

            // Reshape to [seq_len, num_heads * head_size]
            let attn_output = attn_output
                .squeeze(0)?
                .transpose(0, 1)?
                .contiguous()?
                .reshape((seq_len, self.num_heads * self.head_size))?;

            outputs.push(attn_output);
        }

        if outputs.len() == 1 {
            Ok(outputs.into_iter().next().unwrap())
        } else {
            Tensor::cat(&outputs, 0)
        }
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────

/// Precompute RoPE cosine and sine tables for all positions up to `max_seq_len`.
///
/// Returns `(cos, sin)` each of shape `[max_seq_len, head_dim/2]`.
pub fn precompute_rope(
    head_dim: usize,
    theta: f32,
    max_seq_len: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half_dim = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let inv_freq = Tensor::new(&inv_freq[..], device)?;

    let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
    let positions = Tensor::new(&positions[..], device)?.reshape((max_seq_len, 1))?;

    let freqs = positions.matmul(&inv_freq.reshape((1, half_dim))?)?;
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;

    Ok((cos, sin))
}

/// Repeat KV heads for Grouped Query Attention (GQA).
///
/// Input: `[batch, num_kv_heads, seq_len, head_size]`
/// Output: `[batch, num_kv_heads * n_rep, seq_len, head_size]`
fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (batch, num_kv_heads, seq_len, head_size) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((batch, num_kv_heads, n_rep, seq_len, head_size))?
        .reshape((batch, num_kv_heads * n_rep, seq_len, head_size))
}

/// Create a causal attention mask (upper triangle of 1s).
///
/// Returns a `[seq_len, seq_len]` u8 tensor where positions that should be
/// masked (future tokens) are 1 and visible positions are 0.
fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<u8> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| u8::from(j > i)))
        .collect();
    Tensor::from_slice(&mask, (seq_len, seq_len), device)
}

/// Apply mask: where mask == 1, fill with -infinity.
fn masked_fill(tensor: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let neg_inf = Tensor::new(f32::NEG_INFINITY, tensor.device())?.broadcast_as(tensor.shape())?;
    let mask = mask.broadcast_as(tensor.shape())?;
    mask.where_cond(&neg_inf, tensor)
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_precompute_rope_shape() {
        let (cos, sin) = precompute_rope(128, 10000.0, 4096, &Device::Cpu).unwrap();
        assert_eq!(cos.dims(), &[4096, 64]);
        assert_eq!(sin.dims(), &[4096, 64]);
    }

    #[test]
    fn test_precompute_rope_values() {
        let (cos, sin) = precompute_rope(4, 10000.0, 2, &Device::Cpu).unwrap();
        // Position 0: cos(0) = 1.0, sin(0) = 0.0
        let cos_data: Vec<f32> = cos.flatten_all().unwrap().to_vec1().unwrap();
        assert!((cos_data[0] - 1.0).abs() < 1e-6); // cos(0) = 1
        assert!((cos_data[1] - 1.0).abs() < 1e-6); // cos(0) = 1

        let sin_data: Vec<f32> = sin.flatten_all().unwrap().to_vec1().unwrap();
        assert!(sin_data[0].abs() < 1e-6); // sin(0) = 0
        assert!(sin_data[1].abs() < 1e-6); // sin(0) = 0
    }

    #[test]
    fn test_causal_mask() {
        let mask = causal_mask(4, &Device::Cpu).unwrap();
        let data: Vec<u8> = mask.flatten_all().unwrap().to_vec1().unwrap();
        // Upper triangle:
        // 0 1 1 1
        // 0 0 1 1
        // 0 0 0 1
        // 0 0 0 0
        assert_eq!(
            data,
            vec![0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0]
        );
    }

    #[test]
    fn test_causal_mask_size_1() {
        let mask = causal_mask(1, &Device::Cpu).unwrap();
        let data: Vec<u8> = mask.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(data, vec![0]);
    }

    #[test]
    fn test_repeat_kv_no_repeat() {
        let x = Tensor::zeros((1, 8, 10, 128), DType::F32, &Device::Cpu).unwrap();
        let y = repeat_kv(x, 1).unwrap();
        assert_eq!(y.dims(), &[1, 8, 10, 128]);
    }

    #[test]
    fn test_repeat_kv_4x() {
        let x = Tensor::zeros((1, 8, 10, 128), DType::F32, &Device::Cpu).unwrap();
        let y = repeat_kv(x, 4).unwrap();
        assert_eq!(y.dims(), &[1, 32, 10, 128]);
    }

    #[test]
    fn test_repeat_kv_preserves_values() {
        let x = Tensor::arange(0f32, 8.0, &Device::Cpu)
            .unwrap()
            .reshape((1, 2, 1, 4))
            .unwrap();
        let y = repeat_kv(x, 2).unwrap();
        assert_eq!(y.dims(), &[1, 4, 1, 4]);
        let data: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        // Head 0: [0,1,2,3], Head 0 repeat: [0,1,2,3], Head 1: [4,5,6,7], Head 1 repeat: [4,5,6,7]
        assert_eq!(
            data,
            vec![0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 4.0, 5.0, 6.0, 7.0]
        );
    }

    #[test]
    fn test_masked_fill() {
        let tensor = Tensor::ones((2, 2), DType::F32, &Device::Cpu).unwrap();
        let mask = Tensor::new(&[[0u8, 1], [0, 0]], &Device::Cpu).unwrap();
        let result = masked_fill(&tensor, &mask).unwrap();
        let data: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(data[0], 1.0);
        assert!(data[1].is_infinite() && data[1] < 0.0);
        assert_eq!(data[2], 1.0);
        assert_eq!(data[3], 1.0);
    }

    #[test]
    fn test_paged_attention_layer_creation() {
        let layer =
            PagedAttentionLayer::new(32, 8, 128, 10000.0, 128, 4096, &Device::Cpu).unwrap();
        assert_eq!(layer.num_heads, 32);
        assert_eq!(layer.num_kv_heads, 8);
        assert_eq!(layer.head_size, 128);
        assert_eq!(layer.cos.dims(), &[4096, 64]);
    }

    #[test]
    fn test_paged_attention_layer_with_rope() {
        let (cos, sin) = precompute_rope(128, 10000.0, 2048, &Device::Cpu).unwrap();
        let layer = PagedAttentionLayer::with_rope(32, 8, 128, cos.clone(), sin.clone());
        assert_eq!(layer.cos.dims(), cos.dims());
        assert_eq!(layer.sin.dims(), sin.dims());
    }
}
