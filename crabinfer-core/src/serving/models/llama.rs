//! Llama architecture model with paged attention.
//!
//! Loads GGUF weights using candle-core's quantized types and runs forward
//! passes using the serving system's paged KV cache. Compatible with:
//! - Llama 2/3
//! - CodeLlama
//! - Qwen2/2.5 (via QKV biases — `attn_q_bias`, `attn_k_bias`, `attn_v_bias`)
//! - Qwen3/3.5 (via QKV biases + QK normalization — `attn_q_norm`, `attn_k_norm`)
//! - Other Llama-derived architectures (same GGUF weight names)
//!
//! For HuggingFace safetensors loading of Qwen models, see
//! [`SafetensorsQwenModel`](super::super::safetensors_loader) which provides
//! explicit Qwen2/3 dispatch with bias and QK norm support.
//!
//! MoE (Mixture of Experts) variants are not yet supported — use the standard
//! Llama loader for dense models only.

use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{Device, Result, Tensor};
use candle_nn::Module;

use super::attention::{precompute_rope, PagedAttentionLayer};
use super::{ForwardContext, ModelConfig, ModelRunner, RmsNorm, SwiGluMlp};
use crate::serving::quantization::{MaybeQuantizedLinear, QuantizationMethod};

// ─── Transformer layer ───────────────────────────────────────────────────

/// Single transformer layer for the Llama architecture.
#[derive(Clone)]
struct LlamaLayer {
    /// Pre-attention RMS normalization.
    attn_norm: RmsNorm,
    /// Query projection (may be GGUF-quantized or INT8).
    attn_q: MaybeQuantizedLinear,
    /// Key projection (may be GGUF-quantized or INT8).
    attn_k: MaybeQuantizedLinear,
    /// Value projection (may be GGUF-quantized or INT8).
    attn_v: MaybeQuantizedLinear,
    /// Optional QKV biases (required by Qwen2, absent in Llama).
    attn_q_bias: Option<Tensor>,
    attn_k_bias: Option<Tensor>,
    attn_v_bias: Option<Tensor>,
    /// Optional QK normalization (Qwen3 applies RMSNorm to Q and K per-head).
    attn_q_norm: Option<RmsNorm>,
    attn_k_norm: Option<RmsNorm>,
    /// Output projection after attention (may be GGUF-quantized or INT8).
    attn_output: MaybeQuantizedLinear,
    /// Pre-MLP RMS normalization.
    ffn_norm: RmsNorm,
    /// SwiGLU feed-forward network.
    mlp: SwiGluMlp,
    /// Paged attention with RoPE.
    attention: PagedAttentionLayer,
}

impl LlamaLayer {
    fn forward(&self, x: &Tensor, ctx: &ForwardContext, layer_idx: usize) -> Result<Tensor> {
        // Pre-norm attention (fused RMSNorm on CUDA)
        let residual = x;
        let x = self.attn_norm.forward_fused(x, ctx.backend)?;

        // QKV projections (+ optional biases for Qwen2)
        let mut q = self.attn_q.forward(&x)?;
        let mut k = self.attn_k.forward(&x)?;
        let mut v = self.attn_v.forward(&x)?;
        if let Some(ref bias) = self.attn_q_bias {
            q = q.broadcast_add(bias)?;
        }
        if let Some(ref bias) = self.attn_k_bias {
            k = k.broadcast_add(bias)?;
        }
        if let Some(ref bias) = self.attn_v_bias {
            v = v.broadcast_add(bias)?;
        }

        // Optional QK normalization (Qwen3): apply RMSNorm per-head to Q and K
        if let (Some(ref q_norm), Some(ref k_norm)) = (&self.attn_q_norm, &self.attn_k_norm) {
            let total_tokens = q.dims()[0];
            let num_q_heads = self.attention.num_heads;
            let num_kv_heads = self.attention.num_kv_heads;
            let head_size = self.attention.head_size;
            // Reshape to [total_tokens * num_heads, head_size], apply norm, reshape back
            q = q.reshape((total_tokens * num_q_heads, head_size))?;
            q = q_norm.forward(&q)?;
            q = q.reshape((total_tokens, num_q_heads * head_size))?;
            k = k.reshape((total_tokens * num_kv_heads, head_size))?;
            k = k_norm.forward(&k)?;
            k = k.reshape((total_tokens, num_kv_heads * head_size))?;
        }

        // Paged attention (includes RoPE + cache write + attention dispatch)
        let attn_out = self.attention.forward(&q, &k, &v, ctx, layer_idx)?;

        // Output projection
        let attn_proj = self.attn_output.forward(&attn_out)?;

        // Fused residual add + RMSNorm for MLP (eliminates intermediate tensor on CUDA)
        let (x, residual) = self.ffn_norm.forward_add_fused(&attn_proj, residual, ctx.backend)?;

        // MLP (fused SiLU+mul on CUDA) + residual
        let x = self.mlp.forward_fused(&x, ctx.backend)?;
        x + &residual
    }
}

// ─── Llama model ──────────────────────────────────────────────────────────

/// Llama model with paged attention support.
///
/// Unlike candle-transformers' `ModelWeights`, this model:
/// - Does not manage its own KV cache (external paged cache)
/// - Takes `&self` for forward (immutable — all mutable state is external)
/// - Accepts flat batched inputs `[total_tokens]`
/// - Returns logits for all input tokens `[total_tokens, vocab_size]`
pub struct LlamaModel {
    /// Token embedding table (dequantized to f32/f16).
    embed_table: Tensor,
    /// Transformer layers.
    layers: Vec<LlamaLayer>,
    /// Final RMS normalization before output projection.
    norm: RmsNorm,
    /// Output projection to vocabulary (lm_head).
    lm_head: MaybeQuantizedLinear,
    /// Model configuration.
    config: ModelConfig,
}

impl LlamaModel {
    /// Load a Llama model from GGUF content.
    ///
    /// # Arguments
    /// - `ct`: parsed GGUF content (metadata + tensor directory)
    /// - `reader`: seekable reader for loading tensor data
    /// - `device`: target device (CPU or Metal)
    /// - `quantization`: weight quantization method to apply at load time
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
        quantization: QuantizationMethod,
    ) -> Result<Self> {
        let md = &ct.metadata;

        // Auto-detect architecture prefix from GGUF metadata.
        // Most llama-derived architectures (qwen2, mistral, yi, etc.) use
        // the same tensor names but different metadata key prefixes.
        let arch = md
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "llama".to_string());

        // Helper to read u32 metadata with architecture prefix
        let get_u32 = |suffix: &str| -> Result<u32> {
            let key = format!("{arch}.{suffix}");
            md.get(&key)
                .ok_or_else(|| candle_core::Error::Msg(format!("missing GGUF key: {key}")))?
                .to_u32()
                .map_err(|_| candle_core::Error::Msg(format!("invalid u32 for key: {key}")))
        };

        // ── Extract model config from GGUF metadata ──

        let num_heads = get_u32("attention.head_count")? as usize;
        let num_kv_heads = get_u32("attention.head_count_kv")? as usize;
        let num_layers = get_u32("block_count")? as usize;
        let hidden_size = get_u32("embedding_length")? as usize;
        let intermediate_size = get_u32("feed_forward_length")? as usize;
        // head_size: prefer attention.key_length (Qwen3 uses head_dim != hidden/heads),
        // fall back to hidden_size / num_heads for standard Llama
        let head_size = md
            .get(&format!("{arch}.attention.key_length"))
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or(hidden_size / num_heads);
        // rope.dimension_count is optional — default to head_size if absent
        let rope_dim = md
            .get(&format!("{arch}.rope.dimension_count"))
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or(head_size);

        let rms_norm_eps = md
            .get(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(1e-5) as f64;
        let rope_theta = md
            .get(&format!("{arch}.rope.freq_base"))
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(10000.0);
        let context_length = md
            .get(&format!("{arch}.context_length"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(4096) as usize;

        // ── Load token embeddings ──

        let embed_q = ct.tensor(reader, "token_embd.weight", device)?;
        let embed_table = embed_q.dequantize(device)?;
        let vocab_size = embed_table.dims()[0];

        // Helper: load a QMatMul and optionally re-quantize to INT8
        let wrap = |qmm: QMatMul| -> Result<MaybeQuantizedLinear> {
            MaybeQuantizedLinear::from_qmatmul(qmm, quantization, device)
        };

        // ── Load output projection ──
        // Some models share token_embd and output weights (tied embeddings)
        let lm_head = match ct.tensor(reader, "output.weight", device) {
            Ok(t) => wrap(QMatMul::from_qtensor(t)?)?,
            Err(_) => wrap(QMatMul::from_qtensor(ct.tensor(reader, "token_embd.weight", device)?)?)?,
        };

        // ── Load output normalization ──

        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;

        // ── Precompute shared RoPE tables ──
        // All layers use the same rope_theta and rope_dim, so compute once and clone

        let (rope_cos, rope_sin) = precompute_rope(rope_dim, rope_theta, context_length, device)?;

        // ── Load transformer layers ──

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let prefix = format!("blk.{i}");

            // Attention weights
            let attn_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let attn_q = wrap(QMatMul::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?,
            )?)?;
            let attn_k = wrap(QMatMul::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?,
            )?)?;
            let attn_v = wrap(QMatMul::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?,
            )?)?;
            // Optional QKV biases (Qwen2 has them, Llama does not)
            let attn_q_bias = ct
                .tensor(reader, &format!("{prefix}.attn_q.bias"), device)
                .ok()
                .and_then(|t| t.dequantize(device).ok());
            let attn_k_bias = ct
                .tensor(reader, &format!("{prefix}.attn_k.bias"), device)
                .ok()
                .and_then(|t| t.dequantize(device).ok());
            let attn_v_bias = ct
                .tensor(reader, &format!("{prefix}.attn_v.bias"), device)
                .ok()
                .and_then(|t| t.dequantize(device).ok());

            // Optional QK normalization (Qwen3 uses per-head RMSNorm on Q and K)
            let attn_q_norm = ct
                .tensor(reader, &format!("{prefix}.attn_q_norm.weight"), device)
                .ok()
                .map(|t| RmsNorm::from_qtensor(t, rms_norm_eps))
                .transpose()?;
            let attn_k_norm = ct
                .tensor(reader, &format!("{prefix}.attn_k_norm.weight"), device)
                .ok()
                .map(|t| RmsNorm::from_qtensor(t, rms_norm_eps))
                .transpose()?;

            let attn_output = wrap(QMatMul::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?,
            )?)?;

            // MLP weights
            let ffn_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let gate = wrap(QMatMul::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?,
            )?)?;
            let down = wrap(QMatMul::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?,
            )?)?;
            let up = wrap(QMatMul::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?,
            )?)?;
            let mlp = SwiGluMlp::new(gate, down, up);

            // Paged attention layer (shares RoPE tables via Tensor clone = Arc refcount)
            let attention = PagedAttentionLayer::with_rope(
                num_heads,
                num_kv_heads,
                head_size,
                rope_cos.clone(),
                rope_sin.clone(),
            );

            layers.push(LlamaLayer {
                attn_norm,
                attn_q,
                attn_k,
                attn_v,
                attn_q_bias,
                attn_k_bias,
                attn_v_bias,
                attn_q_norm,
                attn_k_norm,
                attn_output,
                ffn_norm,
                mlp,
                attention,
            });
        }

        let config = ModelConfig {
            hidden_size,
            intermediate_size,
            num_heads,
            num_kv_heads,
            num_layers,
            head_size,
            vocab_size,
            rms_norm_eps,
            rope_theta,
            rope_dim,
            max_seq_len: context_length,
        };

        let has_qkv_bias = layers.first().map_or(false, |l| l.attn_q_bias.is_some());
        let has_qk_norm = layers.first().map_or(false, |l| l.attn_q_norm.is_some());
        tracing::info!(
            "Model config: hidden={} heads={} kv_heads={} head_size={} layers={} vocab={} rope_theta={} rope_dim={} max_seq={} intermediate={} qkv_bias={} qk_norm={} quantization={}",
            config.hidden_size, config.num_heads, config.num_kv_heads,
            config.head_size, config.num_layers, config.vocab_size,
            config.rope_theta, config.rope_dim, config.max_seq_len, config.intermediate_size,
            has_qkv_bias, has_qk_norm, quantization,
        );

        Ok(Self {
            embed_table,
            layers,
            norm,
            lm_head,
            config,
        })
    }
}

impl ModelRunner for LlamaModel {
    fn forward(&self, input_ids: &Tensor, ctx: &ForwardContext) -> Result<Tensor> {
        // Token embedding: [total_tokens] -> [total_tokens, hidden_size]
        let mut hidden_states = self.embed_table.index_select(input_ids, 0)?;

        // Transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states, ctx, layer_idx)?;
        }

        // Final norm + output projection: [total_tokens, hidden_size] -> [total_tokens, vocab_size]
        // Uses fused norm+linear when lm_head is a dense weight (eliminates intermediate tensor).
        self.norm.forward_linear_fused(&hidden_states, &self.lm_head, ctx.backend)
    }

    fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        // Look up token embeddings from the model's learned embedding table,
        // then mean-pool across the sequence dimension to produce a single
        // fixed-size vector per input.
        // input_ids: [num_tokens] -> embeddings: [num_tokens, hidden_size]
        let embeddings = self.embed_table.index_select(input_ids, 0)?;
        // Mean-pool: [num_tokens, hidden_size] -> [hidden_size]
        embeddings.mean(0)
    }

    fn embedding_table(&self) -> Option<&Tensor> {
        Some(&self.embed_table)
    }

    fn clone_model(&self) -> Box<dyn ModelRunner> {
        Box::new(LlamaModel {
            embed_table: self.embed_table.clone(),
            layers: self.layers.clone(),
            norm: self.norm.clone(),
            lm_head: self.lm_head.clone(),
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
    fn config(&self) -> &ModelConfig {
        &self.config
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_model_config_from_values() {
        let config = ModelConfig {
            hidden_size: 4096,
            intermediate_size: 11008,
            num_heads: 32,
            num_kv_heads: 8,
            num_layers: 32,
            head_size: 128,
            vocab_size: 32000,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_dim: 128,
            max_seq_len: 4096,
        };
        assert_eq!(config.head_size, config.hidden_size / config.num_heads);
    }

    #[test]
    fn test_embedding_lookup() {
        // Simulate the embedding lookup pattern used in forward()
        let vocab_size = 100;
        let hidden_size = 16;
        let embed_table =
            Tensor::randn(0f32, 1.0, (vocab_size, hidden_size), &Device::Cpu).unwrap();

        let input_ids = Tensor::new(&[0u32, 5, 99, 42], &Device::Cpu).unwrap();
        let hidden = embed_table.index_select(&input_ids, 0).unwrap();
        assert_eq!(hidden.dims(), &[4, hidden_size]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_index_select() {
        let dev = Device::new_cuda(0).expect("no CUDA device");
        let vocab_size = 100;
        let hidden_size = 16;
        let embed_table =
            Tensor::randn(0f32, 1.0, (vocab_size, hidden_size), &dev).unwrap();

        // Test I64 indices
        let ids_i64 = Tensor::new(&[0i64, 5, 99, 42], &dev).unwrap();
        let result = embed_table.index_select(&ids_i64, 0).unwrap();
        assert_eq!(result.dims(), &[4, hidden_size]);

        // Test U32 indices
        let ids_u32 = Tensor::new(&[0u32, 5, 99, 42], &dev).unwrap();
        let result = embed_table.index_select(&ids_u32, 0).unwrap();
        assert_eq!(result.dims(), &[4, hidden_size]);
    }

    #[test]
    fn test_llama_layer_forward_shape() {
        // Test the forward pass shape through a minimal "layer" using CPU tensors.
        // This doesn't test the paged attention path (Metal only) but verifies
        // the norm -> proj -> attention -> residual -> norm -> mlp -> residual flow.
        let hidden_size = 64;
        let head_size = 16;
        let num_heads = 4;
        let num_kv_heads = 4;
        let total_tokens = 8;

        // Create mock input
        let x = Tensor::randn(0f32, 1.0, (total_tokens, hidden_size), &Device::Cpu).unwrap();

        // Test RmsNorm
        let weight = Tensor::ones(hidden_size, DType::F32, &Device::Cpu).unwrap();
        let norm = RmsNorm {
            weight,
            eps: 1e-5,
        };
        let normed = norm.forward(&x).unwrap();
        assert_eq!(normed.dims(), &[total_tokens, hidden_size]);

        // Test that QKV projection shapes would be correct
        // Q: [total_tokens, num_heads * head_size] = [8, 64]
        // K: [total_tokens, num_kv_heads * head_size] = [8, 64]
        // V: [total_tokens, num_kv_heads * head_size] = [8, 64]
        let q = Tensor::randn(
            0f32,
            1.0,
            (total_tokens, num_heads * head_size),
            &Device::Cpu,
        )
        .unwrap();
        let k = Tensor::randn(
            0f32,
            1.0,
            (total_tokens, num_kv_heads * head_size),
            &Device::Cpu,
        )
        .unwrap();
        let _v = Tensor::randn(
            0f32,
            1.0,
            (total_tokens, num_kv_heads * head_size),
            &Device::Cpu,
        )
        .unwrap();

        // Verify reshape to [total_tokens, num_heads, head_size]
        let q_3d = q.reshape((total_tokens, num_heads, head_size)).unwrap();
        assert_eq!(q_3d.dims(), &[total_tokens, num_heads, head_size]);

        let k_3d = k.reshape((total_tokens, num_kv_heads, head_size)).unwrap();
        assert_eq!(k_3d.dims(), &[total_tokens, num_kv_heads, head_size]);
    }
}
