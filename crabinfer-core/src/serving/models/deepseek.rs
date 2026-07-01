//! DeepSeek V2/V3 architecture model with paged attention.
//!
//! Supports models with `"architectures": ["DeepseekV2ForCausalLM"]` or
//! `"DeepseekV3ForCausalLM"` in their HuggingFace `config.json`.
//!
//! Key architectural differences from Llama:
//! - **MLA (Multi-Latent Attention)**: Compresses KV into low-rank latent vectors
//!   via down-projections, then up-projects for attention computation. This reduces
//!   KV cache size by ~10x compared to standard MHA.
//! - **MoE (Mixture of Experts)**: Uses shared experts + routed experts with top-k
//!   gating. Most layers are MoE; a few early layers use dense FFN.
//! - **RoPE on partial dimensions**: Only applied to a portion of Q and K (the
//!   "rope" portion), while the rest uses the compressed latent.
//!
//! Compatible with:
//! - DeepSeek-V2-Lite, DeepSeek-V2
//! - DeepSeek-V3, DeepSeek-R1

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::{Device, Result, Tensor};
use candle_nn::Module;
use crate::serving::kernels::KernelBackend;

use super::attention::{precompute_rope, PagedAttentionLayer};
use super::{ForwardContext, ModelConfig, ModelRunner, RmsNorm, SwiGluMlp};
use crate::serving::quantization::{MaybeQuantizedLinear, QuantizationMethod};
use crate::serving::safetensors_loader::{load_linear, load_tensor, load_tensor_fp32};

// ─── DeepSeek config ─────────────────────────────────────────────────────

/// DeepSeek V2/V3 specific configuration parsed from `config.json`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct DeepSeekConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,

    // ── MLA (Multi-Latent Attention) ──

    /// Latent dimension for compressed KV (q_lora_rank).
    /// If 0 or absent, standard MHA is used for that layer.
    #[serde(default)]
    pub q_lora_rank: Option<usize>,
    /// Latent dimension for compressed KV (kv_lora_rank).
    #[serde(default)]
    pub kv_lora_rank: Option<usize>,
    /// Non-RoPE head dimension (qk_nope_head_dim).
    #[serde(default)]
    pub qk_nope_head_dim: Option<usize>,
    /// RoPE head dimension (qk_rope_head_dim).
    #[serde(default)]
    pub qk_rope_head_dim: Option<usize>,
    /// Value head dimension (v_head_dim).
    #[serde(default)]
    pub v_head_dim: Option<usize>,

    // ── MoE (Mixture of Experts) ──

    /// Number of routed experts.
    #[serde(default)]
    pub n_routed_experts: Option<usize>,
    /// Number of shared experts (always active).
    #[serde(default)]
    pub n_shared_experts: Option<usize>,
    /// Number of experts activated per token (top-k).
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,
    /// First layer index that uses MoE (layers before this use dense FFN).
    #[serde(default)]
    pub first_k_dense_replace: Option<usize>,

    #[serde(default)]
    pub model_type: Option<String>,
}

fn default_rms_norm_eps() -> f64 {
    1e-5
}

fn default_rope_theta() -> f64 {
    10000.0
}

impl DeepSeekConfig {
    pub fn head_size(&self) -> usize {
        // For MLA models, head_size is qk_nope_head_dim + qk_rope_head_dim
        if let (Some(nope), Some(rope)) = (self.qk_nope_head_dim, self.qk_rope_head_dim) {
            nope + rope
        } else {
            self.hidden_size / self.num_attention_heads
        }
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads
            .unwrap_or(self.num_attention_heads)
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_position_embeddings.unwrap_or(4096)
    }

    pub fn rope_dim(&self) -> usize {
        // For MLA, only qk_rope_head_dim dimensions get RoPE
        self.qk_rope_head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Whether this model uses MLA attention.
    pub fn is_mla(&self) -> bool {
        self.kv_lora_rank.is_some() && self.kv_lora_rank.unwrap_or(0) > 0
    }

    /// Whether a given layer uses MoE.
    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        let first_dense = self.first_k_dense_replace.unwrap_or(0);
        let has_moe = self.n_routed_experts.unwrap_or(0) > 0;
        has_moe && layer_idx >= first_dense
    }

    /// Value head dimension for MLA (for the KV cache).
    pub fn value_head_dim(&self) -> usize {
        self.v_head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

// ─── MLA Attention ───────────────────────────────────────────────────────

/// Multi-Latent Attention layer for DeepSeek V2/V3.
///
/// Instead of separate Q, K, V projections, MLA uses:
/// 1. Down-project input to a low-rank latent space (kv_lora_rank)
/// 2. Up-project from latent to K and V
/// 3. Q gets its own down+up projection (q_lora_rank)
/// 4. RoPE is applied only to a subset of Q/K dimensions
///
/// This reduces KV cache from `2 * num_heads * head_size` to `kv_lora_rank + qk_rope_head_dim`,
/// a dramatic memory savings for large models.
#[derive(Clone)]
struct MlaLayer {
    /// Pre-attention RMS normalization.
    attn_norm: RmsNorm,

    // ── Q path ──
    /// Q down-projection: [hidden_size] -> [q_lora_rank] (optional, for compressed Q).
    q_a_proj: Option<MaybeQuantizedLinear>,
    /// Q normalization after down-projection.
    q_a_norm: Option<RmsNorm>,
    /// Q up-projection: [q_lora_rank] -> [num_heads * (qk_nope_head_dim + qk_rope_head_dim)].
    /// Or direct Q projection if no q_lora_rank.
    q_b_proj: MaybeQuantizedLinear,

    // ── KV path ──
    /// KV down-projection: [hidden_size] -> [kv_lora_rank + qk_rope_head_dim].
    kv_a_proj: MaybeQuantizedLinear,
    /// KV normalization after down-projection.
    kv_a_norm: RmsNorm,
    /// KV up-projection: [kv_lora_rank] -> [num_heads * (qk_nope_head_dim + v_head_dim)].
    kv_b_proj: MaybeQuantizedLinear,

    /// Output projection.
    attn_output: MaybeQuantizedLinear,

    /// Pre-MLP RMS normalization.
    ffn_norm: RmsNorm,
    /// Feed-forward network (dense or MoE).
    mlp: DeepSeekMlp,

    /// Paged attention with RoPE (for the rope portion of Q/K).
    attention: PagedAttentionLayer,

    /// Model dimensions for splitting.
    num_heads: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    kv_lora_rank: usize,
    v_head_dim: usize,
}

impl MlaLayer {
    fn forward(&self, x: &Tensor, ctx: &ForwardContext, layer_idx: usize) -> Result<Tensor> {
        let residual = x;
        let x = self.attn_norm.forward_fused(x, ctx.backend)?;
        let total_tokens = x.dims()[0];

        // ── Q path ──
        let q = if let (Some(ref q_a), Some(ref q_a_norm)) = (&self.q_a_proj, &self.q_a_norm) {
            // Compressed Q: down-project -> norm -> up-project
            let q_latent = q_a.forward(&x)?;
            let q_latent = q_a_norm.forward(&q_latent)?;
            self.q_b_proj.forward(&q_latent)?
        } else {
            // Direct Q projection
            self.q_b_proj.forward(&x)?
        };

        // Q shape: [total_tokens, num_heads * (qk_nope_head_dim + qk_rope_head_dim)]
        // Split into nope and rope portions per head
        let q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim;
        let q = q.reshape((total_tokens, self.num_heads, q_head_dim))?;
        let q_nope = q.narrow(2, 0, self.qk_nope_head_dim)?;
        let q_rope = q.narrow(2, self.qk_nope_head_dim, self.qk_rope_head_dim)?;

        // ── KV path ──
        // Down-project: [hidden_size] -> [kv_lora_rank + qk_rope_head_dim]
        let kv_a = self.kv_a_proj.forward(&x)?;
        let kv_latent = kv_a.narrow(1, 0, self.kv_lora_rank)?;
        let k_rope_raw = kv_a.narrow(1, self.kv_lora_rank, self.qk_rope_head_dim)?;

        // Normalize the latent
        let kv_latent = self.kv_a_norm.forward(&kv_latent)?;

        // Up-project: [kv_lora_rank] -> [num_heads * (qk_nope_head_dim + v_head_dim)]
        let kv_b = self.kv_b_proj.forward(&kv_latent)?;
        let kv_b = kv_b.reshape((
            total_tokens,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        ))?;

        let k_nope = kv_b.narrow(2, 0, self.qk_nope_head_dim)?;
        let v = kv_b.narrow(2, self.qk_nope_head_dim, self.v_head_dim)?;

        // Expand k_rope to all heads: [total_tokens, qk_rope_head_dim] -> [total_tokens, num_heads, qk_rope_head_dim]
        let k_rope = k_rope_raw
            .unsqueeze(1)?
            .expand((total_tokens, self.num_heads, self.qk_rope_head_dim))?
            .contiguous()?;

        // Apply RoPE to the rope portions of Q and K
        let q_rope = ctx.backend.fused_rope(
            &q_rope.contiguous()?,
            ctx.positions,
            &self.attention.cos,
            &self.attention.sin,
            self.num_heads,
            self.qk_rope_head_dim,
            self.qk_rope_head_dim,
        )?;
        let k_rope = ctx.backend.fused_rope(
            &k_rope,
            ctx.positions,
            &self.attention.cos,
            &self.attention.sin,
            self.num_heads,
            self.qk_rope_head_dim,
            self.qk_rope_head_dim,
        )?;

        // Concatenate nope and rope portions for full Q and K
        // Q: [total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim]
        let q_full = Tensor::cat(&[&q_nope.contiguous()?, &q_rope], 2)?;
        // K: [total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim]
        let k_full = Tensor::cat(&[&k_nope.contiguous()?, &k_rope], 2)?;

        // For paged attention, we need flat shapes:
        // Q: [total_tokens, num_heads * head_size]
        // K: [total_tokens, num_heads * head_size] (KV heads = num_heads for MLA)
        // V: [total_tokens, num_heads * v_head_dim]
        let head_size = self.qk_nope_head_dim + self.qk_rope_head_dim;
        let _q_flat = q_full.reshape((total_tokens, self.num_heads * head_size))?;
        let k_flat = k_full.reshape((total_tokens, self.num_heads * head_size))?;
        let _v_flat = v.contiguous()?.reshape((total_tokens, self.num_heads * self.v_head_dim))?;

        // Write KV to cache and compute attention
        // Note: For MLA, we store the full (expanded) K and V in the paged cache.
        // A future optimization would store the compressed latent instead.
        let (key_cache, value_cache) = &ctx.kv_caches[layer_idx];
        let k_for_cache = k_flat.reshape((total_tokens, self.num_heads, head_size))?;
        let v_for_cache = v.contiguous()?;
        ctx.backend.reshape_and_cache(
            &k_for_cache,
            &v_for_cache,
            key_cache,
            value_cache,
            ctx.slot_mapping,
        )?;

        // Compute attention via SDPA (MLA head_size differs from standard)
        // For now, use the simple SDPA path since MLA's head geometry
        // doesn't fit the standard paged attention kernel directly.
        let attn_out = self.sdpa_attention(&q_full, &k_full, &v, ctx)?;

        // Output projection
        let attn_proj = self.attn_output.forward(&attn_out)?;

        // Fused residual add + RMSNorm for MLP
        let (x, residual) = self
            .ffn_norm
            .forward_add_fused(&attn_proj, residual, ctx.backend)?;

        // MLP + residual
        let mlp_out = self.mlp.forward(&x, ctx)?;
        mlp_out + &residual
    }

    /// Simple SDPA attention for MLA (per-sequence, handles variable head dimensions).
    fn sdpa_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        ctx: &ForwardContext,
    ) -> Result<Tensor> {
        let head_size = self.qk_nope_head_dim + self.qk_rope_head_dim;
        let scale = 1.0 / (head_size as f64).sqrt();
        let num_seqs = ctx.query_start_loc.len() - 1;
        let mut outputs = Vec::with_capacity(num_seqs);

        for seq_idx in 0..num_seqs {
            let start = ctx.query_start_loc[seq_idx];
            let end = ctx.query_start_loc[seq_idx + 1];
            let seq_len = end - start;
            if seq_len == 0 {
                continue;
            }

            // [seq_len, num_heads, head_size] -> [1, num_heads, seq_len, head_size]
            let q_seq = query.narrow(0, start, seq_len)?.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
            let k_seq = key.narrow(0, start, seq_len)?.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
            let v_seq = value.narrow(0, start, seq_len)?.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;

            // Attention: [1, num_heads, seq_len, seq_len]
            let attn_weights = (q_seq.matmul(&k_seq.t()?)? * scale)?;

            // Causal mask
            let mask: Vec<u8> = (0..seq_len)
                .flat_map(|i| (0..seq_len).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (seq_len, seq_len), query.device())?;
            let neg_inf = Tensor::new(f32::NEG_INFINITY, query.device())?.broadcast_as(attn_weights.shape())?;
            let mask_broad = mask.broadcast_as(attn_weights.shape())?;
            let attn_weights = mask_broad.where_cond(&neg_inf, &attn_weights)?;

            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            let attn_output = attn_weights.matmul(&v_seq)?;

            // [1, num_heads, seq_len, v_head_dim] -> [seq_len, num_heads * v_head_dim]
            let attn_output = attn_output
                .squeeze(0)?
                .transpose(0, 1)?
                .contiguous()?
                .reshape((seq_len, self.num_heads * self.v_head_dim))?;

            outputs.push(attn_output);
        }

        if outputs.len() == 1 {
            Ok(outputs.into_iter().next().unwrap())
        } else {
            Tensor::cat(&outputs, 0)
        }
    }
}

// ─── MoE / Dense MLP ────────────────────────────────────────────────────

/// Feed-forward network that can be either dense (SwiGLU) or MoE.
#[derive(Clone)]
enum DeepSeekMlp {
    /// Standard dense SwiGLU MLP.
    Dense(SwiGluMlp),
    /// Mixture of Experts with shared + routed experts.
    Moe(MoeLayer),
}

impl DeepSeekMlp {
    fn forward(&self, x: &Tensor, ctx: &ForwardContext) -> Result<Tensor> {
        match self {
            Self::Dense(mlp) => mlp.forward_fused(x, ctx.backend),
            Self::Moe(moe) => moe.forward(x),
        }
    }
}

/// Mixture of Experts layer.
///
/// Uses a gating network to route each token to the top-k experts,
/// plus shared experts that always process all tokens.
#[derive(Clone)]
struct MoeLayer {
    /// Gating network: [hidden_size] -> [n_routed_experts] logits.
    gate: MaybeQuantizedLinear,
    /// Routed expert MLP layers.
    experts: Vec<SwiGluMlp>,
    /// Shared expert MLP (always active).
    shared_expert: Option<SwiGluMlp>,
    /// Number of experts activated per token.
    num_experts_per_tok: usize,
}

impl MoeLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let total_tokens = x.dims()[0];
        let hidden_size = x.dims()[1];

        // Compute gate logits and select top-k experts per token
        let gate_logits = self.gate.forward(x)?; // [total_tokens, n_experts]
        let gate_probs = candle_nn::ops::softmax_last_dim(&gate_logits)?;

        // Get top-k expert indices and weights per token
        // For simplicity, iterate per token (batch-optimized routing is a future optimization)
        let mut output = Tensor::zeros((total_tokens, hidden_size), x.dtype(), x.device())?;

        for token_idx in 0..total_tokens {
            let token_probs = gate_probs.get(token_idx)?; // [n_experts]
            let probs_vec: Vec<f32> = token_probs.to_vec1()?;

            // Find top-k expert indices
            let mut indexed: Vec<(usize, f32)> = probs_vec.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let topk: Vec<(usize, f32)> = indexed
                .into_iter()
                .take(self.num_experts_per_tok)
                .collect();

            // Normalize top-k weights
            let weight_sum: f32 = topk.iter().map(|(_, w)| w).sum();
            let token_input = x.get(token_idx)?.unsqueeze(0)?; // [1, hidden_size]

            let mut token_output = Tensor::zeros((1, hidden_size), x.dtype(), x.device())?;
            for (expert_idx, weight) in &topk {
                if *expert_idx < self.experts.len() {
                    let expert_out = self.experts[*expert_idx].forward(&token_input)?;
                    let normalized_weight = weight / weight_sum;
                    token_output = (token_output + (expert_out * normalized_weight as f64)?)?;
                }
            }

            // Add to output
            output = output.slice_assign(&[token_idx..token_idx + 1, 0..hidden_size], &token_output)?;
        }

        // Add shared expert output
        if let Some(ref shared) = self.shared_expert {
            let shared_out = shared.forward(x)?;
            output = (output + shared_out)?;
        }

        Ok(output)
    }
}

// ─── DeepSeek model ──────────────────────────────────────────────────────

/// DeepSeek V2/V3 model with MLA attention and MoE.
pub struct SafetensorsDeepSeekModel {
    embed_table: Tensor,
    layers: Vec<MlaLayer>,
    norm: RmsNorm,
    lm_head: MaybeQuantizedLinear,
    config: ModelConfig,
}

impl SafetensorsDeepSeekModel {
    /// Load a DeepSeek V2/V3 model from safetensors weights.
    pub fn from_safetensors(
        ds_config: &DeepSeekConfig,
        weights: &HashMap<String, Tensor>,
        device: &Device,
        quantization: QuantizationMethod,
        serving_dtype: candle_core::DType,
    ) -> Result<Self> {
        let num_heads = ds_config.num_attention_heads;
        let num_kv_heads = ds_config.num_kv_heads();
        let head_size = ds_config.head_size();
        let hidden_size = ds_config.hidden_size;
        let rope_dim = ds_config.rope_dim();
        let max_seq_len = ds_config.max_seq_len();
        let rms_norm_eps = ds_config.rms_norm_eps;
        let rope_theta = ds_config.rope_theta as f32;
        let qk_nope_head_dim = ds_config.qk_nope_head_dim.unwrap_or(0);
        let qk_rope_head_dim = ds_config.qk_rope_head_dim.unwrap_or(head_size);
        let kv_lora_rank = ds_config.kv_lora_rank.unwrap_or(0);
        let v_head_dim = ds_config.value_head_dim();

        let model_config = ModelConfig {
            hidden_size,
            intermediate_size: ds_config.intermediate_size,
            num_heads,
            num_kv_heads,
            num_layers: ds_config.num_hidden_layers,
            head_size,
            vocab_size: ds_config.vocab_size,
            rms_norm_eps,
            rope_theta,
            rope_dim,
            max_seq_len,
        };

        // Load embeddings (always FP32 — norm/embed preservation per DTYPE-04)
        let embed_table = load_tensor_fp32(weights, "model.embed_tokens.weight", device)?;

        // Load output projection (lm_head stays FP32)
        let lm_head = if weights.contains_key("lm_head.weight") {
            load_linear(weights, "lm_head.weight", device, QuantizationMethod::None, candle_core::DType::F32)?
        } else {
            let qmm = candle_core::quantized::QMatMul::Tensor(embed_table.clone());
            MaybeQuantizedLinear::from_qmatmul(qmm, QuantizationMethod::None, device)?
        };

        // Final norm (always FP32)
        let norm_weight = load_tensor_fp32(weights, "model.norm.weight", device)?;
        let norm = RmsNorm {
            weight: norm_weight,
            eps: rms_norm_eps as f32,
        };

        // Precompute RoPE tables (only for rope portion)
        let (rope_cos, rope_sin) = precompute_rope(rope_dim, rope_theta, max_seq_len, device)?;

        // Load transformer layers
        let mut layers = Vec::with_capacity(ds_config.num_hidden_layers);
        for i in 0..ds_config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");

            // Norm weights always FP32 (DTYPE-04)
            let attn_norm_weight = load_tensor_fp32(
                weights,
                &format!("{prefix}.input_layernorm.weight"),
                device,
            )?;
            let attn_norm = RmsNorm {
                weight: attn_norm_weight,
                eps: rms_norm_eps as f32,
            };

            // ── Q path ──
            let (q_a_proj, q_a_norm, q_b_proj) = if ds_config.q_lora_rank.is_some()
                && ds_config.q_lora_rank.unwrap_or(0) > 0
                && weights.contains_key(&format!("{prefix}.self_attn.q_a_proj.weight"))
            {
                let q_a = load_linear(
                    weights,
                    &format!("{prefix}.self_attn.q_a_proj.weight"),
                    device,
                    quantization,
                    serving_dtype,
                )?;
                // q_a_layernorm is a norm weight — always FP32
                let q_a_norm_w = load_tensor_fp32(
                    weights,
                    &format!("{prefix}.self_attn.q_a_layernorm.weight"),
                    device,
                )?;
                let q_a_norm = RmsNorm {
                    weight: q_a_norm_w,
                    eps: rms_norm_eps as f32,
                };
                let q_b = load_linear(
                    weights,
                    &format!("{prefix}.self_attn.q_b_proj.weight"),
                    device,
                    quantization,
                    serving_dtype,
                )?;
                (Some(q_a), Some(q_a_norm), q_b)
            } else {
                // Direct Q projection (no compression)
                let q_proj = load_linear(
                    weights,
                    &format!("{prefix}.self_attn.q_proj.weight"),
                    device,
                    quantization,
                    serving_dtype,
                )?;
                (None, None, q_proj)
            };

            // ── KV path ──
            let kv_a_proj_name = if weights
                .contains_key(&format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight"))
            {
                format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight")
            } else {
                format!("{prefix}.self_attn.kv_a_proj.weight")
            };
            let kv_a_proj = load_linear(weights, &kv_a_proj_name, device, quantization, serving_dtype)?;

            // kv_a_layernorm is a norm weight — always FP32
            let kv_a_norm_w = load_tensor_fp32(
                weights,
                &format!("{prefix}.self_attn.kv_a_layernorm.weight"),
                device,
            )?;
            let kv_a_norm = RmsNorm {
                weight: kv_a_norm_w,
                eps: rms_norm_eps as f32,
            };

            let kv_b_proj = load_linear(
                weights,
                &format!("{prefix}.self_attn.kv_b_proj.weight"),
                device,
                quantization,
                serving_dtype,
            )?;

            let attn_output = load_linear(
                weights,
                &format!("{prefix}.self_attn.o_proj.weight"),
                device,
                quantization,
                serving_dtype,
            )?;

            // FFN norm — always FP32
            let ffn_norm_weight = load_tensor_fp32(
                weights,
                &format!("{prefix}.post_attention_layernorm.weight"),
                device,
            )?;
            let ffn_norm = RmsNorm {
                weight: ffn_norm_weight,
                eps: rms_norm_eps as f32,
            };

            // ── MLP: Dense or MoE ──
            let mlp = if ds_config.is_moe_layer(i) {
                let n_routed = ds_config.n_routed_experts.unwrap_or(0);
                let n_shared = ds_config.n_shared_experts.unwrap_or(0);
                let num_experts_per_tok = ds_config.num_experts_per_tok.unwrap_or(2);

                // Gate (small routing projection — always dense, cast to serving_dtype)
                let gate = load_linear(
                    weights,
                    &format!("{prefix}.mlp.gate.weight"),
                    device,
                    QuantizationMethod::None,
                    serving_dtype,
                )?;

                // Routed experts
                let mut experts = Vec::with_capacity(n_routed);
                for e in 0..n_routed {
                    let ep = format!("{prefix}.mlp.experts.{e}");
                    let expert_gate = load_linear(
                        weights,
                        &format!("{ep}.gate_proj.weight"),
                        device,
                        quantization,
                        serving_dtype,
                    )?;
                    let expert_down = load_linear(
                        weights,
                        &format!("{ep}.down_proj.weight"),
                        device,
                        quantization,
                        serving_dtype,
                    )?;
                    let expert_up = load_linear(
                        weights,
                        &format!("{ep}.up_proj.weight"),
                        device,
                        quantization,
                        serving_dtype,
                    )?;
                    experts.push(SwiGluMlp::new(expert_gate, expert_down, expert_up));
                }

                // Shared expert
                let shared_expert = if n_shared > 0 {
                    let sp = format!("{prefix}.mlp.shared_experts");
                    let shared_gate = load_linear(
                        weights,
                        &format!("{sp}.gate_proj.weight"),
                        device,
                        quantization,
                        serving_dtype,
                    )?;
                    let shared_down = load_linear(
                        weights,
                        &format!("{sp}.down_proj.weight"),
                        device,
                        quantization,
                        serving_dtype,
                    )?;
                    let shared_up = load_linear(
                        weights,
                        &format!("{sp}.up_proj.weight"),
                        device,
                        quantization,
                        serving_dtype,
                    )?;
                    Some(SwiGluMlp::new(shared_gate, shared_down, shared_up))
                } else {
                    None
                };

                DeepSeekMlp::Moe(MoeLayer {
                    gate,
                    experts,
                    shared_expert,
                    num_experts_per_tok,
                })
            } else {
                // Dense FFN
                let gate = load_linear(
                    weights,
                    &format!("{prefix}.mlp.gate_proj.weight"),
                    device,
                    quantization,
                    serving_dtype,
                )?;
                let down = load_linear(
                    weights,
                    &format!("{prefix}.mlp.down_proj.weight"),
                    device,
                    quantization,
                    serving_dtype,
                )?;
                let up = load_linear(
                    weights,
                    &format!("{prefix}.mlp.up_proj.weight"),
                    device,
                    quantization,
                    serving_dtype,
                )?;
                DeepSeekMlp::Dense(SwiGluMlp::new(gate, down, up))
            };

            let attention = PagedAttentionLayer::with_rope(
                num_heads,
                num_heads, // MLA: all heads have their own KV (expanded from latent)
                head_size,
                rope_cos.clone(),
                rope_sin.clone(),
            );

            layers.push(MlaLayer {
                attn_norm,
                q_a_proj,
                q_a_norm,
                q_b_proj,
                kv_a_proj,
                kv_a_norm,
                kv_b_proj,
                attn_output,
                ffn_norm,
                mlp,
                attention,
                num_heads,
                qk_nope_head_dim,
                qk_rope_head_dim,
                kv_lora_rank,
                v_head_dim,
            });
        }

        tracing::info!(
            "DeepSeek model loaded: hidden={} heads={} kv_heads={} head_size={} layers={} vocab={} mla={} kv_lora_rank={} qk_nope={} qk_rope={} v_head={} n_routed={:?} n_shared={:?}",
            model_config.hidden_size, model_config.num_heads, model_config.num_kv_heads,
            model_config.head_size, model_config.num_layers, model_config.vocab_size,
            ds_config.is_mla(), kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
            ds_config.n_routed_experts, ds_config.n_shared_experts,
        );

        Ok(Self {
            embed_table,
            layers,
            norm,
            lm_head,
            config: model_config,
        })
    }
}

impl ModelRunner for SafetensorsDeepSeekModel {
    fn forward(&self, input_ids: &Tensor, ctx: &ForwardContext) -> Result<Tensor> {
        let mut hidden_states = self.embed_table.index_select(input_ids, 0)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states, ctx, layer_idx)?;
        }

        // Final norm + output projection: fused on CUDA, sequential on CPU/Metal.
        self.norm.forward_linear_fused(&hidden_states, &self.lm_head, ctx.backend)
    }

    fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        let embeddings = self.embed_table.index_select(input_ids, 0)?;
        embeddings.mean(0)
    }

    fn embedding_table(&self) -> Option<&Tensor> {
        Some(&self.embed_table)
    }

    fn clone_model(&self) -> Box<dyn ModelRunner> {
        Box::new(SafetensorsDeepSeekModel {
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

// ─── activate_marlin ─────────────────────────────────────────────────────

impl SafetensorsDeepSeekModel {
    /// Activate Marlin fused kernel for all GPTQ/AWQ layers.
    /// DeepSeek has MLA attention (optional q_a_proj, q_b_proj, kv_a_proj, kv_b_proj)
    /// and either Dense or MoE MLP. MoE has a router gate + Vec<SwiGluMlp> experts
    /// + optional shared_expert.
    /// Returns the number of layers whose qweight_marlin was populated.
    pub(crate) fn activate_marlin(&mut self, backend: &Arc<dyn KernelBackend>) -> Result<usize> {
        let mut marlin_count = 0;

        let mut process_linear = |linear: &mut MaybeQuantizedLinear| -> Result<()> {
            match linear {
                MaybeQuantizedLinear::Gptq(ref mut gptq) => {
                    gptq.backend = Some(Arc::clone(backend));
                    if gptq.reformat_for_marlin(backend.as_ref())? {
                        marlin_count += 1;
                    }
                }
                MaybeQuantizedLinear::Awq(ref mut awq) => {
                    awq.inner_mut().backend = Some(Arc::clone(backend));
                    if awq.inner_mut().reformat_for_marlin(backend.as_ref())? {
                        marlin_count += 1;
                    }
                }
                _ => {}
            }
            Ok(())
        };

        for layer in &mut self.layers {
            // MLA Q path — q_a_proj is Option<MaybeQuantizedLinear>
            if let Some(ref mut q_a) = layer.q_a_proj {
                process_linear(q_a)?;
            }
            process_linear(&mut layer.q_b_proj)?;
            // MLA KV path
            process_linear(&mut layer.kv_a_proj)?;
            process_linear(&mut layer.kv_b_proj)?;
            // Output projection
            process_linear(&mut layer.attn_output)?;
            // MLP: Dense or MoE
            match &mut layer.mlp {
                DeepSeekMlp::Dense(mlp) => {
                    process_linear(&mut mlp.gate)?;
                    process_linear(&mut mlp.down)?;
                    process_linear(&mut mlp.up)?;
                }
                DeepSeekMlp::Moe(moe) => {
                    process_linear(&mut moe.gate)?;
                    for expert in &mut moe.experts {
                        process_linear(&mut expert.gate)?;
                        process_linear(&mut expert.down)?;
                        process_linear(&mut expert.up)?;
                    }
                    if let Some(ref mut shared) = moe.shared_expert {
                        process_linear(&mut shared.gate)?;
                        process_linear(&mut shared.down)?;
                        process_linear(&mut shared.up)?;
                    }
                }
            }
        }

        // lm_head (same pattern as Llama)
        let lm = &mut self.lm_head;
        match lm {
            MaybeQuantizedLinear::Gptq(ref mut gptq) => {
                gptq.backend = Some(Arc::clone(backend));
                if gptq.reformat_for_marlin(backend.as_ref())? {
                    marlin_count += 1;
                }
            }
            MaybeQuantizedLinear::Awq(ref mut awq) => {
                awq.inner_mut().backend = Some(Arc::clone(backend));
                if awq.inner_mut().reformat_for_marlin(backend.as_ref())? {
                    marlin_count += 1;
                }
            }
            _ => {}
        }

        Ok(marlin_count)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deepseek_v2_config_parse() {
        let json = r#"{
            "architectures": ["DeepseekV2ForCausalLM"],
            "hidden_size": 5120,
            "intermediate_size": 12288,
            "num_attention_heads": 128,
            "num_key_value_heads": 128,
            "num_hidden_layers": 60,
            "vocab_size": 102400,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "max_position_embeddings": 4096,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "n_routed_experts": 160,
            "n_shared_experts": 2,
            "num_experts_per_tok": 6,
            "first_k_dense_replace": 1,
            "model_type": "deepseek_v2"
        }"#;
        let config: DeepSeekConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.hidden_size, 5120);
        assert_eq!(config.num_attention_heads, 128);
        assert_eq!(config.num_kv_heads(), 128);
        // head_size = qk_nope_head_dim + qk_rope_head_dim
        assert_eq!(config.head_size(), 192);
        assert_eq!(config.rope_dim(), 64);
        assert_eq!(config.max_seq_len(), 4096);
        assert!(config.is_mla());
        assert_eq!(config.kv_lora_rank, Some(512));
        assert_eq!(config.q_lora_rank, Some(1536));
        assert_eq!(config.value_head_dim(), 128);
        // Layer 0 is dense (first_k_dense_replace = 1)
        assert!(!config.is_moe_layer(0));
        assert!(config.is_moe_layer(1));
        assert!(config.is_moe_layer(59));
    }

    #[test]
    fn test_deepseek_v3_config_parse() {
        let json = r#"{
            "architectures": ["DeepseekV3ForCausalLM"],
            "hidden_size": 7168,
            "intermediate_size": 18432,
            "num_attention_heads": 128,
            "num_hidden_layers": 61,
            "vocab_size": 129280,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "max_position_embeddings": 4096,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "n_routed_experts": 256,
            "n_shared_experts": 1,
            "num_experts_per_tok": 8,
            "first_k_dense_replace": 3,
            "model_type": "deepseek_v3"
        }"#;
        let config: DeepSeekConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.hidden_size, 7168);
        assert_eq!(config.num_hidden_layers, 61);
        assert_eq!(config.head_size(), 192);
        assert!(config.is_mla());
        assert_eq!(config.n_routed_experts, Some(256));
        assert_eq!(config.num_experts_per_tok, Some(8));
        // Layers 0-2 are dense
        assert!(!config.is_moe_layer(0));
        assert!(!config.is_moe_layer(2));
        assert!(config.is_moe_layer(3));
    }

    #[test]
    fn test_deepseek_v2_lite_config() {
        // DeepSeek-V2-Lite uses smaller dimensions
        let json = r#"{
            "architectures": ["DeepseekV2ForCausalLM"],
            "hidden_size": 2048,
            "intermediate_size": 10944,
            "num_attention_heads": 16,
            "num_hidden_layers": 27,
            "vocab_size": 102400,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "max_position_embeddings": 4096,
            "q_lora_rank": 0,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "n_routed_experts": 64,
            "n_shared_experts": 2,
            "num_experts_per_tok": 6,
            "first_k_dense_replace": 1,
            "model_type": "deepseek_v2"
        }"#;
        let config: DeepSeekConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.hidden_size, 2048);
        // q_lora_rank=0 means MLA is still used (kv_lora_rank > 0) but Q is direct
        assert!(config.is_mla());
        assert_eq!(config.q_lora_rank, Some(0));
        assert_eq!(config.kv_lora_rank, Some(512));
    }

    #[test]
    fn test_deepseek_config_defaults() {
        let json = r#"{
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000
        }"#;
        let config: DeepSeekConfig = serde_json::from_str(json).unwrap();
        // Without MLA config, falls back to standard MHA behavior
        assert!(!config.is_mla());
        assert_eq!(config.head_size(), 128); // 4096 / 32
        assert_eq!(config.rope_dim(), 128);
        assert_eq!(config.num_kv_heads(), 32);
        assert!(!config.is_moe_layer(0));
    }

    #[test]
    fn test_deepseek_config_moe_layer_boundary() {
        let json = r#"{
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "n_routed_experts": 64,
            "first_k_dense_replace": 5
        }"#;
        let config: DeepSeekConfig = serde_json::from_str(json).unwrap();
        // Layers 0-4 are dense, 5+ are MoE
        assert!(!config.is_moe_layer(0));
        assert!(!config.is_moe_layer(4));
        assert!(config.is_moe_layer(5));
        assert!(config.is_moe_layer(31));
    }

    #[test]
    fn test_deepseek_config_no_moe() {
        let json = r#"{
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "first_k_dense_replace": 0
        }"#;
        let config: DeepSeekConfig = serde_json::from_str(json).unwrap();
        // No routed experts means no MoE layers
        assert!(!config.is_moe_layer(0));
        assert!(!config.is_moe_layer(31));
    }

    #[test]
    fn test_deepseek_head_size_standard() {
        let json = r#"{
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000
        }"#;
        let config: DeepSeekConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.head_size(), 128); // 4096/32, no MLA
    }

    #[test]
    fn test_deepseek_head_size_mla() {
        let json = r#"{
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "qk_nope_head_dim": 96,
            "qk_rope_head_dim": 32
        }"#;
        let config: DeepSeekConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.head_size(), 128); // 96 + 32
        assert_eq!(config.rope_dim(), 32);
    }

    #[test]
    fn test_deepseek_fused_linear_matches_unfused() {
        // Verify that forward_linear_fused produces the same output as
        // separate rmsnorm + linear.forward within tolerance.
        let dev = &candle_core::Device::Cpu;
        let hidden_size = 64;
        let out_features = 32;

        let weight = candle_core::Tensor::randn(0f32, 1.0, hidden_size, dev).unwrap();
        let norm = RmsNorm {
            weight,
            eps: 1e-5,
        };

        let x = candle_core::Tensor::randn(0f32, 1.0, (4, hidden_size), dev).unwrap();

        // Create a dense MaybeQuantizedLinear
        let linear_weight =
            candle_core::Tensor::randn(0f32, 1.0, (out_features, hidden_size), dev).unwrap();
        let qmm = candle_core::quantized::QMatMul::Tensor(linear_weight);
        let linear = MaybeQuantizedLinear::QMatMul(qmm);

        let backend = crate::serving::kernels::cpu_backend::CpuBackend::new();

        // Fused path
        let fused = norm.forward_linear_fused(&x, &linear, &backend).unwrap();

        // Unfused reference
        let normed = norm.forward(&x).unwrap();
        let unfused = linear.forward(&normed).unwrap();

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
        assert_eq!(fused.dims(), &[4, out_features]);
    }

    // ─── activate_marlin tests ────────────────────────────────────────────

    /// Minimal mock backend that reports name() == "cuda" so reformat_for_marlin
    /// gets past its name check.
    struct MockCudaBackend;

    impl crate::serving::kernels::backend::KernelBackend for MockCudaBackend {
        fn name(&self) -> &'static str {
            "cuda"
        }

        fn paged_attention(
            &self, _: &candle_core::Tensor, _: &candle_core::Tensor, _: &candle_core::Tensor,
            _: &candle_core::Tensor, _: &candle_core::Tensor, _: &candle_core::Tensor,
            _: &crate::serving::kernels::backend::PagedAttentionConfig,
        ) -> candle_core::Result<()> {
            Ok(())
        }

        fn reshape_and_cache(
            &self, _: &candle_core::Tensor, _: &candle_core::Tensor,
            _: &candle_core::Tensor, _: &candle_core::Tensor, _: &candle_core::Tensor,
        ) -> candle_core::Result<()> {
            Ok(())
        }

        fn copy_blocks(
            &self, _: &candle_core::Tensor, _: &candle_core::Tensor,
            _: &candle_core::Tensor, _: usize,
        ) -> candle_core::Result<()> {
            Ok(())
        }

        fn allocate_kv_caches(
            &self, _: usize, _: usize, _: usize, _: usize,
            _: candle_core::DType, _: &candle_core::Device,
        ) -> candle_core::Result<(Vec<candle_core::Tensor>, Vec<candle_core::Tensor>)> {
            Ok((vec![], vec![]))
        }
    }

    /// Build an aligned GPTQ MaybeQuantizedLinear (N=128 multiple of 64, K=256 multiple of 128).
    fn make_gptq(out: usize, in_: usize) -> MaybeQuantizedLinear {
        let w = candle_core::Tensor::randn(0f32, 0.1, (out, in_), &candle_core::Device::Cpu).unwrap();
        MaybeQuantizedLinear::Gptq(
            crate::serving::quantization::GptqLinear::from_float(&w, None, 128).unwrap()
        )
    }

    /// Build a dense (non-quantized) MaybeQuantizedLinear.
    fn make_dense(out: usize, in_: usize) -> MaybeQuantizedLinear {
        let w = candle_core::Tensor::randn(0f32, 0.1, (out, in_), &candle_core::Device::Cpu).unwrap();
        MaybeQuantizedLinear::QMatMul(candle_core::quantized::QMatMul::Tensor(w))
    }

    fn make_rms_norm(size: usize) -> RmsNorm {
        let dev = &candle_core::Device::Cpu;
        let w = candle_core::Tensor::ones(size, candle_core::DType::F32, dev).unwrap();
        RmsNorm { weight: w, eps: 1e-5 }
    }

    fn make_attention() -> PagedAttentionLayer {
        let dev = &candle_core::Device::Cpu;
        let rope_cos = candle_core::Tensor::zeros((64, 64), candle_core::DType::F32, dev).unwrap();
        let rope_sin = candle_core::Tensor::zeros((64, 64), candle_core::DType::F32, dev).unwrap();
        PagedAttentionLayer::with_rope(4, 4, 64, rope_cos, rope_sin)
    }

    /// Build a minimal SafetensorsDeepSeekModel with Dense MLP and no q_a_proj.
    fn make_test_deepseek_model_dense_gptq() -> SafetensorsDeepSeekModel {
        let dev = &candle_core::Device::Cpu;

        let layer = MlaLayer {
            attn_norm: make_rms_norm(256),
            q_a_proj: None,
            q_a_norm: None,
            q_b_proj: make_gptq(128, 256),
            kv_a_proj: make_gptq(128, 256),
            kv_a_norm: make_rms_norm(128),
            kv_b_proj: make_gptq(128, 256),
            attn_output: make_gptq(128, 256),
            ffn_norm: make_rms_norm(256),
            mlp: DeepSeekMlp::Dense(SwiGluMlp::new(
                make_gptq(128, 256),
                make_gptq(256, 128),
                make_gptq(128, 256),
            )),
            attention: make_attention(),
            num_heads: 4,
            qk_nope_head_dim: 32,
            qk_rope_head_dim: 32,
            kv_lora_rank: 64,
            v_head_dim: 64,
        };

        let embed_w = candle_core::Tensor::zeros((100, 256), candle_core::DType::F32, dev).unwrap();
        let lm_head_w = candle_core::Tensor::zeros((100, 256), candle_core::DType::F32, dev).unwrap();
        let lm_head = MaybeQuantizedLinear::QMatMul(candle_core::quantized::QMatMul::Tensor(lm_head_w));

        SafetensorsDeepSeekModel {
            embed_table: embed_w,
            layers: vec![layer],
            norm: make_rms_norm(256),
            lm_head,
            config: ModelConfig {
                hidden_size: 256,
                intermediate_size: 128,
                num_heads: 4,
                num_kv_heads: 4,
                num_layers: 1,
                head_size: 64,
                vocab_size: 100,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                rope_dim: 32,
                max_seq_len: 64,
            },
        }
    }

    /// Build a model with q_a_proj present.
    fn make_test_deepseek_model_with_q_a() -> SafetensorsDeepSeekModel {
        let dev = &candle_core::Device::Cpu;

        let layer = MlaLayer {
            attn_norm: make_rms_norm(256),
            q_a_proj: Some(make_gptq(128, 256)),
            q_a_norm: Some(make_rms_norm(128)),
            q_b_proj: make_gptq(128, 256),
            kv_a_proj: make_gptq(128, 256),
            kv_a_norm: make_rms_norm(128),
            kv_b_proj: make_gptq(128, 256),
            attn_output: make_gptq(128, 256),
            ffn_norm: make_rms_norm(256),
            mlp: DeepSeekMlp::Dense(SwiGluMlp::new(
                make_gptq(128, 256),
                make_gptq(256, 128),
                make_gptq(128, 256),
            )),
            attention: make_attention(),
            num_heads: 4,
            qk_nope_head_dim: 32,
            qk_rope_head_dim: 32,
            kv_lora_rank: 64,
            v_head_dim: 64,
        };

        let embed_w = candle_core::Tensor::zeros((100, 256), candle_core::DType::F32, dev).unwrap();
        let lm_head_w = candle_core::Tensor::zeros((100, 256), candle_core::DType::F32, dev).unwrap();
        let lm_head = MaybeQuantizedLinear::QMatMul(candle_core::quantized::QMatMul::Tensor(lm_head_w));

        SafetensorsDeepSeekModel {
            embed_table: embed_w,
            layers: vec![layer],
            norm: make_rms_norm(256),
            lm_head,
            config: ModelConfig {
                hidden_size: 256,
                intermediate_size: 128,
                num_heads: 4,
                num_kv_heads: 4,
                num_layers: 1,
                head_size: 64,
                vocab_size: 100,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                rope_dim: 32,
                max_seq_len: 64,
            },
        }
    }

    /// Build a model with MoE MLP (4 routed experts + 1 shared).
    fn make_test_deepseek_model_moe() -> SafetensorsDeepSeekModel {
        let dev = &candle_core::Device::Cpu;
        let n_experts = 4;

        let experts = (0..n_experts).map(|_| SwiGluMlp::new(
            make_gptq(128, 256),
            make_gptq(256, 128),
            make_gptq(128, 256),
        )).collect();

        let moe = MoeLayer {
            gate: make_gptq(64, 256), // router: n_experts=64 would be too big; use smaller for test
            experts,
            shared_expert: Some(SwiGluMlp::new(
                make_gptq(128, 256),
                make_gptq(256, 128),
                make_gptq(128, 256),
            )),
            num_experts_per_tok: 2,
        };

        let layer = MlaLayer {
            attn_norm: make_rms_norm(256),
            q_a_proj: None,
            q_a_norm: None,
            q_b_proj: make_gptq(128, 256),
            kv_a_proj: make_gptq(128, 256),
            kv_a_norm: make_rms_norm(128),
            kv_b_proj: make_gptq(128, 256),
            attn_output: make_gptq(128, 256),
            ffn_norm: make_rms_norm(256),
            mlp: DeepSeekMlp::Moe(moe),
            attention: make_attention(),
            num_heads: 4,
            qk_nope_head_dim: 32,
            qk_rope_head_dim: 32,
            kv_lora_rank: 64,
            v_head_dim: 64,
        };

        let embed_w = candle_core::Tensor::zeros((100, 256), candle_core::DType::F32, dev).unwrap();
        let lm_head_w = candle_core::Tensor::zeros((100, 256), candle_core::DType::F32, dev).unwrap();
        let lm_head = MaybeQuantizedLinear::QMatMul(candle_core::quantized::QMatMul::Tensor(lm_head_w));

        SafetensorsDeepSeekModel {
            embed_table: embed_w,
            layers: vec![layer],
            norm: make_rms_norm(256),
            lm_head,
            config: ModelConfig {
                hidden_size: 256,
                intermediate_size: 128,
                num_heads: 4,
                num_kv_heads: 4,
                num_layers: 1,
                head_size: 64,
                vocab_size: 100,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                rope_dim: 32,
                max_seq_len: 64,
            },
        }
    }

    /// Build a model with all-Dense (non-quantized) layers.
    fn make_test_deepseek_model_all_dense() -> SafetensorsDeepSeekModel {
        let dev = &candle_core::Device::Cpu;

        let layer = MlaLayer {
            attn_norm: make_rms_norm(256),
            q_a_proj: None,
            q_a_norm: None,
            q_b_proj: make_dense(128, 256),
            kv_a_proj: make_dense(128, 256),
            kv_a_norm: make_rms_norm(128),
            kv_b_proj: make_dense(128, 256),
            attn_output: make_dense(128, 256),
            ffn_norm: make_rms_norm(256),
            mlp: DeepSeekMlp::Dense(SwiGluMlp::new(
                make_dense(128, 256),
                make_dense(256, 128),
                make_dense(128, 256),
            )),
            attention: make_attention(),
            num_heads: 4,
            qk_nope_head_dim: 32,
            qk_rope_head_dim: 32,
            kv_lora_rank: 64,
            v_head_dim: 64,
        };

        let embed_w = candle_core::Tensor::zeros((100, 256), candle_core::DType::F32, dev).unwrap();
        let lm_head_w = candle_core::Tensor::zeros((100, 256), candle_core::DType::F32, dev).unwrap();
        let lm_head = MaybeQuantizedLinear::QMatMul(candle_core::quantized::QMatMul::Tensor(lm_head_w));

        SafetensorsDeepSeekModel {
            embed_table: embed_w,
            layers: vec![layer],
            norm: make_rms_norm(256),
            lm_head,
            config: ModelConfig {
                hidden_size: 256,
                intermediate_size: 128,
                num_heads: 4,
                num_kv_heads: 4,
                num_layers: 1,
                head_size: 64,
                vocab_size: 100,
                rms_norm_eps: 1e-5,
                rope_theta: 10000.0,
                rope_dim: 32,
                max_seq_len: 64,
            },
        }
    }

    #[test]
    fn test_deepseek_activate_marlin_cuda_dense_mlp() {
        // q_a_proj=None, Dense MLP, 4 MLA + 3 mlp = 7 GPTQ linears
        let mut model = make_test_deepseek_model_dense_gptq();
        let backend: std::sync::Arc<dyn crate::serving::kernels::backend::KernelBackend> =
            std::sync::Arc::new(MockCudaBackend);
        let count = model.activate_marlin(&backend).unwrap();
        // q_b_proj + kv_a_proj + kv_b_proj + attn_output + gate + down + up = 7
        assert_eq!(count, 7, "expected 7 GPTQ layers reformatted (dense MLP, no q_a_proj), got {count}");
        // Verify q_b_proj is reformatted
        if let MaybeQuantizedLinear::Gptq(ref gptq) = model.layers[0].q_b_proj {
            assert!(gptq.qweight_marlin.is_some(), "q_b_proj should be reformatted");
        }
    }

    #[test]
    fn test_deepseek_activate_marlin_cuda_dense_with_q_a() {
        // q_a_proj=Some, Dense MLP, 5 MLA + 3 mlp = 8 GPTQ linears
        let mut model = make_test_deepseek_model_with_q_a();
        let backend: std::sync::Arc<dyn crate::serving::kernels::backend::KernelBackend> =
            std::sync::Arc::new(MockCudaBackend);
        let count = model.activate_marlin(&backend).unwrap();
        // q_a_proj + q_b_proj + kv_a_proj + kv_b_proj + attn_output + gate + down + up = 8
        assert_eq!(count, 8, "expected 8 GPTQ layers reformatted (dense MLP + q_a_proj), got {count}");
        if let Some(MaybeQuantizedLinear::Gptq(ref gptq)) = model.layers[0].q_a_proj {
            assert!(gptq.qweight_marlin.is_some(), "q_a_proj should be reformatted");
        } else {
            panic!("q_a_proj should be Some(Gptq)");
        }
    }

    #[test]
    fn test_deepseek_activate_marlin_cuda_moe() {
        // No q_a_proj, MoE MLP: 4 attn + 1 moe.gate + 4*3 experts + 3 shared = 20 GPTQ linears
        let mut model = make_test_deepseek_model_moe();
        let backend: std::sync::Arc<dyn crate::serving::kernels::backend::KernelBackend> =
            std::sync::Arc::new(MockCudaBackend);
        let count = model.activate_marlin(&backend).unwrap();
        // q_b_proj(1) + kv_a_proj(1) + kv_b_proj(1) + attn_output(1) = 4 attn
        // moe.gate(1) + 4 experts * 3 = 12 + shared 3 = 16 mlp
        // total = 20
        // Note: moe.gate has out=64, K=256 — 64 is a multiple of 64, 256 multiple of 128, so it reformats
        assert_eq!(count, 20, "expected 20 GPTQ layers reformatted (MoE with 4 experts + shared), got {count}");
    }

    #[test]
    fn test_deepseek_activate_marlin_cpu_skips() {
        let mut model = make_test_deepseek_model_dense_gptq();
        let backend: std::sync::Arc<dyn crate::serving::kernels::backend::KernelBackend> =
            std::sync::Arc::new(crate::serving::kernels::cpu_backend::CpuBackend::new());
        let count = model.activate_marlin(&backend).unwrap();
        assert_eq!(count, 0, "CPU backend should skip Marlin reformatting");
        // Backend field IS set on GPTQ layers even when reformat skips
        if let MaybeQuantizedLinear::Gptq(ref gptq) = model.layers[0].q_b_proj {
            assert!(gptq.backend.is_some(), "backend should be set even when reformat skips");
            assert!(gptq.qweight_marlin.is_none(), "qweight_marlin should remain None on CPU");
        }
    }

    #[test]
    fn test_deepseek_activate_marlin_dense_noop() {
        let mut model = make_test_deepseek_model_all_dense();
        let backend: std::sync::Arc<dyn crate::serving::kernels::backend::KernelBackend> =
            std::sync::Arc::new(MockCudaBackend);
        let count = model.activate_marlin(&backend).unwrap();
        assert_eq!(count, 0, "Dense (non-GPTQ) layers should not be counted");
    }
}
