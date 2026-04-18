//! Phi-3/Phi-4 architecture model with paged attention.
//!
//! Supports models with `"architectures": ["Phi3ForCausalLM"]` or `"PhiForCausalLM"`
//! in their HuggingFace `config.json`.
//!
//! Key differences from Llama:
//! - QKV packed as a single fused projection (`qkv_proj`) that is split post-projection
//! - SU/YaRN-based RoPE with partial rotation (`partial_rotary_factor`)
//! - Optional QK layer normalization
//! - Gate+Up projection fused as a single `gate_up_proj` weight
//!
//! Compatible with:
//! - Phi-3-mini, Phi-3-small, Phi-3-medium
//! - Phi-4 (phi-4)

use std::collections::HashMap;

use candle_core::{Device, Result, Tensor};
use candle_nn::Module;

use super::attention::{precompute_rope, PagedAttentionLayer};
use super::{ForwardContext, ModelConfig, ModelRunner, RmsNorm, SwiGluMlp};
use crate::serving::quantization::{MaybeQuantizedLinear, QuantizationMethod};
use crate::serving::safetensors_loader::{load_linear, load_tensor, load_tensor_fp32};

// ─── Phi-3 config ─────────────────────────────────────────────────────────

/// Phi-3/Phi-4 specific configuration parsed from `config.json`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Phi3Config {
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
    #[serde(default)]
    pub head_dim: Option<usize>,
    /// Fraction of head_dim that uses rotary embeddings (default 1.0 = full rotation).
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
    /// Whether QK normalization is applied (Phi-3-small uses this).
    #[serde(default)]
    pub qk_layernorm: bool,
    /// Sliding window attention size (None = full attention).
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub model_type: Option<String>,
    /// Original max position embeddings (for RoPE scaling).
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
}

fn default_rms_norm_eps() -> f64 {
    1e-5
}

fn default_rope_theta() -> f64 {
    10000.0
}

fn default_partial_rotary_factor() -> f64 {
    1.0
}

impl Phi3Config {
    pub fn head_size(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads
            .unwrap_or(self.num_attention_heads)
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_position_embeddings.unwrap_or(4096)
    }

    /// Compute the rotary embedding dimension based on partial_rotary_factor.
    pub fn rope_dim(&self) -> usize {
        let head_size = self.head_size();
        (head_size as f64 * self.partial_rotary_factor) as usize
    }
}

// ─── Transformer layer ───────────────────────────────────────────────────

/// Single transformer layer for Phi-3 architecture.
///
/// Phi-3 uses fused QKV projection and fused gate_up projection, reducing
/// the number of weight matrices per layer.
#[derive(Clone)]
struct Phi3Layer {
    /// Pre-attention RMS normalization.
    attn_norm: RmsNorm,
    /// Fused QKV projection: output is `[total_tokens, (num_heads + 2*num_kv_heads) * head_size]`.
    qkv_proj: MaybeQuantizedLinear,
    /// Output projection after attention.
    attn_output: MaybeQuantizedLinear,
    /// Optional QK layer normalization (Phi-3-small).
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    /// Pre-MLP RMS normalization.
    ffn_norm: RmsNorm,
    /// SwiGLU MLP (Phi-3 uses separate gate/up/down like Llama).
    mlp: SwiGluMlp,
    /// Paged attention with RoPE.
    attention: PagedAttentionLayer,
    /// Number of query heads (for splitting fused QKV).
    num_heads: usize,
    /// Number of KV heads (for splitting fused QKV).
    num_kv_heads: usize,
    /// Head dimension.
    head_size: usize,
}

impl Phi3Layer {
    fn forward(&self, x: &Tensor, ctx: &ForwardContext, layer_idx: usize) -> Result<Tensor> {
        let residual = x;
        let x = self.attn_norm.forward_fused(x, ctx.backend)?;

        // Fused QKV projection
        let qkv = self.qkv_proj.forward(&x)?;
        let total_tokens = qkv.dims()[0];

        // Split QKV: [total_tokens, (num_heads + 2*num_kv_heads) * head_size]
        let q_size = self.num_heads * self.head_size;
        let kv_size = self.num_kv_heads * self.head_size;

        let mut q = qkv.narrow(1, 0, q_size)?;
        let mut k = qkv.narrow(1, q_size, kv_size)?;
        let v = qkv.narrow(1, q_size + kv_size, kv_size)?;

        // Optional QK normalization
        if let (Some(ref q_norm), Some(ref k_norm)) = (&self.q_norm, &self.k_norm) {
            q = q.reshape((total_tokens * self.num_heads, self.head_size))?;
            q = q_norm.forward(&q)?;
            q = q.reshape((total_tokens, q_size))?;
            k = k.reshape((total_tokens * self.num_kv_heads, self.head_size))?;
            k = k_norm.forward(&k)?;
            k = k.reshape((total_tokens, kv_size))?;
        }

        // Paged attention (includes RoPE + cache write + attention dispatch)
        let attn_out = self.attention.forward(&q, &k, &v, ctx, layer_idx)?;

        // Output projection
        let attn_proj = self.attn_output.forward(&attn_out)?;

        // Fused residual add + RMSNorm for MLP
        let (x, residual) = self
            .ffn_norm
            .forward_add_fused(&attn_proj, residual, ctx.backend)?;

        // MLP + residual
        let x = self.mlp.forward_fused(&x, ctx.backend)?;
        x + &residual
    }
}

// ─── Phi-3 model ──────────────────────────────────────────────────────────

/// Phi-3/Phi-4 model with paged attention support.
pub struct SafetensorsPhi3Model {
    embed_table: Tensor,
    layers: Vec<Phi3Layer>,
    norm: RmsNorm,
    lm_head: MaybeQuantizedLinear,
    config: ModelConfig,
}

impl SafetensorsPhi3Model {
    /// Load a Phi-3/Phi-4 model from safetensors weights.
    pub fn from_safetensors(
        phi3_config: &Phi3Config,
        weights: &HashMap<String, Tensor>,
        device: &Device,
        quantization: QuantizationMethod,
        serving_dtype: candle_core::DType,
    ) -> Result<Self> {
        let num_heads = phi3_config.num_attention_heads;
        let num_kv_heads = phi3_config.num_kv_heads();
        let head_size = phi3_config.head_size();
        let hidden_size = phi3_config.hidden_size;
        let rope_dim = phi3_config.rope_dim();
        let max_seq_len = phi3_config.max_seq_len();
        let rms_norm_eps = phi3_config.rms_norm_eps;
        let rope_theta = phi3_config.rope_theta as f32;

        let model_config = ModelConfig {
            hidden_size,
            intermediate_size: phi3_config.intermediate_size,
            num_heads,
            num_kv_heads,
            num_layers: phi3_config.num_hidden_layers,
            head_size,
            vocab_size: phi3_config.vocab_size,
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

        // Precompute RoPE tables
        let (rope_cos, rope_sin) = precompute_rope(rope_dim, rope_theta, max_seq_len, device)?;

        // Load transformer layers
        let mut layers = Vec::with_capacity(phi3_config.num_hidden_layers);
        for i in 0..phi3_config.num_hidden_layers {
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

            // Phi-3 uses fused qkv_proj or separate q/k/v projections depending on variant.
            // Try fused first, fall back to separate.
            let qkv_proj = if weights.contains_key(&format!(
                "{prefix}.self_attn.qkv_proj.weight"
            )) {
                load_linear(
                    weights,
                    &format!("{prefix}.self_attn.qkv_proj.weight"),
                    device,
                    quantization,
                    serving_dtype,
                )?
            } else {
                // Separate Q/K/V -- build a fused weight by concatenating.
                // This path handles Phi-4 and some Phi-3 variants.
                let q_w = load_tensor(
                    weights,
                    &format!("{prefix}.self_attn.q_proj.weight"),
                    device,
                    serving_dtype,
                )?;
                let k_w = load_tensor(
                    weights,
                    &format!("{prefix}.self_attn.k_proj.weight"),
                    device,
                    serving_dtype,
                )?;
                let v_w = load_tensor(
                    weights,
                    &format!("{prefix}.self_attn.v_proj.weight"),
                    device,
                    serving_dtype,
                )?;
                let qkv_w = Tensor::cat(&[&q_w, &k_w, &v_w], 0)?;
                let qmm = candle_core::quantized::QMatMul::Tensor(qkv_w);
                MaybeQuantizedLinear::from_qmatmul(qmm, quantization, device)?
            };

            let attn_output = load_linear(
                weights,
                &format!("{prefix}.self_attn.o_proj.weight"),
                device,
                quantization,
                serving_dtype,
            )?;

            // Optional QK normalization (always FP32)
            let q_norm = if phi3_config.qk_layernorm {
                let w = load_tensor_fp32(
                    weights,
                    &format!("{prefix}.self_attn.q_layernorm.weight"),
                    device,
                )
                .ok();
                w.map(|weight| RmsNorm {
                    weight,
                    eps: rms_norm_eps as f32,
                })
            } else {
                None
            };
            let k_norm = if phi3_config.qk_layernorm {
                let w = load_tensor_fp32(
                    weights,
                    &format!("{prefix}.self_attn.k_layernorm.weight"),
                    device,
                )
                .ok();
                w.map(|weight| RmsNorm {
                    weight,
                    eps: rms_norm_eps as f32,
                })
            } else {
                None
            };

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

            // MLP: Phi-3 may use fused gate_up_proj or separate gate/up projections
            let mlp = if weights.contains_key(&format!("{prefix}.mlp.gate_up_proj.weight")) {
                // Fused gate_up: split the weight in half (cast to serving_dtype)
                let gate_up_w = load_tensor(
                    weights,
                    &format!("{prefix}.mlp.gate_up_proj.weight"),
                    device,
                    serving_dtype,
                )?;
                let gate_up_size = gate_up_w.dims()[0];
                let half = gate_up_size / 2;
                let gate_w = gate_up_w.narrow(0, 0, half)?;
                let up_w = gate_up_w.narrow(0, half, half)?;
                let gate = MaybeQuantizedLinear::from_qmatmul(
                    candle_core::quantized::QMatMul::Tensor(gate_w),
                    quantization,
                    device,
                )?;
                let up = MaybeQuantizedLinear::from_qmatmul(
                    candle_core::quantized::QMatMul::Tensor(up_w),
                    quantization,
                    device,
                )?;
                let down = load_linear(
                    weights,
                    &format!("{prefix}.mlp.down_proj.weight"),
                    device,
                    quantization,
                    serving_dtype,
                )?;
                SwiGluMlp::new(gate, down, up)
            } else {
                // Separate gate/up/down
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
                SwiGluMlp::new(gate, down, up)
            };

            let attention = PagedAttentionLayer::with_rope(
                num_heads,
                num_kv_heads,
                head_size,
                rope_cos.clone(),
                rope_sin.clone(),
            );

            layers.push(Phi3Layer {
                attn_norm,
                qkv_proj,
                attn_output,
                q_norm,
                k_norm,
                ffn_norm,
                mlp,
                attention,
                num_heads,
                num_kv_heads,
                head_size,
            });
        }

        tracing::info!(
            "Phi-3 model loaded: hidden={} heads={} kv_heads={} head_size={} layers={} vocab={} partial_rotary={} qk_layernorm={} sliding_window={:?}",
            model_config.hidden_size, model_config.num_heads, model_config.num_kv_heads,
            model_config.head_size, model_config.num_layers, model_config.vocab_size,
            phi3_config.partial_rotary_factor, phi3_config.qk_layernorm, phi3_config.sliding_window,
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

impl ModelRunner for SafetensorsPhi3Model {
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
        Box::new(SafetensorsPhi3Model {
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

    #[test]
    fn test_phi3_config_parse_mini() {
        let json = r#"{
            "architectures": ["Phi3ForCausalLM"],
            "hidden_size": 3072,
            "intermediate_size": 8192,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32064,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "max_position_embeddings": 4096,
            "partial_rotary_factor": 1.0,
            "model_type": "phi3"
        }"#;
        let config: Phi3Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.hidden_size, 3072);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_kv_heads(), 32);
        assert_eq!(config.head_size(), 96);
        assert_eq!(config.rope_dim(), 96); // partial_rotary = 1.0, so full head_size
        assert_eq!(config.max_seq_len(), 4096);
        assert!(!config.qk_layernorm);
        assert!(config.sliding_window.is_none());
    }

    #[test]
    fn test_phi3_config_parse_small_with_qk_norm() {
        let json = r#"{
            "architectures": ["Phi3ForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 32,
            "vocab_size": 32064,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "max_position_embeddings": 131072,
            "partial_rotary_factor": 0.5,
            "qk_layernorm": true,
            "sliding_window": 2048,
            "model_type": "phi3"
        }"#;
        let config: Phi3Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_kv_heads(), 8);
        assert_eq!(config.head_size(), 128);
        assert_eq!(config.rope_dim(), 64); // 128 * 0.5 = 64
        assert!(config.qk_layernorm);
        assert_eq!(config.sliding_window, Some(2048));
        assert_eq!(config.max_seq_len(), 131072);
    }

    #[test]
    fn test_phi3_config_defaults() {
        let json = r#"{
            "hidden_size": 3072,
            "intermediate_size": 8192,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32064
        }"#;
        let config: Phi3Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.num_kv_heads(), 32); // defaults to num_attention_heads
        assert_eq!(config.max_seq_len(), 4096); // default
        assert!((config.rms_norm_eps - 1e-5).abs() < 1e-10);
        assert!((config.rope_theta - 10000.0).abs() < 1.0);
        assert!((config.partial_rotary_factor - 1.0).abs() < 1e-10);
        assert!(!config.qk_layernorm);
    }

    #[test]
    fn test_phi4_config_parse() {
        let json = r#"{
            "architectures": ["Phi3ForCausalLM"],
            "hidden_size": 5120,
            "intermediate_size": 17920,
            "num_attention_heads": 40,
            "num_key_value_heads": 10,
            "num_hidden_layers": 40,
            "vocab_size": 100352,
            "rms_norm_eps": 1e-5,
            "rope_theta": 250000.0,
            "max_position_embeddings": 16384,
            "partial_rotary_factor": 1.0,
            "model_type": "phi3"
        }"#;
        let config: Phi3Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.hidden_size, 5120);
        assert_eq!(config.num_attention_heads, 40);
        assert_eq!(config.num_kv_heads(), 10);
        assert_eq!(config.head_size(), 128);
        assert_eq!(config.vocab_size, 100352);
        assert!((config.rope_theta - 250000.0).abs() < 1.0);
    }

    #[test]
    fn test_phi3_config_rope_dim_partial() {
        let json = r#"{
            "hidden_size": 4096,
            "intermediate_size": 8192,
            "num_attention_heads": 32,
            "num_hidden_layers": 1,
            "vocab_size": 100,
            "partial_rotary_factor": 0.25
        }"#;
        let config: Phi3Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.head_size(), 128);
        // 128 * 0.25 = 32
        assert_eq!(config.rope_dim(), 32);
    }

    #[test]
    fn test_phi3_config_with_head_dim_override() {
        let json = r#"{
            "hidden_size": 3072,
            "intermediate_size": 8192,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32064,
            "head_dim": 128,
            "partial_rotary_factor": 0.5
        }"#;
        let config: Phi3Config = serde_json::from_str(json).unwrap();
        // head_dim override takes precedence
        assert_eq!(config.head_size(), 128);
        assert_eq!(config.rope_dim(), 64); // 128 * 0.5
    }

    #[test]
    fn test_phi3_fused_linear_matches_unfused() {
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
}
