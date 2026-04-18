//! Mistral architecture model with paged attention.
//!
//! Supports models with `"architectures": ["MistralForCausalLM"]` in their
//! HuggingFace `config.json`.
//!
//! Mistral is very close to Llama with the key difference being sliding window
//! attention. The model reuses `SafetensorsLlamaModel` internally with a
//! sliding window parameter since the weight layout is identical.
//!
//! Compatible with:
//! - Mistral 7B / Mistral 3 (12B, 24B)
//! - Mistral NeMo
//! - Codestral

use std::collections::HashMap;

use candle_core::{Device, Result, Tensor};
use candle_nn::Module;

use super::attention::{precompute_rope, PagedAttentionLayer};
use super::{ForwardContext, ModelConfig, ModelRunner, RmsNorm, SwiGluMlp};
use crate::serving::quantization::{MaybeQuantizedLinear, QuantizationMethod};
use crate::serving::safetensors_loader::{load_linear, load_tensor_fp32};

// ─── Mistral config ──────────────────────────────────────────────────────

/// Mistral-specific configuration parsed from `config.json`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct MistralConfig {
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
    /// Sliding window size for attention. `None` means full attention.
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub model_type: Option<String>,
}

fn default_rms_norm_eps() -> f64 {
    1e-5
}

fn default_rope_theta() -> f64 {
    10000.0
}

impl MistralConfig {
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

    pub fn rope_dim(&self) -> usize {
        self.head_size()
    }
}

// ─── Transformer layer ───────────────────────────────────────────────────

/// Single transformer layer for Mistral architecture.
///
/// Identical to Llama layer structure. Sliding window attention is handled
/// at the scheduler/cache level rather than in the model itself.
#[derive(Clone)]
struct MistralLayer {
    attn_norm: RmsNorm,
    attn_q: MaybeQuantizedLinear,
    attn_k: MaybeQuantizedLinear,
    attn_v: MaybeQuantizedLinear,
    attn_output: MaybeQuantizedLinear,
    ffn_norm: RmsNorm,
    mlp: SwiGluMlp,
    attention: PagedAttentionLayer,
}

impl MistralLayer {
    fn forward(&self, x: &Tensor, ctx: &ForwardContext, layer_idx: usize) -> Result<Tensor> {
        // Pre-norm attention
        let residual = x;
        let x = self.attn_norm.forward_fused(x, ctx.backend)?;

        // QKV projections
        let q = self.attn_q.forward(&x)?;
        let k = self.attn_k.forward(&x)?;
        let v = self.attn_v.forward(&x)?;

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

// ─── Mistral model ───────────────────────────────────────────────────────

/// Mistral model with paged attention support.
///
/// The weight layout is identical to Llama. The key difference is sliding
/// window attention, which is tracked in the config but enforced at the
/// scheduler/cache eviction level, not inside the model forward pass.
pub struct SafetensorsMistralModel {
    embed_table: Tensor,
    layers: Vec<MistralLayer>,
    norm: RmsNorm,
    lm_head: MaybeQuantizedLinear,
    config: ModelConfig,
    /// Sliding window size (informational, enforced by scheduler).
    #[allow(dead_code)]
    sliding_window: Option<usize>,
}

impl SafetensorsMistralModel {
    /// Load a Mistral model from safetensors weights.
    pub fn from_safetensors(
        mistral_config: &MistralConfig,
        weights: &HashMap<String, Tensor>,
        device: &Device,
        quantization: QuantizationMethod,
        serving_dtype: candle_core::DType,
    ) -> Result<Self> {
        let num_heads = mistral_config.num_attention_heads;
        let num_kv_heads = mistral_config.num_kv_heads();
        let head_size = mistral_config.head_size();
        let hidden_size = mistral_config.hidden_size;
        let rope_dim = mistral_config.rope_dim();
        let max_seq_len = mistral_config.max_seq_len();
        let rms_norm_eps = mistral_config.rms_norm_eps;
        let rope_theta = mistral_config.rope_theta as f32;

        let model_config = ModelConfig {
            hidden_size,
            intermediate_size: mistral_config.intermediate_size,
            num_heads,
            num_kv_heads,
            num_layers: mistral_config.num_hidden_layers,
            head_size,
            vocab_size: mistral_config.vocab_size,
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
        let mut layers = Vec::with_capacity(mistral_config.num_hidden_layers);
        for i in 0..mistral_config.num_hidden_layers {
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

            let attn_q = load_linear(
                weights,
                &format!("{prefix}.self_attn.q_proj.weight"),
                device,
                quantization,
                serving_dtype,
            )?;
            let attn_k = load_linear(
                weights,
                &format!("{prefix}.self_attn.k_proj.weight"),
                device,
                quantization,
                serving_dtype,
            )?;
            let attn_v = load_linear(
                weights,
                &format!("{prefix}.self_attn.v_proj.weight"),
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
            let mlp = SwiGluMlp::new(gate, down, up);

            let attention = PagedAttentionLayer::with_rope(
                num_heads,
                num_kv_heads,
                head_size,
                rope_cos.clone(),
                rope_sin.clone(),
            );

            layers.push(MistralLayer {
                attn_norm,
                attn_q,
                attn_k,
                attn_v,
                attn_output,
                ffn_norm,
                mlp,
                attention,
            });
        }

        tracing::info!(
            "Mistral model loaded: hidden={} heads={} kv_heads={} head_size={} layers={} vocab={} sliding_window={:?}",
            model_config.hidden_size, model_config.num_heads, model_config.num_kv_heads,
            model_config.head_size, model_config.num_layers, model_config.vocab_size,
            mistral_config.sliding_window,
        );

        Ok(Self {
            embed_table,
            layers,
            norm,
            lm_head,
            config: model_config,
            sliding_window: mistral_config.sliding_window,
        })
    }
}

impl ModelRunner for SafetensorsMistralModel {
    fn forward(&self, input_ids: &Tensor, ctx: &ForwardContext) -> Result<Tensor> {
        let mut hidden_states = self.embed_table.index_select(input_ids, 0)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states, ctx, layer_idx)?;
        }

        let hidden_states = self.norm.forward_fused(&hidden_states, ctx.backend)?;
        self.lm_head.forward(&hidden_states)
    }

    fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        let embeddings = self.embed_table.index_select(input_ids, 0)?;
        embeddings.mean(0)
    }

    fn embedding_table(&self) -> Option<&Tensor> {
        Some(&self.embed_table)
    }

    fn clone_model(&self) -> Box<dyn ModelRunner> {
        Box::new(SafetensorsMistralModel {
            embed_table: self.embed_table.clone(),
            layers: self.layers.clone(),
            norm: self.norm.clone(),
            lm_head: self.lm_head.clone(),
            config: self.config.clone(),
            sliding_window: self.sliding_window,
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
    fn test_mistral_config_parse_7b() {
        let json = r#"{
            "architectures": ["MistralForCausalLM"],
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "max_position_embeddings": 32768,
            "sliding_window": 4096,
            "model_type": "mistral"
        }"#;
        let config: MistralConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_kv_heads(), 8);
        assert_eq!(config.head_size(), 128);
        assert_eq!(config.rope_dim(), 128);
        assert_eq!(config.max_seq_len(), 32768);
        assert_eq!(config.sliding_window, Some(4096));
    }

    #[test]
    fn test_mistral_config_parse_nemo() {
        let json = r#"{
            "architectures": ["MistralForCausalLM"],
            "hidden_size": 5120,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 40,
            "vocab_size": 131072,
            "rms_norm_eps": 1e-5,
            "rope_theta": 1000000.0,
            "max_position_embeddings": 1024000,
            "head_dim": 128,
            "model_type": "mistral"
        }"#;
        let config: MistralConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.hidden_size, 5120);
        assert_eq!(config.head_size(), 128); // explicit head_dim
        assert_eq!(config.num_kv_heads(), 8);
        assert_eq!(config.max_seq_len(), 1024000);
        assert!(config.sliding_window.is_none());
        assert!((config.rope_theta - 1000000.0).abs() < 1.0);
    }

    #[test]
    fn test_mistral_config_parse_v3_small() {
        let json = r#"{
            "architectures": ["MistralForCausalLM"],
            "hidden_size": 6144,
            "intermediate_size": 16384,
            "num_attention_heads": 48,
            "num_key_value_heads": 8,
            "num_hidden_layers": 56,
            "vocab_size": 32768,
            "rms_norm_eps": 1e-5,
            "rope_theta": 1000000.0,
            "max_position_embeddings": 32768,
            "head_dim": 128,
            "sliding_window": null,
            "model_type": "mistral"
        }"#;
        let config: MistralConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.hidden_size, 6144);
        assert_eq!(config.num_attention_heads, 48);
        assert_eq!(config.num_kv_heads(), 8);
        assert_eq!(config.head_size(), 128);
        assert_eq!(config.num_hidden_layers, 56);
        assert!(config.sliding_window.is_none());
    }

    #[test]
    fn test_mistral_config_defaults() {
        let json = r#"{
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000
        }"#;
        let config: MistralConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.num_kv_heads(), 32); // defaults to num_attention_heads
        assert_eq!(config.max_seq_len(), 4096); // default
        assert!((config.rms_norm_eps - 1e-5).abs() < 1e-10);
        assert!((config.rope_theta - 10000.0).abs() < 1.0);
        assert!(config.sliding_window.is_none());
    }

    #[test]
    fn test_mistral_config_head_dim_override() {
        let json = r#"{
            "hidden_size": 5120,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "head_dim": 160
        }"#;
        let config: MistralConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.head_size(), 160);
        assert_eq!(config.rope_dim(), 160);
    }

    #[test]
    fn test_mistral_fused_linear_matches_unfused() {
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
