//! HuggingFace safetensors model loader.
//!
//! Loads model weights from HuggingFace safetensors format (`.safetensors` files)
//! instead of GGUF. This enables loading the thousands of models on HuggingFace Hub
//! that are published in safetensors format (the HF default since 2023).
//!
//! Usage:
//! ```ignore
//! // Point to a directory containing safetensors files + config.json
//! let model = load_model_from_safetensors("/path/to/model/", &device, quantization)?;
//! ```
//!
//! Supports Llama-family architectures (Llama, Mistral, Yi, Gemma) and explicit
//! Qwen2/2.5 and Qwen3/3.5 support with QKV biases and QK normalization.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Module;

use super::models::{ModelConfig, ModelRunner, RmsNorm, SwiGluMlp};
use super::models::attention::{precompute_rope, PagedAttentionLayer};
use super::models::ForwardContext;
use super::kernels::backend::KernelBackend;
use super::quantization::{MaybeQuantizedLinear, QuantizationMethod};

// ─── HuggingFace config.json ─────────────────────────────────────────────

/// GPTQ quantization configuration from `quantize_config.json`.
#[derive(Debug, serde::Deserialize)]
pub(crate) struct GptqConfig {
    pub(crate) bits: usize,
    pub(crate) group_size: usize,
    #[serde(default)]
    pub(crate) desc_act: bool,
    #[serde(default = "default_quant_method_gptq")]
    pub(crate) quant_method: String,
}

fn default_quant_method_gptq() -> String {
    "gptq".to_string()
}

/// AWQ quantization configuration from `quantize_config.json` or embedded in `config.json`.
#[derive(Debug, serde::Deserialize)]
pub(crate) struct AwqConfig {
    #[serde(alias = "w_bit")]
    pub(crate) bits: usize,
    #[serde(alias = "q_group_size")]
    pub(crate) group_size: usize,
    #[serde(default)]
    pub(crate) version: Option<String>,
}

/// Wrapper to parse the `quantization_config` field from HuggingFace `config.json`.
#[derive(Debug, serde::Deserialize)]
struct HfConfigWithQuantization {
    #[serde(default)]
    quantization_config: Option<serde_json::Value>,
}

/// Detect quantization method from a model directory.
///
/// Returns `Some((method, group_size, bits))` if a quantization config is found,
/// or `None` for non-quantized models.
///
/// Detection order:
/// 1. `quantize_config.json` in the model directory
/// 2. `quantization_config` embedded in `config.json`
///
/// # Errors
/// Returns an error if `desc_act=true` is detected (not supported).
pub(crate) fn detect_quant_config(
    model_dir: &Path,
) -> candle_core::Result<Option<(super::quantization::QuantizationMethod, usize, usize)>> {
    use super::quantization::QuantizationMethod;

    // 1. Try quantize_config.json first
    let qcfg_path = model_dir.join("quantize_config.json");
    if qcfg_path.exists() {
        let text = std::fs::read_to_string(&qcfg_path).map_err(|e| {
            candle_core::Error::Msg(format!(
                "Failed to read quantize_config.json: {e}"
            ))
        })?;
        let raw: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
            candle_core::Error::Msg(format!("Failed to parse quantize_config.json: {e}"))
        })?;

        let quant_method = raw
            .get("quant_method")
            .and_then(|v| v.as_str())
            .unwrap_or("gptq")
            .to_lowercase();

        match quant_method.as_str() {
            "gptq" => {
                let cfg: GptqConfig = serde_json::from_value(raw).map_err(|e| {
                    candle_core::Error::Msg(format!(
                        "Failed to parse GPTQ config in quantize_config.json: {e}"
                    ))
                })?;
                if cfg.desc_act {
                    return Err(candle_core::Error::Msg(
                        "GPTQ models with desc_act=true are not yet supported. \
                         Use a model quantized with desc_act=false."
                            .to_string(),
                    ));
                }
                return Ok(Some((QuantizationMethod::Gptq, cfg.group_size, cfg.bits)));
            }
            "awq" => {
                let cfg: AwqConfig = serde_json::from_value(raw).map_err(|e| {
                    candle_core::Error::Msg(format!(
                        "Failed to parse AWQ config in quantize_config.json: {e}"
                    ))
                })?;
                return Ok(Some((QuantizationMethod::Awq, cfg.group_size, cfg.bits)));
            }
            other => {
                tracing::warn!("Unknown quant_method in quantize_config.json: {other}, ignoring");
            }
        }
    }

    // 2. Fall back to config.json quantization_config
    let cfg_path = model_dir.join("config.json");
    if cfg_path.exists() {
        let text = std::fs::read_to_string(&cfg_path).map_err(|e| {
            candle_core::Error::Msg(format!("Failed to read config.json: {e}"))
        })?;
        let hf_cfg: HfConfigWithQuantization =
            serde_json::from_str(&text).map_err(|e| {
                candle_core::Error::Msg(format!(
                    "Failed to parse config.json for quantization_config: {e}"
                ))
            })?;

        if let Some(qcfg_val) = hf_cfg.quantization_config {
            let quant_method = qcfg_val
                .get("quant_method")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_lowercase();

            match quant_method.as_str() {
                "gptq" => {
                    let cfg: GptqConfig = serde_json::from_value(qcfg_val).map_err(|e| {
                        candle_core::Error::Msg(format!(
                            "Failed to parse embedded GPTQ config: {e}"
                        ))
                    })?;
                    if cfg.desc_act {
                        return Err(candle_core::Error::Msg(
                            "GPTQ models with desc_act=true are not yet supported. \
                             Use a model quantized with desc_act=false."
                                .to_string(),
                        ));
                    }
                    return Ok(Some((
                        QuantizationMethod::Gptq,
                        cfg.group_size,
                        cfg.bits,
                    )));
                }
                "awq" => {
                    let cfg: AwqConfig = serde_json::from_value(qcfg_val).map_err(|e| {
                        candle_core::Error::Msg(format!(
                            "Failed to parse embedded AWQ config: {e}"
                        ))
                    })?;
                    return Ok(Some((
                        QuantizationMethod::Awq,
                        cfg.group_size,
                        cfg.bits,
                    )));
                }
                _ => {
                    // Not a recognized quantization method — treat as non-quantized
                }
            }
        }
    }

    Ok(None)
}

/// Architecture detection from config.json.
///
/// Reads the `architectures` and `model_type` fields to determine which model
/// implementation to use.
///
/// This type is exposed publicly so the server crate can perform architecture
/// detection without re-reading config.json.
#[derive(Debug, Default, serde::Deserialize)]
pub struct HfArchitectureProbe {
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub model_type: Option<String>,
}

/// Detected model architecture family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArchitecture {
    Llama,
    Phi3,
    Mistral,
    DeepSeekV2,
    /// Qwen2 and Qwen2.5 — Llama-compatible with QKV biases and optional sliding window.
    Qwen2,
    /// Qwen3 and Qwen3.5 — Llama-compatible with QKV biases, QK normalization, and optional sliding window.
    Qwen3,
    /// Standard BERT encoder (BertModel, RoBERTa, gte-small, etc.)
    Bert,
    /// NomicBert encoder (nomic-embed-text-v1.5) — custom RoPE+SwiGLU architecture.
    NomicBert,
}

impl ModelArchitecture {
    /// Detect architecture from config.json fields.
    pub fn detect(architectures: &[String], model_type: Option<&str>) -> Self {
        // Check architectures array first (most reliable)
        for arch in architectures {
            let arch_lower = arch.to_lowercase();
            // NomicBert must be checked BEFORE Bert (more specific)
            if arch_lower.contains("nomicbert") {
                return Self::NomicBert;
            }
            if arch_lower == "bertmodel" || arch_lower == "robertamodel" {
                return Self::Bert;
            }
            // Qwen3/3.5 must be checked before Qwen2 (since "qwen3" doesn't contain "qwen2")
            if arch_lower.contains("qwen3") {
                return Self::Qwen3;
            }
            if arch_lower.contains("qwen2") {
                return Self::Qwen2;
            }
            if arch_lower.contains("phi3") || arch_lower.contains("phifor") {
                return Self::Phi3;
            }
            if arch_lower.contains("deepseekv2") || arch_lower.contains("deepseekv3") {
                return Self::DeepSeekV2;
            }
            if arch_lower.contains("mistral") {
                return Self::Mistral;
            }
        }

        // Fall back to model_type
        if let Some(mt) = model_type {
            let mt_lower = mt.to_lowercase();
            // NomicBert before Bert (more specific)
            if mt_lower == "nomic_bert" || mt_lower == "nomicbert" {
                return Self::NomicBert;
            }
            if mt_lower == "bert" || mt_lower == "roberta" {
                return Self::Bert;
            }
            // Qwen3/3.5 before Qwen2
            if mt_lower == "qwen3" || mt_lower == "qwen3_5" || mt_lower == "qwen3.5" {
                return Self::Qwen3;
            }
            if mt_lower == "qwen2" || mt_lower == "qwen2_5" || mt_lower == "qwen2.5" {
                return Self::Qwen2;
            }
            if mt_lower.contains("phi3") || mt_lower == "phi" {
                return Self::Phi3;
            }
            if mt_lower.contains("deepseek") {
                return Self::DeepSeekV2;
            }
            if mt_lower.contains("mistral") {
                return Self::Mistral;
            }
        }

        // Default to Llama (handles Llama, Yi, Gemma, CodeLlama, etc.)
        Self::Llama
    }

    /// Returns true for encoder-only embedding models that don't use paged KV cache.
    pub fn is_embedding_only(&self) -> bool {
        matches!(self, Self::Bert | Self::NomicBert)
    }
}

/// Minimal HuggingFace `config.json` for Llama-family models.
///
/// Also supports Qwen2/3-specific fields: `sliding_window`, `use_sliding_window`,
/// and `qk_norm` for QK normalization (Qwen3).
#[derive(Debug, serde::Deserialize)]
struct HfConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    num_hidden_layers: usize,
    vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    rope_theta: f64,
    #[serde(default)]
    max_position_embeddings: Option<usize>,
    /// Explicit head dimension (Qwen3 uses this when head_dim != hidden_size / num_heads).
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default)]
    model_type: Option<String>,
    /// Sliding window size (used by some Qwen2/3 and Mistral configs).
    #[serde(default)]
    sliding_window: Option<usize>,
    /// Whether sliding window attention is enabled (Qwen2.5, Qwen3).
    #[serde(default)]
    use_sliding_window: Option<bool>,
    /// Whether QK normalization is used (Qwen3 = true, applies per-head RMSNorm to Q and K).
    #[serde(default)]
    qk_norm: Option<bool>,
}

fn default_rms_norm_eps() -> f64 {
    1e-5
}

fn default_rope_theta() -> f64 {
    10000.0
}

impl HfConfig {
    fn head_size(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    fn max_seq_len(&self) -> usize {
        self.max_position_embeddings.unwrap_or(4096)
    }

    fn rope_dim(&self) -> usize {
        // Most Llama-family models use head_size as rope_dim
        self.head_size()
    }
}

// ─── Weight name mapping ─────────────────────────────────────────────────

/// Map HuggingFace weight names to our internal names.
///
/// HuggingFace Llama naming:
///   model.embed_tokens.weight
///   model.layers.{i}.self_attn.q_proj.weight
///   model.layers.{i}.self_attn.k_proj.weight
///   model.layers.{i}.self_attn.v_proj.weight
///   model.layers.{i}.self_attn.o_proj.weight
///   model.layers.{i}.mlp.gate_proj.weight
///   model.layers.{i}.mlp.down_proj.weight
///   model.layers.{i}.mlp.up_proj.weight
///   model.layers.{i}.input_layernorm.weight
///   model.layers.{i}.post_attention_layernorm.weight
///   model.norm.weight
///   lm_head.weight

// ─── Safetensors Llama model ──────────────────────────────────────────────

#[derive(Clone)]
struct SafetensorsLlamaLayer {
    attn_norm: RmsNorm,
    attn_q: MaybeQuantizedLinear,
    attn_k: MaybeQuantizedLinear,
    attn_v: MaybeQuantizedLinear,
    attn_output: MaybeQuantizedLinear,
    ffn_norm: RmsNorm,
    mlp: SwiGluMlp,
    attention: PagedAttentionLayer,
}

impl SafetensorsLlamaLayer {
    fn forward(&self, x: &Tensor, ctx: &ForwardContext, layer_idx: usize) -> Result<Tensor> {
        // Pre-norm attention (fused RMSNorm on CUDA)
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

        // Fused residual add + RMSNorm for MLP (eliminates intermediate tensor on CUDA)
        let (x, residual) = self.ffn_norm.forward_add_fused(&attn_proj, residual, ctx.backend)?;

        // MLP (fused SiLU+mul on CUDA) + residual
        let x = self.mlp.forward_fused(&x, ctx.backend)?;
        x + &residual
    }
}

pub struct SafetensorsLlamaModel {
    embed_table: Tensor,
    layers: Vec<SafetensorsLlamaLayer>,
    norm: RmsNorm,
    lm_head: MaybeQuantizedLinear,
    config: ModelConfig,
}

impl ModelRunner for SafetensorsLlamaModel {
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
        Box::new(SafetensorsLlamaModel {
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

impl SafetensorsLlamaModel {
    /// Activate Marlin fused kernel for all GPTQ/AWQ layers.
    ///
    /// Iterates all quantized linear layers, sets their backend reference,
    /// and calls `reformat_for_marlin()` to convert weights to Marlin tile layout.
    ///
    /// Returns the number of layers successfully reformatted.
    fn activate_marlin(&mut self, backend: &Arc<dyn KernelBackend>) -> Result<usize> {
        let mut marlin_count = 0;

        // Helper closure to process a single MaybeQuantizedLinear
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
            process_linear(&mut layer.attn_q)?;
            process_linear(&mut layer.attn_k)?;
            process_linear(&mut layer.attn_v)?;
            process_linear(&mut layer.attn_output)?;
            // MLP projections are inside SwiGluMlp — access via the MaybeQuantizedLinear fields
            process_linear(&mut layer.mlp.gate)?;
            process_linear(&mut layer.mlp.down)?;
            process_linear(&mut layer.mlp.up)?;
        }

        // Also check lm_head (typically not quantized, but be thorough)
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

// ─── Safetensors Qwen model (Qwen2/2.5/3/3.5) ──────────────────────────

/// Single transformer layer for Qwen2/3 architectures.
///
/// Extends the Llama layer with:
/// - QKV biases (required by Qwen2 and Qwen3)
/// - Optional QK normalization (Qwen3 applies per-head RMSNorm to Q and K)
#[derive(Clone)]
struct SafetensorsQwenLayer {
    attn_norm: RmsNorm,
    attn_q: MaybeQuantizedLinear,
    attn_k: MaybeQuantizedLinear,
    attn_v: MaybeQuantizedLinear,
    attn_output: MaybeQuantizedLinear,
    /// QKV biases (Qwen2 and Qwen3 use biases on Q, K, V projections).
    attn_q_bias: Option<Tensor>,
    attn_k_bias: Option<Tensor>,
    attn_v_bias: Option<Tensor>,
    /// QK normalization (Qwen3: per-head RMSNorm on Q and K).
    attn_q_norm: Option<RmsNorm>,
    attn_k_norm: Option<RmsNorm>,
    ffn_norm: RmsNorm,
    mlp: SwiGluMlp,
    attention: PagedAttentionLayer,
}

impl SafetensorsQwenLayer {
    fn forward(&self, x: &Tensor, ctx: &ForwardContext, layer_idx: usize) -> Result<Tensor> {
        // Pre-norm attention (fused RMSNorm on CUDA)
        let residual = x;
        let x = self.attn_norm.forward_fused(x, ctx.backend)?;

        // QKV projections + biases
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

/// Qwen2/3 model with paged attention support.
///
/// Reuses the Llama transformer architecture with additional support for:
/// - QKV biases (Qwen2 and Qwen3)
/// - QK normalization / per-head RMSNorm (Qwen3)
/// - Sliding window attention (some Qwen2.5/3 configs)
/// - Explicit `head_dim` (Qwen3 may differ from `hidden_size / num_heads`)
pub struct SafetensorsQwenModel {
    embed_table: Tensor,
    layers: Vec<SafetensorsQwenLayer>,
    norm: RmsNorm,
    lm_head: MaybeQuantizedLinear,
    config: ModelConfig,
}

impl ModelRunner for SafetensorsQwenModel {
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
        Box::new(SafetensorsQwenModel {
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

impl SafetensorsQwenModel {
    /// Activate Marlin fused kernel for all GPTQ/AWQ layers.
    fn activate_marlin(&mut self, backend: &Arc<dyn KernelBackend>) -> Result<usize> {
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
            process_linear(&mut layer.attn_q)?;
            process_linear(&mut layer.attn_k)?;
            process_linear(&mut layer.attn_v)?;
            process_linear(&mut layer.attn_output)?;
            process_linear(&mut layer.mlp.gate)?;
            process_linear(&mut layer.mlp.down)?;
            process_linear(&mut layer.mlp.up)?;
        }

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

// ─── Loading ─────────────────────────────────────────────────────────────

/// Load a tensor from the weight map, converting to the target dtype.
///
/// Casting rules:
/// - If the on-disk dtype already matches `dtype`, the tensor is returned as-is.
/// - If the tensor is stored as a different half-precision format (BF16 vs F16),
///   the cross-cast is skipped with an INFO log and the original dtype is kept.
/// - For all other cases (e.g. F32 → BF16), the cast is applied.
pub fn load_tensor(
    weights: &HashMap<String, Tensor>,
    name: &str,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let t = weights
        .get(name)
        .ok_or_else(|| candle_core::Error::Msg(format!("missing weight: {name}")))?
        .to_device(device)?;

    let current = t.dtype();
    if current == dtype {
        return Ok(t);
    }

    // Detect cross-cast between half-precision formats (BF16 ↔ F16).
    let is_half_precision = |d: DType| matches!(d, DType::F16 | DType::BF16);
    if is_half_precision(current) && is_half_precision(dtype) {
        tracing::info!(
            "Weight '{}' is already {:?} on disk; skipping cross-cast to {:?}",
            name, current, dtype,
        );
        return Ok(t);
    }

    t.to_dtype(dtype)
}

/// Load a tensor from the weight map, always returning it as FP32.
///
/// Use this for norm weights, embedding tables, and other parameters that
/// must remain in full precision regardless of the serving dtype.
pub fn load_tensor_fp32(
    weights: &HashMap<String, Tensor>,
    name: &str,
    device: &Device,
) -> Result<Tensor> {
    load_tensor(weights, name, device, DType::F32)
}

/// Load a weight as a MaybeQuantizedLinear.
pub fn load_linear(
    weights: &HashMap<String, Tensor>,
    name: &str,
    device: &Device,
    quantization: QuantizationMethod,
    dtype: DType,
) -> Result<MaybeQuantizedLinear> {
    let w = load_tensor(weights, name, device, dtype)?;
    let qmm = candle_core::quantized::QMatMul::Tensor(w);
    MaybeQuantizedLinear::from_qmatmul(qmm, quantization, device)
}

/// Load a raw tensor without dtype conversion (for packed int4 tensors).
fn load_raw_tensor(
    weights: &HashMap<String, Tensor>,
    name: &str,
    device: &Device,
) -> Result<Tensor> {
    weights
        .get(name)
        .ok_or_else(|| candle_core::Error::Msg(format!("missing weight: {name}")))?
        .to_device(device)
}

/// Load a GPTQ quantized linear layer from HuggingFace safetensors format.
///
/// Maps the HF tensor naming convention:
/// - `{prefix}.qweight` → packed INT4 weights `[in_features/8, out_features]` as U32
/// - `{prefix}.qzeros`  → packed INT4 zero points
/// - `{prefix}.scales`  → per-group scale factors
/// - `{prefix}.bias`    → optional bias
///
fn load_gptq_linear(
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    group_size: usize,
    device: &Device,
) -> Result<MaybeQuantizedLinear> {
    use super::quantization::GptqLinear;

    let qweight = load_raw_tensor(weights, &format!("{prefix}.qweight"), device)?;
    let qzeros = load_raw_tensor(weights, &format!("{prefix}.qzeros"), device)?;
    let scales = load_raw_tensor(weights, &format!("{prefix}.scales"), device)?;
    let bias = if weights.contains_key(&format!("{prefix}.bias")) {
        Some(load_raw_tensor(weights, &format!("{prefix}.bias"), device)?)
    } else {
        None
    };

    let layer = GptqLinear::from_parts(qweight, scales, qzeros, bias, group_size)?;
    Ok(MaybeQuantizedLinear::Gptq(layer))
}

/// Load an AWQ quantized linear layer from HuggingFace safetensors format.
///
/// Uses the same weight layout as GPTQ — the difference is only in how the model
/// was calibrated offline.
fn load_awq_linear(
    weights: &HashMap<String, Tensor>,
    prefix: &str,
    group_size: usize,
    device: &Device,
) -> Result<MaybeQuantizedLinear> {
    use super::quantization::AwqLinear;

    let qweight = load_raw_tensor(weights, &format!("{prefix}.qweight"), device)?;
    let qzeros = load_raw_tensor(weights, &format!("{prefix}.qzeros"), device)?;
    let scales = load_raw_tensor(weights, &format!("{prefix}.scales"), device)?;
    let bias = if weights.contains_key(&format!("{prefix}.bias")) {
        Some(load_raw_tensor(weights, &format!("{prefix}.bias"), device)?)
    } else {
        None
    };

    let layer = AwqLinear::from_parts(qweight, scales, qzeros, bias, group_size)?;
    Ok(MaybeQuantizedLinear::Awq(layer))
}

/// Load a model from a directory containing safetensors files and config.json.
///
/// The directory should contain:
/// - `config.json` — HuggingFace model config
/// - `*.safetensors` — model weight files (one or more shards)
///
/// If a kernel backend is provided and the device is CUDA, Marlin fused kernels
/// will be activated for all GPTQ/AWQ layers (reformatting weights to tile layout).
///
/// Returns a boxed `ModelRunner` for use with the serving engine.
pub fn load_model_from_safetensors(
    model_dir: &Path,
    device: &Device,
    quantization: QuantizationMethod,
    serving_dtype: DType,
) -> Result<Box<dyn ModelRunner>> {
    load_model_from_safetensors_with_backend(model_dir, device, quantization, None, serving_dtype)
}

/// Load a model from safetensors with an optional kernel backend for Marlin activation.
pub fn load_model_from_safetensors_with_backend(
    model_dir: &Path,
    device: &Device,
    quantization: QuantizationMethod,
    kernel_backend: Option<Arc<dyn KernelBackend>>,
    serving_dtype: DType,
) -> Result<Box<dyn ModelRunner>> {
    // ── Read config.json ──
    let config_path = model_dir.join("config.json");
    let config_text = std::fs::read_to_string(&config_path).map_err(|e| {
        candle_core::Error::Msg(format!(
            "Failed to read {}: {e}",
            config_path.display()
        ))
    })?;
    let hf_config: HfConfig = serde_json::from_str(&config_text).map_err(|e| {
        candle_core::Error::Msg(format!("Failed to parse config.json: {e}"))
    })?;

    tracing::info!(
        "HF config: hidden={} heads={} kv_heads={} layers={} vocab={} model_type={:?}",
        hf_config.hidden_size,
        hf_config.num_attention_heads,
        hf_config.num_kv_heads(),
        hf_config.num_hidden_layers,
        hf_config.vocab_size,
        hf_config.model_type,
    );

    // ── Auto-detect quantization from model directory ──
    // If caller passes None/auto, detect from quantize_config.json or config.json.
    // If caller passes an explicit method, use it (but still read group_size from config).
    let (effective_quantization, quant_group_size) = match detect_quant_config(model_dir)? {
        Some((detected_method, group_size, bits)) => {
            // Use caller's override if explicitly set; else use auto-detected method
            let method = if quantization != QuantizationMethod::None {
                quantization
            } else {
                detected_method
            };
            tracing::info!(
                "Detected {} quantization: bits={bits}, group_size={group_size}",
                method,
            );
            (method, group_size)
        }
        None => {
            // No quantization config found — use what the caller passed
            (quantization, 128)
        }
    };

    // ── Determine weight dtype ──
    // Quantized models (GPTQ/AWQ) have their own packed format — dtype casting
    // would corrupt the packed INT4/INT8 data. Always use F32 for those.
    let weight_dtype = match effective_quantization {
        QuantizationMethod::None | QuantizationMethod::Int8WeightOnly | QuantizationMethod::Fp8 => {
            serving_dtype
        }
        QuantizationMethod::Gptq | QuantizationMethod::Awq => DType::F32,
    };

    tracing::info!(
        "Loading model weights in {:?} (quantization: {})",
        weight_dtype, effective_quantization,
    );

    // ── Find and load safetensors files ──
    let mut st_files: Vec<std::path::PathBuf> = std::fs::read_dir(model_dir)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to read dir: {e}")))?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    st_files.sort();

    if st_files.is_empty() {
        return Err(candle_core::Error::Msg(format!(
            "No .safetensors files found in {}",
            model_dir.display()
        )));
    }

    tracing::info!("Loading {} safetensors file(s)...", st_files.len());

    // Use candle's MmapedSafetensors for efficient memory-mapped loading
    // SAFETY: memory-mapped files remain valid for the duration of model use
    let tensors = if st_files.len() == 1 {
        unsafe { candle_core::safetensors::MmapedSafetensors::new(&st_files[0])? }
    } else {
        unsafe { candle_core::safetensors::MmapedSafetensors::multi(&st_files)? }
    };

    // Load tensors into a HashMap. When the target device is CUDA, load I32/I16
    // tensors directly to GPU (candle fork now has const_set_i32/i16 kernels from
    // Plan 01). Non-CUDA builds or other dtypes load to CPU first.
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    for (name, _view) in tensors.tensors() {
        let tensor = if device.is_cuda() {
            // Direct GPU loading — no CPU roundtrip (requires I32/I16 CUDA kernels from Plan 01)
            match tensors.load(&name, device) {
                Ok(t) => t,
                Err(e) => {
                    // Fallback to CPU load + transfer if direct GPU loading fails for any tensor
                    tracing::debug!(
                        "Direct GPU load failed for tensor '{}', falling back to CPU: {e}",
                        name
                    );
                    tensors.load(&name, &Device::Cpu)?
                }
            }
        } else {
            tensors.load(&name, &Device::Cpu)?
        };
        weights.insert(name, tensor);
    }

    tracing::info!("Loaded {} tensors from safetensors", weights.len());

    // ── Detect model architecture ──
    let arch_probe: HfArchitectureProbe = serde_json::from_str(&config_text).map_err(|e| {
        candle_core::Error::Msg(format!("Failed to parse config.json architectures: {e}"))
    })?;
    let architecture = ModelArchitecture::detect(
        &arch_probe.architectures,
        arch_probe.model_type.as_deref(),
    );
    tracing::info!("Detected model architecture: {:?}", architecture);

    // ── Dispatch to architecture-specific loader ──
    match architecture {
        ModelArchitecture::Bert => {
            let model = super::models::bert::BertEmbeddingRunner::from_safetensors(
                model_dir, device, weight_dtype,
            )?;
            return Ok(Box::new(model));
        }
        ModelArchitecture::NomicBert => {
            let model = super::models::bert::NomicBertRunner::from_safetensors(
                model_dir, device, weight_dtype,
            )?;
            return Ok(Box::new(model));
        }
        ModelArchitecture::Phi3 => {
            let phi3_config: super::models::phi3::Phi3Config =
                serde_json::from_str(&config_text).map_err(|e| {
                    candle_core::Error::Msg(format!("Failed to parse Phi-3 config: {e}"))
                })?;
            let mut model = super::models::phi3::SafetensorsPhi3Model::from_safetensors(
                &phi3_config,
                &weights,
                device,
                effective_quantization,
                weight_dtype,
            )?;

            // Activate Marlin fused kernels for GPTQ/AWQ layers on CUDA
            if device.is_cuda() {
                if let Some(ref backend) = kernel_backend {
                    let marlin_count = model.activate_marlin(backend)?;
                    if marlin_count > 0 {
                        tracing::info!(
                            "Activated Marlin fused kernel for {marlin_count} quantized layers"
                        );
                    }
                }
            }

            return Ok(Box::new(model));
        }
        ModelArchitecture::Mistral => {
            let mistral_config: super::models::mistral::MistralConfig =
                serde_json::from_str(&config_text).map_err(|e| {
                    candle_core::Error::Msg(format!("Failed to parse Mistral config: {e}"))
                })?;
            let mut model = super::models::mistral::SafetensorsMistralModel::from_safetensors(
                &mistral_config,
                &weights,
                device,
                effective_quantization,
                weight_dtype,
            )?;

            // Activate Marlin fused kernels for GPTQ/AWQ layers on CUDA
            if device.is_cuda() {
                if let Some(ref backend) = kernel_backend {
                    let marlin_count = model.activate_marlin(backend)?;
                    if marlin_count > 0 {
                        tracing::info!(
                            "Activated Marlin fused kernel for {marlin_count} quantized layers"
                        );
                    }
                }
            }

            return Ok(Box::new(model));
        }
        ModelArchitecture::DeepSeekV2 => {
            let ds_config: super::models::deepseek::DeepSeekConfig =
                serde_json::from_str(&config_text).map_err(|e| {
                    candle_core::Error::Msg(format!("Failed to parse DeepSeek config: {e}"))
                })?;
            let mut model = super::models::deepseek::SafetensorsDeepSeekModel::from_safetensors(
                &ds_config,
                &weights,
                device,
                effective_quantization,
                weight_dtype,
            )?;

            // Activate Marlin fused kernels for GPTQ/AWQ layers on CUDA
            if device.is_cuda() {
                if let Some(ref backend) = kernel_backend {
                    let marlin_count = model.activate_marlin(backend)?;
                    if marlin_count > 0 {
                        tracing::info!(
                            "Activated Marlin fused kernel for {marlin_count} quantized layers"
                        );
                    }
                }
            }

            return Ok(Box::new(model));
        }
        ModelArchitecture::Qwen2 | ModelArchitecture::Qwen3 => {
            let is_qwen3 = architecture == ModelArchitecture::Qwen3;
            let variant_name = if is_qwen3 { "Qwen3" } else { "Qwen2" };
            let has_qk_norm = is_qwen3 || hf_config.qk_norm.unwrap_or(false);

            tracing::info!(
                "Loading {} model (biases=true, qk_norm={}, sliding_window={:?})",
                variant_name,
                has_qk_norm,
                hf_config.sliding_window,
            );

            let num_heads = hf_config.num_attention_heads;
            let num_kv_heads = hf_config.num_kv_heads();
            let head_size = hf_config.head_size();
            let hidden_size = hf_config.hidden_size;
            let rope_dim = hf_config.rope_dim();
            let max_seq_len = hf_config.max_seq_len();
            let rms_norm_eps = hf_config.rms_norm_eps;
            let rope_theta = hf_config.rope_theta as f32;

            let model_config = ModelConfig {
                hidden_size,
                intermediate_size: hf_config.intermediate_size,
                num_heads,
                num_kv_heads,
                num_layers: hf_config.num_hidden_layers,
                head_size,
                vocab_size: hf_config.vocab_size,
                rms_norm_eps,
                rope_theta,
                rope_dim,
                max_seq_len,
            };

            // Load embeddings (always FP32 — norm/embed preservation per DTYPE-04)
            let embed_table = load_tensor_fp32(&weights, "model.embed_tokens.weight", device)?;

            // Load output projection (lm_head stays FP32 — tied to embed_table)
            let lm_head = if weights.contains_key("lm_head.weight") {
                load_linear(&weights, "lm_head.weight", device, QuantizationMethod::None, DType::F32)?
            } else {
                let qmm = candle_core::quantized::QMatMul::Tensor(embed_table.clone());
                MaybeQuantizedLinear::from_qmatmul(qmm, QuantizationMethod::None, device)?
            };

            // Final norm (always FP32)
            let norm_weight = load_tensor_fp32(&weights, "model.norm.weight", device)?;
            let norm = RmsNorm { weight: norm_weight, eps: rms_norm_eps as f32 };

            // Precompute shared RoPE tables
            let (rope_cos, rope_sin) = precompute_rope(rope_dim, rope_theta, max_seq_len, device)?;

            // Build linear layer loader — projection weights cast to weight_dtype
            let load_proj = |weights: &HashMap<String, Tensor>, prefix: &str| -> Result<MaybeQuantizedLinear> {
                match effective_quantization {
                    QuantizationMethod::Gptq => load_gptq_linear(weights, prefix, quant_group_size, device),
                    QuantizationMethod::Awq => load_awq_linear(weights, prefix, quant_group_size, device),
                    other => load_linear(weights, &format!("{prefix}.weight"), device, other, weight_dtype),
                }
            };

            // Load transformer layers
            let mut layers = Vec::with_capacity(hf_config.num_hidden_layers);
            for i in 0..hf_config.num_hidden_layers {
                let prefix = format!("model.layers.{i}");

                // Norm weights are always FP32 (DTYPE-04)
                let attn_norm_weight = load_tensor_fp32(
                    &weights,
                    &format!("{prefix}.input_layernorm.weight"),
                    device,
                )?;
                let attn_norm = RmsNorm {
                    weight: attn_norm_weight,
                    eps: rms_norm_eps as f32,
                };

                let attn_q = load_proj(&weights, &format!("{prefix}.self_attn.q_proj"))?;
                let attn_k = load_proj(&weights, &format!("{prefix}.self_attn.k_proj"))?;
                let attn_v = load_proj(&weights, &format!("{prefix}.self_attn.v_proj"))?;
                let attn_output = load_proj(&weights, &format!("{prefix}.self_attn.o_proj"))?;

                // QKV biases (Qwen2 and Qwen3 both use them) — always FP32
                let attn_q_bias = weights
                    .get(&format!("{prefix}.self_attn.q_proj.bias"))
                    .map(|t| t.to_device(device).and_then(|t| t.to_dtype(DType::F32)))
                    .transpose()?;
                let attn_k_bias = weights
                    .get(&format!("{prefix}.self_attn.k_proj.bias"))
                    .map(|t| t.to_device(device).and_then(|t| t.to_dtype(DType::F32)))
                    .transpose()?;
                let attn_v_bias = weights
                    .get(&format!("{prefix}.self_attn.v_proj.bias"))
                    .map(|t| t.to_device(device).and_then(|t| t.to_dtype(DType::F32)))
                    .transpose()?;

                // QK normalization (Qwen3 only) — always FP32
                let (attn_q_norm, attn_k_norm) = if has_qk_norm {
                    let q_norm_w = weights
                        .get(&format!("{prefix}.self_attn.q_norm.weight"))
                        .map(|t| t.to_device(device).and_then(|t| t.to_dtype(DType::F32)))
                        .transpose()?;
                    let k_norm_w = weights
                        .get(&format!("{prefix}.self_attn.k_norm.weight"))
                        .map(|t| t.to_device(device).and_then(|t| t.to_dtype(DType::F32)))
                        .transpose()?;
                    (
                        q_norm_w.map(|w| RmsNorm { weight: w, eps: rms_norm_eps as f32 }),
                        k_norm_w.map(|w| RmsNorm { weight: w, eps: rms_norm_eps as f32 }),
                    )
                } else {
                    (None, None)
                };

                // FFN norm — always FP32
                let ffn_norm_weight = load_tensor_fp32(
                    &weights,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    device,
                )?;
                let ffn_norm = RmsNorm {
                    weight: ffn_norm_weight,
                    eps: rms_norm_eps as f32,
                };

                let gate = load_proj(&weights, &format!("{prefix}.mlp.gate_proj"))?;
                let down = load_proj(&weights, &format!("{prefix}.mlp.down_proj"))?;
                let up = load_proj(&weights, &format!("{prefix}.mlp.up_proj"))?;
                let mlp = SwiGluMlp::new(gate, down, up);

                let attention = PagedAttentionLayer::with_rope(
                    num_heads,
                    num_kv_heads,
                    head_size,
                    rope_cos.clone(),
                    rope_sin.clone(),
                );

                layers.push(SafetensorsQwenLayer {
                    attn_norm,
                    attn_q,
                    attn_k,
                    attn_v,
                    attn_output,
                    attn_q_bias,
                    attn_k_bias,
                    attn_v_bias,
                    attn_q_norm,
                    attn_k_norm,
                    ffn_norm,
                    mlp,
                    attention,
                });
            }

            let has_biases = layers.first().map_or(false, |l| l.attn_q_bias.is_some());
            let has_norms = layers.first().map_or(false, |l| l.attn_q_norm.is_some());
            tracing::info!(
                "{} model loaded: hidden={} heads={} kv_heads={} head_size={} layers={} vocab={} qkv_bias={} qk_norm={} quantization={}",
                variant_name,
                model_config.hidden_size, model_config.num_heads, model_config.num_kv_heads,
                model_config.head_size, model_config.num_layers, model_config.vocab_size,
                has_biases, has_norms, effective_quantization,
            );

            let mut model = SafetensorsQwenModel {
                embed_table,
                layers,
                norm,
                lm_head,
                config: model_config,
            };

            // Activate Marlin fused kernels for GPTQ/AWQ layers on CUDA
            if device.is_cuda() {
                if let Some(ref backend) = kernel_backend {
                    let marlin_count = model.activate_marlin(backend)?;
                    if marlin_count > 0 {
                        tracing::info!(
                            "Activated Marlin fused kernel for {marlin_count} quantized layers"
                        );
                    }
                }
            }

            return Ok(Box::new(model));
        }
        ModelArchitecture::Llama => {
            // Fall through to existing Llama loader below
        }
    }

    // ── Llama architecture (default path) ──

    // ── Build model config ──
    let num_heads = hf_config.num_attention_heads;
    let num_kv_heads = hf_config.num_kv_heads();
    let head_size = hf_config.head_size();
    let hidden_size = hf_config.hidden_size;
    let rope_dim = hf_config.rope_dim();
    let max_seq_len = hf_config.max_seq_len();
    let rms_norm_eps = hf_config.rms_norm_eps;
    let rope_theta = hf_config.rope_theta as f32;

    let model_config = ModelConfig {
        hidden_size,
        intermediate_size: hf_config.intermediate_size,
        num_heads,
        num_kv_heads,
        num_layers: hf_config.num_hidden_layers,
        head_size,
        vocab_size: hf_config.vocab_size,
        rms_norm_eps,
        rope_theta,
        rope_dim,
        max_seq_len,
    };

    // ── Load embeddings (always FP32 — norm/embed preservation per DTYPE-04) ──
    let embed_table = load_tensor_fp32(&weights, "model.embed_tokens.weight", device)?;

    // ── Load output projection ──
    // lm_head is never quantized — it maps hidden states to vocab logits.
    // Keep as FP32 (tied to embed_table which is FP32; QMatMul handles mixed dtypes).
    let lm_head = if weights.contains_key("lm_head.weight") {
        load_linear(&weights, "lm_head.weight", device, QuantizationMethod::None, DType::F32)?
    } else {
        // Tied embeddings: reuse embed_tokens
        let qmm = candle_core::quantized::QMatMul::Tensor(embed_table.clone());
        MaybeQuantizedLinear::from_qmatmul(qmm, QuantizationMethod::None, device)?
    };

    // ── Load final norm (always FP32) ──
    let norm_weight = load_tensor_fp32(&weights, "model.norm.weight", device)?;
    let norm = RmsNorm { weight: norm_weight, eps: rms_norm_eps as f32 };

    // ── Precompute shared RoPE tables ──
    let (rope_cos, rope_sin) = precompute_rope(rope_dim, rope_theta, max_seq_len, device)?;

    // ── Build linear layer loader ──
    // Projection weights (q/k/v/o/gate/down/up) cast to weight_dtype.
    // Norm weights and embeddings are always loaded as FP32 (DTYPE-04).
    let load_proj = |weights: &HashMap<String, Tensor>, prefix: &str| -> Result<MaybeQuantizedLinear> {
        match effective_quantization {
            QuantizationMethod::Gptq => load_gptq_linear(weights, prefix, quant_group_size, device),
            QuantizationMethod::Awq => load_awq_linear(weights, prefix, quant_group_size, device),
            other => load_linear(weights, &format!("{prefix}.weight"), device, other, weight_dtype),
        }
    };

    // ── Load transformer layers ──
    let mut layers = Vec::with_capacity(hf_config.num_hidden_layers);
    for i in 0..hf_config.num_hidden_layers {
        let prefix = format!("model.layers.{i}");

        // Norm weights are always FP32 (DTYPE-04)
        let attn_norm_weight = load_tensor_fp32(
            &weights,
            &format!("{prefix}.input_layernorm.weight"),
            device,
        )?;
        let attn_norm = RmsNorm {
            weight: attn_norm_weight,
            eps: rms_norm_eps as f32,
        };

        let attn_q = load_proj(&weights, &format!("{prefix}.self_attn.q_proj"))?;
        let attn_k = load_proj(&weights, &format!("{prefix}.self_attn.k_proj"))?;
        let attn_v = load_proj(&weights, &format!("{prefix}.self_attn.v_proj"))?;
        let attn_output = load_proj(&weights, &format!("{prefix}.self_attn.o_proj"))?;

        // FFN norm — always FP32
        let ffn_norm_weight = load_tensor_fp32(
            &weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
            device,
        )?;
        let ffn_norm = RmsNorm {
            weight: ffn_norm_weight,
            eps: rms_norm_eps as f32,
        };

        let gate = load_proj(&weights, &format!("{prefix}.mlp.gate_proj"))?;
        let down = load_proj(&weights, &format!("{prefix}.mlp.down_proj"))?;
        let up = load_proj(&weights, &format!("{prefix}.mlp.up_proj"))?;
        let mlp = SwiGluMlp::new(gate, down, up);

        let attention = PagedAttentionLayer::with_rope(
            num_heads,
            num_kv_heads,
            head_size,
            rope_cos.clone(),
            rope_sin.clone(),
        );

        layers.push(SafetensorsLlamaLayer {
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
        "Safetensors model loaded: hidden={} heads={} kv_heads={} head_size={} layers={} vocab={} quantization={}",
        model_config.hidden_size, model_config.num_heads, model_config.num_kv_heads,
        model_config.head_size, model_config.num_layers, model_config.vocab_size,
        effective_quantization,
    );

    let mut model = SafetensorsLlamaModel {
        embed_table,
        layers,
        norm,
        lm_head,
        config: model_config,
    };

    // Activate Marlin fused kernels for GPTQ/AWQ layers on CUDA
    if device.is_cuda() {
        if let Some(ref backend) = kernel_backend {
            let marlin_count = model.activate_marlin(backend)?;
            if marlin_count > 0 {
                tracing::info!(
                    "Activated Marlin fused kernel for {marlin_count} quantized layers"
                );
            }
        }
    }

    Ok(Box::new(model))
}

/// Check if a path is a directory containing safetensors files.
pub fn is_safetensors_dir(path: &Path) -> bool {
    if !path.is_dir() {
        return false;
    }
    path.join("config.json").exists()
        && std::fs::read_dir(path)
            .ok()
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .any(|e| {
                        e.path()
                            .extension()
                            .and_then(|ext| ext.to_str())
                            == Some("safetensors")
                    })
            })
            .unwrap_or(false)
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── GPTQ/AWQ config parsing tests ────────────────────────────────────

    #[test]
    fn test_gptq_config_parse() {
        let json = r#"{"bits": 4, "group_size": 128, "desc_act": false, "sym": true, "quant_method": "gptq"}"#;
        let cfg: GptqConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.bits, 4);
        assert_eq!(cfg.group_size, 128);
        assert!(!cfg.desc_act);
        assert_eq!(cfg.quant_method, "gptq");
    }

    #[test]
    fn test_gptq_config_defaults_desc_act() {
        // desc_act defaults to false when missing
        let json = r#"{"bits": 4, "group_size": 128}"#;
        let cfg: GptqConfig = serde_json::from_str(json).unwrap();
        assert!(!cfg.desc_act);
    }

    #[test]
    fn test_awq_config_parse_standalone() {
        // AWQ quantize_config.json format
        let json = r#"{"quant_method": "awq", "q_group_size": 128, "w_bit": 4, "version": "GEMM"}"#;
        let cfg: AwqConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.bits, 4);
        assert_eq!(cfg.group_size, 128);
        assert_eq!(cfg.version.as_deref(), Some("GEMM"));
    }

    #[test]
    fn test_awq_config_parse_alt_fields() {
        // AWQ using bits/group_size directly instead of aliases
        let json = r#"{"bits": 4, "group_size": 64}"#;
        let cfg: AwqConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.bits, 4);
        assert_eq!(cfg.group_size, 64);
    }

    #[test]
    fn test_gptq_config_desc_act_error() {
        let json = r#"{"bits": 4, "group_size": 128, "desc_act": true, "quant_method": "gptq"}"#;
        let cfg: GptqConfig = serde_json::from_str(json).unwrap();
        // detect_quant_config should error on desc_act=true — we test the config value here
        assert!(cfg.desc_act);
    }

    #[test]
    fn test_detect_quant_method_gptq_from_quantize_config() {
        let tmp = std::env::temp_dir().join("crabinfer_test_gptq_detect");
        std::fs::create_dir_all(&tmp).unwrap();

        // Write quantize_config.json
        let qcfg = r#"{"bits": 4, "group_size": 128, "desc_act": false, "quant_method": "gptq"}"#;
        std::fs::write(tmp.join("quantize_config.json"), qcfg).unwrap();

        // Also need a config.json so detect_quant_config can fall back
        let cfg = r#"{"hidden_size": 4096, "intermediate_size": 11008, "num_attention_heads": 32, "num_hidden_layers": 32, "vocab_size": 32000}"#;
        std::fs::write(tmp.join("config.json"), cfg).unwrap();

        let result = detect_quant_config(&tmp).unwrap();
        assert!(result.is_some());
        let (method, group_size, bits) = result.unwrap();
        assert_eq!(method, QuantizationMethod::Gptq);
        assert_eq!(group_size, 128);
        assert_eq!(bits, 4);

        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn test_detect_quant_method_awq_from_quantize_config() {
        let tmp = std::env::temp_dir().join("crabinfer_test_awq_detect");
        std::fs::create_dir_all(&tmp).unwrap();

        let qcfg = r#"{"quant_method": "awq", "q_group_size": 128, "w_bit": 4, "version": "GEMM"}"#;
        std::fs::write(tmp.join("quantize_config.json"), qcfg).unwrap();

        let cfg = r#"{"hidden_size": 4096, "intermediate_size": 11008, "num_attention_heads": 32, "num_hidden_layers": 32, "vocab_size": 32000}"#;
        std::fs::write(tmp.join("config.json"), cfg).unwrap();

        let result = detect_quant_config(&tmp).unwrap();
        assert!(result.is_some());
        let (method, group_size, bits) = result.unwrap();
        assert_eq!(method, QuantizationMethod::Awq);
        assert_eq!(group_size, 128);
        assert_eq!(bits, 4);

        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn test_detect_quant_method_embedded_in_config_json() {
        let tmp = std::env::temp_dir().join("crabinfer_test_awq_embedded_detect");
        std::fs::create_dir_all(&tmp).unwrap();

        // AWQ config embedded in config.json
        let cfg = r#"{
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "quantization_config": {
                "quant_method": "awq",
                "q_group_size": 64,
                "w_bit": 4,
                "version": "GEMM"
            }
        }"#;
        std::fs::write(tmp.join("config.json"), cfg).unwrap();

        let result = detect_quant_config(&tmp).unwrap();
        assert!(result.is_some());
        let (method, group_size, bits) = result.unwrap();
        assert_eq!(method, QuantizationMethod::Awq);
        assert_eq!(group_size, 64);
        assert_eq!(bits, 4);

        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn test_detect_quant_method_none_for_plain_model() {
        let tmp = std::env::temp_dir().join("crabinfer_test_no_quant_detect");
        std::fs::create_dir_all(&tmp).unwrap();

        let cfg = r#"{"hidden_size": 4096, "intermediate_size": 11008, "num_attention_heads": 32, "num_hidden_layers": 32, "vocab_size": 32000}"#;
        std::fs::write(tmp.join("config.json"), cfg).unwrap();

        let result = detect_quant_config(&tmp).unwrap();
        assert!(result.is_none());

        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn test_detect_quant_method_desc_act_error() {
        let tmp = std::env::temp_dir().join("crabinfer_test_desc_act_error");
        std::fs::create_dir_all(&tmp).unwrap();

        let qcfg = r#"{"bits": 4, "group_size": 128, "desc_act": true, "quant_method": "gptq"}"#;
        std::fs::write(tmp.join("quantize_config.json"), qcfg).unwrap();
        std::fs::write(tmp.join("config.json"), r#"{"hidden_size": 4096, "intermediate_size": 11008, "num_attention_heads": 32, "num_hidden_layers": 32, "vocab_size": 32000}"#).unwrap();

        let result = detect_quant_config(&tmp);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("desc_act"), "error should mention desc_act, got: {err_msg}");
    }

    #[test]
    fn test_hf_config_defaults() {
        let json = r#"{
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "vocab_size": 32000
        }"#;
        let config: HfConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_kv_heads(), 32); // defaults to num_attention_heads
        assert_eq!(config.head_size(), 128); // 4096 / 32
        assert_eq!(config.max_seq_len(), 4096); // default
        assert!((config.rms_norm_eps - 1e-5).abs() < 1e-10);
        assert!((config.rope_theta - 10000.0).abs() < 1.0);
    }

    #[test]
    fn test_hf_config_gqa() {
        let json = r#"{
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 32,
            "vocab_size": 128256,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500000.0,
            "max_position_embeddings": 131072
        }"#;
        let config: HfConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.num_kv_heads(), 8);
        assert_eq!(config.head_size(), 128);
        assert_eq!(config.max_seq_len(), 131072);
        assert!((config.rope_theta - 500000.0).abs() < 1.0);
    }

    #[test]
    fn test_hf_config_custom_head_dim() {
        let json = r#"{
            "hidden_size": 3584,
            "intermediate_size": 18944,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "num_hidden_layers": 28,
            "vocab_size": 151936,
            "head_dim": 128
        }"#;
        let config: HfConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.head_size(), 128); // explicit head_dim overrides
        assert_eq!(config.num_kv_heads(), 4);
    }

    #[test]
    fn test_is_safetensors_dir_nonexistent() {
        assert!(!is_safetensors_dir(Path::new("/nonexistent/path")));
    }

    #[test]
    fn test_is_safetensors_dir_not_dir() {
        // A file that exists but isn't a directory
        assert!(!is_safetensors_dir(Path::new("/etc/hostname")));
    }

    #[test]
    fn test_load_missing_config() {
        let tmp = std::env::temp_dir().join("crabinfer_test_no_config");
        let _ = std::fs::create_dir_all(&tmp);
        let result = load_model_from_safetensors(&tmp, &Device::Cpu, QuantizationMethod::None, DType::F32);
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    // ── Architecture detection tests ──────────────────────────────────────

    #[test]
    fn test_detect_architecture_llama() {
        let arch = ModelArchitecture::detect(
            &["LlamaForCausalLM".to_string()],
            Some("llama"),
        );
        assert_eq!(arch, ModelArchitecture::Llama);
    }

    #[test]
    fn test_detect_architecture_phi3() {
        let arch = ModelArchitecture::detect(
            &["Phi3ForCausalLM".to_string()],
            Some("phi3"),
        );
        assert_eq!(arch, ModelArchitecture::Phi3);
    }

    #[test]
    fn test_detect_architecture_phi_legacy() {
        let arch = ModelArchitecture::detect(
            &["PhiForCausalLM".to_string()],
            Some("phi"),
        );
        assert_eq!(arch, ModelArchitecture::Phi3);
    }

    #[test]
    fn test_detect_architecture_mistral() {
        let arch = ModelArchitecture::detect(
            &["MistralForCausalLM".to_string()],
            Some("mistral"),
        );
        assert_eq!(arch, ModelArchitecture::Mistral);
    }

    #[test]
    fn test_detect_architecture_deepseek_v2() {
        let arch = ModelArchitecture::detect(
            &["DeepseekV2ForCausalLM".to_string()],
            Some("deepseek_v2"),
        );
        assert_eq!(arch, ModelArchitecture::DeepSeekV2);
    }

    #[test]
    fn test_detect_architecture_deepseek_v3() {
        let arch = ModelArchitecture::detect(
            &["DeepseekV3ForCausalLM".to_string()],
            Some("deepseek_v3"),
        );
        assert_eq!(arch, ModelArchitecture::DeepSeekV2);
    }

    #[test]
    fn test_detect_architecture_default_to_llama() {
        // Unknown architectures default to Llama (handles Yi, Gemma, etc.)
        let arch = ModelArchitecture::detect(
            &["YiForCausalLM".to_string()],
            Some("yi"),
        );
        assert_eq!(arch, ModelArchitecture::Llama);
    }

    #[test]
    fn test_detect_architecture_empty() {
        let arch = ModelArchitecture::detect(&[], None);
        assert_eq!(arch, ModelArchitecture::Llama);
    }

    #[test]
    fn test_detect_architecture_model_type_fallback() {
        // When architectures list is empty, use model_type
        let arch = ModelArchitecture::detect(&[], Some("mistral"));
        assert_eq!(arch, ModelArchitecture::Mistral);
    }

    #[test]
    fn test_detect_architecture_model_type_deepseek() {
        let arch = ModelArchitecture::detect(&[], Some("deepseek_v2"));
        assert_eq!(arch, ModelArchitecture::DeepSeekV2);
    }

    // ── Qwen architecture detection tests ────────────────────────────────

    #[test]
    fn test_detect_architecture_qwen2() {
        let arch = ModelArchitecture::detect(
            &["Qwen2ForCausalLM".to_string()],
            Some("qwen2"),
        );
        assert_eq!(arch, ModelArchitecture::Qwen2);
    }

    #[test]
    fn test_detect_architecture_qwen2_5() {
        let arch = ModelArchitecture::detect(
            &["Qwen2_5ForCausalLM".to_string()],
            Some("qwen2_5"),
        );
        assert_eq!(arch, ModelArchitecture::Qwen2);
    }

    #[test]
    fn test_detect_architecture_qwen3() {
        let arch = ModelArchitecture::detect(
            &["Qwen3ForCausalLM".to_string()],
            Some("qwen3"),
        );
        assert_eq!(arch, ModelArchitecture::Qwen3);
    }

    #[test]
    fn test_detect_architecture_qwen3_5() {
        let arch = ModelArchitecture::detect(
            &["Qwen3_5ForCausalLM".to_string()],
            Some("qwen3_5"),
        );
        assert_eq!(arch, ModelArchitecture::Qwen3);
    }

    #[test]
    fn test_detect_architecture_qwen3_model_type() {
        // Detect Qwen3 from model_type alone (no architectures array)
        let arch = ModelArchitecture::detect(&[], Some("qwen3"));
        assert_eq!(arch, ModelArchitecture::Qwen3);
    }

    #[test]
    fn test_detect_architecture_qwen2_model_type() {
        // Detect Qwen2 from model_type alone
        let arch = ModelArchitecture::detect(&[], Some("qwen2"));
        assert_eq!(arch, ModelArchitecture::Qwen2);
    }

    #[test]
    fn test_detect_architecture_qwen2_5_model_type() {
        // Detect Qwen2.5 from model_type alone
        let arch = ModelArchitecture::detect(&[], Some("qwen2_5"));
        assert_eq!(arch, ModelArchitecture::Qwen2);
    }

    #[test]
    fn test_detect_architecture_qwen3_5_model_type() {
        // Detect Qwen3.5 from model_type alone
        let arch = ModelArchitecture::detect(&[], Some("qwen3_5"));
        assert_eq!(arch, ModelArchitecture::Qwen3);
    }

    // ── Qwen HfConfig parsing tests ─────────────────────────────────────

    #[test]
    fn test_hf_config_qwen2_fields() {
        let json = r#"{
            "hidden_size": 3584,
            "intermediate_size": 18944,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "num_hidden_layers": 28,
            "vocab_size": 151936,
            "head_dim": 128,
            "sliding_window": 32768,
            "use_sliding_window": true,
            "model_type": "qwen2"
        }"#;
        let config: HfConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.head_size(), 128);
        assert_eq!(config.sliding_window, Some(32768));
        assert_eq!(config.use_sliding_window, Some(true));
        assert_eq!(config.qk_norm, None);
    }

    #[test]
    fn test_hf_config_qwen3_fields() {
        let json = r#"{
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 36,
            "vocab_size": 151936,
            "head_dim": 128,
            "qk_norm": true,
            "sliding_window": 32768,
            "use_sliding_window": false,
            "model_type": "qwen3"
        }"#;
        let config: HfConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.head_size(), 128);
        assert_eq!(config.qk_norm, Some(true));
        assert_eq!(config.sliding_window, Some(32768));
        assert_eq!(config.use_sliding_window, Some(false));
    }

    // ── Bert/NomicBert architecture detection tests ──────────────────────

    #[test]
    fn test_detect_architecture_bert() {
        let arch = ModelArchitecture::detect(
            &["BertModel".to_string()],
            Some("bert"),
        );
        assert_eq!(arch, ModelArchitecture::Bert);
    }

    #[test]
    fn test_detect_architecture_nomic_bert() {
        let arch = ModelArchitecture::detect(
            &["NomicBertModel".to_string()],
            Some("nomic_bert"),
        );
        assert_eq!(arch, ModelArchitecture::NomicBert);
    }

    #[test]
    fn test_detect_architecture_bert_model_type_only() {
        let arch = ModelArchitecture::detect(&[], Some("bert"));
        assert_eq!(arch, ModelArchitecture::Bert);
    }

    #[test]
    fn test_detect_architecture_roberta() {
        let arch = ModelArchitecture::detect(
            &["RobertaModel".to_string()],
            Some("roberta"),
        );
        assert_eq!(arch, ModelArchitecture::Bert);
    }

    // ─── Wave 0: BF16/FP16 dtype stubs (DTYPE-03) ────────────────────────

    #[test]
    fn test_load_tensor_dtype_bf16_cpu() {
        // DTYPE-03: load_tensor(weights, name, &Device::Cpu, DType::BF16)
        //           should return a BF16 tensor on success (F32 on-disk → BF16)
        let t = Tensor::zeros((4, 4), DType::F32, &Device::Cpu).unwrap();
        let mut weights = HashMap::new();
        weights.insert("proj.weight".to_string(), t);
        let result = load_tensor(&weights, "proj.weight", &Device::Cpu, DType::BF16).unwrap();
        assert_eq!(result.dtype(), DType::BF16);
    }

    #[test]
    fn test_load_tensor_fp32_preserves_dtype() {
        // DTYPE-03: load_tensor_fp32(weights, name, &Device::Cpu)
        //           should always return DType::F32 regardless of on-disk dtype
        let t = Tensor::zeros((4, 4), DType::BF16, &Device::Cpu).unwrap();
        let mut weights = HashMap::new();
        weights.insert("norm.weight".to_string(), t);
        let result = load_tensor_fp32(&weights, "norm.weight", &Device::Cpu).unwrap();
        assert_eq!(result.dtype(), DType::F32);
    }

    #[test]
    fn test_load_tensor_cross_cast_skip() {
        // DTYPE-03: if tensor on disk is already BF16 and target dtype is F16,
        //           the loader should NOT cross-cast (log info, keep original BF16)
        let t = Tensor::zeros((4, 4), DType::BF16, &Device::Cpu).unwrap();
        let mut weights = HashMap::new();
        weights.insert("proj.weight".to_string(), t);
        // Request F16, but tensor is BF16 → cross-cast skipped, stays BF16
        let result = load_tensor(&weights, "proj.weight", &Device::Cpu, DType::F16).unwrap();
        assert_eq!(result.dtype(), DType::BF16);
    }
}
