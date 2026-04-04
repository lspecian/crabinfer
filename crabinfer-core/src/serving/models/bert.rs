//! BERT and NomicBert embedding model runners.
//!
//! Implements `ModelRunner` for encoder-only embedding models:
//! - `BertEmbeddingRunner` — wraps candle-transformers `BertModel`, supports gte-small
//!   and any standard BERT model.
//! - `NomicBertRunner` — custom encoder with RoPE (base=1000) and SwiGLU, supports
//!   nomic-embed-text-v1.5.
//!
//! Both runners implement `embed()` with masked mean pooling and L2 normalization.
//! `forward()` returns an error (encoder-only, no causal generation).

use std::path::{Path, PathBuf};
use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, Module};

use super::{ModelConfig, ModelRunner};
use super::super::models::ForwardContext;

// ─── BertEmbeddingRunner ──────────────────────────────────────────────────

/// BERT embedding runner using candle-transformers BertModel.
///
/// Supports any standard BERT-compatible model (BERT, RoBERTa, etc.)
/// loaded from HuggingFace safetensors format.
pub struct BertEmbeddingRunner {
    model: candle_transformers::models::bert::BertModel,
    config: ModelConfig,
    model_dir: PathBuf,
    device: Device,
}

impl BertEmbeddingRunner {
    /// Load a BERT model from a safetensors directory.
    ///
    /// Reads `config.json` and all `.safetensors` files from `model_dir`.
    pub fn from_safetensors(
        model_dir: &Path,
        device: &Device,
        serving_dtype: DType,
    ) -> Result<Self> {
        use candle_transformers::models::bert;

        // Read config.json
        let config_path = model_dir.join("config.json");
        let config_text = std::fs::read_to_string(&config_path).map_err(|e| {
            candle_core::Error::Msg(format!("Failed to read BERT config.json: {e}"))
        })?;

        let bert_config: bert::Config = serde_json::from_str(&config_text).map_err(|e| {
            candle_core::Error::Msg(format!("Failed to parse BERT config.json: {e}"))
        })?;

        // Find safetensors files
        let mut st_files: Vec<std::path::PathBuf> = std::fs::read_dir(model_dir)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to read model dir: {e}")))?
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

        // Load via VarBuilder from mmaped safetensors
        // SAFETY: memory-mapped files remain valid for the duration of model use
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&st_files, serving_dtype, device)?
        };

        let model = bert::BertModel::load(vb, &bert_config)?;

        // Build ModelConfig
        let num_heads = bert_config.num_attention_heads;
        let head_size = bert_config.hidden_size / num_heads;
        let config = ModelConfig {
            hidden_size: bert_config.hidden_size,
            intermediate_size: bert_config.intermediate_size,
            num_heads,
            num_kv_heads: num_heads,
            num_layers: bert_config.num_hidden_layers,
            head_size,
            vocab_size: bert_config.vocab_size,
            rms_norm_eps: bert_config.layer_norm_eps,
            rope_theta: 0.0,
            rope_dim: 0,
            max_seq_len: bert_config.max_position_embeddings,
        };

        Ok(Self {
            model,
            config,
            model_dir: model_dir.to_path_buf(),
            device: device.clone(),
        })
    }
}

impl ModelRunner for BertEmbeddingRunner {
    fn forward(&self, _input_ids: &Tensor, _ctx: &ForwardContext) -> Result<Tensor> {
        Err(candle_core::Error::Msg(
            "BertEmbeddingRunner is encoder-only; use embed() instead".to_string(),
        ))
    }

    fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        let device = &self.device;

        // input_ids is [num_tokens] — reshape to [1, seq_len]
        let seq_len = input_ids.dims()[0];
        let input_ids_2d = input_ids
            .to_dtype(DType::U32)?
            .reshape((1, seq_len))?;

        // Create zero token_type_ids [1, seq_len]
        let token_type_ids = Tensor::zeros((1, seq_len), DType::U32, device)?;

        // Create ones attention_mask [1, seq_len]
        let attention_mask = Tensor::ones((1, seq_len), DType::U32, device)?;

        // Forward through the encoder: output is [1, seq_len, hidden_size]
        let hidden_states = self.model.forward(
            &input_ids_2d,
            &token_type_ids,
            Some(&attention_mask),
        )?;

        // Masked mean pool and L2 normalize → [hidden_size]
        let pooled = masked_mean_pool(&hidden_states, &attention_mask)?;
        // pooled is [1, hidden_size] — squeeze to [hidden_size]
        pooled.squeeze(0)
    }

    fn clone_model(&self) -> Box<dyn ModelRunner> {
        // Reload from disk on clone (BertModel is not cheaply clonable)
        let runner = BertEmbeddingRunner::from_safetensors(
            &self.model_dir,
            &self.device,
            DType::F32,
        )
        .expect("BertEmbeddingRunner: failed to reload model on clone");
        Box::new(runner)
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

// ─── Pooling ──────────────────────────────────────────────────────────────

/// Masked mean pooling with L2 normalization.
///
/// # Arguments
/// - `hidden_states`: `[batch, seq_len, hidden_size]` — encoder output
/// - `attention_mask`: `[batch, seq_len]` U32 — 1 for real tokens, 0 for padding
///
/// # Returns
/// `[batch, hidden_size]` L2-normalized embedding vectors
pub fn masked_mean_pool(hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    let (_batch, _seq_len, hidden_size) = hidden_states.dims3()?;

    // Cast attention_mask to F32 and expand to [batch, seq_len, 1]
    let mask_f32 = attention_mask.to_dtype(DType::F32)?.unsqueeze(2)?;

    // Mask out padding tokens
    let masked = hidden_states.broadcast_mul(&mask_f32)?;

    // Sum over seq_len dimension → [batch, hidden_size]
    let sum = masked.sum(1)?;

    // Sum of mask per row → [batch, 1]
    let mask_sum = mask_f32.sum(1)?;

    // Clamp to avoid division by zero
    let mask_sum_clamped = mask_sum.clamp(1e-9_f32, f32::MAX)?;

    // Mean
    let mean = sum.broadcast_div(&mask_sum_clamped)?;

    // L2 normalize: divide by sqrt(sum of squares)
    let sq = (&mean * &mean)?;
    let sq_sum = sq.sum(1)?.unsqueeze(1)?; // [batch, 1]
    let norm = sq_sum.sqrt()?;
    let norm_clamped = norm.clamp(1e-12_f32, f32::MAX)?;
    mean.broadcast_div(&norm_clamped)
}

// ─── NomicBert config ─────────────────────────────────────────────────────

fn default_n_positions() -> usize {
    8192
}

fn default_nomic_ln_eps() -> f64 {
    1e-12
}

fn default_rotary_fraction() -> f32 {
    1.0
}

fn default_nomic_rope_base() -> f32 {
    1000.0
}

/// NomicBert HuggingFace config.json — uses non-standard field names.
///
/// See: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/blob/main/config.json
#[derive(Debug, serde::Deserialize)]
pub struct NomicBertHfConfig {
    /// hidden_size (e.g. 768)
    pub n_embd: usize,
    /// num_attention_heads (e.g. 12)
    pub n_head: usize,
    /// num_hidden_layers (e.g. 12)
    pub n_layer: usize,
    /// intermediate_size (e.g. 3072, used as n_inner for fc1/fc2)
    pub n_inner: usize,
    /// vocab_size (e.g. 30528)
    pub vocab_size: usize,
    /// max_position_embeddings
    #[serde(default = "default_n_positions")]
    pub n_positions: usize,
    /// LayerNorm epsilon
    #[serde(default = "default_nomic_ln_eps")]
    pub layer_norm_epsilon: f64,
    /// Fraction of head_dim used for RoPE (usually 1.0 = full head)
    #[serde(default = "default_rotary_fraction")]
    pub rotary_emb_fraction: f32,
    /// RoPE base (nomic-bert uses 1000.0, NOT 10000.0)
    #[serde(default = "default_nomic_rope_base")]
    pub rotary_emb_base: f32,
}

// ─── NomicBert custom encoder layers ─────────────────────────────────────

/// A single NomicBert encoder layer with pre-norm, bidirectional attention,
/// RoPE, and SwiGLU MLP.
struct NomicBertLayer {
    /// Pre-norm for attention
    attn_ln: candle_nn::LayerNorm,
    /// Fused QKV weight [3*hidden, hidden]
    wqkv: Tensor,
    /// Attention output projection
    out_proj: candle_nn::Linear,
    /// Pre-norm for MLP
    mlp_ln: candle_nn::LayerNorm,
    /// fc1: [2*n_inner, n_embd] (gate + up combined)
    fc1: Tensor,
    /// fc2: [n_embd, n_inner]
    fc2: candle_nn::Linear,
    num_heads: usize,
    head_dim: usize,
    /// Precomputed RoPE cosines [max_seq_len, rope_dim]
    rope_cos: Tensor,
    /// Precomputed RoPE sines [max_seq_len, rope_dim]
    rope_sin: Tensor,
    rope_dim: usize,
}

impl NomicBertLayer {
    fn load(
        vb: VarBuilder,
        config: &NomicBertHfConfig,
        rope_cos: Tensor,
        rope_sin: Tensor,
    ) -> Result<Self> {
        let hidden = config.n_embd;
        let num_heads = config.n_head;
        let head_dim = hidden / num_heads;
        let n_inner = config.n_inner;
        let rope_dim = ((config.rotary_emb_fraction * head_dim as f32) as usize).max(1);

        // Layer norms (pre-norm style, standard LayerNorm not RMSNorm)
        let attn_ln = candle_nn::layer_norm(hidden, config.layer_norm_epsilon, vb.pp("norm1"))?;
        let mlp_ln = candle_nn::layer_norm(hidden, config.layer_norm_epsilon, vb.pp("norm2"))?;

        // Fused QKV: weight is [3*hidden, hidden]
        let wqkv = vb.get((3 * hidden, hidden), "mixer.Wqkv.weight")?;

        // Output projection
        let out_proj = candle_nn::linear(hidden, hidden, vb.pp("mixer.out_proj"))?;

        // SwiGLU MLP: fc1 is [2*n_inner, n_embd], fc2 is [n_embd, n_inner]
        let fc1 = vb.get((2 * n_inner, hidden), "mlp.fc1.weight")?;
        let fc2 = candle_nn::linear_no_bias(n_inner, hidden, vb.pp("mlp.fc2"))?;

        Ok(Self {
            attn_ln,
            wqkv,
            out_proj,
            mlp_ln,
            fc1,
            fc2,
            num_heads,
            head_dim,
            rope_cos,
            rope_sin,
            rope_dim,
        })
    }

    fn apply_rope(&self, x: &Tensor, seq_len: usize) -> Result<Tensor> {
        // x: [batch, num_heads, seq_len, head_dim]
        let rope_dim = self.rope_dim;
        let cos = self.rope_cos.narrow(0, 0, seq_len)?; // [seq_len, rope_dim]
        let sin = self.rope_sin.narrow(0, 0, seq_len)?; // [seq_len, rope_dim]

        // Only rotate the first rope_dim dims
        let x_rot = x.narrow(candle_core::D::Minus1, 0, rope_dim)?;
        let x_pass = x.narrow(candle_core::D::Minus1, rope_dim, self.head_dim - rope_dim)?;

        // Rotate x_rot: split into first/second half, apply cos/sin
        let half = rope_dim / 2;
        let x1 = x_rot.narrow(candle_core::D::Minus1, 0, half)?;
        let x2 = x_rot.narrow(candle_core::D::Minus1, half, half)?;

        // Expand cos/sin for broadcast: [1, 1, seq_len, rope_dim]
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        // cos/sin contain the full rope_dim entries
        let cos1 = cos.narrow(candle_core::D::Minus1, 0, half)?;
        let cos2 = cos.narrow(candle_core::D::Minus1, half, half)?;
        let sin1 = sin.narrow(candle_core::D::Minus1, 0, half)?;
        let sin2 = sin.narrow(candle_core::D::Minus1, half, half)?;

        let rotated_x1 = (x1.broadcast_mul(&cos1)? - x2.broadcast_mul(&sin1)?)?;
        let rotated_x2 = (x1.broadcast_mul(&sin2)? + x2.broadcast_mul(&cos2)?)?;

        let x_rot_out = Tensor::cat(&[&rotated_x1, &rotated_x2], candle_core::D::Minus1)?;

        if self.head_dim > rope_dim {
            Tensor::cat(&[&x_rot_out, &x_pass], candle_core::D::Minus1)
        } else {
            Ok(x_rot_out)
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, hidden) = x.dims3()?;
        let num_heads = self.num_heads;
        let head_dim = self.head_dim;

        // ── Attention (pre-norm) ──
        let residual = x;
        let x_norm = self.attn_ln.forward(x)?;

        // Fused QKV: [batch, seq_len, 3*hidden]
        // Use broadcast_matmul to handle [batch, seq, hidden] @ [hidden, 3*hidden]
        let wqkv_t = self.wqkv.t()?.unsqueeze(0)?; // [1, hidden, 3*hidden]
        let qkv = x_norm.broadcast_matmul(&wqkv_t)?;

        // Split Q, K, V: each [batch, seq_len, hidden]
        let q = qkv.narrow(2, 0, hidden)?;
        let k = qkv.narrow(2, hidden, hidden)?;
        let v = qkv.narrow(2, 2 * hidden, hidden)?;

        // Reshape to [batch, num_heads, seq_len, head_dim]
        let q = q.reshape((batch, seq_len, num_heads, head_dim))?.transpose(1, 2)?;
        let k = k.reshape((batch, seq_len, num_heads, head_dim))?.transpose(1, 2)?;
        let v = v.reshape((batch, seq_len, num_heads, head_dim))?.transpose(1, 2)?;

        // Apply RoPE to Q and K
        let q = self.apply_rope(&q, seq_len)?;
        let k = self.apply_rope(&k, seq_len)?;

        // Bidirectional attention: Q * K^T / sqrt(head_dim)
        let scale = (head_dim as f64).sqrt();
        let attn_scores = (q.matmul(&k.transpose(2, 3)?)? / scale)?;
        let attn_probs = candle_nn::ops::softmax(&attn_scores, candle_core::D::Minus1)?;

        // Weighted sum of V: [batch, num_heads, seq_len, head_dim]
        let attn_out = attn_probs.matmul(&v)?;

        // Reshape back: [batch, seq_len, hidden]
        let attn_out = attn_out.transpose(1, 2)?.contiguous()?.reshape((batch, seq_len, hidden))?;

        // Output projection + residual
        let attn_proj = self.out_proj.forward(&attn_out)?;
        let x = (residual + &attn_proj)?;

        // ── SwiGLU MLP (pre-norm) ──
        let residual = &x;
        let x_norm = self.mlp_ln.forward(&x)?;

        // fc1: [2*n_inner, n_embd] — split into gate [n_inner] and up [n_inner]
        let n_inner = self.fc1.dims()[0] / 2;
        let fc1_t = self.fc1.t()?.unsqueeze(0)?; // [1, hidden, 2*n_inner]
        let fc1_out = x_norm.broadcast_matmul(&fc1_t)?; // [batch, seq_len, 2*n_inner]
        let gate = fc1_out.narrow(2, 0, n_inner)?;
        let up = fc1_out.narrow(2, n_inner, n_inner)?;

        // SwiGLU: silu(gate) * up
        let gated = (candle_nn::ops::silu(&gate)? * up)?;

        // fc2: [n_embd, n_inner]
        let mlp_out = self.fc2.forward(&gated)?;

        residual + &mlp_out
    }
}

// ─── NomicBertRunner ─────────────────────────────────────────────────────

/// NomicBert embedding runner.
///
/// Implements a custom encoder matching the nomic-embed-text-v1.5 architecture:
/// - Standard word embeddings (no position embeddings — RoPE in attention)
/// - Pre-norm LayerNorm (NOT RmsNorm)
/// - Bidirectional attention with fused QKV (`Wqkv` weight)
/// - RoPE with base=1000 (not 10000!)
/// - SwiGLU MLP with combined fc1 weight [2*n_inner, n_embd]
/// - Final LayerNorm after all layers
pub struct NomicBertRunner {
    word_embeddings: candle_nn::Embedding,
    layers: Vec<NomicBertLayer>,
    final_ln: candle_nn::LayerNorm,
    config: ModelConfig,
    model_dir: PathBuf,
    device: Device,
}

impl NomicBertRunner {
    /// Precompute RoPE sin/cos tables for NomicBert.
    ///
    /// NomicBert uses RoPE base = 1000.0 (NOT 10000.0 like Llama).
    fn precompute_rope(
        rope_dim: usize,
        base: f32,
        max_seq_len: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        // theta_i = base^(-2i/rope_dim) for i in [0, rope_dim/2)
        let half = rope_dim / 2;
        let thetas: Vec<f32> = (0..half)
            .map(|i| 1.0 / base.powf(2.0 * i as f32 / rope_dim as f32))
            .collect();
        let thetas = Tensor::new(thetas.as_slice(), device)?;

        // positions: [max_seq_len]
        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let positions = Tensor::new(positions.as_slice(), device)?;

        // outer product: [max_seq_len, half]
        let angles = positions.unsqueeze(1)?.broadcast_mul(&thetas.unsqueeze(0)?)?;

        // cos [max_seq_len, half] and sin [max_seq_len, half]
        // Then concat to get [max_seq_len, rope_dim]
        let cos_half = angles.cos()?;
        let sin_half = angles.sin()?;
        let cos = Tensor::cat(&[&cos_half, &cos_half], 1)?;
        let sin_neg = (&Tensor::zeros_like(&sin_half)? - &sin_half)?;
        let sin = Tensor::cat(&[&sin_neg, &sin_half], 1)?;

        Ok((cos, sin))
    }

    /// Load NomicBert model from safetensors directory.
    pub fn from_safetensors(
        model_dir: &Path,
        device: &Device,
        serving_dtype: DType,
    ) -> Result<Self> {
        // Read config.json
        let config_path = model_dir.join("config.json");
        let config_text = std::fs::read_to_string(&config_path).map_err(|e| {
            candle_core::Error::Msg(format!("Failed to read NomicBert config.json: {e}"))
        })?;

        let nomic_config: NomicBertHfConfig = serde_json::from_str(&config_text).map_err(|e| {
            candle_core::Error::Msg(format!("Failed to parse NomicBert config.json: {e}"))
        })?;

        // Find safetensors files
        let mut st_files: Vec<std::path::PathBuf> = std::fs::read_dir(model_dir)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to read model dir: {e}")))?
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

        // Load tensors via mmaped safetensors
        // SAFETY: memory-mapped files remain valid for the duration of model use
        let vb_root = unsafe {
            VarBuilder::from_mmaped_safetensors(&st_files, serving_dtype, device)?
        };

        // nomic-embed uses `model.` prefix
        let vb = vb_root.pp("model");

        let hidden = nomic_config.n_embd;
        let num_heads = nomic_config.n_head;
        let head_dim = hidden / num_heads;
        let rope_dim = ((nomic_config.rotary_emb_fraction * head_dim as f32) as usize).max(1);

        // Precompute RoPE tables
        let (rope_cos, rope_sin) = Self::precompute_rope(
            rope_dim,
            nomic_config.rotary_emb_base,
            nomic_config.n_positions,
            device,
        )?;

        // Word embeddings (no position embeddings — RoPE handles positions)
        let word_embeddings = candle_nn::embedding(
            nomic_config.vocab_size,
            hidden,
            vb.pp("embeddings.word_embeddings"),
        )?;

        // Load transformer layers
        let layers: Result<Vec<_>> = (0..nomic_config.n_layer)
            .map(|i| {
                NomicBertLayer::load(
                    vb.pp(format!("encoder.layers.{i}")),
                    &nomic_config,
                    rope_cos.clone(),
                    rope_sin.clone(),
                )
            })
            .collect();
        let layers = layers?;

        // Final LayerNorm
        let final_ln = candle_nn::layer_norm(hidden, nomic_config.layer_norm_epsilon, vb.pp("emb_ln"))?;

        let config = ModelConfig {
            hidden_size: hidden,
            intermediate_size: nomic_config.n_inner,
            num_heads,
            num_kv_heads: num_heads,
            num_layers: nomic_config.n_layer,
            head_size: head_dim,
            vocab_size: nomic_config.vocab_size,
            rms_norm_eps: nomic_config.layer_norm_epsilon,
            rope_theta: nomic_config.rotary_emb_base,
            rope_dim,
            max_seq_len: nomic_config.n_positions,
        };

        Ok(Self {
            word_embeddings,
            layers,
            final_ln,
            config,
            model_dir: model_dir.to_path_buf(),
            device: device.clone(),
        })
    }

    /// Run the full encoder forward pass.
    ///
    /// Input: `[batch, seq_len]` u32 token IDs.
    /// Output: `[batch, seq_len, hidden_size]` hidden states.
    fn encoder_forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let mut hidden = self.word_embeddings.forward(input_ids)?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }
        self.final_ln.forward(&hidden)
    }
}

impl ModelRunner for NomicBertRunner {
    fn forward(&self, _input_ids: &Tensor, _ctx: &ForwardContext) -> Result<Tensor> {
        Err(candle_core::Error::Msg(
            "NomicBertRunner is encoder-only; use embed() instead".to_string(),
        ))
    }

    fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        let device = &self.device;

        // Reshape to [1, seq_len]
        let seq_len = input_ids.dims()[0];
        let input_ids_2d = input_ids.to_dtype(DType::U32)?.reshape((1, seq_len))?;

        // All-ones attention mask (no padding in single-sequence inference)
        let attention_mask = Tensor::ones((1, seq_len), DType::U32, device)?;

        // Full encoder forward: [1, seq_len, hidden_size]
        let hidden_states = self.encoder_forward(&input_ids_2d)?;

        // Masked mean pool and L2 normalize → [hidden_size]
        let pooled = masked_mean_pool(&hidden_states, &attention_mask)?;
        pooled.squeeze(0)
    }

    fn clone_model(&self) -> Box<dyn ModelRunner> {
        let runner = NomicBertRunner::from_safetensors(
            &self.model_dir,
            &self.device,
            DType::F32,
        )
        .expect("NomicBertRunner: failed to reload model on clone");
        Box::new(runner)
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
    use candle_core::{DType, Device, Tensor};

    // ── Task 1: BertEmbeddingRunner tests ─────────────────────────────────

    /// Test that masked_mean_pool excludes padding tokens and produces L2-normalized output.
    #[test]
    fn test_bert_mean_pool() {
        let dev = &Device::Cpu;
        let hidden_size = 8;
        let seq_len = 4;

        // Create hidden states: [1, 4, 8] — all ones for easy math
        let hidden = Tensor::ones((1, seq_len, hidden_size), DType::F32, dev).unwrap();

        // Mask: first 3 tokens are real, last is padding
        let mask = Tensor::new(&[[1u32, 1, 1, 0]], dev).unwrap();

        let pooled = masked_mean_pool(&hidden, &mask).unwrap();

        // Shape should be [1, 8]
        assert_eq!(pooled.dims(), &[1, hidden_size]);

        // All values should be equal (we had all-ones input, uniform mask)
        let vals: Vec<f32> = pooled.flatten_all().unwrap().to_vec1().unwrap();
        // L2 norm of [1,1,1,1,1,1,1,1] is sqrt(8), so normalized value = 1/sqrt(8)
        let expected = 1.0_f32 / (hidden_size as f32).sqrt();
        for v in &vals {
            assert!((v - expected).abs() < 1e-5, "expected {expected}, got {v}");
        }
    }

    /// Test that masked_mean_pool with all padding (mask=0) handles gracefully (no NaN).
    #[test]
    fn test_bert_mean_pool_no_nan() {
        let dev = &Device::Cpu;
        let hidden = Tensor::ones((1, 4, 8), DType::F32, dev).unwrap();
        let mask = Tensor::ones((1, 4), DType::U32, dev).unwrap();

        let pooled = masked_mean_pool(&hidden, &mask).unwrap();
        let vals: Vec<f32> = pooled.flatten_all().unwrap().to_vec1().unwrap();
        for v in &vals {
            assert!(v.is_finite(), "pooled output should be finite, got {v}");
        }
    }

    /// Test that BertEmbeddingRunner::forward() returns an error (encoder-only).
    ///
    /// We can test this without loading a real model by checking the error path
    /// through the embed() method indirectly via a random-weight stub.
    #[test]
    fn test_bert_forward_returns_error() {
        // We create a minimal in-memory BERT model using random weights
        // via VarBuilder::from_tensors.
        let dev = &Device::Cpu;
        let hidden_size = 16usize;
        let num_layers = 1usize;
        let num_heads = 2usize;
        let intermediate_size = 32usize;
        let vocab_size = 100usize;
        let max_pos = 32usize;
        let type_vocab_size = 2usize;
        let head_size = hidden_size / num_heads;

        let mut tensors: HashMap<String, Tensor> = HashMap::new();

        // Build a minimal BERT model tensor map
        let add_linear = |tensors: &mut HashMap<String, Tensor>, name: &str, out: usize, inp: usize| {
            tensors.insert(format!("{name}.weight"), Tensor::randn(0f32, 0.02, (out, inp), dev).unwrap());
            tensors.insert(format!("{name}.bias"), Tensor::zeros(out, DType::F32, dev).unwrap());
        };
        let add_ln = |tensors: &mut HashMap<String, Tensor>, name: &str, size: usize| {
            tensors.insert(format!("{name}.weight"), Tensor::ones(size, DType::F32, dev).unwrap());
            tensors.insert(format!("{name}.bias"), Tensor::zeros(size, DType::F32, dev).unwrap());
        };
        let add_embed = |tensors: &mut HashMap<String, Tensor>, name: &str, rows: usize, cols: usize| {
            tensors.insert(format!("{name}.weight"), Tensor::randn(0f32, 0.02, (rows, cols), dev).unwrap());
        };

        // Embeddings
        add_embed(&mut tensors, "embeddings.word_embeddings", vocab_size, hidden_size);
        add_embed(&mut tensors, "embeddings.position_embeddings", max_pos, hidden_size);
        add_embed(&mut tensors, "embeddings.token_type_embeddings", type_vocab_size, hidden_size);
        add_ln(&mut tensors, "embeddings.LayerNorm", hidden_size);

        // Encoder layer 0
        for layer_i in 0..num_layers {
            let lp = format!("encoder.layer.{layer_i}");
            // Self-attention
            add_linear(&mut tensors, &format!("{lp}.attention.self.query"), hidden_size, hidden_size);
            add_linear(&mut tensors, &format!("{lp}.attention.self.key"), hidden_size, hidden_size);
            add_linear(&mut tensors, &format!("{lp}.attention.self.value"), hidden_size, hidden_size);
            add_linear(&mut tensors, &format!("{lp}.attention.output.dense"), hidden_size, hidden_size);
            add_ln(&mut tensors, &format!("{lp}.attention.output.LayerNorm"), hidden_size);
            // Intermediate + output
            add_linear(&mut tensors, &format!("{lp}.intermediate.dense"), intermediate_size, hidden_size);
            add_linear(&mut tensors, &format!("{lp}.output.dense"), hidden_size, intermediate_size);
            add_ln(&mut tensors, &format!("{lp}.output.LayerNorm"), hidden_size);
        }

        let vb = VarBuilder::from_tensors(tensors, DType::F32, dev);

        let bert_config = candle_transformers::models::bert::Config {
            vocab_size,
            hidden_size,
            num_hidden_layers: num_layers,
            num_attention_heads: num_heads,
            intermediate_size,
            hidden_act: candle_transformers::models::bert::HiddenAct::Gelu,
            hidden_dropout_prob: 0.0,
            max_position_embeddings: max_pos,
            type_vocab_size,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            ..Default::default()
        };

        let model = candle_transformers::models::bert::BertModel::load(vb, &bert_config).unwrap();
        let mc = ModelConfig {
            hidden_size,
            intermediate_size,
            num_heads,
            num_kv_heads: num_heads,
            num_layers,
            head_size,
            vocab_size,
            rms_norm_eps: 1e-12,
            rope_theta: 0.0,
            rope_dim: 0,
            max_seq_len: max_pos,
        };
        let runner = BertEmbeddingRunner {
            model,
            config: mc,
            model_dir: PathBuf::from("/tmp"),
            device: dev.clone(),
        };

        // forward() should return an error
        let dummy_ids = Tensor::zeros(4, DType::U32, dev).unwrap();
        let dummy_ctx = make_dummy_ctx(dev);
        let result = runner.forward(&dummy_ids, &dummy_ctx);
        assert!(result.is_err(), "forward() should return Err for BERT");
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("encoder-only"), "error message should mention encoder-only, got: {err_msg}");
    }

    /// Test that BertEmbeddingRunner::embed() returns a [hidden_size] tensor.
    #[test]
    fn test_bert_embed_shape() {
        let dev = &Device::Cpu;
        let hidden_size = 16usize;
        let num_layers = 1usize;
        let num_heads = 2usize;
        let intermediate_size = 32usize;
        let vocab_size = 100usize;
        let max_pos = 32usize;
        let type_vocab_size = 2usize;

        let runner = make_bert_runner(dev, hidden_size, num_layers, num_heads, intermediate_size, vocab_size, max_pos, type_vocab_size);

        // embed() should return [hidden_size]
        let input_ids = Tensor::new(&[1u32, 2, 3, 4], dev).unwrap();
        let result = runner.embed(&input_ids).unwrap();
        assert_eq!(result.dims(), &[hidden_size], "embed() should return [hidden_size]");

        // Result should be L2-normalized (norm ~1.0)
        let vals: Vec<f32> = result.to_vec1().unwrap();
        let norm: f32 = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "L2 norm should be ~1.0, got {norm}");
    }

    // ── Task 2: NomicBert tests ────────────────────────────────────────────

    /// Test that NomicBertHfConfig correctly parses nomic-bert config fields.
    #[test]
    fn test_nomic_bert_config_parse() {
        let json = r#"{
            "model_type": "nomic_bert",
            "n_embd": 768,
            "n_head": 12,
            "n_layer": 12,
            "n_inner": 3072,
            "vocab_size": 30528,
            "n_positions": 8192,
            "layer_norm_epsilon": 1e-12,
            "rotary_emb_fraction": 1.0,
            "rotary_emb_base": 1000.0
        }"#;
        let cfg: NomicBertHfConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.n_embd, 768);
        assert_eq!(cfg.n_head, 12);
        assert_eq!(cfg.n_layer, 12);
        assert_eq!(cfg.n_inner, 3072);
        assert_eq!(cfg.vocab_size, 30528);
        assert_eq!(cfg.n_positions, 8192);
        assert!((cfg.layer_norm_epsilon - 1e-12).abs() < 1e-20);
        assert!((cfg.rotary_emb_fraction - 1.0).abs() < 1e-6);
        assert!((cfg.rotary_emb_base - 1000.0).abs() < 1e-3);
    }

    /// Test NomicBertHfConfig defaults (n_positions, layer_norm_epsilon, rotary fields).
    #[test]
    fn test_nomic_bert_config_defaults() {
        let json = r#"{
            "n_embd": 768,
            "n_head": 12,
            "n_layer": 12,
            "n_inner": 3072,
            "vocab_size": 30528
        }"#;
        let cfg: NomicBertHfConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.n_positions, 8192);
        assert!((cfg.layer_norm_epsilon - 1e-12).abs() < 1e-20);
        assert!((cfg.rotary_emb_fraction - 1.0).abs() < 1e-6);
        assert!((cfg.rotary_emb_base - 1000.0).abs() < 1e-3);
    }

    /// Test that NomicBertRunner::embed() returns a [hidden_size] tensor.
    #[test]
    fn test_nomic_bert_embed_shape() {
        let dev = &Device::Cpu;
        let runner = make_nomic_bert_runner(dev);

        let input_ids = Tensor::new(&[1u32, 2, 3, 4], dev).unwrap();
        let result = runner.embed(&input_ids).unwrap();
        assert_eq!(
            result.dims(),
            &[runner.config.hidden_size],
            "embed() should return [hidden_size]"
        );

        // Result should be L2-normalized (norm ~1.0)
        let vals: Vec<f32> = result.to_vec1().unwrap();
        let norm: f32 = vals.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "L2 norm should be ~1.0, got {norm}"
        );
    }

    /// Test that NomicBertRunner::forward() returns Err (encoder-only).
    #[test]
    fn test_nomic_bert_forward_returns_error() {
        let dev = &Device::Cpu;
        let runner = make_nomic_bert_runner(dev);

        let dummy_ids = Tensor::zeros(4, DType::U32, dev).unwrap();
        let dummy_ctx = make_dummy_ctx(dev);
        let result = runner.forward(&dummy_ids, &dummy_ctx);
        assert!(result.is_err(), "forward() should return Err for NomicBert");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("encoder-only"),
            "error message should mention encoder-only, got: {err_msg}"
        );
    }

    // ── Test helpers ──────────────────────────────────────────────────────

    fn make_bert_runner(
        dev: &Device,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        intermediate_size: usize,
        vocab_size: usize,
        max_pos: usize,
        type_vocab_size: usize,
    ) -> BertEmbeddingRunner {
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        let head_size = hidden_size / num_heads;

        let add_linear = |tensors: &mut HashMap<String, Tensor>, name: &str, out: usize, inp: usize| {
            tensors.insert(format!("{name}.weight"), Tensor::randn(0f32, 0.02, (out, inp), dev).unwrap());
            tensors.insert(format!("{name}.bias"), Tensor::zeros(out, DType::F32, dev).unwrap());
        };
        let add_ln = |tensors: &mut HashMap<String, Tensor>, name: &str, size: usize| {
            tensors.insert(format!("{name}.weight"), Tensor::ones(size, DType::F32, dev).unwrap());
            tensors.insert(format!("{name}.bias"), Tensor::zeros(size, DType::F32, dev).unwrap());
        };
        let add_embed = |tensors: &mut HashMap<String, Tensor>, name: &str, rows: usize, cols: usize| {
            tensors.insert(format!("{name}.weight"), Tensor::randn(0f32, 0.02, (rows, cols), dev).unwrap());
        };

        add_embed(&mut tensors, "embeddings.word_embeddings", vocab_size, hidden_size);
        add_embed(&mut tensors, "embeddings.position_embeddings", max_pos, hidden_size);
        add_embed(&mut tensors, "embeddings.token_type_embeddings", type_vocab_size, hidden_size);
        add_ln(&mut tensors, "embeddings.LayerNorm", hidden_size);

        for i in 0..num_layers {
            let lp = format!("encoder.layer.{i}");
            add_linear(&mut tensors, &format!("{lp}.attention.self.query"), hidden_size, hidden_size);
            add_linear(&mut tensors, &format!("{lp}.attention.self.key"), hidden_size, hidden_size);
            add_linear(&mut tensors, &format!("{lp}.attention.self.value"), hidden_size, hidden_size);
            add_linear(&mut tensors, &format!("{lp}.attention.output.dense"), hidden_size, hidden_size);
            add_ln(&mut tensors, &format!("{lp}.attention.output.LayerNorm"), hidden_size);
            add_linear(&mut tensors, &format!("{lp}.intermediate.dense"), intermediate_size, hidden_size);
            add_linear(&mut tensors, &format!("{lp}.output.dense"), hidden_size, intermediate_size);
            add_ln(&mut tensors, &format!("{lp}.output.LayerNorm"), hidden_size);
        }

        let vb = VarBuilder::from_tensors(tensors, DType::F32, dev);
        let bert_config = candle_transformers::models::bert::Config {
            vocab_size,
            hidden_size,
            num_hidden_layers: num_layers,
            num_attention_heads: num_heads,
            intermediate_size,
            hidden_act: candle_transformers::models::bert::HiddenAct::Gelu,
            hidden_dropout_prob: 0.0,
            max_position_embeddings: max_pos,
            type_vocab_size,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            ..Default::default()
        };

        let model = candle_transformers::models::bert::BertModel::load(vb, &bert_config).unwrap();
        let mc = ModelConfig {
            hidden_size,
            intermediate_size,
            num_heads,
            num_kv_heads: num_heads,
            num_layers,
            head_size,
            vocab_size,
            rms_norm_eps: 1e-12,
            rope_theta: 0.0,
            rope_dim: 0,
            max_seq_len: max_pos,
        };
        BertEmbeddingRunner {
            model,
            config: mc,
            model_dir: PathBuf::from("/tmp"),
            device: dev.clone(),
        }
    }

    fn make_nomic_bert_runner(dev: &Device) -> NomicBertRunner {
        // Small dimensions for fast tests
        let hidden_size = 16usize;
        let num_heads = 2usize;
        let head_dim = hidden_size / num_heads;
        let n_inner = 32usize;
        let vocab_size = 100usize;
        let n_positions = 32usize;
        let n_layer = 1usize;
        let rope_dim = head_dim; // rotary_emb_fraction = 1.0
        let ln_eps = 1e-12f64;
        let rope_base = 1000.0f32;

        let nomic_config = NomicBertHfConfig {
            n_embd: hidden_size,
            n_head: num_heads,
            n_layer,
            n_inner,
            vocab_size,
            n_positions,
            layer_norm_epsilon: ln_eps,
            rotary_emb_fraction: 1.0,
            rotary_emb_base: rope_base,
        };

        // Precompute RoPE
        let (rope_cos, rope_sin) =
            NomicBertRunner::precompute_rope(rope_dim, rope_base, n_positions, dev).unwrap();

        // Build tensors for 1 layer
        let mut tensors: HashMap<String, Tensor> = HashMap::new();

        let add_tensor = |tensors: &mut HashMap<String, Tensor>, name: &str, shape: &[usize]| {
            tensors.insert(
                name.to_string(),
                Tensor::randn(0f32, 0.02, shape, dev).unwrap(),
            );
        };
        let add_zeros = |tensors: &mut HashMap<String, Tensor>, name: &str, shape: &[usize]| {
            tensors.insert(
                name.to_string(),
                Tensor::zeros(shape, DType::F32, dev).unwrap(),
            );
        };
        let add_ones = |tensors: &mut HashMap<String, Tensor>, name: &str, shape: &[usize]| {
            tensors.insert(
                name.to_string(),
                Tensor::ones(shape, DType::F32, dev).unwrap(),
            );
        };

        // Embeddings (under model. prefix, loaded via vb.pp("model"))
        add_tensor(&mut tensors, "model.embeddings.word_embeddings.weight", &[vocab_size, hidden_size]);

        // Layer 0
        {
            let p = "model.encoder.layers.0";
            // Attention norms
            add_ones(&mut tensors, &format!("{p}.norm1.weight"), &[hidden_size]);
            add_zeros(&mut tensors, &format!("{p}.norm1.bias"), &[hidden_size]);
            add_ones(&mut tensors, &format!("{p}.norm2.weight"), &[hidden_size]);
            add_zeros(&mut tensors, &format!("{p}.norm2.bias"), &[hidden_size]);
            // Fused QKV
            add_tensor(&mut tensors, &format!("{p}.mixer.Wqkv.weight"), &[3 * hidden_size, hidden_size]);
            // Out proj
            add_tensor(&mut tensors, &format!("{p}.mixer.out_proj.weight"), &[hidden_size, hidden_size]);
            add_zeros(&mut tensors, &format!("{p}.mixer.out_proj.bias"), &[hidden_size]);
            // MLP
            add_tensor(&mut tensors, &format!("{p}.mlp.fc1.weight"), &[2 * n_inner, hidden_size]);
            add_tensor(&mut tensors, &format!("{p}.mlp.fc2.weight"), &[hidden_size, n_inner]);
        }

        // Final layer norm
        add_ones(&mut tensors, "model.emb_ln.weight", &[hidden_size]);
        add_zeros(&mut tensors, "model.emb_ln.bias", &[hidden_size]);

        let vb_root = VarBuilder::from_tensors(tensors, DType::F32, dev);
        let vb = vb_root.pp("model");

        // Word embeddings
        let word_embeddings = candle_nn::embedding(vocab_size, hidden_size, vb.pp("embeddings.word_embeddings")).unwrap();

        // Layer
        let layer = NomicBertLayer::load(
            vb.pp("encoder.layers.0"),
            &nomic_config,
            rope_cos,
            rope_sin,
        ).unwrap();

        // Final LN
        let final_ln = candle_nn::layer_norm(hidden_size, ln_eps, vb.pp("emb_ln")).unwrap();

        let config = ModelConfig {
            hidden_size,
            intermediate_size: n_inner,
            num_heads,
            num_kv_heads: num_heads,
            num_layers: n_layer,
            head_size: head_dim,
            vocab_size,
            rms_norm_eps: ln_eps,
            rope_theta: rope_base,
            rope_dim,
            max_seq_len: n_positions,
        };

        NomicBertRunner {
            word_embeddings,
            layers: vec![layer],
            final_ln,
            config,
            model_dir: PathBuf::from("/tmp"),
            device: dev.clone(),
        }
    }

    /// Create a minimal ForwardContext for testing (won't be used, but forward() takes it).
    fn make_dummy_ctx(dev: &Device) -> ForwardContext<'static> {
        // We can't easily create a valid ForwardContext for tests, but we don't need one
        // since forward() returns Err immediately without using the context.
        // Use a trick: leak a few tensors so they have 'static lifetime for the test.
        let positions = Box::leak(Box::new(Tensor::zeros(1, DType::U32, dev).unwrap()));
        let block_table = Box::leak(Box::new(Tensor::zeros((1, 1), DType::I32, dev).unwrap()));
        let slot_mapping = Box::leak(Box::new(Tensor::zeros(1, DType::I32, dev).unwrap()));
        let context_lens = Box::leak(Box::new(Tensor::zeros(1, DType::I32, dev).unwrap()));
        let kv_caches: &'static [(Tensor, Tensor)] = Box::leak(Box::new(vec![]));

        use crate::serving::kernels::cpu_backend::CpuBackend;
        let backend: &'static dyn crate::serving::kernels::backend::KernelBackend =
            Box::leak(Box::new(CpuBackend::new()));

        ForwardContext {
            positions,
            block_table,
            slot_mapping,
            context_lens,
            query_start_loc: &[],
            seq_lens: &[],
            kv_caches,
            max_context_len: 0,
            is_all_decode: false,
            backend,
        }
    }
}
