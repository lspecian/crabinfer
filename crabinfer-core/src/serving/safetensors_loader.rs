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
//! Supports Llama-family architectures (Llama, Qwen2/3, Mistral, Yi, Gemma).

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Module;

use super::models::{ModelConfig, ModelRunner, RmsNorm, SwiGluMlp};
use super::models::attention::{precompute_rope, PagedAttentionLayer};
use super::models::ForwardContext;
use super::quantization::{MaybeQuantizedLinear, QuantizationMethod};

// ─── HuggingFace config.json ─────────────────────────────────────────────

/// Minimal HuggingFace `config.json` for Llama-family models.
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
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default)]
    model_type: Option<String>,
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

// ─── Loading ─────────────────────────────────────────────────────────────

/// Load a tensor from the weight map, converting to the target dtype.
fn load_tensor(
    weights: &HashMap<String, Tensor>,
    name: &str,
    device: &Device,
) -> Result<Tensor> {
    weights
        .get(name)
        .ok_or_else(|| candle_core::Error::Msg(format!("missing weight: {name}")))?
        .to_device(device)?
        .to_dtype(DType::F32)
}

/// Load a weight as a MaybeQuantizedLinear.
fn load_linear(
    weights: &HashMap<String, Tensor>,
    name: &str,
    device: &Device,
    quantization: QuantizationMethod,
) -> Result<MaybeQuantizedLinear> {
    let w = load_tensor(weights, name, device)?;
    let qmm = candle_core::quantized::QMatMul::Tensor(w);
    MaybeQuantizedLinear::from_qmatmul(qmm, quantization, device)
}

/// Load a model from a directory containing safetensors files and config.json.
///
/// The directory should contain:
/// - `config.json` — HuggingFace model config
/// - `*.safetensors` — model weight files (one or more shards)
///
/// Returns a boxed `ModelRunner` for use with the serving engine.
pub fn load_model_from_safetensors(
    model_dir: &Path,
    device: &Device,
    quantization: QuantizationMethod,
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

    // Load all tensors into a HashMap for easy lookup
    let mut weights: HashMap<String, Tensor> = HashMap::new();
    for (name, _view) in tensors.tensors() {
        let tensor = tensors.load(&name, device)?;
        weights.insert(name, tensor);
    }

    tracing::info!("Loaded {} tensors from safetensors", weights.len());

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

    // ── Load embeddings ──
    let embed_table = load_tensor(&weights, "model.embed_tokens.weight", device)?;

    // ── Load output projection ──
    let lm_head = if weights.contains_key("lm_head.weight") {
        load_linear(&weights, "lm_head.weight", device, quantization)?
    } else {
        // Tied embeddings: reuse embed_tokens
        let qmm = candle_core::quantized::QMatMul::Tensor(embed_table.clone());
        MaybeQuantizedLinear::from_qmatmul(qmm, quantization, device)?
    };

    // ── Load final norm ──
    let norm_weight = load_tensor(&weights, "model.norm.weight", device)?;
    let norm = RmsNorm { weight: norm_weight, eps: rms_norm_eps as f32 };

    // ── Precompute shared RoPE tables ──
    let (rope_cos, rope_sin) = precompute_rope(rope_dim, rope_theta, max_seq_len, device)?;

    // ── Load transformer layers ──
    let mut layers = Vec::with_capacity(hf_config.num_hidden_layers);
    for i in 0..hf_config.num_hidden_layers {
        let prefix = format!("model.layers.{i}");

        let attn_norm_weight = load_tensor(
            &weights,
            &format!("{prefix}.input_layernorm.weight"),
            device,
        )?;
        let attn_norm = RmsNorm {
            weight: attn_norm_weight,
            eps: rms_norm_eps as f32,
        };

        let attn_q = load_linear(
            &weights,
            &format!("{prefix}.self_attn.q_proj.weight"),
            device,
            quantization,
        )?;
        let attn_k = load_linear(
            &weights,
            &format!("{prefix}.self_attn.k_proj.weight"),
            device,
            quantization,
        )?;
        let attn_v = load_linear(
            &weights,
            &format!("{prefix}.self_attn.v_proj.weight"),
            device,
            quantization,
        )?;
        let attn_output = load_linear(
            &weights,
            &format!("{prefix}.self_attn.o_proj.weight"),
            device,
            quantization,
        )?;

        let ffn_norm_weight = load_tensor(
            &weights,
            &format!("{prefix}.post_attention_layernorm.weight"),
            device,
        )?;
        let ffn_norm = RmsNorm {
            weight: ffn_norm_weight,
            eps: rms_norm_eps as f32,
        };

        let gate = load_linear(
            &weights,
            &format!("{prefix}.mlp.gate_proj.weight"),
            device,
            quantization,
        )?;
        let down = load_linear(
            &weights,
            &format!("{prefix}.mlp.down_proj.weight"),
            device,
            quantization,
        )?;
        let up = load_linear(
            &weights,
            &format!("{prefix}.mlp.up_proj.weight"),
            device,
            quantization,
        )?;
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
        quantization,
    );

    Ok(Box::new(SafetensorsLlamaModel {
        embed_table,
        layers,
        norm,
        lm_head,
        config: model_config,
    }))
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
        let result = load_model_from_safetensors(&tmp, &Device::Cpu, QuantizationMethod::None);
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
