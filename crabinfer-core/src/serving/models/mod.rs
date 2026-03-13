//! Paged-attention-native model implementations.
//!
//! These models load GGUF weights and use the serving system's paged KV cache
//! instead of managing their own. They accept flat batched inputs `[total_tokens]`
//! and dispatch to Metal paged attention kernels for decode.
//!
//! Key differences from candle-transformers models:
//! - No internal KV cache — the paged cache is external
//! - Forward takes `&self` (immutable) since all mutable state is in the cache
//! - Flat batched inputs: all sequences' tokens concatenated
//! - Positions passed as a tensor, not a scalar offset

pub mod attention;
pub mod deepseek;
pub mod llama;
pub mod mistral;
pub mod phi3;
pub mod vision;

use candle_core::quantized::gguf_file;
use candle_core::{Device, Result, Tensor};
use candle_nn::Module;

use crate::serving::kernels::KernelBackend;
use crate::serving::quantization::{MaybeQuantizedLinear, QuantizationMethod};

// ─── Forward context ──────────────────────────────────────────────────────

/// Context for a batched forward pass with paged KV cache.
///
/// Carries all the paged attention state needed by the model layers.
/// Built by the engine loop from scheduler output.
pub struct ForwardContext<'a> {
    /// Per-token position in each sequence's context. `[total_tokens]` u32.
    pub positions: &'a Tensor,
    /// Block table mapping logical blocks to physical blocks. `[num_seqs, max_blocks_per_seq]` i32.
    pub block_table: &'a Tensor,
    /// Slot mapping for writing new K/V tokens to cache. `[total_tokens]` i32.
    pub slot_mapping: &'a Tensor,
    /// Total context length per sequence (including cached tokens). `[num_seqs]` i32.
    pub context_lens: &'a Tensor,
    /// CSR offsets: `query_start_loc[i]` = start of seq i's tokens in the flat tensor.
    /// Length = num_seqs + 1 (last element = total_tokens).
    pub query_start_loc: &'a [usize],
    /// Sequence lengths (same values as context_lens, as usize for convenience).
    pub seq_lens: &'a [usize],
    /// Per-layer KV caches: `(key_cache, value_cache)`.
    pub kv_caches: &'a [(Tensor, Tensor)],
    /// Maximum context length across all sequences (for kernel dispatch sizing).
    pub max_context_len: usize,
    /// Whether all sequences are in decode mode (exactly 1 new token each).
    /// When true, the paged attention kernel is used directly.
    /// When false, standard SDPA is used for prefill sequences.
    pub is_all_decode: bool,
    /// Kernel backend for dispatching paged attention operations.
    pub backend: &'a dyn KernelBackend,
}

// ─── Model config ─────────────────────────────────────────────────────────

/// Model configuration extracted from GGUF metadata.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_layers: usize,
    pub head_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub rope_dim: usize,
    pub max_seq_len: usize,
}

// ─── Model runner trait ───────────────────────────────────────────────────

/// Trait for paged-attention-native model implementations.
///
/// Models implementing this trait:
/// - Load weights from GGUF via `from_gguf()`
/// - Accept flat batched inputs with paged attention context
/// - Return logits for all input tokens
///
/// The trait requires `Send + Sync` because the engine loop may hold
/// the model across await points.
pub trait ModelRunner: Send + Sync {
    /// Batched forward pass with paged KV cache.
    ///
    /// # Arguments
    /// - `input_ids`: `[total_tokens]` u32 — token IDs for all sequences concatenated
    /// - `ctx`: forward context with paged attention state
    ///
    /// # Returns
    /// Logits tensor `[total_tokens, vocab_size]`
    fn forward(&self, input_ids: &Tensor, ctx: &ForwardContext) -> Result<Tensor>;

    /// Compute embeddings for the given input tokens.
    ///
    /// Default implementation returns an error — only embedding-capable models
    /// override this. The returned tensor should be `[num_tokens, hidden_size]`.
    fn embed(&self, _input_ids: &Tensor) -> Result<Tensor> {
        Err(candle_core::Error::Msg(
            "Model does not support embeddings".to_string(),
        ))
    }

    /// Return the token embedding table, if available.
    ///
    /// Used by `EngineHandle::embed()` to produce mean-pooled embeddings
    /// from the model's learned token vectors. Shape: `[vocab_size, hidden_size]`.
    fn embedding_table(&self) -> Option<&Tensor> {
        None
    }

    /// Create a clone of this model, sharing weight tensors via Arc.
    ///
    /// Candle tensors use `Arc<Storage>` internally, so cloning a tensor is
    /// cheap (just an Arc bump). Each cloned model can be given to a separate
    /// `EngineHandle` worker — they share weights but have independent KV caches.
    fn clone_model(&self) -> Box<dyn ModelRunner>;

    fn num_layers(&self) -> usize;
    fn num_kv_heads(&self) -> usize;
    fn head_size(&self) -> usize;
    fn num_heads(&self) -> usize;
    fn config(&self) -> &ModelConfig;
}

// ─── Architecture registry ───────────────────────────────────────────────

/// Factory function type for loading a model from a GGUF file.
///
/// Takes the parsed GGUF content, the open file handle, the target device,
/// and quantization method. Returns a boxed ModelRunner.
pub type ModelFactory = fn(
    gguf_file::Content,
    &mut std::fs::File,
    &Device,
    QuantizationMethod,
) -> Result<Box<dyn ModelRunner>>;

/// Registry entry for a model architecture.
struct ArchitectureEntry {
    /// GGUF `general.architecture` value(s) this loader handles.
    names: &'static [&'static str],
    /// Factory function to load the model.
    factory: ModelFactory,
}

/// Global architecture registry.
///
/// Add new architectures by appending to this array.
static ARCHITECTURE_REGISTRY: &[ArchitectureEntry] = &[ArchitectureEntry {
    names: &["llama"],
    factory: llama_factory,
}];

/// Factory wrapper for LlamaModel.
fn llama_factory(
    ct: gguf_file::Content,
    file: &mut std::fs::File,
    device: &Device,
    quantization: QuantizationMethod,
) -> Result<Box<dyn ModelRunner>> {
    let model = llama::LlamaModel::from_gguf(ct, file, device, quantization)?;
    Ok(Box::new(model))
}

/// Look up a model factory by GGUF architecture name.
///
/// Returns the factory function if a matching architecture is registered,
/// or `None` if the architecture is not supported.
pub fn find_architecture(arch: &str) -> Option<ModelFactory> {
    let arch_lower = arch.to_lowercase();
    ARCHITECTURE_REGISTRY
        .iter()
        .find(|entry| entry.names.iter().any(|n| *n == arch_lower))
        .map(|entry| entry.factory)
}

/// Load a model from a GGUF file using the architecture registry.
///
/// Reads `general.architecture` from GGUF metadata, looks up the appropriate
/// loader, and returns a boxed `ModelRunner`.
pub fn load_model_from_gguf(
    ct: gguf_file::Content,
    file: &mut std::fs::File,
    device: &Device,
    quantization: QuantizationMethod,
) -> Result<Box<dyn ModelRunner>> {
    let arch = ct
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "llama".to_string());

    let factory = find_architecture(&arch).ok_or_else(|| {
        candle_core::Error::Msg(format!(
            "Unsupported model architecture: '{}'. Supported: {}",
            arch,
            supported_architectures().join(", ")
        ))
    })?;

    factory(ct, file, device, quantization)
}

/// List all supported architecture names.
pub fn supported_architectures() -> Vec<&'static str> {
    ARCHITECTURE_REGISTRY
        .iter()
        .flat_map(|entry| entry.names.iter().copied())
        .collect()
}

// ─── Shared building blocks ───────────────────────────────────────────────

/// RMS Layer Normalization.
///
/// Dequantizes the GGUF weight at construction time and applies
/// `candle_nn::ops::rms_norm` during forward.
#[derive(Clone)]
pub struct RmsNorm {
    pub(crate) weight: Tensor,
    pub(crate) eps: f32,
}

impl RmsNorm {
    /// Load from a GGUF quantized tensor.
    pub fn from_qtensor(qtensor: candle_core::quantized::QTensor, eps: f64) -> Result<Self> {
        let weight = qtensor.dequantize(&qtensor.device())?;
        Ok(Self {
            weight,
            eps: eps as f32,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(x, &self.weight, self.eps)
    }

    /// Forward pass using fused RMSNorm kernel when available.
    ///
    /// On CUDA, this computes the entire normalization in a single kernel pass.
    pub fn forward_fused(&self, x: &Tensor, backend: &dyn KernelBackend) -> Result<Tensor> {
        backend.fused_rmsnorm(x, &self.weight, self.eps)
    }

    /// Fused RMSNorm + linear projection.
    ///
    /// On CUDA, this runs in a single kernel. On CPU/Metal, falls back to
    /// sequential norm + matmul via the default trait implementation.
    pub fn forward_linear_fused(
        &self,
        x: &Tensor,
        linear: &MaybeQuantizedLinear,
        backend: &dyn KernelBackend,
    ) -> Result<Tensor> {
        // Only use fused path for dense (non-quantized) linear layers
        // because the fused kernel expects a dense weight matrix.
        match linear.weight_tensor() {
            Some(weight) => {
                backend.fused_layernorm_linear(x, &self.weight, self.eps, weight)
            }
            None => {
                // Quantized or GGUF linear: use unfused path
                let normed = backend.fused_rmsnorm(x, &self.weight, self.eps)?;
                linear.forward(&normed)
            }
        }
    }

    /// Fused residual add + RMSNorm.
    ///
    /// Computes `x_out = x + residual`, then normalizes in one kernel pass on CUDA.
    /// Returns `(normed, x_out)` where `x_out` is used as the next residual.
    pub fn forward_add_fused(
        &self,
        x: &Tensor,
        residual: &Tensor,
        backend: &dyn KernelBackend,
    ) -> Result<(Tensor, Tensor)> {
        backend.fused_add_rmsnorm(x, residual, &self.weight, self.eps)
    }
}

/// SwiGLU MLP: `output = down(silu(gate(x)) * up(x))`
///
/// Used by Llama, Qwen2, Phi3, and Gemma architectures.
#[derive(Clone)]
pub struct SwiGluMlp {
    pub(crate) gate: MaybeQuantizedLinear,
    pub(crate) down: MaybeQuantizedLinear,
    pub(crate) up: MaybeQuantizedLinear,
}

impl SwiGluMlp {
    pub fn new(gate: MaybeQuantizedLinear, down: MaybeQuantizedLinear, up: MaybeQuantizedLinear) -> Self {
        Self { gate, down, up }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate.forward(x)?)?;
        let up = self.up.forward(x)?;
        self.down.forward(&(gate * up)?)
    }

    /// Forward pass using fused SiLU+multiply kernel when available.
    ///
    /// On CUDA, this eliminates 2 intermediate tensor allocations by computing
    /// `silu(gate(x)) * up(x)` in a single kernel pass.
    pub fn forward_fused(&self, x: &Tensor, backend: &dyn KernelBackend) -> Result<Tensor> {
        let gate_out = self.gate.forward(x)?;
        let up_out = self.up.forward(x)?;
        let fused = backend.fused_silu_mul(&gate_out, &up_out)?;
        self.down.forward(&fused)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_rmsnorm_forward_shape() {
        let weight = Tensor::ones(64, DType::F32, &Device::Cpu).unwrap();
        let norm = RmsNorm {
            weight,
            eps: 1e-5,
        };
        let x = Tensor::randn(0f32, 1.0, (4, 64), &Device::Cpu).unwrap();
        let y = norm.forward(&x).unwrap();
        assert_eq!(y.dims(), &[4, 64]);
    }

    #[test]
    fn test_rmsnorm_values() {
        // RmsNorm with weight=1 should normalize the vector
        let weight = Tensor::ones(4, DType::F32, &Device::Cpu).unwrap();
        let norm = RmsNorm {
            weight,
            eps: 1e-6,
        };
        let x = Tensor::new(&[[2.0f32, 2.0, 2.0, 2.0]], &Device::Cpu).unwrap();
        let y = norm.forward(&x).unwrap();
        let data: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        // RMS of [2,2,2,2] = 2, so normalized = [1,1,1,1]
        for v in data {
            assert!((v - 1.0).abs() < 1e-4, "expected ~1.0, got {v}");
        }
    }

    #[test]
    fn test_rmsnorm_forward_linear_fused_matches_unfused() {
        // Verify that forward_linear_fused produces the same output as
        // separate rmsnorm + linear.forward within tolerance.
        let dev = &Device::Cpu;
        let hidden_size = 64;
        let out_features = 32;

        let weight = Tensor::randn(0f32, 1.0, hidden_size, dev).unwrap();
        let norm = RmsNorm {
            weight,
            eps: 1e-5,
        };

        let x = Tensor::randn(0f32, 1.0, (4, hidden_size), dev).unwrap();

        // Create a dense MaybeQuantizedLinear
        let linear_weight =
            Tensor::randn(0f32, 1.0, (out_features, hidden_size), dev).unwrap();
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

    #[test]
    fn test_rmsnorm_forward_linear_fused_quantized_fallback() {
        // Verify that forward_linear_fused falls back gracefully for
        // quantized (QTensor) linear layers.
        let dev = &Device::Cpu;
        let hidden_size = 64;

        let weight = Tensor::ones(hidden_size, DType::F32, dev).unwrap();
        let norm = RmsNorm {
            weight,
            eps: 1e-5,
        };

        let x = Tensor::randn(0f32, 1.0, (2, hidden_size), dev).unwrap();

        // Use a QMatMul with a quantized tensor to trigger fallback
        let linear_weight =
            Tensor::randn(0f32, 1.0, (32, hidden_size), dev).unwrap();
        let qtensor = candle_core::quantized::QTensor::quantize(
            &linear_weight,
            candle_core::quantized::GgmlDType::Q4_0,
        )
        .unwrap();
        let qmm = candle_core::quantized::QMatMul::from_qtensor(qtensor).unwrap();
        let linear = MaybeQuantizedLinear::QMatMul(qmm);

        let backend = crate::serving::kernels::cpu_backend::CpuBackend::new();

        // Should still work (unfused fallback)
        let result = norm
            .forward_linear_fused(&x, &linear, &backend)
            .unwrap();
        assert_eq!(result.dims()[0], 2);
        // Output features from Q4_0 quantized
        assert!(result.dims()[1] > 0);
    }

    #[test]
    fn test_find_architecture_llama() {
        assert!(find_architecture("llama").is_some());
        assert!(find_architecture("Llama").is_some());
        assert!(find_architecture("LLAMA").is_some());
    }

    #[test]
    fn test_find_architecture_unknown() {
        assert!(find_architecture("gpt2").is_none());
        assert!(find_architecture("mamba").is_none());
    }

    #[test]
    fn test_supported_architectures() {
        let archs = supported_architectures();
        assert!(archs.contains(&"llama"));
        assert!(!archs.is_empty());
    }

    #[test]
    fn test_model_config_debug() {
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
        // Just verify it compiles and formats
        let _ = format!("{:?}", config);
    }
}
