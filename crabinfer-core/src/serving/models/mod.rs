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
pub mod llama;

use candle_core::quantized::QMatMul;
use candle_core::{Result, Tensor};
use candle_nn::Module;

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

    fn num_layers(&self) -> usize;
    fn num_kv_heads(&self) -> usize;
    fn head_size(&self) -> usize;
    fn num_heads(&self) -> usize;
    fn config(&self) -> &ModelConfig;
}

// ─── Shared building blocks ───────────────────────────────────────────────

/// RMS Layer Normalization.
///
/// Dequantizes the GGUF weight at construction time and applies
/// `candle_nn::ops::rms_norm` during forward.
pub struct RmsNorm {
    weight: Tensor,
    eps: f32,
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
}

/// SwiGLU MLP: `output = down(silu(gate(x)) * up(x))`
///
/// Used by Llama, Qwen2, Phi3, and Gemma architectures.
pub struct SwiGluMlp {
    gate: QMatMul,
    down: QMatMul,
    up: QMatMul,
}

impl SwiGluMlp {
    pub fn new(gate: QMatMul, down: QMatMul, up: QMatMul) -> Self {
        Self { gate, down, up }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate.forward(x)?)?;
        let up = self.up.forward(x)?;
        self.down.forward(&(gate * up)?)
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
