//! Backend trait abstraction for inference engines.
//!
//! This module defines the [`Backend`] and [`LoadedModel`] traits that decouple
//! the CrabInfer engine from any specific ML framework (Candle, llama.cpp, etc.).
//!
//! The engine interacts only with these traits, so swapping the underlying
//! framework requires implementing a new backend — no changes to engine.rs.

use crate::{CrabInferError, ModelInfo};

/// Which compute device to use for inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendDevice {
    /// Run on CPU only.
    Cpu,
    /// Use Apple Metal GPU acceleration.
    Metal,
}

/// A backend that can load GGUF models and create inference sessions.
///
/// Implementations wrap a specific ML runtime (e.g. Candle, llama.cpp) and
/// translate its types into the CrabInfer trait interface.
///
/// # Thread Safety
///
/// Backends must be `Send + Sync` so they can be shared behind an `Arc` and
/// called from any thread (including Swift's cooperative thread pool).
pub trait Backend: Send + Sync {
    /// Human-readable name of this backend (e.g., `"candle"`, `"llama.cpp"`).
    fn name(&self) -> &str;

    /// Load a GGUF model from disk.
    ///
    /// # Arguments
    ///
    /// * `model_path` -- path to the `.gguf` file.
    /// * `device` -- whether to use CPU or Metal acceleration.
    /// * `metallib_path` -- optional path to a directory of pre-compiled
    ///   `.metallib` files.  When provided, the backend should prefer loading
    ///   pre-compiled shaders instead of compiling at runtime (which can fail
    ///   on iOS due to XPC sandbox restrictions).
    ///
    /// # Errors
    ///
    /// Returns `CrabInferError::ModelNotFound` if the file does not exist,
    /// `CrabInferError::ModelLoadFailed` for any other loading issue.
    fn load_model(
        &self,
        model_path: &str,
        device: BackendDevice,
        metallib_path: Option<&str>,
    ) -> Result<Box<dyn LoadedModel>, CrabInferError>;

    /// Return the current Metal GPU memory allocation in bytes, if available.
    ///
    /// Returns `None` if the backend does not use Metal or cannot report
    /// memory usage.
    fn metal_allocated_bytes(&self) -> Option<usize> {
        None
    }

    /// Release unused GPU buffers to free memory.
    ///
    /// No-op for backends that do not manage a GPU buffer pool.
    fn release_gpu_buffers(&self) -> Result<(), CrabInferError> {
        Ok(())
    }

    /// Wait for all GPU operations to complete.
    ///
    /// This is a synchronization barrier.  After this returns, all previously
    /// submitted GPU work has finished and results are safe to read.
    fn wait_until_completed(&self) -> Result<(), CrabInferError> {
        Ok(())
    }
}

/// A loaded model ready for inference.
///
/// This trait abstracts over different model architectures (Phi3, Qwen3, Llama,
/// Gemma3, etc.) and provides a uniform interface for token generation.
///
/// # Mutability
///
/// `forward()` takes `&mut self` because it mutates internal KV caches.
/// Callers must hold exclusive access (e.g. via a `Mutex`).
pub trait LoadedModel: Send {
    /// Run a forward pass on the given token IDs.
    ///
    /// # Arguments
    ///
    /// * `tokens` -- slice of token IDs to process.
    /// * `position` -- current position in the sequence for KV cache indexing.
    ///   Pass `0` on the first call (prefill) and increment by the number of
    ///   tokens processed on subsequent calls.
    ///
    /// # Returns
    ///
    /// The logits for the next token as a flat `Vec<f32>` of size `vocab_size`.
    fn forward(&mut self, tokens: &[u32], position: usize) -> Result<Vec<f32>, CrabInferError>;

    /// Clear the KV cache (e.g., between generations).
    ///
    /// Must be called before starting a new generation sequence to avoid
    /// stale cached state from a previous run.
    fn clear_kv_cache(&mut self);

    /// Get metadata about the loaded model (name, architecture, quant, etc.).
    fn model_info(&self) -> ModelInfo;

    /// Get the end-of-sequence token ID for this model.
    ///
    /// The engine uses this to detect when generation should stop.
    fn eos_token_id(&self) -> u32;

    /// Get the tokenizer associated with this model.
    ///
    /// Returns a reference to the tokenizer for encoding/decoding text.
    fn tokenizer(&self) -> &tokenizers::Tokenizer;
}
