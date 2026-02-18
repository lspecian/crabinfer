/// CrabInfer — Safe, memory-aware LLM inference for iOS
///
/// This crate provides the core inference engine that bridges
/// Candle (Hugging Face's Rust ML framework) with iOS via UniFFI.

pub mod device;
pub mod engine;
pub mod memory;

// UniFFI proc-macro scaffolding — generates the FFI glue code
uniffi::setup_scaffolding!();

/// Device capabilities information
#[derive(Debug, Clone, uniffi::Record)]
pub struct DeviceInfo {
    pub device_model: String,
    pub total_memory_bytes: u64,
    pub available_memory_bytes: u64,
    pub has_metal_gpu: bool,
    pub has_neural_engine: bool,
    pub recommended_quant: String,
    pub max_model_size_b: u32,
}

/// Engine configuration
#[derive(Debug, Clone, uniffi::Record)]
pub struct EngineConfig {
    pub model_path: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub context_length: u32,
    pub use_metal: bool,
    pub memory_limit_bytes: u64,
    /// Path to directory containing pre-compiled .metallib files.
    /// When set, Metal kernel loading uses these binaries instead of
    /// compiling shaders at runtime, avoiding XPC failures on iOS.
    /// Empty string means no metallib dir (use runtime compilation).
    pub metallib_path: String,
}

/// Generated token output
#[derive(Debug, Clone, uniffi::Record)]
pub struct TokenOutput {
    pub text: String,
    pub token_id: u32,
    pub probability: f32,
    pub is_end_of_sequence: bool,
}

/// Model metadata
#[derive(Debug, Clone, uniffi::Record)]
pub struct ModelInfo {
    pub model_name: String,
    pub architecture: String,
    pub parameter_count: u64,
    pub quantization: String,
    pub file_size_bytes: u64,
    pub context_length: u32,
    pub vocab_size: u32,
}

/// Generation statistics
#[derive(Debug, Clone, uniffi::Record)]
pub struct GenerationStats {
    pub tokens_generated: u32,
    pub tokens_per_second: f64,
    pub time_to_first_token_ms: f64,
    pub total_time_ms: f64,
    pub peak_memory_bytes: u64,
    pub compute_backend: String,
}

/// Memory pressure levels
#[derive(Debug, Clone, PartialEq, uniffi::Enum)]
pub enum MemoryPressure {
    Normal,
    Warning,
    Critical,
    Terminal,
}

/// Errors
#[derive(Debug, thiserror::Error, uniffi::Error)]
#[uniffi(flat_error)]
pub enum CrabInferError {
    #[error("Model file not found")]
    ModelNotFound,
    #[error("Failed to load model: {reason}")]
    ModelLoadFailed { reason: String },
    #[error("Out of memory")]
    OutOfMemory,
    #[error("Metal GPU not available")]
    MetalNotAvailable,
    #[error("Tokenization failed")]
    TokenizationFailed,
    #[error("Inference failed")]
    InferenceFailed,
    #[error("Context window overflow")]
    ContextOverflow,
    #[error("Invalid configuration")]
    InvalidConfig,
    #[error("Device not supported")]
    DeviceNotSupported,
}

// === Top-level functions exported to Swift ===

/// Get the CrabInfer version
#[uniffi::export]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Detect the current device's capabilities
#[uniffi::export]
pub fn detect_device() -> DeviceInfo {
    device::detect_device()
}

// === CrabInferEngine exported to Swift ===

#[derive(uniffi::Object)]
pub struct CrabInferEngine {
    inner: engine::CrabInferEngine,
}

#[uniffi::export]
impl CrabInferEngine {
    #[uniffi::constructor]
    pub fn new(config: EngineConfig) -> Result<Self, CrabInferError> {
        Ok(Self {
            inner: engine::CrabInferEngine::new(config)?,
        })
    }

    pub fn load_model(&self, model_path: String) -> Result<(), CrabInferError> {
        self.inner.load_model(model_path)
    }

    pub fn model_info(&self) -> Result<ModelInfo, CrabInferError> {
        self.inner.model_info()
    }

    pub fn complete(
        &self,
        prompt: String,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<String, CrabInferError> {
        self.inner.complete(prompt, max_tokens, temperature)
    }

    pub fn next_token(&self, prompt: String) -> Result<Option<TokenOutput>, CrabInferError> {
        self.inner.next_token(prompt)
    }

    pub fn reset(&self) {
        self.inner.reset()
    }

    pub fn last_stats(&self) -> Option<GenerationStats> {
        self.inner.last_stats()
    }

    pub fn memory_pressure(&self) -> MemoryPressure {
        self.inner.memory_pressure()
    }

    pub fn reduce_memory(&self) {
        self.inner.reduce_memory()
    }

    pub fn unload_model(&self) {
        self.inner.unload_model()
    }

    pub fn is_model_loaded(&self) -> bool {
        self.inner.is_model_loaded()
    }

    /// Run a load/unload stress test to detect memory leaks.
    ///
    /// Returns a log of RSS measurements after each cycle.
    pub fn stress_test(
        &self,
        model_path: String,
        cycles: u32,
        tokens_per_cycle: u32,
    ) -> Result<Vec<String>, CrabInferError> {
        self.inner.stress_test(model_path, cycles, tokens_per_cycle)
    }
}
