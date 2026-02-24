/// CrabInfer — Safe, memory-aware LLM inference for iOS
///
/// This crate provides the core inference engine that bridges
/// Candle (Hugging Face's Rust ML framework) with iOS via UniFFI.

pub mod backend;
pub mod backends;
pub mod catalog;
pub mod chat_template;
pub mod device;
pub mod engine;
pub mod memory_pressure;
pub mod prompt;
pub mod conversation;
pub mod facts;
pub mod chunker;
pub mod embedding;
pub mod vectorstore;
pub mod knowledge;
pub mod provider;
pub mod providers;
pub mod router;
pub mod tools;
pub mod agent;
pub mod mcp;

#[cfg(feature = "providers")]
pub mod credentials;
#[cfg(feature = "providers")]
pub mod download;

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
    /// Maximum safe model file size in bytes for this device tier.
    /// Loading a file larger than this will be rejected to prevent jetsam kills.
    pub max_model_file_size_bytes: u64,
    /// Apple Silicon chip name (e.g. "Apple M4 Max"), or "Unknown" on non-Apple platforms.
    pub chip_name: String,
    /// Apple Silicon variant: "Base", "Pro", "Max", "Ultra", or empty on non-Apple.
    pub chip_variant: String,
    /// Recommended context window size based on available RAM.
    pub recommended_context_length: u32,
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
    /// Whether this is a Mixture of Experts model.
    pub is_moe: bool,
    /// Total number of expert modules (0 if not MoE).
    pub expert_count: u32,
    /// Number of experts activated per token (0 if not MoE).
    /// For memory estimation: weight memory uses `parameter_count` (all experts),
    /// but KV cache and activation memory scale with `active_params`.
    pub expert_used_count: u32,
    /// Active parameters per token. For dense models this equals `parameter_count`.
    /// For MoE models this is roughly `parameter_count * expert_used_count / expert_count`
    /// (shared layers + active expert fraction).
    pub active_parameter_count: u64,
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
    #[error("Model file not found. Ensure the GGUF file exists at the specified path and is readable.")]
    ModelNotFound,
    #[error("Failed to load model: {reason}. Check that the file is a valid GGUF and the architecture is supported (Llama, Phi3, Qwen2/3, Gemma).")]
    ModelLoadFailed { reason: String },
    #[error("Out of memory: {reason}. Try a smaller model, reduce context_length, or call unload_model() first.")]
    OutOfMemory { reason: String },
    #[error("Metal GPU not available. Set use_metal=false to use CPU inference, or ensure you are running on Apple Silicon.")]
    MetalNotAvailable,
    #[error("Tokenization failed. Ensure the model's GGUF file contains a valid tokenizer or provide a tokenizer.json alongside it.")]
    TokenizationFailed,
    #[error("Inference failed. Ensure a model is loaded (is_model_loaded()), no concurrent generation is running, and the prompt fits within context_length.")]
    InferenceFailed,
    #[error("Context window overflow. The prompt exceeds the configured context_length. Reduce the prompt length, lower max_tokens, or increase context_length (requires more RAM).")]
    ContextOverflow,
    #[error("Invalid configuration. Check that temperature is 0.0-2.0, top_p is 0.0-1.0, context_length > 0, and model_path is non-empty.")]
    InvalidConfig,
    #[error("Device not supported. CrabInfer requires Apple Silicon (Metal GPU) or a 64-bit CPU. Check detect_device() for capabilities.")]
    DeviceNotSupported,
    #[error("Model too large: {file_size_mb} MB exceeds device limit of {max_allowed_mb} MB. Use a smaller quantization (e.g. Q4_K_M instead of Q8_0) or a model with fewer parameters.")]
    ModelTooLarge { file_size_mb: u64, max_allowed_mb: u64 },
    #[error("Network error: {reason}. Check connectivity, verify the provider URL, or switch to a local model with LocalFirst/LocalOnly routing.")]
    NetworkError { reason: String },
    #[error("API error from {provider} (status {status_code}): {message}. For 401/403: check your API key. For 429: rate limited, wait before retrying. For 5xx: provider may be down.")]
    ApiError { provider: String, status_code: u32, message: String },
    #[error("Authentication failed for {provider}. Set a valid API key via set_credential(\"{provider}\", key) or check that your key has not expired.")]
    AuthenticationFailed { provider: String },
    #[error("Rate limited by {provider}, retry after {retry_after_seconds}s. Consider using a local model or reducing request frequency.")]
    RateLimited { provider: String, retry_after_seconds: u32 },
    #[error("Provider {provider} is not available. Verify the server is running and accessible, or use the Router with fallback providers.")]
    ProviderNotAvailable { provider: String },
    #[error("Download failed: {reason}. Check internet connectivity and disk space. The download will resume from where it left off on retry.")]
    DownloadFailed { reason: String },
    #[error("Integrity check failed: {reason}. The file may be corrupted or incomplete. Delete it and re-download.")]
    IntegrityCheckFailed { reason: String },
    #[error("Storage error: {reason}. Ensure the directory exists, is writable, and has sufficient disk space.")]
    StorageError { reason: String },
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

/// Estimate peak memory usage for a GGUF model before loading.
///
/// Reads only the GGUF header (fast, ~ms) and returns estimated peak
/// memory in bytes using the calibrated multi-component model
/// (weights + KV cache + activations + embedding inflation × 1.35 safety).
#[uniffi::export]
pub fn estimate_model_memory(model_path: String, context_length: u32) -> Result<u64, CrabInferError> {
    let (info, emb_overhead) = engine::peek_model_metadata(&model_path, context_length)?;
    let total_b = info.parameter_count as f32 / 1e9;
    let active_b = info.active_parameter_count as f32 / 1e9;
    Ok(memory_pressure::MemoryPressureManager::estimate_model_memory_corrected(
        total_b, active_b, &info.quantization, context_length, emb_overhead,
    ))
}

// === Model catalog functions exported to Swift ===

use catalog::CatalogEntry;

/// Return the full curated model catalog.
#[uniffi::export]
pub fn model_catalog() -> Vec<CatalogEntry> {
    catalog::all()
}

/// Return catalog models that fit on the given device.
#[uniffi::export]
pub fn models_for_device(device: DeviceInfo) -> Vec<CatalogEntry> {
    catalog::for_device(&device)
}

/// Return the best (largest fitting) model per category for this device.
#[uniffi::export]
pub fn recommended_models(device: DeviceInfo) -> Vec<CatalogEntry> {
    catalog::recommended(&device)
}

/// Look up a single catalog entry by ID.
#[uniffi::export]
pub fn catalog_entry(id: String) -> Option<CatalogEntry> {
    catalog::get(&id)
}

// === Credential management functions exported to Swift ===

/// Store an API key for a cloud provider.
/// Provider names: "openai", "anthropic", "google", "ollama".
#[cfg(feature = "providers")]
#[uniffi::export]
pub fn set_credential(provider: String, api_key: String) {
    credentials::set_api_key(&provider, &api_key);
}

/// Retrieve the stored API key for a provider.
#[cfg(feature = "providers")]
#[uniffi::export]
pub fn get_credential(provider: String) -> Option<String> {
    credentials::get_api_key(&provider)
}

/// Remove a stored API key.
#[cfg(feature = "providers")]
#[uniffi::export]
pub fn remove_credential(provider: String) {
    credentials::remove_api_key(&provider);
}

/// Check if a key is stored for the given provider.
#[cfg(feature = "providers")]
#[uniffi::export]
pub fn has_credential(provider: String) -> bool {
    credentials::has_api_key(&provider)
}

/// List provider names that have stored keys.
#[cfg(feature = "providers")]
#[uniffi::export]
pub fn configured_providers() -> Vec<String> {
    credentials::configured_providers()
}

/// Validate a stored API key by making a lightweight API call.
/// Returns true if valid, false if authentication fails.
#[cfg(feature = "providers")]
#[uniffi::export]
pub fn validate_credential(provider: String) -> Result<bool, CrabInferError> {
    credentials::validate_api_key(&provider)
}

// === Memory pressure callback interface ===

/// Callback interface for Swift to receive memory pressure change notifications.
///
/// Swift implements this protocol and passes it via `set_memory_pressure_listener()`.
#[uniffi::export(callback_interface)]
pub trait MemoryPressureListener: Send + Sync {
    fn on_pressure_change(&self, old_level: MemoryPressure, new_level: MemoryPressure);
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

    /// Register a listener for memory pressure changes.
    ///
    /// The listener's `on_pressure_change` method is called whenever the
    /// memory pressure level transitions (e.g. Normal → Warning).
    pub fn set_memory_pressure_listener(&self, listener: Box<dyn MemoryPressureListener>) {
        let listener = std::sync::Arc::new(listener);
        self.inner.set_pressure_callback(move |old, new| {
            listener.on_pressure_change(old, new);
        });
    }

    /// Get current Metal GPU memory allocation in bytes.
    ///
    /// Returns 0 if no Metal device is active.
    pub fn metal_allocated_bytes(&self) -> u64 {
        self.inner.metal_allocated_bytes()
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

// === CrabInferProvider exported to Swift ===

use provider::{
    CompletionRequest, CompletionResponse, ModelDescriptor, Provider, ProviderConfig,
};
use std::sync::{Arc, Mutex};

/// Unified provider interface for local and cloud LLM inference.
///
/// Swift API:
/// ```swift
/// let provider = try CrabInferProvider.newLocal(config: engineConfig)
/// let response = try provider.complete(request: request)
/// ```
#[derive(uniffi::Object)]
pub struct CrabInferProvider {
    inner: Box<dyn Provider>,
    stream_state: Mutex<
        Option<Box<dyn Iterator<Item = Result<TokenOutput, CrabInferError>> + Send>>,
    >,
}

#[uniffi::export]
impl CrabInferProvider {
    /// Create a cloud provider from configuration.
    #[uniffi::constructor]
    pub fn new(config: ProviderConfig) -> Result<Self, CrabInferError> {
        #[cfg(feature = "providers")]
        let inner: Box<dyn Provider> = match config.provider_type.as_str() {
            "openai" => Box::new(providers::openai::OpenAIProvider::new(config)?),
            "anthropic" => Box::new(providers::anthropic::AnthropicProvider::new(config)?),
            "google" => Box::new(providers::google::GoogleProvider::new(config)?),
            "ollama" => Box::new(providers::ollama::OllamaProvider::new(config)?),
            "vllm" => Box::new(providers::vllm::VllmProvider::new(config)?),
            _ => return Err(CrabInferError::InvalidConfig),
        };

        #[cfg(not(feature = "providers"))]
        {
            let _ = config;
            return Err(CrabInferError::InvalidConfig);
        }

        #[cfg(feature = "providers")]
        Ok(Self {
            inner,
            stream_state: Mutex::new(None),
        })
    }

    /// Create a local inference provider.
    #[uniffi::constructor(name = "new_local")]
    pub fn new_local(engine_config: EngineConfig) -> Result<Self, CrabInferError> {
        let inner = Box::new(providers::local::LocalProvider::new(engine_config)?);
        Ok(Self {
            inner,
            stream_state: Mutex::new(None),
        })
    }

    /// Provider name ("local", "openai", "anthropic", etc.)
    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// Generate a complete response (blocking).
    pub fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, CrabInferError> {
        self.inner.complete(&request)
    }

    /// Start streaming token generation. Call `stream_next()` repeatedly after this.
    pub fn stream_start(&self, request: CompletionRequest) -> Result<(), CrabInferError> {
        let iter = self.inner.stream(&request)?;
        *self.stream_state.lock().unwrap() = Some(iter);
        Ok(())
    }

    /// Get the next streamed token, or None when generation is complete.
    pub fn stream_next(&self) -> Result<Option<TokenOutput>, CrabInferError> {
        let mut guard = self.stream_state.lock().unwrap();
        let iter = match guard.as_mut() {
            Some(it) => it,
            None => return Ok(None),
        };
        match iter.next() {
            Some(Ok(token)) => {
                if token.is_end_of_sequence {
                    *guard = None;
                }
                Ok(Some(token))
            }
            Some(Err(e)) => {
                *guard = None;
                Err(e)
            }
            None => {
                *guard = None;
                Ok(None)
            }
        }
    }

    /// List models available from this provider.
    pub fn available_models(&self) -> Result<Vec<ModelDescriptor>, CrabInferError> {
        self.inner.available_models()
    }

    /// Check if the provider is reachable/configured.
    pub fn is_available(&self) -> bool {
        self.inner.is_available()
    }
}

// === CrabInferRouter exported to Swift ===

use router::{Router, RouterConfig, RoutingDecision};

/// Smart router that automatically selects between local and cloud providers.
///
/// Swift API:
/// ```swift
/// let router = try CrabInferRouter(
///     config: RouterConfig(policy: .localFirst, privacyMode: false, ollamaIsLocal: true),
///     localConfig: engineConfig,
///     cloudConfigs: [openaiConfig, anthropicConfig]
/// )
/// let response = try router.complete(request: request)
/// let decision = router.lastRoutingDecision()
/// ```
#[derive(uniffi::Object)]
pub struct CrabInferRouter {
    inner: Router,
    stream_state: Mutex<
        Option<Box<dyn Iterator<Item = Result<TokenOutput, CrabInferError>> + Send>>,
    >,
}

#[uniffi::export]
impl CrabInferRouter {
    /// Create a router with optional local and cloud providers.
    #[uniffi::constructor]
    pub fn new(
        config: RouterConfig,
        local_config: Option<EngineConfig>,
        cloud_configs: Vec<ProviderConfig>,
    ) -> Result<Self, CrabInferError> {
        let local: Option<Box<dyn Provider>> = match local_config {
            Some(ec) => Some(Box::new(providers::local::LocalProvider::new(ec)?)),
            None => None,
        };

        #[allow(unused_mut)]
        let mut self_hosted: Vec<Box<dyn Provider>> = Vec::new();
        #[allow(unused_mut)]
        let mut cloud: Vec<Box<dyn Provider>> = Vec::new();
        #[cfg(feature = "providers")]
        for cc in cloud_configs {
            let tier = router::resolve_tier(&cc);
            let provider: Box<dyn Provider> = match cc.provider_type.as_str() {
                "openai" => Box::new(providers::openai::OpenAIProvider::new(cc)?),
                "anthropic" => Box::new(providers::anthropic::AnthropicProvider::new(cc)?),
                "google" => Box::new(providers::google::GoogleProvider::new(cc)?),
                "ollama" => Box::new(providers::ollama::OllamaProvider::new(cc)?),
                "vllm" => Box::new(providers::vllm::VllmProvider::new(cc)?),
                _ => return Err(CrabInferError::InvalidConfig),
            };
            match tier {
                router::ProviderTier::SelfHosted => self_hosted.push(provider),
                _ => cloud.push(provider),
            }
        }

        #[cfg(not(feature = "providers"))]
        {
            if !cloud_configs.is_empty() {
                return Err(CrabInferError::InvalidConfig);
            }
        }

        Ok(Self {
            inner: Router::new(config, local, self_hosted, cloud),
            stream_state: Mutex::new(None),
        })
    }

    /// Router name.
    pub fn name(&self) -> String {
        "router".to_string()
    }

    /// Generate a complete response, routing automatically.
    pub fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, CrabInferError> {
        self.inner.complete(&request)
    }

    /// Start streaming, routing automatically. Call `stream_next()` repeatedly after this.
    pub fn stream_start(&self, request: CompletionRequest) -> Result<(), CrabInferError> {
        let iter = self.inner.stream(&request)?;
        *self.stream_state.lock().unwrap() = Some(iter);
        Ok(())
    }

    /// Get the next streamed token, or None when done.
    pub fn stream_next(&self) -> Result<Option<TokenOutput>, CrabInferError> {
        let mut guard = self.stream_state.lock().unwrap();
        let iter = match guard.as_mut() {
            Some(it) => it,
            None => return Ok(None),
        };
        match iter.next() {
            Some(Ok(token)) => {
                if token.is_end_of_sequence {
                    *guard = None;
                }
                Ok(Some(token))
            }
            Some(Err(e)) => {
                *guard = None;
                Err(e)
            }
            None => {
                *guard = None;
                Ok(None)
            }
        }
    }

    /// List models available from all configured providers.
    pub fn available_models(&self) -> Result<Vec<ModelDescriptor>, CrabInferError> {
        self.inner.available_models()
    }

    /// Check if any provider is available.
    pub fn is_available(&self) -> bool {
        self.inner.is_available()
    }

    /// Set network availability (call from Swift when NWPathMonitor reports changes).
    pub fn set_network_available(&self, available: bool) {
        self.inner.set_network_available(available);
    }

    /// Get the routing decision from the last `complete()` or `stream_start()` call.
    pub fn last_routing_decision(&self) -> Option<RoutingDecision> {
        self.inner.last_routing_decision()
    }
}

// === CrabInferVllm — vLLM-specific provider exported to Swift ===

#[cfg(feature = "providers")]
use providers::vllm::{VllmCompletionOptions, VllmProvider, VllmServerMetrics};

/// vLLM provider with both standard Provider methods and vLLM-specific
/// features (health checks, Prometheus metrics, guided decoding).
///
/// Swift API:
/// ```swift
/// let vllm = try CrabInferVllm(config: ProviderConfig(
///     providerType: "vllm",
///     apiKey: "",
///     baseUrl: "http://my-vllm-server:8000",
///     defaultModel: "Qwen/Qwen3-8B",
///     timeoutSeconds: 30
/// ))
/// let healthy = try vllm.health()
/// let metrics = try vllm.serverMetrics()
/// let response = try vllm.complete(request: request)
/// ```
#[cfg(feature = "providers")]
#[derive(uniffi::Object)]
pub struct CrabInferVllm {
    inner: VllmProvider,
    stream_state: Mutex<
        Option<Box<dyn Iterator<Item = Result<TokenOutput, CrabInferError>> + Send>>,
    >,
}

#[cfg(feature = "providers")]
#[uniffi::export]
impl CrabInferVllm {
    /// Create a vLLM provider. `config.provider_type` should be `"vllm"`.
    #[uniffi::constructor]
    pub fn new(config: ProviderConfig) -> Result<Self, CrabInferError> {
        let inner = VllmProvider::new(config)?;
        Ok(Self {
            inner,
            stream_state: Mutex::new(None),
        })
    }

    /// Provider name: `"vllm"`.
    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// Generate a complete response (blocking, standard OpenAI-compatible).
    pub fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, CrabInferError> {
        self.inner.complete(&request)
    }

    /// Generate a complete response with vLLM-specific options
    /// (repetition_penalty, min_p, guided decoding).
    pub fn complete_with_options(
        &self,
        request: CompletionRequest,
        options: VllmCompletionOptions,
    ) -> Result<CompletionResponse, CrabInferError> {
        self.inner.complete_with_options(&request, &options)
    }

    /// Start streaming token generation. Call `stream_next()` repeatedly after this.
    pub fn stream_start(&self, request: CompletionRequest) -> Result<(), CrabInferError> {
        let iter = self.inner.stream(&request)?;
        *self.stream_state.lock().unwrap() = Some(iter);
        Ok(())
    }

    /// Get the next streamed token, or None when generation is complete.
    pub fn stream_next(&self) -> Result<Option<TokenOutput>, CrabInferError> {
        let mut guard = self.stream_state.lock().unwrap();
        let iter = match guard.as_mut() {
            Some(it) => it,
            None => return Ok(None),
        };
        match iter.next() {
            Some(Ok(token)) => {
                if token.is_end_of_sequence {
                    *guard = None;
                }
                Ok(Some(token))
            }
            Some(Err(e)) => {
                *guard = None;
                Err(e)
            }
            None => {
                *guard = None;
                Ok(None)
            }
        }
    }

    /// List models currently served by the vLLM instance (live API call).
    pub fn available_models(&self) -> Result<Vec<ModelDescriptor>, CrabInferError> {
        self.inner.available_models()
    }

    /// Check if the vLLM server is reachable (calls GET /health).
    pub fn is_available(&self) -> bool {
        self.inner.is_available()
    }

    /// Check vLLM server health via GET /health.
    pub fn health(&self) -> Result<bool, CrabInferError> {
        self.inner.health()
    }

    /// Scrape key Prometheus metrics from the vLLM server (GET /metrics).
    pub fn server_metrics(&self) -> Result<VllmServerMetrics, CrabInferError> {
        self.inner.server_metrics()
    }
}

// === CrabInferVllmMlx — vLLM-MLX-specific provider exported to Swift ===

// === Agent Runtime types exported to Swift ===

/// Agent configuration for Swift.
#[derive(Debug, Clone, uniffi::Record)]
pub struct AgentConfigRecord {
    /// Maximum number of tool-calling rounds per turn.
    pub max_tool_rounds: u32,
    /// Maximum tokens for each LLM call.
    pub max_tokens: u32,
    /// Temperature for generation.
    pub temperature: f32,
    /// Top-p sampling.
    pub top_p: f32,
    /// Number of RAG results to retrieve per query.
    pub rag_top_k: u32,
}

/// Record of a tool execution during an agent turn.
#[derive(Debug, Clone, uniffi::Record)]
pub struct AgentToolExecution {
    pub tool_name: String,
    /// JSON string of the tool arguments.
    pub arguments_json: String,
    pub output: String,
    pub is_error: bool,
}

/// Response from a single agent turn.
#[derive(Debug, Clone, uniffi::Record)]
pub struct AgentResponseRecord {
    /// The final text response.
    pub text: String,
    /// Tool executions that happened during this turn.
    pub tool_executions: Vec<AgentToolExecution>,
    /// Number of LLM rounds used.
    pub rounds: u32,
}

/// A remembered fact (key-value pair).
#[derive(Debug, Clone, uniffi::Record)]
pub struct FactRecord {
    pub key: String,
    pub value: String,
}

// === CrabInferAgent exported to Swift ===

/// AI Agent with tool calling, memory, and knowledge.
///
/// The agent combines an LLM provider (local, cloud, or router) with a tool
/// registry, conversation memory, persistent facts, and optional knowledge base
/// to provide an autonomous assistant that can reason and act.
///
/// Swift API:
/// ```swift
/// let agent = try CrabInferAgent(
///     providerConfig: ProviderConfig(providerType: "openai", ...),
///     config: AgentConfigRecord(maxToolRounds: 10, ...)
/// )
/// let response = try agent.run(userInput: "What files are in my home directory?")
/// // response.toolExecutions shows file_list was called
/// // response.text has the answer
/// ```
#[derive(uniffi::Object)]
pub struct CrabInferAgent {
    inner: Mutex<agent::Agent>,
}

#[uniffi::export]
impl CrabInferAgent {
    /// Create an agent with a cloud provider.
    #[uniffi::constructor]
    pub fn new(
        provider_config: ProviderConfig,
        config: AgentConfigRecord,
    ) -> Result<Self, CrabInferError> {
        #[cfg(feature = "providers")]
        let provider: Box<dyn Provider> = match provider_config.provider_type.as_str() {
            "openai" => Box::new(providers::openai::OpenAIProvider::new(provider_config)?),
            "anthropic" => Box::new(providers::anthropic::AnthropicProvider::new(provider_config)?),
            "google" => Box::new(providers::google::GoogleProvider::new(provider_config)?),
            "ollama" => Box::new(providers::ollama::OllamaProvider::new(provider_config)?),
            "vllm" => Box::new(providers::vllm::VllmProvider::new(provider_config)?),
            _ => return Err(CrabInferError::InvalidConfig),
        };

        #[cfg(not(feature = "providers"))]
        {
            let _ = provider_config;
            let _ = config;
            return Err(CrabInferError::InvalidConfig);
        }

        #[cfg(feature = "providers")]
        {
            let agent_config = agent::AgentConfig {
                max_tool_rounds: config.max_tool_rounds,
                max_tokens: config.max_tokens,
                temperature: config.temperature,
                top_p: config.top_p,
                rag_top_k: config.rag_top_k as usize,
            };

            let agent = agent::Agent::new(Arc::from(provider))
                .with_config(agent_config);

            Ok(Self {
                inner: Mutex::new(agent),
            })
        }
    }

    /// Create an agent with a local inference provider.
    #[uniffi::constructor(name = "new_local")]
    pub fn new_local(
        engine_config: EngineConfig,
        config: AgentConfigRecord,
    ) -> Result<Self, CrabInferError> {
        let provider: Box<dyn Provider> =
            Box::new(providers::local::LocalProvider::new(engine_config)?);

        let agent_config = agent::AgentConfig {
            max_tool_rounds: config.max_tool_rounds,
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            top_p: config.top_p,
            rag_top_k: config.rag_top_k as usize,
        };

        let agent = agent::Agent::new(Arc::from(provider))
            .with_config(agent_config);

        Ok(Self {
            inner: Mutex::new(agent),
        })
    }

    /// Run the agent with user input and return the response.
    ///
    /// This executes the full agent loop: LLM → parse tool calls → execute →
    /// feed results back → repeat until text-only response.
    pub fn run(&self, user_input: String) -> Result<AgentResponseRecord, CrabInferError> {
        let mut agent = self.inner.lock().unwrap();
        let response = agent.run(&user_input)?;
        Ok(AgentResponseRecord {
            text: response.text,
            tool_executions: response
                .tool_calls
                .into_iter()
                .map(|tc| AgentToolExecution {
                    tool_name: tc.tool_name,
                    arguments_json: serde_json::to_string(&tc.arguments)
                        .unwrap_or_default(),
                    output: tc.output,
                    is_error: tc.is_error,
                })
                .collect(),
            rounds: response.rounds,
        })
    }

    /// Get the names of all available tools.
    pub fn tool_names(&self) -> Vec<String> {
        let mut agent = self.inner.lock().unwrap();
        agent.tools_mut().tool_names()
    }

    /// Get the number of available tools.
    pub fn tool_count(&self) -> u32 {
        let mut agent = self.inner.lock().unwrap();
        agent.tools_mut().len() as u32
    }

    /// Add a persistent fact (key-value pair).
    pub fn add_fact(&self, key: String, value: String) {
        let mut agent = self.inner.lock().unwrap();
        agent.facts_mut().add_fact(&key, &value);
    }

    /// Remove a persistent fact. Returns true if it existed.
    pub fn remove_fact(&self, key: String) -> bool {
        let mut agent = self.inner.lock().unwrap();
        agent.facts_mut().remove_fact(&key)
    }

    /// Get all stored facts.
    pub fn get_facts(&self) -> Vec<FactRecord> {
        let agent = self.inner.lock().unwrap();
        agent.facts_mut_ref()
            .all_facts()
            .iter()
            .map(|f| FactRecord {
                key: f.key.clone(),
                value: f.value.clone(),
            })
            .collect()
    }

    /// Get the number of stored facts.
    pub fn fact_count(&self) -> u32 {
        let agent = self.inner.lock().unwrap();
        agent.facts_mut_ref().count() as u32
    }

    /// Clear conversation history.
    pub fn clear_conversation(&self) {
        let mut agent = self.inner.lock().unwrap();
        agent.clear_conversation();
    }

    /// Get the number of messages in the conversation.
    pub fn message_count(&self) -> u32 {
        let agent = self.inner.lock().unwrap();
        agent.conversation().message_count() as u32
    }

    /// Save persistent state (conversation + facts).
    pub fn save(&self) -> Result<(), CrabInferError> {
        let agent = self.inner.lock().unwrap();
        agent.save()
    }

    /// Set the system prompt identity.
    pub fn set_identity(&self, identity: String) {
        let mut agent = self.inner.lock().unwrap();
        let prompt = prompt::SystemPrompt::new().identity(&identity);
        agent.set_system_prompt(prompt);
    }

    /// Set the conversation persistence path.
    pub fn set_conversation_path(&self, path: String) {
        let mut agent = self.inner.lock().unwrap();
        agent.set_conversation_path(&path);
    }

    /// Set the facts persistence path.
    pub fn set_facts_path(&self, path: String) {
        let mut agent = self.inner.lock().unwrap();
        agent.set_facts_path(&path);
    }

    /// Connect to an MCP stdio server and register its tools.
    /// Returns the number of tools registered.
    pub fn connect_mcp_stdio(&self, command: String, args: Vec<String>) -> Result<u32, CrabInferError> {
        let mut agent = self.inner.lock().unwrap();
        let str_args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
        let client = mcp::McpStdioClient::connect(&command, &str_args)?;
        let client = Arc::new(client);
        let count = mcp::register_mcp_tools(agent.tools_mut(), client)?;
        Ok(count as u32)
    }
}
