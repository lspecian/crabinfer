//! Record types mapped from crabinfer-core to napi.
//!
//! napi-rs auto-converts snake_case Rust fields to camelCase JS properties.
//! u64 fields are mapped to i64 (JS number is safe up to 2^53).

use napi_derive::napi;

use crate::enums::{ModelCategory, MemoryPressure, ProviderTier, RoutingPolicy, RoutingReason};

// ---------------------------------------------------------------------------
// Device & Engine config
// ---------------------------------------------------------------------------

/// Device capabilities information.
#[napi(object)]
pub struct DeviceInfo {
    pub device_model: String,
    pub total_memory_bytes: i64,
    pub available_memory_bytes: i64,
    pub has_metal_gpu: bool,
    pub has_neural_engine: bool,
    pub recommended_quant: String,
    pub max_model_size_b: u32,
    pub max_model_file_size_bytes: i64,
    pub chip_name: String,
    pub chip_variant: String,
    pub recommended_context_length: u32,
}

impl From<crabinfer_core::DeviceInfo> for DeviceInfo {
    fn from(d: crabinfer_core::DeviceInfo) -> Self {
        Self {
            device_model: d.device_model,
            total_memory_bytes: d.total_memory_bytes as i64,
            available_memory_bytes: d.available_memory_bytes as i64,
            has_metal_gpu: d.has_metal_gpu,
            has_neural_engine: d.has_neural_engine,
            recommended_quant: d.recommended_quant,
            max_model_size_b: d.max_model_size_b,
            max_model_file_size_bytes: d.max_model_file_size_bytes as i64,
            chip_name: d.chip_name,
            chip_variant: d.chip_variant,
            recommended_context_length: d.recommended_context_length,
        }
    }
}

/// Engine configuration.
#[napi(object)]
pub struct EngineConfig {
    pub model_path: String,
    pub max_tokens: u32,
    pub temperature: f64,
    pub top_p: f64,
    pub context_length: u32,
    pub use_metal: bool,
    pub memory_limit_bytes: i64,
    pub metallib_path: String,
}

impl From<EngineConfig> for crabinfer_core::EngineConfig {
    fn from(c: EngineConfig) -> Self {
        Self {
            model_path: c.model_path,
            max_tokens: c.max_tokens,
            temperature: c.temperature as f32,
            top_p: c.top_p as f32,
            context_length: c.context_length,
            use_metal: c.use_metal,
            memory_limit_bytes: c.memory_limit_bytes as u64,
            metallib_path: c.metallib_path,
        }
    }
}

// ---------------------------------------------------------------------------
// Token output & stats
// ---------------------------------------------------------------------------

/// Generated token output.
#[napi(object)]
pub struct TokenOutput {
    pub text: String,
    pub token_id: u32,
    pub probability: f64,
    pub is_end_of_sequence: bool,
}

impl From<crabinfer_core::TokenOutput> for TokenOutput {
    fn from(t: crabinfer_core::TokenOutput) -> Self {
        Self {
            text: t.text,
            token_id: t.token_id,
            probability: t.probability as f64,
            is_end_of_sequence: t.is_end_of_sequence,
        }
    }
}

/// Generation statistics.
#[napi(object)]
pub struct GenerationStats {
    pub tokens_generated: u32,
    pub tokens_per_second: f64,
    pub time_to_first_token_ms: f64,
    pub total_time_ms: f64,
    pub peak_memory_bytes: i64,
    pub compute_backend: String,
}

impl From<crabinfer_core::GenerationStats> for GenerationStats {
    fn from(s: crabinfer_core::GenerationStats) -> Self {
        Self {
            tokens_generated: s.tokens_generated,
            tokens_per_second: s.tokens_per_second,
            time_to_first_token_ms: s.time_to_first_token_ms,
            total_time_ms: s.total_time_ms,
            peak_memory_bytes: s.peak_memory_bytes as i64,
            compute_backend: s.compute_backend,
        }
    }
}

/// Model metadata.
#[napi(object)]
pub struct ModelInfo {
    pub model_name: String,
    pub architecture: String,
    pub parameter_count: i64,
    pub quantization: String,
    pub file_size_bytes: i64,
    pub context_length: u32,
    pub vocab_size: u32,
    pub is_moe: bool,
    pub expert_count: u32,
    pub expert_used_count: u32,
    pub active_parameter_count: i64,
}

impl From<crabinfer_core::ModelInfo> for ModelInfo {
    fn from(m: crabinfer_core::ModelInfo) -> Self {
        Self {
            model_name: m.model_name,
            architecture: m.architecture,
            parameter_count: m.parameter_count as i64,
            quantization: m.quantization,
            file_size_bytes: m.file_size_bytes as i64,
            context_length: m.context_length,
            vocab_size: m.vocab_size,
            is_moe: m.is_moe,
            expert_count: m.expert_count,
            expert_used_count: m.expert_used_count,
            active_parameter_count: m.active_parameter_count as i64,
        }
    }
}

// ---------------------------------------------------------------------------
// Provider types
// ---------------------------------------------------------------------------

/// A chat message in a conversation.
#[napi(object)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl From<ChatMessage> for crabinfer_core::provider::ChatMessage {
    fn from(m: ChatMessage) -> Self {
        Self {
            role: m.role,
            content: m.content,
        }
    }
}

/// Unified completion request.
#[napi(object)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: u32,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub system_prompt: Option<String>,
    pub api_key_override: Option<String>,
}

impl From<CompletionRequest> for crabinfer_core::provider::CompletionRequest {
    fn from(r: CompletionRequest) -> Self {
        Self {
            model: r.model.unwrap_or_default(),
            messages: r.messages.into_iter().map(Into::into).collect(),
            max_tokens: r.max_tokens,
            temperature: r.temperature.unwrap_or(0.7) as f32,
            top_p: r.top_p.unwrap_or(1.0) as f32,
            system_prompt: r.system_prompt.unwrap_or_default(),
            api_key_override: r.api_key_override.unwrap_or_default(),
        }
    }
}

/// Unified completion response.
#[napi(object)]
pub struct CompletionResponse {
    pub content: String,
    pub model: String,
    pub provider_name: String,
    pub stop_reason: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub routing_info: String,
}

impl From<crabinfer_core::provider::CompletionResponse> for CompletionResponse {
    fn from(r: crabinfer_core::provider::CompletionResponse) -> Self {
        Self {
            content: r.content,
            model: r.model,
            provider_name: r.provider_name,
            stop_reason: r.stop_reason,
            input_tokens: r.input_tokens,
            output_tokens: r.output_tokens,
            routing_info: r.routing_info,
        }
    }
}

/// Describes a model available from a provider.
#[napi(object)]
pub struct ModelDescriptor {
    pub id: String,
    pub name: String,
    pub provider: String,
    pub is_local: bool,
    pub context_length: u32,
}

impl From<crabinfer_core::provider::ModelDescriptor> for ModelDescriptor {
    fn from(m: crabinfer_core::provider::ModelDescriptor) -> Self {
        Self {
            id: m.id,
            name: m.name,
            provider: m.provider,
            is_local: m.is_local,
            context_length: m.context_length,
        }
    }
}

/// Provider configuration.
#[napi(object)]
pub struct ProviderConfig {
    pub provider_type: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub default_model: Option<String>,
    pub timeout_seconds: Option<u32>,
    pub tier_override: Option<String>,
}

impl From<ProviderConfig> for crabinfer_core::provider::ProviderConfig {
    fn from(c: ProviderConfig) -> Self {
        Self {
            provider_type: c.provider_type,
            api_key: c.api_key.unwrap_or_default(),
            base_url: c.base_url.unwrap_or_default(),
            default_model: c.default_model.unwrap_or_default(),
            timeout_seconds: c.timeout_seconds.unwrap_or(30),
            tier_override: c.tier_override.unwrap_or_default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Router types
// ---------------------------------------------------------------------------

/// Router configuration.
#[napi(object)]
pub struct RouterConfig {
    pub policy: RoutingPolicy,
    pub privacy_mode: Option<bool>,
    pub ollama_is_local: Option<bool>,
    pub data_sovereignty: Option<bool>,
}

impl From<RouterConfig> for crabinfer_core::router::RouterConfig {
    fn from(c: RouterConfig) -> Self {
        Self {
            policy: c.policy.into(),
            privacy_mode: c.privacy_mode.unwrap_or(false),
            ollama_is_local: c.ollama_is_local.unwrap_or(true),
            data_sovereignty: c.data_sovereignty.unwrap_or(false),
        }
    }
}

/// Routing decision details.
#[napi(object)]
pub struct RoutingDecision {
    pub provider_name: String,
    pub reason: RoutingReason,
    pub providers_tried: u32,
    pub is_local: bool,
    pub provider_tier: ProviderTier,
    pub memory_pressure: MemoryPressure,
    pub network_available: bool,
    pub latency_ms: u32,
}

impl From<crabinfer_core::router::RoutingDecision> for RoutingDecision {
    fn from(d: crabinfer_core::router::RoutingDecision) -> Self {
        Self {
            provider_name: d.provider_name,
            reason: d.reason.into(),
            providers_tried: d.providers_tried,
            is_local: d.is_local,
            provider_tier: d.provider_tier.into(),
            memory_pressure: d.memory_pressure.into(),
            network_available: d.network_available,
            latency_ms: d.latency_ms,
        }
    }
}

// ---------------------------------------------------------------------------
// Catalog types
// ---------------------------------------------------------------------------

/// A model in the curated catalog.
#[napi(object)]
pub struct CatalogEntry {
    pub id: String,
    pub name: String,
    pub hf_repo: String,
    pub gguf_file: String,
    pub tokenizer_repo: String,
    pub size_bytes: i64,
    pub param_count_b: f64,
    pub active_param_count_b: f64,
    pub quant: String,
    pub architecture: String,
    pub min_ram_gb: u32,
    pub category: ModelCategory,
    pub is_moe: bool,
    pub sha256: String,
    pub requires_auth: bool,
}

impl From<crabinfer_core::catalog::CatalogEntry> for CatalogEntry {
    fn from(e: crabinfer_core::catalog::CatalogEntry) -> Self {
        Self {
            id: e.id,
            name: e.name,
            hf_repo: e.hf_repo,
            gguf_file: e.gguf_file,
            tokenizer_repo: e.tokenizer_repo,
            size_bytes: e.size_bytes as i64,
            param_count_b: e.param_count_b as f64,
            active_param_count_b: e.active_param_count_b as f64,
            quant: e.quant,
            architecture: e.architecture,
            min_ram_gb: e.min_ram_gb,
            category: e.category.into(),
            is_moe: e.is_moe,
            sha256: e.sha256,
            requires_auth: e.requires_auth,
        }
    }
}

impl From<CatalogEntry> for crabinfer_core::catalog::CatalogEntry {
    fn from(e: CatalogEntry) -> Self {
        let category = match e.category {
            ModelCategory::General => crabinfer_core::catalog::ModelCategory::General,
            ModelCategory::Code => crabinfer_core::catalog::ModelCategory::Code,
            ModelCategory::Reasoning => crabinfer_core::catalog::ModelCategory::Reasoning,
        };
        Self {
            id: e.id,
            name: e.name,
            hf_repo: e.hf_repo,
            gguf_file: e.gguf_file,
            tokenizer_repo: e.tokenizer_repo,
            size_bytes: e.size_bytes as u64,
            param_count_b: e.param_count_b as f32,
            active_param_count_b: e.active_param_count_b as f32,
            quant: e.quant,
            architecture: e.architecture,
            min_ram_gb: e.min_ram_gb,
            category,
            is_moe: e.is_moe,
            sha256: e.sha256,
            requires_auth: e.requires_auth,
        }
    }
}

// ---------------------------------------------------------------------------
// vLLM types
// ---------------------------------------------------------------------------

/// vLLM server metrics.
#[cfg(feature = "providers")]
#[napi(object)]
pub struct VllmServerMetrics {
    pub gpu_cache_usage: f64,
    pub requests_running: u32,
    pub requests_waiting: u32,
    pub requests_swapped: u32,
}

#[cfg(feature = "providers")]
impl From<crabinfer_core::providers::vllm::VllmServerMetrics> for VllmServerMetrics {
    fn from(m: crabinfer_core::providers::vllm::VllmServerMetrics) -> Self {
        Self {
            gpu_cache_usage: m.gpu_cache_usage as f64,
            requests_running: m.requests_running,
            requests_waiting: m.requests_waiting,
            requests_swapped: m.requests_swapped,
        }
    }
}

/// vLLM-specific completion options.
#[cfg(feature = "providers")]
#[napi(object)]
pub struct VllmCompletionOptions {
    pub repetition_penalty: Option<f64>,
    pub min_p: Option<f64>,
    pub guided_json: Option<String>,
    pub guided_regex: Option<String>,
}

#[cfg(feature = "providers")]
impl From<VllmCompletionOptions> for crabinfer_core::providers::vllm::VllmCompletionOptions {
    fn from(o: VllmCompletionOptions) -> Self {
        Self {
            repetition_penalty: o.repetition_penalty.unwrap_or(1.0) as f32,
            min_p: o.min_p.unwrap_or(0.0) as f32,
            guided_json: o.guided_json.unwrap_or_default(),
            guided_regex: o.guided_regex.unwrap_or_default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Download types
// ---------------------------------------------------------------------------

/// A successfully downloaded model with local paths.
#[cfg(feature = "providers")]
#[napi(object)]
pub struct DownloadedModel {
    pub id: String,
    pub model_path: String,
    pub tokenizer_path: String,
    pub size_bytes: i64,
    pub downloaded_at: String,
}

#[cfg(feature = "providers")]
impl From<crabinfer_core::download::DownloadedModel> for DownloadedModel {
    fn from(d: crabinfer_core::download::DownloadedModel) -> Self {
        Self {
            id: d.id,
            model_path: d.model_path,
            tokenizer_path: d.tokenizer_path,
            size_bytes: d.size_bytes as i64,
            downloaded_at: d.downloaded_at,
        }
    }
}
