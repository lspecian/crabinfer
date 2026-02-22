//! Provider trait and unified types for multi-provider LLM inference.
//!
//! The Provider abstraction allows CrabInfer to route between local models
//! and cloud LLM providers (OpenAI, Anthropic, Google, Ollama) through a
//! single API. Swift sees the `CrabInferProvider` wrapper object; the trait
//! itself is Rust-internal.

use crate::{CrabInferError, TokenOutput};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Unified types (UniFFI-compatible)
// ---------------------------------------------------------------------------

/// A chat message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize, uniffi::Record)]
pub struct ChatMessage {
    /// Role: "system", "user", or "assistant"
    pub role: String,
    /// Message content
    pub content: String,
}

/// Unified completion request that works for both local and cloud providers.
#[derive(Debug, Clone, uniffi::Record)]
pub struct CompletionRequest {
    /// Model identifier. For local: ignored (uses loaded model).
    /// For cloud: "gpt-4o", "claude-sonnet-4-20250514", "gemini-2.5-pro", etc.
    pub model: String,
    /// Conversation messages.
    pub messages: Vec<ChatMessage>,
    /// Maximum tokens to generate.
    pub max_tokens: u32,
    /// Sampling temperature (0.0 - 2.0).
    pub temperature: f32,
    /// Nucleus sampling threshold (0.0 - 1.0). Use 1.0 to disable.
    pub top_p: f32,
    /// System prompt. Anthropic uses natively; others prepend to messages.
    /// Empty string means no system prompt.
    pub system_prompt: String,
    /// Optional API key override for this specific request.
    /// Used in multi-tenant scenarios where different requests use different
    /// credentials. Empty string = use provider default.
    pub api_key_override: String,
}

/// Unified completion response.
#[derive(Debug, Clone, uniffi::Record)]
pub struct CompletionResponse {
    /// The generated text content.
    pub content: String,
    /// Which model actually served the request.
    pub model: String,
    /// Which provider served the request ("local", "openai", "anthropic", etc.).
    pub provider_name: String,
    /// Stop reason: "end_turn", "max_tokens", "stop_sequence".
    pub stop_reason: String,
    /// Input/prompt token count (estimated for local).
    pub input_tokens: u32,
    /// Output/completion token count.
    pub output_tokens: u32,
    /// Routing info: which provider was selected and why.
    /// Empty string if the request was not routed (direct provider call).
    pub routing_info: String,
}

/// Describes a model available from a provider.
#[derive(Debug, Clone, uniffi::Record)]
pub struct ModelDescriptor {
    /// Model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514", or local model name).
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Provider that offers this model.
    pub provider: String,
    /// Whether this is a local or cloud model.
    pub is_local: bool,
    /// Context window size (0 if unknown).
    pub context_length: u32,
}

/// Provider configuration for cloud and self-hosted providers.
#[derive(Clone, uniffi::Record)]
pub struct ProviderConfig {
    /// Provider type: "openai", "anthropic", "google", "ollama", "vllm"
    pub provider_type: String,
    /// API key (empty for ollama).
    pub api_key: String,
    /// Base URL override (empty = use provider default).
    /// Useful for proxies, Azure OpenAI, or custom Ollama hosts.
    pub base_url: String,
    /// Default model ID for this provider.
    pub default_model: String,
    /// Request timeout in seconds (0 = default 30s).
    pub timeout_seconds: u32,
    /// Provider tier override: "self_hosted", "cloud", or "" (auto-detect from provider_type).
    /// Auto-detection: "ollama"/"vllm" → SelfHosted, everything else → Cloud.
    pub tier_override: String,
}

impl std::fmt::Debug for ProviderConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderConfig")
            .field("provider_type", &self.provider_type)
            .field("api_key", &redact_key(&self.api_key))
            .field("base_url", &self.base_url)
            .field("default_model", &self.default_model)
            .field("timeout_seconds", &self.timeout_seconds)
            .field("tier_override", &self.tier_override)
            .finish()
    }
}

/// Redact an API key for safe logging. Shows first 4 chars + "***".
fn redact_key(key: &str) -> String {
    if key.is_empty() {
        "<empty>".to_string()
    } else if key.len() <= 4 {
        "***".to_string()
    } else {
        format!("{}***", &key[..4])
    }
}

// ---------------------------------------------------------------------------
// Provider trait (Rust-internal, NOT UniFFI-exported)
// ---------------------------------------------------------------------------

/// The Provider trait — internal to Rust, NOT exported to UniFFI.
/// UniFFI sees the `CrabInferProvider` wrapper object instead.
pub trait Provider: Send + Sync {
    /// Human-readable provider name.
    fn name(&self) -> &str;

    /// Generate a complete response (blocking).
    fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse, CrabInferError>;

    /// Generate tokens one at a time (blocking iterator).
    /// Returns an iterator that yields `TokenOutput` items.
    /// For cloud providers, this consumes an SSE stream internally.
    fn stream(
        &self,
        request: &CompletionRequest,
    ) -> Result<
        Box<dyn Iterator<Item = Result<TokenOutput, CrabInferError>> + Send>,
        CrabInferError,
    >;

    /// List models available from this provider.
    fn available_models(&self) -> Result<Vec<ModelDescriptor>, CrabInferError>;

    /// Check if the provider is reachable/configured.
    fn is_available(&self) -> bool;
}
