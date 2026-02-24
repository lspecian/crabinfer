//! CrabInferProvider class — unified provider for local and cloud inference.

use std::sync::Arc;

use napi_derive::napi;

use crabinfer_core::provider::Provider;

use crate::error::to_napi_error;
use crate::stream::TokenStream;
use crate::types::{
    CompletionRequest, CompletionResponse, EngineConfig, ModelDescriptor, ProviderConfig,
};

/// Unified provider interface for local and cloud LLM inference.
#[napi]
pub struct CrabInferProvider {
    inner: Arc<Box<dyn Provider>>,
}

#[napi]
impl CrabInferProvider {
    /// Create a cloud provider from configuration.
    #[napi(factory)]
    pub fn create(config: ProviderConfig) -> napi::Result<Self> {
        let core_config: crabinfer_core::provider::ProviderConfig = config.into();

        #[cfg(feature = "providers")]
        {
            let inner: Box<dyn Provider> = match core_config.provider_type.as_str() {
                "openai" => Box::new(
                    crabinfer_core::providers::openai::OpenAIProvider::new(core_config)
                        .map_err(to_napi_error)?,
                ),
                "anthropic" => Box::new(
                    crabinfer_core::providers::anthropic::AnthropicProvider::new(core_config)
                        .map_err(to_napi_error)?,
                ),
                "google" => Box::new(
                    crabinfer_core::providers::google::GoogleProvider::new(core_config)
                        .map_err(to_napi_error)?,
                ),
                "ollama" => Box::new(
                    crabinfer_core::providers::ollama::OllamaProvider::new(core_config)
                        .map_err(to_napi_error)?,
                ),
                "vllm" => Box::new(
                    crabinfer_core::providers::vllm::VllmProvider::new(core_config)
                        .map_err(to_napi_error)?,
                ),
                _ => return Err(to_napi_error(crabinfer_core::CrabInferError::InvalidConfig)),
            };
            Ok(Self {
                inner: Arc::new(inner),
            })
        }

        #[cfg(not(feature = "providers"))]
        {
            let _ = core_config;
            Err(to_napi_error(crabinfer_core::CrabInferError::InvalidConfig))
        }
    }

    /// Create a local inference provider.
    #[napi(factory)]
    pub fn create_local(engine_config: EngineConfig) -> napi::Result<Self> {
        let core_config: crabinfer_core::EngineConfig = engine_config.into();
        let inner: Box<dyn Provider> = Box::new(
            crabinfer_core::providers::local::LocalProvider::new(core_config)
                .map_err(to_napi_error)?,
        );
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Provider name ("local", "openai", "anthropic", etc.).
    #[napi]
    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// Generate a complete response.
    #[napi]
    pub fn complete(&self, request: CompletionRequest) -> napi::Result<CompletionResponse> {
        let core_request: crabinfer_core::provider::CompletionRequest = request.into();
        self.inner
            .complete(&core_request)
            .map(Into::into)
            .map_err(to_napi_error)
    }

    /// Start streaming and return a TokenStream (async iterator).
    #[napi]
    pub fn stream(&self, request: CompletionRequest) -> napi::Result<TokenStream> {
        let core_request: crabinfer_core::provider::CompletionRequest = request.into();
        let iter = self.inner.stream(&core_request).map_err(to_napi_error)?;
        Ok(TokenStream::new(iter))
    }

    /// List models available from this provider.
    #[napi]
    pub fn available_models(&self) -> napi::Result<Vec<ModelDescriptor>> {
        self.inner
            .available_models()
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(to_napi_error)
    }

    /// Check if the provider is reachable/configured.
    #[napi]
    pub fn is_available(&self) -> bool {
        self.inner.is_available()
    }
}
