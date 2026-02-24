//! CrabInferRouter class — smart routing between local, self-hosted, and cloud providers.

use napi_derive::napi;

use crabinfer_core::provider::Provider;

use crate::error::to_napi_error;
use crate::stream::TokenStream;
use crate::types::{
    CompletionRequest, CompletionResponse, EngineConfig, ModelDescriptor, ProviderConfig,
    RouterConfig, RoutingDecision,
};

/// Smart router that automatically selects between local, self-hosted, and cloud providers.
#[napi]
pub struct CrabInferRouter {
    inner: crabinfer_core::router::Router,
}

#[napi]
impl CrabInferRouter {
    /// Create a router with optional local and cloud providers.
    #[napi(constructor)]
    pub fn new(
        config: RouterConfig,
        local_config: Option<EngineConfig>,
        cloud_configs: Vec<ProviderConfig>,
    ) -> napi::Result<Self> {
        let core_config: crabinfer_core::router::RouterConfig = config.into();

        let local: Option<Box<dyn Provider>> = match local_config {
            Some(ec) => {
                let core_ec: crabinfer_core::EngineConfig = ec.into();
                Some(Box::new(
                    crabinfer_core::providers::local::LocalProvider::new(core_ec)
                        .map_err(to_napi_error)?,
                ))
            }
            None => None,
        };

        #[allow(unused_mut)]
        let mut self_hosted: Vec<Box<dyn Provider>> = Vec::new();
        #[allow(unused_mut)]
        let mut cloud: Vec<Box<dyn Provider>> = Vec::new();

        #[cfg(feature = "providers")]
        for cc in cloud_configs {
            let core_cc: crabinfer_core::provider::ProviderConfig = cc.into();
            let tier = crabinfer_core::router::resolve_tier(&core_cc);
            let provider: Box<dyn Provider> = match core_cc.provider_type.as_str() {
                "openai" => Box::new(
                    crabinfer_core::providers::openai::OpenAIProvider::new(core_cc)
                        .map_err(to_napi_error)?,
                ),
                "anthropic" => Box::new(
                    crabinfer_core::providers::anthropic::AnthropicProvider::new(core_cc)
                        .map_err(to_napi_error)?,
                ),
                "google" => Box::new(
                    crabinfer_core::providers::google::GoogleProvider::new(core_cc)
                        .map_err(to_napi_error)?,
                ),
                "ollama" => Box::new(
                    crabinfer_core::providers::ollama::OllamaProvider::new(core_cc)
                        .map_err(to_napi_error)?,
                ),
                "vllm" => Box::new(
                    crabinfer_core::providers::vllm::VllmProvider::new(core_cc)
                        .map_err(to_napi_error)?,
                ),
                _ => return Err(to_napi_error(crabinfer_core::CrabInferError::InvalidConfig)),
            };
            match tier {
                crabinfer_core::router::ProviderTier::SelfHosted => self_hosted.push(provider),
                _ => cloud.push(provider),
            }
        }

        #[cfg(not(feature = "providers"))]
        {
            if !cloud_configs.is_empty() {
                return Err(to_napi_error(crabinfer_core::CrabInferError::InvalidConfig));
            }
        }

        Ok(Self {
            inner: crabinfer_core::router::Router::new(core_config, local, self_hosted, cloud),
        })
    }

    /// Router name.
    #[napi]
    pub fn name(&self) -> String {
        "router".to_string()
    }

    /// Generate a complete response, routing automatically.
    #[napi]
    pub fn complete(&self, request: CompletionRequest) -> napi::Result<CompletionResponse> {
        let core_request: crabinfer_core::provider::CompletionRequest = request.into();
        self.inner
            .complete(&core_request)
            .map(Into::into)
            .map_err(to_napi_error)
    }

    /// Start streaming, routing automatically. Returns a TokenStream.
    #[napi]
    pub fn stream(&self, request: CompletionRequest) -> napi::Result<TokenStream> {
        let core_request: crabinfer_core::provider::CompletionRequest = request.into();
        let iter = self.inner.stream(&core_request).map_err(to_napi_error)?;
        Ok(TokenStream::new(iter))
    }

    /// List models available from all configured providers.
    #[napi]
    pub fn available_models(&self) -> napi::Result<Vec<ModelDescriptor>> {
        self.inner
            .available_models()
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(to_napi_error)
    }

    /// Check if any provider is available.
    #[napi]
    pub fn is_available(&self) -> bool {
        self.inner.is_available()
    }

    /// Set network availability (call when connectivity changes).
    #[napi]
    pub fn set_network_available(&self, available: bool) {
        self.inner.set_network_available(available);
    }

    /// Get the routing decision from the last complete() or stream() call.
    #[napi]
    pub fn last_routing_decision(&self) -> Option<RoutingDecision> {
        self.inner.last_routing_decision().map(Into::into)
    }
}
