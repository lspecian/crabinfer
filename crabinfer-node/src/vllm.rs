//! CrabInferVllm class — vLLM-specific provider with health checks and metrics.

#[cfg(feature = "providers")]
use napi_derive::napi;

#[cfg(feature = "providers")]
use crabinfer_core::provider::Provider;

#[cfg(feature = "providers")]
use crate::error::to_napi_error;
#[cfg(feature = "providers")]
use crate::stream::TokenStream;
#[cfg(feature = "providers")]
use crate::types::{
    CompletionRequest, CompletionResponse, ModelDescriptor, ProviderConfig,
    VllmCompletionOptions, VllmServerMetrics,
};

/// vLLM provider with health checks, Prometheus metrics, and guided decoding.
#[cfg(feature = "providers")]
#[napi]
pub struct CrabInferVllm {
    inner: crabinfer_core::providers::vllm::VllmProvider,
}

#[cfg(feature = "providers")]
#[napi]
impl CrabInferVllm {
    /// Create a vLLM provider.
    #[napi(constructor)]
    pub fn new(config: ProviderConfig) -> napi::Result<Self> {
        let core_config: crabinfer_core::provider::ProviderConfig = config.into();
        let inner =
            crabinfer_core::providers::vllm::VllmProvider::new(core_config).map_err(to_napi_error)?;
        Ok(Self { inner })
    }

    /// Provider name: "vllm".
    #[napi]
    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// Generate a complete response.
    #[napi]
    pub fn complete(&self, request: CompletionRequest) -> napi::Result<CompletionResponse> {
        let core_request: crabinfer_core::provider::CompletionRequest = request.into();
        Provider::complete(&self.inner, &core_request)
            .map(Into::into)
            .map_err(to_napi_error)
    }

    /// Generate a complete response with vLLM-specific options.
    #[napi]
    pub fn complete_with_options(
        &self,
        request: CompletionRequest,
        options: VllmCompletionOptions,
    ) -> napi::Result<CompletionResponse> {
        let core_request: crabinfer_core::provider::CompletionRequest = request.into();
        let core_options: crabinfer_core::providers::vllm::VllmCompletionOptions = options.into();
        self.inner
            .complete_with_options(&core_request, &core_options)
            .map(Into::into)
            .map_err(to_napi_error)
    }

    /// Start streaming and return a TokenStream.
    #[napi]
    pub fn stream(&self, request: CompletionRequest) -> napi::Result<TokenStream> {
        let core_request: crabinfer_core::provider::CompletionRequest = request.into();
        let iter = Provider::stream(&self.inner, &core_request).map_err(to_napi_error)?;
        Ok(TokenStream::new(iter))
    }

    /// List models served by the vLLM instance.
    #[napi]
    pub fn available_models(&self) -> napi::Result<Vec<ModelDescriptor>> {
        Provider::available_models(&self.inner)
            .map(|v| v.into_iter().map(Into::into).collect())
            .map_err(to_napi_error)
    }

    /// Check if the vLLM server is reachable.
    #[napi]
    pub fn is_available(&self) -> bool {
        Provider::is_available(&self.inner)
    }

    /// Check vLLM server health via GET /health.
    #[napi]
    pub fn health(&self) -> napi::Result<bool> {
        self.inner.health().map_err(to_napi_error)
    }

    /// Scrape Prometheus metrics from the vLLM server.
    #[napi]
    pub fn server_metrics(&self) -> napi::Result<VllmServerMetrics> {
        self.inner
            .server_metrics()
            .map(Into::into)
            .map_err(to_napi_error)
    }
}
