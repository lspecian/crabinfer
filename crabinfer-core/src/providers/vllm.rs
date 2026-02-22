//! vLLM provider — connects to self-hosted vLLM inference servers.
//!
//! vLLM speaks the OpenAI `/v1/chat/completions` format, so the core
//! completion/streaming logic reuses the same SSE parsing. This provider
//! adds health checking, dynamic model discovery, Prometheus metrics
//! scraping, and vLLM-specific sampling parameters (guided decoding,
//! repetition_penalty, min_p, etc.).

use crate::provider::{
    CompletionRequest, CompletionResponse, ModelDescriptor, Provider, ProviderConfig,
};
use crate::providers::http_utils::{build_client, SseReader};
use crate::{CrabInferError, TokenOutput};
use reqwest::blocking::Client;
use serde::Deserialize;

// ---------------------------------------------------------------------------
// vLLM-specific types (UniFFI-exported via lib.rs)
// ---------------------------------------------------------------------------

/// Server-side metrics scraped from vLLM's Prometheus `/metrics` endpoint.
#[derive(Debug, Clone, Default, uniffi::Record)]
pub struct VllmServerMetrics {
    /// GPU KV-cache usage (0.0–1.0, where 1.0 = 100%).
    pub gpu_cache_usage: f32,
    /// Number of requests currently running on GPU.
    pub requests_running: u32,
    /// Number of requests waiting to be processed.
    pub requests_waiting: u32,
    /// Number of requests swapped to CPU.
    pub requests_swapped: u32,
}

/// Extra sampling/generation parameters specific to vLLM.
///
/// Pass these to `complete_with_options()` for vLLM-specific features.
/// Default values (0.0 / 1.0 / empty) mean "disabled / use server default".
#[derive(Debug, Clone, uniffi::Record)]
pub struct VllmCompletionOptions {
    /// Repetition penalty. 1.0 = disabled.
    pub repetition_penalty: f32,
    /// Minimum probability relative to the top token. 0.0 = disabled.
    pub min_p: f32,
    /// JSON schema string for guided/constrained generation. Empty = disabled.
    pub guided_json: String,
    /// Regex pattern for guided/constrained generation. Empty = disabled.
    pub guided_regex: String,
}

impl Default for VllmCompletionOptions {
    fn default() -> Self {
        Self {
            repetition_penalty: 1.0,
            min_p: 0.0,
            guided_json: String::new(),
            guided_regex: String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

pub struct VllmProvider {
    client: Client,
    api_key: String,
    base_url: String,
    default_model: String,
}

impl VllmProvider {
    pub fn new(config: ProviderConfig) -> Result<Self, CrabInferError> {
        let client = build_client(config.timeout_seconds)?;
        // vLLM has no public cloud endpoint — base_url is required.
        let base_url = if config.base_url.is_empty() {
            "http://localhost:8000".to_string()
        } else {
            config.base_url.trim_end_matches('/').to_string()
        };
        let api_key = crate::credentials::resolve_api_key("vllm", &config.api_key)
            .unwrap_or_default();
        Ok(Self {
            client,
            api_key,
            base_url,
            default_model: config.default_model,
        })
    }

    fn resolve_model(&self, request_model: &str) -> String {
        if request_model.is_empty() {
            self.default_model.clone()
        } else {
            request_model.to_string()
        }
    }

    fn build_messages(request: &CompletionRequest) -> Vec<serde_json::Value> {
        let mut messages = Vec::new();
        if !request.system_prompt.is_empty() {
            messages.push(serde_json::json!({
                "role": "system",
                "content": request.system_prompt,
            }));
        }
        for msg in &request.messages {
            messages.push(serde_json::json!({
                "role": msg.role,
                "content": msg.content,
            }));
        }
        messages
    }

    fn effective_key<'a>(&'a self, request: &'a CompletionRequest) -> &'a str {
        if request.api_key_override.is_empty() {
            &self.api_key
        } else {
            &request.api_key_override
        }
    }

    fn build_request_body(
        &self,
        request: &CompletionRequest,
        stream: bool,
        options: Option<&VllmCompletionOptions>,
    ) -> serde_json::Value {
        let model = self.resolve_model(&request.model);
        let mut body = serde_json::json!({
            "model": model,
            "messages": Self::build_messages(request),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": stream,
        });

        // Inject vLLM-specific extra parameters when provided.
        if let Some(opts) = options {
            if opts.repetition_penalty != 1.0 {
                body["repetition_penalty"] = serde_json::json!(opts.repetition_penalty);
            }
            if opts.min_p > 0.0 {
                body["min_p"] = serde_json::json!(opts.min_p);
            }
            if !opts.guided_json.is_empty() {
                body["guided_json"] = serde_json::json!(opts.guided_json);
            }
            if !opts.guided_regex.is_empty() {
                body["guided_regex"] = serde_json::json!(opts.guided_regex);
            }
        }

        body
    }

    fn send_completions(
        &self,
        request: &CompletionRequest,
        stream: bool,
        options: Option<&VllmCompletionOptions>,
    ) -> Result<reqwest::blocking::Response, CrabInferError> {
        let body = self.build_request_body(request, stream, options);
        let key = self.effective_key(request);

        let mut req = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .header("Content-Type", "application/json");

        if !key.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", key));
        }

        let resp = req.json(&body).send().map_err(|e| CrabInferError::NetworkError {
            reason: e.to_string(),
        })?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body_text = resp.text().unwrap_or_default();
            return match status {
                401 => Err(CrabInferError::AuthenticationFailed {
                    provider: "vllm".to_string(),
                }),
                429 => Err(CrabInferError::RateLimited {
                    provider: "vllm".to_string(),
                    retry_after_seconds: 10,
                }),
                _ => Err(CrabInferError::ApiError {
                    provider: "vllm".to_string(),
                    status_code: status as u32,
                    message: body_text,
                }),
            };
        }

        Ok(resp)
    }

    // --- vLLM-specific public methods ---

    /// Check if the vLLM server is healthy via `GET /health`.
    pub fn health(&self) -> Result<bool, CrabInferError> {
        let resp = self
            .client
            .get(format!("{}/health", self.base_url))
            .send()
            .map_err(|e| CrabInferError::NetworkError {
                reason: e.to_string(),
            })?;
        Ok(resp.status().is_success())
    }

    /// Scrape key Prometheus metrics from `GET /metrics`.
    pub fn server_metrics(&self) -> Result<VllmServerMetrics, CrabInferError> {
        let resp = self
            .client
            .get(format!("{}/metrics", self.base_url))
            .send()
            .map_err(|e| CrabInferError::NetworkError {
                reason: e.to_string(),
            })?;

        if !resp.status().is_success() {
            return Err(CrabInferError::ApiError {
                provider: "vllm".to_string(),
                status_code: resp.status().as_u16() as u32,
                message: "failed to fetch /metrics".to_string(),
            });
        }

        let text = resp.text().map_err(|e| CrabInferError::NetworkError {
            reason: e.to_string(),
        })?;

        Ok(parse_prometheus_metrics(&text))
    }

    /// Generate a complete response with vLLM-specific options.
    pub fn complete_with_options(
        &self,
        request: &CompletionRequest,
        options: &VllmCompletionOptions,
    ) -> Result<CompletionResponse, CrabInferError> {
        let resp = self.send_completions(request, false, Some(options))?;
        parse_completion_response(resp)
    }
}

// ---------------------------------------------------------------------------
// Provider trait
// ---------------------------------------------------------------------------

impl Provider for VllmProvider {
    fn name(&self) -> &str {
        "vllm"
    }

    fn complete(
        &self,
        request: &CompletionRequest,
    ) -> Result<CompletionResponse, CrabInferError> {
        let resp = self.send_completions(request, false, None)?;
        parse_completion_response(resp)
    }

    fn stream(
        &self,
        request: &CompletionRequest,
    ) -> Result<
        Box<dyn Iterator<Item = Result<TokenOutput, CrabInferError>> + Send>,
        CrabInferError,
    > {
        let resp = self.send_completions(request, true, None)?;
        let sse_reader = SseReader::new(resp);
        Ok(Box::new(VllmStreamIterator {
            sse_reader,
            done: false,
        }))
    }

    fn available_models(&self) -> Result<Vec<ModelDescriptor>, CrabInferError> {
        // Live API call — vLLM serves real model list
        let mut req = self
            .client
            .get(format!("{}/v1/models", self.base_url));

        if !self.api_key.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", self.api_key));
        }

        let resp = req.send().map_err(|e| CrabInferError::NetworkError {
            reason: e.to_string(),
        })?;

        if !resp.status().is_success() {
            return Err(CrabInferError::ApiError {
                provider: "vllm".to_string(),
                status_code: resp.status().as_u16() as u32,
                message: "failed to fetch /v1/models".to_string(),
            });
        }

        let data: VllmModelsResponse =
            resp.json().map_err(|e| CrabInferError::ApiError {
                provider: "vllm".to_string(),
                status_code: 0,
                message: format!("failed to parse /v1/models: {}", e),
            })?;

        Ok(data
            .data
            .into_iter()
            .map(|m| ModelDescriptor {
                id: m.id.clone(),
                name: m.id,
                provider: "vllm".to_string(),
                is_local: false,
                context_length: m.max_model_len.unwrap_or(0),
            })
            .collect())
    }

    fn is_available(&self) -> bool {
        self.health().unwrap_or(false)
    }
}

// ---------------------------------------------------------------------------
// Response parsing (OpenAI-compatible)
// ---------------------------------------------------------------------------

fn parse_completion_response(
    resp: reqwest::blocking::Response,
) -> Result<CompletionResponse, CrabInferError> {
    let data: OpenAIResponse = resp.json().map_err(|e| CrabInferError::ApiError {
        provider: "vllm".to_string(),
        status_code: 0,
        message: format!("failed to parse response: {}", e),
    })?;

    let choice = data.choices.first().ok_or(CrabInferError::ApiError {
        provider: "vllm".to_string(),
        status_code: 0,
        message: "no choices in response".to_string(),
    })?;

    Ok(CompletionResponse {
        content: choice.message.content.clone(),
        model: data.model,
        provider_name: "vllm".to_string(),
        stop_reason: choice
            .finish_reason
            .clone()
            .unwrap_or_else(|| "end_turn".to_string()),
        input_tokens: data.usage.prompt_tokens,
        output_tokens: data.usage.completion_tokens,
        routing_info: String::new(),
    })
}

// ---------------------------------------------------------------------------
// Stream iterator (identical to OpenAI SSE format)
// ---------------------------------------------------------------------------

struct VllmStreamIterator {
    sse_reader: SseReader<reqwest::blocking::Response>,
    done: bool,
}

impl Iterator for VllmStreamIterator {
    type Item = Result<TokenOutput, CrabInferError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        loop {
            let event = match self.sse_reader.next()? {
                Ok(e) => e,
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            };

            if event.data == "[DONE]" {
                self.done = true;
                return None;
            }

            let chunk: OpenAIChunk = match serde_json::from_str(&event.data) {
                Ok(c) => c,
                Err(_) => continue,
            };

            if let Some(choice) = chunk.choices.first() {
                if let Some(ref content) = choice.delta.content {
                    if !content.is_empty() {
                        return Some(Ok(TokenOutput {
                            text: content.clone(),
                            token_id: 0,
                            probability: 0.0,
                            is_end_of_sequence: false,
                        }));
                    }
                }
                if choice.finish_reason.is_some() {
                    self.done = true;
                    return Some(Ok(TokenOutput {
                        text: String::new(),
                        token_id: 0,
                        probability: 0.0,
                        is_end_of_sequence: true,
                    }));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Prometheus metrics parser
// ---------------------------------------------------------------------------

/// Parse key vLLM Prometheus gauges from the text exposition format.
///
/// We only extract the 4 gauges we care about. The format is:
/// ```text
/// # HELP vllm:gpu_cache_usage_perc GPU KV-cache usage.
/// # TYPE vllm:gpu_cache_usage_perc gauge
/// vllm:gpu_cache_usage_perc 0.42
/// ```
fn parse_prometheus_metrics(text: &str) -> VllmServerMetrics {
    let mut metrics = VllmServerMetrics::default();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Prometheus text format: `metric_name{labels} value` or `metric_name value`
        // Strip labels by finding the last space-separated token as the value.
        // For gauges with labels we take the first match (no label filtering).
        if let Some(val) = extract_metric_value(line, "vllm:gpu_cache_usage_perc") {
            metrics.gpu_cache_usage = val;
        } else if let Some(val) = extract_metric_value(line, "vllm:num_requests_running") {
            metrics.requests_running = val as u32;
        } else if let Some(val) = extract_metric_value(line, "vllm:num_requests_waiting") {
            metrics.requests_waiting = val as u32;
        } else if let Some(val) = extract_metric_value(line, "vllm:num_requests_swapped") {
            metrics.requests_swapped = val as u32;
        }
    }

    metrics
}

/// Extract a float value for a metric line that starts with `name` (possibly followed by labels).
fn extract_metric_value(line: &str, name: &str) -> Option<f32> {
    if !line.starts_with(name) {
        return None;
    }

    // After the metric name, there's either a space, '{' (labels), or nothing.
    let rest = &line[name.len()..];
    let after_labels = if rest.starts_with('{') {
        // Skip past the closing '}'
        rest.find('}').map(|i| &rest[i + 1..])
    } else {
        Some(rest)
    };

    after_labels
        .and_then(|s| s.trim().split_whitespace().next())
        .and_then(|v| v.parse::<f32>().ok())
}

// ---------------------------------------------------------------------------
// Response types (OpenAI-compatible, same as vLLM output)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct OpenAIResponse {
    model: String,
    choices: Vec<OpenAIChoice>,
    usage: OpenAIUsage,
}

#[derive(Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAIMessage {
    content: String,
}

#[derive(Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Deserialize)]
struct OpenAIChunk {
    choices: Vec<OpenAIChunkChoice>,
}

#[derive(Deserialize)]
struct OpenAIChunkChoice {
    delta: OpenAIDelta,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAIDelta {
    content: Option<String>,
}

// vLLM /v1/models response
#[derive(Deserialize)]
struct VllmModelsResponse {
    data: Vec<VllmModelEntry>,
}

#[derive(Deserialize)]
struct VllmModelEntry {
    id: String,
    #[serde(default)]
    max_model_len: Option<u32>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_openai_compatible_response() {
        let json = r#"{
            "id": "cmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "Qwen/Qwen3-8B",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from vLLM!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}
        }"#;
        let resp: OpenAIResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.model, "Qwen/Qwen3-8B");
        assert_eq!(resp.choices[0].message.content, "Hello from vLLM!");
        assert_eq!(resp.usage.prompt_tokens, 12);
        assert_eq!(resp.usage.completion_tokens, 8);
    }

    #[test]
    fn test_parse_streaming_chunk() {
        let json = r#"{"id":"cmpl-1","object":"chat.completion.chunk","created":1234,"model":"Qwen/Qwen3-8B","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}"#;
        let chunk: OpenAIChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Hi"));
        assert!(chunk.choices[0].finish_reason.is_none());
    }

    #[test]
    fn test_parse_vllm_models_response() {
        let json = r#"{
            "object": "list",
            "data": [
                {"id": "Qwen/Qwen3-8B", "object": "model", "created": 1234, "owned_by": "vllm", "max_model_len": 32768},
                {"id": "meta-llama/Llama-3.1-8B", "object": "model", "created": 1234, "owned_by": "vllm"}
            ]
        }"#;
        let resp: VllmModelsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[0].id, "Qwen/Qwen3-8B");
        assert_eq!(resp.data[0].max_model_len, Some(32768));
        assert_eq!(resp.data[1].id, "meta-llama/Llama-3.1-8B");
        assert_eq!(resp.data[1].max_model_len, None);
    }

    #[test]
    fn test_parse_prometheus_metrics() {
        let text = r#"
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc 0.42
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 3
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting 7
# HELP vllm:num_requests_swapped Number of requests swapped to CPU.
# TYPE vllm:num_requests_swapped gauge
vllm:num_requests_swapped 1
# HELP vllm:num_preemptions_total Cumulative number of preemption from the engine.
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total 0
"#;
        let m = parse_prometheus_metrics(text);
        assert!((m.gpu_cache_usage - 0.42).abs() < 0.001);
        assert_eq!(m.requests_running, 3);
        assert_eq!(m.requests_waiting, 7);
        assert_eq!(m.requests_swapped, 1);
    }

    #[test]
    fn test_parse_prometheus_metrics_with_labels() {
        // Some vLLM metrics have labels like model_name
        let text = r#"
vllm:gpu_cache_usage_perc{model_name="Qwen/Qwen3-8B"} 0.65
vllm:num_requests_running{model_name="Qwen/Qwen3-8B"} 5
vllm:num_requests_waiting{model_name="Qwen/Qwen3-8B"} 0
vllm:num_requests_swapped{model_name="Qwen/Qwen3-8B"} 2
"#;
        let m = parse_prometheus_metrics(text);
        assert!((m.gpu_cache_usage - 0.65).abs() < 0.001);
        assert_eq!(m.requests_running, 5);
        assert_eq!(m.requests_waiting, 0);
        assert_eq!(m.requests_swapped, 2);
    }

    #[test]
    fn test_vllm_completion_options_default() {
        let opts = VllmCompletionOptions::default();
        assert!((opts.repetition_penalty - 1.0).abs() < f32::EPSILON);
        assert!((opts.min_p - 0.0).abs() < f32::EPSILON);
        assert!(opts.guided_json.is_empty());
        assert!(opts.guided_regex.is_empty());
    }

    #[test]
    fn test_extract_metric_value() {
        assert_eq!(extract_metric_value("vllm:gpu_cache_usage_perc 0.42", "vllm:gpu_cache_usage_perc"), Some(0.42));
        assert_eq!(extract_metric_value("vllm:gpu_cache_usage_perc{model=\"x\"} 0.7", "vllm:gpu_cache_usage_perc"), Some(0.7));
        assert_eq!(extract_metric_value("vllm:num_requests_running 10", "vllm:num_requests_running"), Some(10.0));
        assert_eq!(extract_metric_value("# comment", "vllm:gpu_cache_usage_perc"), None);
        assert_eq!(extract_metric_value("other_metric 1.0", "vllm:gpu_cache_usage_perc"), None);
    }
}
