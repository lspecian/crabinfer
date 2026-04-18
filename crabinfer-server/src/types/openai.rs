use super::common::ChatMessage;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<StopCondition>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<u32>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Request priority (lower = higher priority, default 0). CrabInfer extension.
    #[serde(default)]
    pub priority: Option<i32>,
    /// Per-request cache salt for tenant isolation. CrabInfer extension.
    /// When set, prefix cache block hashes are namespaced to this salt,
    /// preventing KV cache sharing across tenants even for identical prompts.
    #[serde(default)]
    pub cache_salt: Option<String>,
    /// Tool definitions available for the model to call.
    #[serde(default)]
    pub tools: Option<Vec<ToolDefinition>>,
    /// Controls whether the model should call tools.
    /// - `"auto"` (default): model decides
    /// - `"none"`: never call tools
    /// - `"required"`: must call at least one tool
    /// - `{"type": "function", "function": {"name": "..."}}`: call a specific tool
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
    /// Response format constraint.
    /// - `{"type": "json_object"}` — constrain output to valid JSON
    /// - `{"type": "json_schema", "json_schema": {"name": "...", "schema": {...}}}` — constrain to specific schema
    /// - `{"type": "text"}` (default) — no constraint
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
}

#[derive(Debug, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
#[allow(dead_code)]
pub enum StopCondition {
    Single(String),
    Multiple(Vec<String>),
}

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

/// A tool the model may call (currently only "function" type).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub type_field: String,
    pub function: FunctionDefinition,
}

/// Function definition within a tool.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// JSON Schema for the function parameters.
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
}

/// Controls tool calling behavior.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// "auto", "none", or "required"
    String(String),
    /// Specific tool: {"type": "function", "function": {"name": "..."}}
    Specific(ToolChoiceSpecific),
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolChoiceSpecific {
    #[serde(rename = "type")]
    pub type_field: String,
    pub function: ToolChoiceFunction,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolChoiceFunction {
    pub name: String,
}

// ---------------------------------------------------------------------------
// Response format
// ---------------------------------------------------------------------------

/// Response format constraint (OpenAI-compatible).
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseFormat {
    /// "text" (default), "json_object", or "json_schema"
    #[serde(rename = "type")]
    pub type_field: String,
    /// JSON schema specification (only when type is "json_schema").
    #[serde(default)]
    pub json_schema: Option<JsonSchemaSpec>,
}

/// JSON schema specification for structured output.
#[derive(Debug, Clone, Deserialize)]
pub struct JsonSchemaSpec {
    /// Schema name (used in prompting).
    pub name: String,
    /// Optional description.
    #[serde(default)]
    pub description: Option<String>,
    /// The JSON Schema itself.
    #[serde(default)]
    pub schema: Option<serde_json::Value>,
    /// Whether to enable strict schema adherence. Default: false.
    #[serde(default)]
    pub strict: Option<bool>,
}

// ---------------------------------------------------------------------------
// Vision / multimodal content parts
// ---------------------------------------------------------------------------

/// A single part of a multimodal message content array.
///
/// OpenAI vision format: `[{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {...}}]`
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    /// Text content part.
    #[serde(rename = "text")]
    Text {
        text: String,
    },
    /// Image URL content part (base64 data URI or HTTP URL).
    #[serde(rename = "image_url")]
    ImageUrl {
        image_url: ImageUrl,
    },
}

/// Image URL with optional detail level.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageUrl {
    /// Image source: `data:image/png;base64,...` or `https://...`
    pub url: String,
    /// Detail level for image processing. Default: "auto".
    /// - `"auto"` — server decides based on image size
    /// - `"low"` — resize to 512x512, fewer tokens
    /// - `"high"` — use higher resolution, more tokens
    #[serde(default = "default_detail")]
    pub detail: String,
}

fn default_detail() -> String {
    "auto".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_salt_deserialization() {
        let json = r#"{"model":"gpt-4","messages":[],"cache_salt":"tenant-foo"}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.cache_salt.as_deref(), Some("tenant-foo"));
    }

    #[test]
    fn test_cache_salt_default_none() {
        let json = r#"{"model":"gpt-4","messages":[]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.cache_salt.is_none());
    }

    #[test]
    fn test_completion_request_deser() {
        let json = r#"{"model":"m","prompt":"hello"}"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "m");
        assert_eq!(req.prompt, "hello");
        assert!(req.max_tokens.is_none());
        assert!(req.temperature.is_none());
        assert!(req.top_p.is_none());
        assert!(req.stream.is_none());
        assert!(req.cache_salt.is_none());
    }

    #[test]
    fn test_completion_request_with_optionals() {
        let json = r#"{"model":"m","prompt":"hi","max_tokens":64,"temperature":0.7,"top_p":0.9,"stream":true,"cache_salt":"x"}"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.max_tokens, Some(64));
        assert!((req.temperature.unwrap() - 0.7).abs() < 1e-5);
        assert!((req.top_p.unwrap() - 0.9).abs() < 1e-5);
        assert_eq!(req.stream, Some(true));
        assert_eq!(req.cache_salt.as_deref(), Some("x"));
    }

    #[test]
    fn test_completion_response_ser() {
        let resp = CompletionResponse {
            id: "cmpl-test".to_string(),
            object: "text_completion".to_string(),
            created: 1234567890,
            model: "m".to_string(),
            choices: vec![CompletionChoice {
                text: "hello world".to_string(),
                index: 0,
                finish_reason: "stop".to_string(),
                logprobs: None,
            }],
            usage: Usage {
                prompt_tokens: 5,
                completion_tokens: 3,
                total_tokens: 8,
            },
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["object"], "text_completion");
        assert_eq!(json["choices"][0]["text"], "hello world");
        assert_eq!(json["choices"][0]["index"], 0);
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert_eq!(json["usage"]["total_tokens"], 8);
    }

    #[test]
    fn test_completion_choice_logprobs_serializes_as_null() {
        let choice = CompletionChoice {
            text: "hi".to_string(),
            index: 0,
            finish_reason: "stop".to_string(),
            logprobs: None,
        };
        let json = serde_json::to_value(&choice).unwrap();
        // The key MUST be present even when None; check it's null, not absent
        assert!(json.get("logprobs").is_some(), "logprobs key must be present");
        assert!(json["logprobs"].is_null(), "logprobs must serialize as JSON null");
    }
}

// ---------------------------------------------------------------------------
// Non-streaming response
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ChoiceMessage,
    pub finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
}

/// Logprobs object attached to a choice (non-streaming).
#[derive(Serialize)]
pub struct ChoiceLogprobs {
    pub content: Vec<TokenLogprobInfo>,
}

/// Per-token log probability information (OpenAI-compatible).
#[derive(Serialize, Clone)]
pub struct TokenLogprobInfo {
    pub token: String,
    pub logprob: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<Vec<TopLogprobEntry>>,
}

/// A single entry in the top_logprobs array.
#[derive(Serialize, Clone)]
pub struct TopLogprobEntry {
    pub token: String,
    pub logprob: f32,
}

#[derive(Serialize)]
pub struct ChoiceMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Tool calls produced by the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// A tool call in the response.
#[derive(Debug, Clone, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub type_field: String,
    pub function: FunctionCall,
}

/// Function invocation details in a tool call.
#[derive(Debug, Clone, Serialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ---------------------------------------------------------------------------
// Streaming chunk
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
}

#[derive(Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Tool calls in streaming (each chunk may contain partial tool call data).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<DeltaToolCall>>,
}

/// A tool call delta in streaming responses.
#[derive(Debug, Clone, Serialize)]
pub struct DeltaToolCall {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub type_field: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<DeltaFunctionCall>,
}

/// Partial function call in streaming.
#[derive(Debug, Clone, Serialize)]
pub struct DeltaFunctionCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

// ---------------------------------------------------------------------------
// Legacy /v1/completions (OpenAI text completion endpoint)
// ---------------------------------------------------------------------------

/// Request body for POST /v1/completions (OpenAI legacy completions API).
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    /// Per-request cache salt for tenant isolation. CrabInfer extension.
    /// See PCCH-01 wiring in 07-02 for the equivalent ChatCompletionRequest field.
    #[serde(default)]
    pub cache_salt: Option<String>,
}

/// Response body for POST /v1/completions.
#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,   // always "text_completion"
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,     // reuse existing Usage struct
}

/// One choice in a completion response.
#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    pub finish_reason: String,
    /// OpenAI clients expect `logprobs` as a null field, not an absent field.
    /// We do NOT use `skip_serializing_if` here — the key must be emitted.
    pub logprobs: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Models list
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelObject>,
}

#[derive(Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}
