use super::common::ChatMessage;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct MessagesRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: u32,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub system: Option<String>,
    /// Per-request cache salt for tenant isolation. CrabInfer extension.
    #[serde(default)]
    pub cache_salt: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anthropic_cache_salt_deserialization() {
        let json = r#"{"model":"claude","messages":[],"max_tokens":100,"cache_salt":"tenant-bar"}"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.cache_salt.as_deref(), Some("tenant-bar"));
    }

    #[test]
    fn test_anthropic_cache_salt_default_none() {
        let json = r#"{"model":"claude","messages":[],"max_tokens":100}"#;
        let req: MessagesRequest = serde_json::from_str(json).unwrap();
        assert!(req.cache_salt.is_none());
    }
}

// ---------------------------------------------------------------------------
// Non-streaming response
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub type_field: String,
    pub role: String,
    pub content: Vec<ContentBlock>,
    pub model: String,
    pub stop_reason: String,
    pub usage: AnthropicUsage,
}

#[derive(Serialize)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub type_field: String,
    pub text: String,
}

#[derive(Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

// ---------------------------------------------------------------------------
// Streaming events
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct MessageStartEvent {
    #[serde(rename = "type")]
    pub type_field: String,
    pub message: MessageStartBody,
}

#[derive(Serialize)]
pub struct MessageStartBody {
    pub id: String,
    #[serde(rename = "type")]
    pub type_field: String,
    pub role: String,
    pub content: Vec<()>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Serialize)]
pub struct ContentBlockStartEvent {
    #[serde(rename = "type")]
    pub type_field: String,
    pub index: u32,
    pub content_block: ContentBlock,
}

#[derive(Serialize)]
pub struct ContentBlockDeltaEvent {
    #[serde(rename = "type")]
    pub type_field: String,
    pub index: u32,
    pub delta: TextDelta,
}

#[derive(Serialize)]
pub struct TextDelta {
    #[serde(rename = "type")]
    pub type_field: String,
    pub text: String,
}

#[derive(Serialize)]
pub struct ContentBlockStopEvent {
    #[serde(rename = "type")]
    pub type_field: String,
    pub index: u32,
}

#[derive(Serialize)]
pub struct MessageDeltaEvent {
    #[serde(rename = "type")]
    pub type_field: String,
    pub delta: MessageDelta,
    pub usage: AnthropicUsage,
}

#[derive(Serialize)]
pub struct MessageDelta {
    pub stop_reason: String,
}

#[derive(Serialize)]
pub struct MessageStopEvent {
    #[serde(rename = "type")]
    pub type_field: String,
}
