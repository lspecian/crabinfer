use serde::{Deserialize, Serialize};

/// A single chat message. Supports text, assistant tool_calls, and tool results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    /// Text content (may be None for pure tool_calls messages).
    #[serde(default)]
    pub content: Option<String>,
    /// Tool call ID — present when role is "tool" (result of a tool invocation).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Tool calls produced by the assistant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallMessage>>,
    /// Function name — present when role is "tool" (identifies which tool produced this result).
    /// Some clients send this alongside tool_call_id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    /// Get content as a string, defaulting to empty.
    pub fn content_str(&self) -> &str {
        self.content.as_deref().unwrap_or("")
    }
}

/// A tool call embedded in a chat message (input side).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallMessage {
    pub id: String,
    #[serde(rename = "type")]
    pub type_field: String,
    pub function: FunctionCallMessage,
}

/// Function name + arguments in a tool call message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallMessage {
    pub name: String,
    pub arguments: String,
}
