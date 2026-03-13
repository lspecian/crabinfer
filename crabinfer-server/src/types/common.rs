use crate::types::openai::ContentPart;
use serde::{Deserialize, Serialize};

/// Message content that can be either a plain string or an array of content parts
/// (for multimodal messages with images).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text content.
    Text(String),
    /// Array of content parts (text + images).
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Extract the text content, concatenating all text parts if multimodal.
    pub fn text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    /// Extract image URLs from content parts, if any.
    pub fn image_urls(&self) -> Vec<&crate::types::openai::ImageUrl> {
        match self {
            MessageContent::Text(_) => vec![],
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::ImageUrl { image_url } => Some(image_url),
                    _ => None,
                })
                .collect(),
        }
    }

    /// Check if this content contains any images.
    pub fn has_images(&self) -> bool {
        match self {
            MessageContent::Text(_) => false,
            MessageContent::Parts(parts) => parts
                .iter()
                .any(|p| matches!(p, ContentPart::ImageUrl { .. })),
        }
    }
}

impl From<String> for MessageContent {
    fn from(s: String) -> Self {
        MessageContent::Text(s)
    }
}

impl From<&str> for MessageContent {
    fn from(s: &str) -> Self {
        MessageContent::Text(s.to_string())
    }
}

/// A single chat message. Supports text, multimodal content (images),
/// assistant tool_calls, and tool results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    /// Message content: either a string or an array of content parts (text + images).
    /// May be None for pure tool_calls messages.
    #[serde(default)]
    pub content: Option<MessageContent>,
    /// Tool call ID -- present when role is "tool" (result of a tool invocation).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Tool calls produced by the assistant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallMessage>>,
    /// Function name -- present when role is "tool" (identifies which tool produced this result).
    /// Some clients send this alongside tool_call_id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    /// Get content as a string, defaulting to empty.
    /// For multimodal messages, concatenates all text parts.
    pub fn content_str(&self) -> String {
        match &self.content {
            Some(content) => content.text(),
            None => String::new(),
        }
    }

    /// Extract image URLs from this message, if any.
    pub fn image_urls(&self) -> Vec<&crate::types::openai::ImageUrl> {
        match &self.content {
            Some(content) => content.image_urls(),
            None => vec![],
        }
    }

    /// Check if this message contains images.
    pub fn has_images(&self) -> bool {
        self.content.as_ref().map_or(false, |c| c.has_images())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_message_deserialize_string_content() {
        let json = r#"{"role": "user", "content": "Hello world"}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content_str(), "Hello world");
        assert!(!msg.has_images());
    }

    #[test]
    fn test_chat_message_deserialize_multimodal_content() {
        let json = r#"{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}}
            ]
        }"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content_str(), "What is in this image?");
        assert!(msg.has_images());
        let urls = msg.image_urls();
        assert_eq!(urls.len(), 1);
        assert_eq!(urls[0].url, "data:image/png;base64,abc123");
        assert_eq!(urls[0].detail, "auto");
    }

    #[test]
    fn test_chat_message_deserialize_multimodal_with_detail() {
        let json = r#"{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg", "detail": "high"}}
            ]
        }"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        let urls = msg.image_urls();
        assert_eq!(urls[0].detail, "high");
    }

    #[test]
    fn test_chat_message_deserialize_text_only_parts() {
        let json = r#"{
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"}
            ]
        }"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.content_str(), "Hello world");
        assert!(!msg.has_images());
    }

    #[test]
    fn test_chat_message_deserialize_null_content() {
        let json = r#"{"role": "assistant", "content": null}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.content_str(), "");
        assert!(!msg.has_images());
    }

    #[test]
    fn test_chat_message_deserialize_missing_content() {
        let json = r#"{"role": "assistant"}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.content_str(), "");
    }

    #[test]
    fn test_message_content_from_string() {
        let content: MessageContent = "hello".into();
        assert_eq!(content.text(), "hello");
        assert!(!content.has_images());
    }

    #[test]
    fn test_chat_message_multiple_images() {
        let json = r#"{
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare these:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,img1"}},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,img2"}}
            ]
        }"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert!(msg.has_images());
        assert_eq!(msg.image_urls().len(), 2);
        assert_eq!(msg.content_str(), "Compare these:");
    }

    #[test]
    fn test_chat_message_serialize_string_content() {
        let msg = ChatMessage {
            role: "user".to_string(),
            content: Some("Hello".into()),
            tool_call_id: None,
            tool_calls: None,
            name: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains(r#""content":"Hello""#));
    }

    #[test]
    fn test_content_part_roundtrip() {
        let json = r#"{"type":"text","text":"hi"}"#;
        let part: ContentPart = serde_json::from_str(json).unwrap();
        match &part {
            ContentPart::Text { text } => assert_eq!(text, "hi"),
            _ => panic!("expected text part"),
        }
        let back = serde_json::to_string(&part).unwrap();
        assert!(back.contains(r#""type":"text""#));
        assert!(back.contains(r#""text":"hi""#));
    }

    #[test]
    fn test_content_part_image_url_roundtrip() {
        let json = r#"{"type":"image_url","image_url":{"url":"https://example.com/img.jpg"}}"#;
        let part: ContentPart = serde_json::from_str(json).unwrap();
        match &part {
            ContentPart::ImageUrl { image_url } => {
                assert_eq!(image_url.url, "https://example.com/img.jpg");
                assert_eq!(image_url.detail, "auto"); // default
            }
            _ => panic!("expected image_url part"),
        }
    }
}
