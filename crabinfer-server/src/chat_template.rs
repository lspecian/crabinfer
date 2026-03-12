//! Chat template — delegates to crabinfer-core's canonical implementation.

use crate::types::common::ChatMessage;

/// Convert a list of chat messages into a raw prompt string
/// using the appropriate template for the model architecture.
pub fn apply_chat_template(architecture: &str, messages: &[ChatMessage]) -> String {
    let core_messages: Vec<crabinfer_core::provider::ChatMessage> = messages
        .iter()
        .map(|m| crabinfer_core::provider::ChatMessage {
            role: m.role.clone(),
            content: m.content_str().to_string(),
        })
        .collect();
    crabinfer_core::chat_template::apply_chat_template(architecture, &core_messages)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_messages() -> Vec<ChatMessage> {
        vec![
            ChatMessage {
                role: "system".to_string(),
                content: Some("You are helpful.".to_string()),
                tool_call_id: None,
                tool_calls: None,
                name: None,
            },
            ChatMessage {
                role: "user".to_string(),
                content: Some("Hello".to_string()),
                tool_call_id: None,
                tool_calls: None,
                name: None,
            },
        ]
    }

    #[test]
    fn test_chatml() {
        let result = apply_chat_template("qwen2", &sample_messages());
        assert!(result.contains("<|im_start|>system\nYou are helpful.<|im_end|>"));
        assert!(result.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_phi3() {
        let result = apply_chat_template("phi3", &sample_messages());
        assert!(result.contains("<|system|>\nYou are helpful.<|end|>"));
        assert!(result.contains("<|user|>\nHello<|end|>"));
        assert!(result.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn test_llama3() {
        let result = apply_chat_template("llama", &sample_messages());
        assert!(result.starts_with("<|begin_of_text|>"));
        assert!(result.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(result.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(result.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
    }

    #[test]
    fn test_gemma() {
        let result = apply_chat_template("gemma3", &sample_messages());
        assert!(result.contains("<start_of_turn>user\nYou are helpful.<end_of_turn>"));
        assert!(result.contains("<start_of_turn>user\nHello<end_of_turn>"));
        assert!(result.ends_with("<start_of_turn>model\n"));
    }

    #[test]
    fn test_fallback() {
        let result = apply_chat_template("unknown_arch", &sample_messages());
        assert!(result.contains("system: You are helpful."));
        assert!(result.contains("user: Hello"));
        assert!(result.ends_with("assistant: "));
    }
}
