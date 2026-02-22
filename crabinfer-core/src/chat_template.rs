//! Chat template conversion: messages[] -> raw prompt per model architecture.
//!
//! Used by LocalProvider to convert ChatMessage sequences into architecture-specific
//! prompt strings that the inference engine expects.

use crate::provider::ChatMessage;

/// Convert a list of chat messages into a raw prompt string
/// using the appropriate template for the model architecture.
pub fn apply_chat_template(architecture: &str, messages: &[ChatMessage]) -> String {
    match architecture.to_lowercase().as_str() {
        "qwen2" | "qwen3" => chatml_template(messages),
        "phi3" => phi3_template(messages),
        "llama" => llama3_template(messages),
        "gemma3" | "gemma2" | "gemma" => gemma_template(messages),
        _ => fallback_template(messages),
    }
}

/// ChatML format (Qwen2, Qwen3)
/// <|im_start|>role\ncontent<|im_end|>\n
fn chatml_template(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            msg.role, msg.content
        ));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// Phi-3 format
/// <|system|>\ncontent<|end|>\n<|user|>\ncontent<|end|>\n<|assistant|>\n
fn phi3_template(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let tag = match msg.role.as_str() {
            "system" => "system",
            "user" => "user",
            "assistant" => "assistant",
            other => other,
        };
        prompt.push_str(&format!("<|{}|>\n{}<|end|>\n", tag, msg.content));
    }
    prompt.push_str("<|assistant|>\n");
    prompt
}

/// Llama 3 format
/// <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
fn llama3_template(messages: &[ChatMessage]) -> String {
    let mut prompt = String::from("<|begin_of_text|>");
    for msg in messages {
        prompt.push_str(&format!(
            "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
            msg.role, msg.content
        ));
    }
    prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    prompt
}

/// Gemma format
/// <start_of_turn>role\ncontent<end_of_turn>\n
fn gemma_template(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let role = match msg.role.as_str() {
            "system" => "user", // Gemma has no system role; prepend to user
            "assistant" => "model",
            other => other,
        };
        prompt.push_str(&format!(
            "<start_of_turn>{}\n{}<end_of_turn>\n",
            role, msg.content
        ));
    }
    prompt.push_str("<start_of_turn>model\n");
    prompt
}

/// Fallback: simple "role: content" format
fn fallback_template(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str(&format!("{}: {}\n\n", msg.role, msg.content));
    }
    prompt.push_str("assistant: ");
    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_messages() -> Vec<ChatMessage> {
        vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are helpful.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
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
