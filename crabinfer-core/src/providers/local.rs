//! Local inference provider wrapping CrabInferEngine.

use crate::chat_template::apply_chat_template;
use crate::engine::CrabInferEngine;
use crate::provider::{
    ChatMessage, CompletionRequest, CompletionResponse, ModelDescriptor, Provider,
};
use crate::{CrabInferError, EngineConfig, TokenOutput};
use std::sync::Arc;

/// Provider that runs inference locally using the CrabInfer engine.
pub struct LocalProvider {
    engine: Arc<CrabInferEngine>,
}

impl LocalProvider {
    pub fn new(config: EngineConfig) -> Result<Self, CrabInferError> {
        let engine = CrabInferEngine::new(config)?;
        Ok(Self {
            engine: Arc::new(engine),
        })
    }
}

impl Provider for LocalProvider {
    fn name(&self) -> &str {
        "local"
    }

    fn complete(
        &self,
        request: &CompletionRequest,
    ) -> Result<CompletionResponse, CrabInferError> {
        let info = self.engine.model_info()?;

        // Build messages with system prompt prepended if set
        let messages = prepend_system_prompt(&request.system_prompt, &request.messages);
        let prompt = apply_chat_template(&info.architecture, &messages);

        self.engine.reset();
        let text = self
            .engine
            .complete(prompt, request.max_tokens, request.temperature)?;

        let input_tokens = estimate_tokens(&messages);
        let output_tokens = (text.len() / 4).max(1) as u32;

        Ok(CompletionResponse {
            content: text,
            model: info.model_name,
            provider_name: "local".to_string(),
            stop_reason: "end_turn".to_string(),
            input_tokens,
            output_tokens,
            routing_info: String::new(),
        })
    }

    fn stream(
        &self,
        request: &CompletionRequest,
    ) -> Result<
        Box<dyn Iterator<Item = Result<TokenOutput, CrabInferError>> + Send>,
        CrabInferError,
    > {
        let info = self.engine.model_info()?;

        let messages = prepend_system_prompt(&request.system_prompt, &request.messages);
        let prompt = apply_chat_template(&info.architecture, &messages);
        let max_tokens = request.max_tokens;

        let engine = Arc::clone(&self.engine);
        engine.reset();

        Ok(Box::new(LocalStreamIterator {
            engine,
            prompt,
            max_tokens,
            tokens_generated: 0,
            done: false,
        }))
    }

    fn available_models(&self) -> Result<Vec<ModelDescriptor>, CrabInferError> {
        match self.engine.model_info() {
            Ok(info) => Ok(vec![ModelDescriptor {
                id: info.model_name.clone(),
                name: info.model_name,
                provider: "local".to_string(),
                is_local: true,
                context_length: info.context_length,
            }]),
            Err(_) => Ok(vec![]),
        }
    }

    fn is_available(&self) -> bool {
        self.engine.is_model_loaded()
    }
}

/// Iterator that streams tokens from the local engine.
struct LocalStreamIterator {
    engine: Arc<CrabInferEngine>,
    prompt: String,
    max_tokens: u32,
    tokens_generated: u32,
    done: bool,
}

impl Iterator for LocalStreamIterator {
    type Item = Result<TokenOutput, CrabInferError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.tokens_generated >= self.max_tokens {
            return None;
        }

        match self.engine.next_token(self.prompt.clone()) {
            Ok(Some(token)) => {
                self.tokens_generated += 1;
                if token.is_end_of_sequence {
                    self.done = true;
                }
                Some(Ok(token))
            }
            Ok(None) => {
                self.done = true;
                None
            }
            Err(e) => {
                self.done = true;
                Some(Err(e))
            }
        }
    }
}

/// Prepend system prompt as a system message if non-empty.
fn prepend_system_prompt(system_prompt: &str, messages: &[ChatMessage]) -> Vec<ChatMessage> {
    if system_prompt.is_empty() {
        return messages.to_vec();
    }
    let mut result = vec![ChatMessage {
        role: "system".to_string(),
        content: system_prompt.to_string(),
    }];
    result.extend(messages.iter().cloned());
    result
}

/// Approximate token count from messages (text.len() / 4).
fn estimate_tokens(messages: &[ChatMessage]) -> u32 {
    let total_chars: usize = messages.iter().map(|m| m.content.len() + m.role.len()).sum();
    (total_chars / 4).max(1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prepend_system_prompt_empty() {
        let msgs = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hi".to_string(),
        }];
        let result = prepend_system_prompt("", &msgs);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].role, "user");
    }

    #[test]
    fn test_prepend_system_prompt() {
        let msgs = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hi".to_string(),
        }];
        let result = prepend_system_prompt("Be helpful", &msgs);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].role, "system");
        assert_eq!(result[0].content, "Be helpful");
        assert_eq!(result[1].role, "user");
    }

    #[test]
    fn test_estimate_tokens() {
        let msgs = vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello, how are you doing today?".to_string(),
        }];
        let tokens = estimate_tokens(&msgs);
        assert!(tokens > 0);
    }

    #[test]
    fn test_local_provider_no_model() {
        // Engine with no model path — should create but fail on complete
        let config = EngineConfig {
            model_path: String::new(),
            max_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
            context_length: 2048,
            use_metal: false,
            memory_limit_bytes: 2 * 1024 * 1024 * 1024,
            metallib_path: String::new(),
        };
        let provider = LocalProvider::new(config).unwrap();
        assert_eq!(provider.name(), "local");
        assert!(!provider.is_available()); // No model loaded
    }
}
