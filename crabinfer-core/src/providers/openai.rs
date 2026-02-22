//! OpenAI-compatible provider.
//!
//! Supports any OpenAI API-compatible endpoint (OpenAI, Azure OpenAI, local servers).

use crate::provider::{
    CompletionRequest, CompletionResponse, ModelDescriptor, Provider, ProviderConfig,
};
use crate::providers::http_utils::{build_client, SseReader};
use crate::{CrabInferError, TokenOutput};
use reqwest::blocking::Client;
use serde::Deserialize;

pub struct OpenAIProvider {
    client: Client,
    api_key: String,
    base_url: String,
    default_model: String,
}

impl OpenAIProvider {
    pub fn new(config: ProviderConfig) -> Result<Self, CrabInferError> {
        let client = build_client(config.timeout_seconds)?;
        let base_url = if config.base_url.is_empty() {
            "https://api.openai.com".to_string()
        } else {
            config.base_url.trim_end_matches('/').to_string()
        };
        let api_key = crate::credentials::resolve_api_key("openai", &config.api_key)
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

    fn build_messages(
        &self,
        request: &CompletionRequest,
    ) -> Vec<serde_json::Value> {
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
}

impl Provider for OpenAIProvider {
    fn name(&self) -> &str {
        "openai"
    }

    fn complete(
        &self,
        request: &CompletionRequest,
    ) -> Result<CompletionResponse, CrabInferError> {
        let model = self.resolve_model(&request.model);
        let effective_key = if request.api_key_override.is_empty() {
            &self.api_key
        } else {
            &request.api_key_override
        };
        let body = serde_json::json!({
            "model": model,
            "messages": self.build_messages(request),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": false,
        });

        let resp = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", effective_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| CrabInferError::NetworkError {
                reason: e.to_string(),
            })?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body_text = resp.text().unwrap_or_default();
            return match status {
                401 => Err(CrabInferError::AuthenticationFailed {
                    provider: "openai".to_string(),
                }),
                429 => Err(CrabInferError::RateLimited {
                    provider: "openai".to_string(),
                    retry_after_seconds: 10,
                }),
                _ => Err(CrabInferError::ApiError {
                    provider: "openai".to_string(),
                    status_code: status as u32,
                    message: body_text,
                }),
            };
        }

        let data: OpenAIResponse = resp.json().map_err(|e| CrabInferError::ApiError {
            provider: "openai".to_string(),
            status_code: 0,
            message: format!("failed to parse response: {}", e),
        })?;

        let choice = data.choices.first().ok_or(CrabInferError::ApiError {
            provider: "openai".to_string(),
            status_code: 0,
            message: "no choices in response".to_string(),
        })?;

        Ok(CompletionResponse {
            content: choice.message.content.clone(),
            model: data.model,
            provider_name: "openai".to_string(),
            stop_reason: choice
                .finish_reason
                .clone()
                .unwrap_or_else(|| "end_turn".to_string()),
            input_tokens: data.usage.prompt_tokens,
            output_tokens: data.usage.completion_tokens,
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
        let model = self.resolve_model(&request.model);
        let effective_key = if request.api_key_override.is_empty() {
            &self.api_key
        } else {
            &request.api_key_override
        };
        let body = serde_json::json!({
            "model": model,
            "messages": self.build_messages(request),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": true,
        });

        let resp = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", effective_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| CrabInferError::NetworkError {
                reason: e.to_string(),
            })?;

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body_text = resp.text().unwrap_or_default();
            return Err(CrabInferError::ApiError {
                provider: "openai".to_string(),
                status_code: status as u32,
                message: body_text,
            });
        }

        let sse_reader = SseReader::new(resp);

        Ok(Box::new(OpenAIStreamIterator {
            sse_reader,
            done: false,
        }))
    }

    fn available_models(&self) -> Result<Vec<ModelDescriptor>, CrabInferError> {
        // Return common models without making an API call
        Ok(vec![
            model_desc("gpt-4o", "GPT-4o", 128000),
            model_desc("gpt-4o-mini", "GPT-4o Mini", 128000),
            model_desc("o1", "o1", 200000),
            model_desc("o3-mini", "o3 Mini", 200000),
        ])
    }

    fn is_available(&self) -> bool {
        !self.api_key.is_empty()
    }
}

fn model_desc(id: &str, name: &str, ctx: u32) -> ModelDescriptor {
    ModelDescriptor {
        id: id.to_string(),
        name: name.to_string(),
        provider: "openai".to_string(),
        is_local: false,
        context_length: ctx,
    }
}

// --- SSE stream iterator ---

struct OpenAIStreamIterator {
    sse_reader: SseReader<reqwest::blocking::Response>,
    done: bool,
}

impl Iterator for OpenAIStreamIterator {
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
                Err(_) => continue, // Skip unparseable chunks
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

// --- Response types ---

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_openai_response() {
        let json = r#"{
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }"#;
        let resp: OpenAIResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.model, "gpt-4o");
        assert_eq!(resp.choices[0].message.content, "Hello!");
        assert_eq!(resp.usage.prompt_tokens, 10);
    }

    #[test]
    fn test_parse_openai_chunk() {
        let json = r#"{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1234,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}"#;
        let chunk: OpenAIChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Hi"));
    }
}
