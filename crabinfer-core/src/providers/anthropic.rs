//! Anthropic Messages API provider.

use crate::provider::{
    CompletionRequest, CompletionResponse, ModelDescriptor, Provider, ProviderConfig,
};
use crate::providers::http_utils::{build_client, SseReader};
use crate::{CrabInferError, TokenOutput};
use reqwest::blocking::Client;
use serde::Deserialize;

pub struct AnthropicProvider {
    client: Client,
    api_key: String,
    base_url: String,
    default_model: String,
}

impl AnthropicProvider {
    pub fn new(config: ProviderConfig) -> Result<Self, CrabInferError> {
        let client = build_client(config.timeout_seconds)?;
        let base_url = if config.base_url.is_empty() {
            "https://api.anthropic.com".to_string()
        } else {
            config.base_url.trim_end_matches('/').to_string()
        };
        let api_key = crate::credentials::resolve_api_key("anthropic", &config.api_key)
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
}

impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
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

        let messages: Vec<serde_json::Value> = request
            .messages
            .iter()
            .filter(|m| m.role != "system")
            .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
            .collect();

        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "stream": false,
        });
        if request.temperature > 0.0 {
            body["temperature"] = serde_json::json!(request.temperature);
        }

        // System prompt
        let system = if !request.system_prompt.is_empty() {
            request.system_prompt.clone()
        } else {
            request
                .messages
                .iter()
                .find(|m| m.role == "system")
                .map(|m| m.content.clone())
                .unwrap_or_default()
        };
        if !system.is_empty() {
            body["system"] = serde_json::json!(system);
        }

        let resp = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", effective_key)
            .header("anthropic-version", "2023-06-01")
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
                    provider: "anthropic".to_string(),
                }),
                429 => Err(CrabInferError::RateLimited {
                    provider: "anthropic".to_string(),
                    retry_after_seconds: 10,
                }),
                _ => Err(CrabInferError::ApiError {
                    provider: "anthropic".to_string(),
                    status_code: status as u32,
                    message: body_text,
                }),
            };
        }

        let data: AnthropicResponse =
            resp.json().map_err(|e| CrabInferError::ApiError {
                provider: "anthropic".to_string(),
                status_code: 0,
                message: format!("failed to parse response: {}", e),
            })?;

        let content = data
            .content
            .iter()
            .filter_map(|b| {
                if b.block_type == "text" {
                    b.text.as_deref()
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("");

        Ok(CompletionResponse {
            content,
            model: data.model,
            provider_name: "anthropic".to_string(),
            stop_reason: data.stop_reason.unwrap_or_else(|| "end_turn".to_string()),
            input_tokens: data.usage.input_tokens,
            output_tokens: data.usage.output_tokens,
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

        let messages: Vec<serde_json::Value> = request
            .messages
            .iter()
            .filter(|m| m.role != "system")
            .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
            .collect();

        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "stream": true,
        });
        if request.temperature > 0.0 {
            body["temperature"] = serde_json::json!(request.temperature);
        }

        let system = if !request.system_prompt.is_empty() {
            request.system_prompt.clone()
        } else {
            request
                .messages
                .iter()
                .find(|m| m.role == "system")
                .map(|m| m.content.clone())
                .unwrap_or_default()
        };
        if !system.is_empty() {
            body["system"] = serde_json::json!(system);
        }

        let resp = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", effective_key)
            .header("anthropic-version", "2023-06-01")
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
                provider: "anthropic".to_string(),
                status_code: status as u32,
                message: body_text,
            });
        }

        let sse_reader = SseReader::new(resp);

        Ok(Box::new(AnthropicStreamIterator {
            sse_reader,
            done: false,
        }))
    }

    fn available_models(&self) -> Result<Vec<ModelDescriptor>, CrabInferError> {
        Ok(vec![
            model_desc("claude-sonnet-4-20250514", "Claude Sonnet 4", 200000),
            model_desc("claude-haiku-4-5-20251001", "Claude Haiku 4.5", 200000),
            model_desc("claude-opus-4-20250514", "Claude Opus 4", 200000),
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
        provider: "anthropic".to_string(),
        is_local: false,
        context_length: ctx,
    }
}

// --- Stream iterator ---

struct AnthropicStreamIterator {
    sse_reader: SseReader<reqwest::blocking::Response>,
    done: bool,
}

impl Iterator for AnthropicStreamIterator {
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

            let event_type = event.event.as_deref().unwrap_or("");

            match event_type {
                "content_block_delta" => {
                    if let Ok(delta) = serde_json::from_str::<ContentBlockDelta>(&event.data) {
                        if !delta.delta.text.is_empty() {
                            return Some(Ok(TokenOutput {
                                text: delta.delta.text,
                                token_id: 0,
                                probability: 0.0,
                                is_end_of_sequence: false,
                            }));
                        }
                    }
                }
                "message_stop" => {
                    self.done = true;
                    return Some(Ok(TokenOutput {
                        text: String::new(),
                        token_id: 0,
                        probability: 0.0,
                        is_end_of_sequence: true,
                    }));
                }
                _ => continue, // Skip message_start, content_block_start, content_block_stop, message_delta
            }
        }
    }
}

// --- Response types ---

#[derive(Deserialize)]
struct AnthropicResponse {
    model: String,
    content: Vec<ContentBlock>,
    stop_reason: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    text: Option<String>,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Deserialize)]
struct ContentBlockDelta {
    delta: TextDelta,
}

#[derive(Deserialize)]
struct TextDelta {
    #[serde(default)]
    text: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_anthropic_response() {
        let json = r#"{
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }"#;
        let resp: AnthropicResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.model, "claude-sonnet-4-20250514");
        assert_eq!(resp.content[0].text.as_deref(), Some("Hello!"));
    }

    #[test]
    fn test_parse_content_block_delta() {
        let json = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}"#;
        let delta: ContentBlockDelta = serde_json::from_str(json).unwrap();
        assert_eq!(delta.delta.text, "Hi");
    }
}
