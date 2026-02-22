//! Ollama provider for local Ollama server.

use crate::provider::{
    CompletionRequest, CompletionResponse, ModelDescriptor, Provider, ProviderConfig,
};
use crate::providers::http_utils::{build_client, JsonLineReader};
use crate::{CrabInferError, TokenOutput};
use reqwest::blocking::Client;
use serde::Deserialize;

pub struct OllamaProvider {
    client: Client,
    base_url: String,
    default_model: String,
}

impl OllamaProvider {
    pub fn new(config: ProviderConfig) -> Result<Self, CrabInferError> {
        let client = build_client(config.timeout_seconds)?;
        let base_url = if config.base_url.is_empty() {
            "http://localhost:11434".to_string()
        } else {
            config.base_url.trim_end_matches('/').to_string()
        };
        Ok(Self {
            client,
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

impl Provider for OllamaProvider {
    fn name(&self) -> &str {
        "ollama"
    }

    fn complete(
        &self,
        request: &CompletionRequest,
    ) -> Result<CompletionResponse, CrabInferError> {
        let model = self.resolve_model(&request.model);

        let mut messages: Vec<serde_json::Value> = Vec::new();
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

        let body = serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": false,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
                "top_p": request.top_p,
            }
        });

        let resp = self
            .client
            .post(format!("{}/api/chat", self.base_url))
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
                provider: "ollama".to_string(),
                status_code: status as u32,
                message: body_text,
            });
        }

        let data: OllamaChatResponse =
            resp.json().map_err(|e| CrabInferError::ApiError {
                provider: "ollama".to_string(),
                status_code: 0,
                message: format!("failed to parse response: {}", e),
            })?;

        let output_tokens = data.eval_count.unwrap_or(0);
        let input_tokens = data.prompt_eval_count.unwrap_or(0);

        Ok(CompletionResponse {
            content: data.message.content,
            model: data.model,
            provider_name: "ollama".to_string(),
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
        let model = self.resolve_model(&request.model);

        let mut messages: Vec<serde_json::Value> = Vec::new();
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

        let body = serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": true,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
                "top_p": request.top_p,
            }
        });

        let resp = self
            .client
            .post(format!("{}/api/chat", self.base_url))
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
                provider: "ollama".to_string(),
                status_code: status as u32,
                message: body_text,
            });
        }

        let line_reader = JsonLineReader::new(resp);

        Ok(Box::new(OllamaStreamIterator {
            line_reader,
            done: false,
        }))
    }

    fn available_models(&self) -> Result<Vec<ModelDescriptor>, CrabInferError> {
        let resp = self
            .client
            .get(format!("{}/api/tags", self.base_url))
            .send()
            .map_err(|e| CrabInferError::NetworkError {
                reason: e.to_string(),
            })?;

        if !resp.status().is_success() {
            return Err(CrabInferError::ProviderNotAvailable {
                provider: "ollama".to_string(),
            });
        }

        let data: OllamaTagsResponse =
            resp.json().map_err(|e| CrabInferError::ApiError {
                provider: "ollama".to_string(),
                status_code: 0,
                message: format!("failed to parse tags: {}", e),
            })?;

        Ok(data
            .models
            .into_iter()
            .map(|m| ModelDescriptor {
                id: m.name.clone(),
                name: m.name,
                provider: "ollama".to_string(),
                is_local: false, // Ollama is a separate process, not embedded
                context_length: 0,
            })
            .collect())
    }

    fn is_available(&self) -> bool {
        self.client
            .get(format!("{}/api/tags", self.base_url))
            .send()
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }
}

// --- Stream iterator ---

struct OllamaStreamIterator {
    line_reader: JsonLineReader<reqwest::blocking::Response>,
    done: bool,
}

impl Iterator for OllamaStreamIterator {
    type Item = Result<TokenOutput, CrabInferError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        loop {
            let line = match self.line_reader.next()? {
                Ok(l) => l,
                Err(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            };

            let chunk: OllamaStreamChunk = match serde_json::from_str(&line) {
                Ok(c) => c,
                Err(_) => continue,
            };

            if chunk.done {
                self.done = true;
                return Some(Ok(TokenOutput {
                    text: chunk.message.content,
                    token_id: 0,
                    probability: 0.0,
                    is_end_of_sequence: true,
                }));
            }

            if !chunk.message.content.is_empty() {
                return Some(Ok(TokenOutput {
                    text: chunk.message.content,
                    token_id: 0,
                    probability: 0.0,
                    is_end_of_sequence: false,
                }));
            }
        }
    }
}

// --- Response types ---

#[derive(Deserialize)]
struct OllamaChatResponse {
    model: String,
    message: OllamaMessage,
    #[serde(default)]
    eval_count: Option<u32>,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
}

#[derive(Deserialize)]
struct OllamaMessage {
    content: String,
}

#[derive(Deserialize)]
struct OllamaStreamChunk {
    message: OllamaMessage,
    done: bool,
}

#[derive(Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaModelInfo>,
}

#[derive(Deserialize)]
struct OllamaModelInfo {
    name: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ollama_response() {
        let json = r#"{
            "model": "llama2",
            "created_at": "2023-08-14T00:00:00Z",
            "message": {"role": "assistant", "content": "Hello!"},
            "done": true,
            "eval_count": 50,
            "prompt_eval_count": 12
        }"#;
        let resp: OllamaChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.model, "llama2");
        assert_eq!(resp.message.content, "Hello!");
        assert_eq!(resp.eval_count, Some(50));
    }

    #[test]
    fn test_parse_ollama_stream_chunk() {
        let json = r#"{"model":"llama2","created_at":"2023-08-14T00:00:00Z","message":{"role":"assistant","content":"Hi"},"done":false}"#;
        let chunk: OllamaStreamChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.message.content, "Hi");
        assert!(!chunk.done);
    }

    #[test]
    fn test_parse_ollama_tags() {
        let json = r#"{
            "models": [
                {"name": "llama2:latest", "model": "llama2", "modified_at": "2023-08-14T00:00:00Z", "size": 3791879987}
            ]
        }"#;
        let tags: OllamaTagsResponse = serde_json::from_str(json).unwrap();
        assert_eq!(tags.models.len(), 1);
        assert_eq!(tags.models[0].name, "llama2:latest");
    }
}
