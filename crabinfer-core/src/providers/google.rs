//! Google Gemini API provider.

use crate::provider::{
    CompletionRequest, CompletionResponse, ModelDescriptor, Provider, ProviderConfig,
};
use crate::providers::http_utils::{build_client, SseReader};
use crate::{CrabInferError, TokenOutput};
use reqwest::blocking::Client;
use serde::Deserialize;

pub struct GoogleProvider {
    client: Client,
    api_key: String,
    base_url: String,
    default_model: String,
}

impl GoogleProvider {
    pub fn new(config: ProviderConfig) -> Result<Self, CrabInferError> {
        let client = build_client(config.timeout_seconds)?;
        let base_url = if config.base_url.is_empty() {
            "https://generativelanguage.googleapis.com".to_string()
        } else {
            config.base_url.trim_end_matches('/').to_string()
        };
        let api_key = crate::credentials::resolve_api_key("google", &config.api_key)
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

    fn build_contents(&self, request: &CompletionRequest) -> Vec<serde_json::Value> {
        let mut contents = Vec::new();

        // Gemini has no system role — prepend system prompt to first user message
        let system_text = if !request.system_prompt.is_empty() {
            Some(request.system_prompt.clone())
        } else {
            request
                .messages
                .iter()
                .find(|m| m.role == "system")
                .map(|m| m.content.clone())
        };

        let mut prepended = false;
        for msg in &request.messages {
            if msg.role == "system" {
                continue; // Handled via prepend
            }
            let role = match msg.role.as_str() {
                "assistant" => "model",
                other => other,
            };
            let text = if !prepended && role == "user" {
                if let Some(ref sys) = system_text {
                    prepended = true;
                    format!("{}\n\n{}", sys, msg.content)
                } else {
                    msg.content.clone()
                }
            } else {
                msg.content.clone()
            };
            contents.push(serde_json::json!({
                "role": role,
                "parts": [{"text": text}]
            }));
        }
        contents
    }
}

impl Provider for GoogleProvider {
    fn name(&self) -> &str {
        "google"
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
            "contents": self.build_contents(request),
            "generationConfig": {
                "temperature": request.temperature,
                "topP": request.top_p,
                "maxOutputTokens": request.max_tokens,
            }
        });

        let url = format!(
            "{}/v1/models/{}:generateContent",
            self.base_url, model
        );

        let resp = self
            .client
            .post(&url)
            .header("x-goog-api-key", effective_key)
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
                401 | 403 => Err(CrabInferError::AuthenticationFailed {
                    provider: "google".to_string(),
                }),
                429 => Err(CrabInferError::RateLimited {
                    provider: "google".to_string(),
                    retry_after_seconds: 10,
                }),
                _ => Err(CrabInferError::ApiError {
                    provider: "google".to_string(),
                    status_code: status as u32,
                    message: body_text,
                }),
            };
        }

        let data: GeminiResponse = resp.json().map_err(|e| CrabInferError::ApiError {
            provider: "google".to_string(),
            status_code: 0,
            message: format!("failed to parse response: {}", e),
        })?;

        let content = data
            .candidates
            .first()
            .and_then(|c| c.content.parts.first())
            .map(|p| p.text.clone())
            .unwrap_or_default();

        let usage = data.usage_metadata.unwrap_or_default();

        Ok(CompletionResponse {
            content,
            model,
            provider_name: "google".to_string(),
            stop_reason: data
                .candidates
                .first()
                .and_then(|c| c.finish_reason.clone())
                .unwrap_or_else(|| "STOP".to_string()),
            input_tokens: usage.prompt_token_count,
            output_tokens: usage.candidates_token_count,
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
            "contents": self.build_contents(request),
            "generationConfig": {
                "temperature": request.temperature,
                "topP": request.top_p,
                "maxOutputTokens": request.max_tokens,
            }
        });

        let url = format!(
            "{}/v1/models/{}:streamGenerateContent?alt=sse",
            self.base_url, model
        );

        let resp = self
            .client
            .post(&url)
            .header("x-goog-api-key", effective_key)
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
                provider: "google".to_string(),
                status_code: status as u32,
                message: body_text,
            });
        }

        let sse_reader = SseReader::new(resp);

        Ok(Box::new(GeminiStreamIterator {
            sse_reader,
            done: false,
        }))
    }

    fn available_models(&self) -> Result<Vec<ModelDescriptor>, CrabInferError> {
        Ok(vec![
            model_desc("gemini-2.5-pro", "Gemini 2.5 Pro", 1048576),
            model_desc("gemini-2.5-flash", "Gemini 2.5 Flash", 1048576),
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
        provider: "google".to_string(),
        is_local: false,
        context_length: ctx,
    }
}

// --- Stream iterator ---

struct GeminiStreamIterator {
    sse_reader: SseReader<reqwest::blocking::Response>,
    done: bool,
}

impl Iterator for GeminiStreamIterator {
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

            let chunk: GeminiResponse = match serde_json::from_str(&event.data) {
                Ok(c) => c,
                Err(_) => continue,
            };

            if let Some(candidate) = chunk.candidates.first() {
                if let Some(part) = candidate.content.parts.first() {
                    if !part.text.is_empty() {
                        let is_done = candidate.finish_reason.is_some();
                        if is_done {
                            self.done = true;
                        }
                        return Some(Ok(TokenOutput {
                            text: part.text.clone(),
                            token_id: 0,
                            probability: 0.0,
                            is_end_of_sequence: is_done,
                        }));
                    }
                }
                if candidate.finish_reason.is_some() {
                    self.done = true;
                    return None;
                }
            }
        }
    }
}

// --- Response types ---

#[derive(Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsage>,
}

#[derive(Deserialize)]
struct GeminiCandidate {
    content: GeminiContent,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Deserialize)]
struct GeminiPart {
    text: String,
}

#[derive(Deserialize, Default)]
struct GeminiUsage {
    #[serde(rename = "promptTokenCount", default)]
    prompt_token_count: u32,
    #[serde(rename = "candidatesTokenCount", default)]
    candidates_token_count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gemini_response() {
        let json = r#"{
            "candidates": [{
                "content": {"parts": [{"text": "Hello!"}], "role": "model"},
                "finishReason": "STOP",
                "index": 0
            }],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}
        }"#;
        let resp: GeminiResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.candidates[0].content.parts[0].text, "Hello!");
        assert_eq!(resp.usage_metadata.unwrap().prompt_token_count, 10);
    }
}
