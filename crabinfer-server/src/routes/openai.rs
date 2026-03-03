use crate::chat_template::apply_chat_template;
use crate::error::ServerError;
use crate::state::AppState;
use crate::types::openai::*;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio_stream::wrappers::ReceiverStream;

/// Hard cap on max_tokens to prevent runaway generation.
const MAX_TOKENS_CAP: usize = 16384;

/// Per-request timeout for token generation (seconds).
const REQUEST_TIMEOUT_SECS: u64 = 120;

/// GET /v1/models
pub async fn list_models(
    State(state): State<Arc<AppState>>,
) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelObject {
            id: state.model_id.clone(),
            object: "model".to_string(),
            created: state.created_at,
            owned_by: "crabinfer".to_string(),
        }],
    })
}

/// POST /v1/chat/completions
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, ServerError> {
    state.metrics.inc_request();

    if req.stream == Some(true) {
        return Ok(chat_completions_stream(state, req).await?.into_response());
    }

    // ── Serving engine path (continuous batching) ──
    if let Some(ref engine) = state.serving_engine {
        return serving_chat_completions(&state, engine, &req).await;
    }

    // ── Legacy engine path ──
    let engine = state
        .engine
        .as_ref()
        .ok_or_else(|| ServerError::internal("no engine available"))?;

    let _lock = state.inference_lock.lock().await;

    let architecture = &state.model_info.architecture;
    let prompt = apply_chat_template(architecture, &req.messages);
    let max_tokens = req.max_tokens.unwrap_or(256);
    let temperature = req.temperature.unwrap_or(0.7);

    let engine = Arc::clone(engine);
    let result = tokio::task::spawn_blocking(move || {
        engine.reset();
        engine.complete(prompt, max_tokens, temperature)
    })
    .await
    .map_err(|e| {
        state.metrics.inc_error();
        ServerError::internal(format!("task join error: {}", e))
    })?
    .map_err(|e| {
        state.metrics.inc_error();
        ServerError::from(e)
    })?;

    let prompt_tokens = estimate_tokens(&req.messages);
    let completion_tokens = (result.len() / 4) as u32;
    state.metrics.inc_success();
    state
        .metrics
        .add_tokens(prompt_tokens as u64, completion_tokens as u64);

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", now),
        object: "chat.completion".to_string(),
        created: now,
        model: state.model_id.clone(),
        choices: vec![Choice {
            index: 0,
            message: ChoiceMessage {
                role: "assistant".to_string(),
                content: result,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    };

    Ok(Json(response).into_response())
}

/// Non-streaming chat completions via the serving engine.
async fn serving_chat_completions(
    state: &AppState,
    engine: &crabinfer_core::serving::engine_loop::EngineHandle,
    req: &ChatCompletionRequest,
) -> Result<axum::response::Response, ServerError> {
    use crabinfer_core::serving::sequence::SamplingParams;

    let architecture = &state.model_info.architecture;
    let prompt = apply_chat_template(architecture, &req.messages);
    tracing::debug!("Chat prompt ({} chars): {:?}", prompt.len(), &prompt[..prompt.len().min(200)]);

    let prompt_tokens = engine
        .encode(&prompt)
        .map_err(|e| ServerError::internal(format!("tokenization failed: {e}")))?;
    let prompt_token_count = prompt_tokens.len() as u32;
    tracing::debug!("Encoded {} tokens", prompt_token_count);

    let max_tokens = (req.max_tokens.unwrap_or(256) as usize).min(MAX_TOKENS_CAP);
    let temperature = req.temperature.unwrap_or(0.7);
    let top_p = req.top_p.unwrap_or(0.9);

    let params = SamplingParams {
        temperature,
        top_p,
        max_tokens,
        ..SamplingParams::default()
    };

    let mut rx = engine.submit(prompt_tokens, params).map_err(|e| {
        use crabinfer_core::serving::engine_loop::EngineError;
        match e {
            EngineError::Overloaded => ServerError::too_many_requests("server is overloaded, try again later"),
            _ => ServerError::internal(format!("engine error: {e}")),
        }
    })?;

    // Collect all generated tokens with timeout
    let mut generated_ids: Vec<u32> = Vec::new();
    let mut finish_reason_str = "stop".to_string();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(REQUEST_TIMEOUT_SECS);

    loop {
        match tokio::time::timeout_at(deadline, rx.recv()).await {
            Ok(Some(tok)) => {
                generated_ids.push(tok.token_id);
                if let Some(reason) = tok.finish_reason {
                    finish_reason_str = finish_reason_to_openai(reason).to_string();
                    break;
                }
            }
            Ok(None) => break, // Channel closed
            Err(_) => {
                // Timeout — drop rx to signal cancellation to engine
                finish_reason_str = "length".to_string();
                tracing::warn!(
                    "Request timed out after {}s ({} tokens generated)",
                    REQUEST_TIMEOUT_SECS,
                    generated_ids.len(),
                );
                break;
            }
        }
    }

    let completion_text = engine.decode(&generated_ids).unwrap_or_default();
    let completion_tokens = generated_ids.len() as u32;

    state.metrics.inc_success();
    state
        .metrics
        .add_tokens(prompt_token_count as u64, completion_tokens as u64);

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", now),
        object: "chat.completion".to_string(),
        created: now,
        model: state.model_id.clone(),
        choices: vec![Choice {
            index: 0,
            message: ChoiceMessage {
                role: "assistant".to_string(),
                content: completion_text,
            },
            finish_reason: finish_reason_str,
        }],
        usage: Usage {
            prompt_tokens: prompt_token_count,
            completion_tokens,
            total_tokens: prompt_token_count + completion_tokens,
        },
    };

    Ok(Json(response).into_response())
}

/// Streaming variant of chat completions
async fn chat_completions_stream(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
) -> Result<
    Sse<axum::response::sse::KeepAliveStream<ReceiverStream<Result<Event, std::convert::Infallible>>>>,
    ServerError,
> {
    // ── Serving engine path ──
    if let Some(ref engine) = state.serving_engine {
        return serving_chat_completions_stream(state.clone(), engine.clone(), req).await;
    }

    // ── Legacy engine path ──
    let engine = state
        .engine
        .as_ref()
        .ok_or_else(|| ServerError::internal("no engine available"))?;

    let architecture = state.model_info.architecture.clone();
    let prompt = apply_chat_template(&architecture, &req.messages);
    let max_tokens = req.max_tokens.unwrap_or(256);
    let model_id = state.model_id.clone();

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let chunk_id = format!("chatcmpl-{}", now);

    let (tx, rx) = tokio::sync::mpsc::channel(32);
    let engine = Arc::clone(engine);

    tokio::spawn(async move {
        let _lock = state.inference_lock.lock().await;

        let prompt_clone = prompt.clone();

        let engine_reset = Arc::clone(&engine);
        let _ = tokio::task::spawn_blocking(move || engine_reset.reset()).await;

        // Send initial chunk with role
        let initial_chunk = ChatCompletionChunk {
            id: chunk_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: now,
            model: model_id.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
        };
        if let Ok(json) = serde_json::to_string(&initial_chunk) {
            let _ = tx.send(Ok(Event::default().data(json))).await;
        }

        // Stream tokens
        let mut tokens_generated = 0u32;
        loop {
            if tokens_generated >= max_tokens {
                break;
            }

            let engine_tok = Arc::clone(&engine);
            let prompt_for_tok = prompt_clone.clone();

            let token_result = tokio::task::spawn_blocking(move || {
                engine_tok.next_token(prompt_for_tok)
            })
            .await;

            match token_result {
                Ok(Ok(Some(tok))) => {
                    if tok.is_end_of_sequence {
                        break;
                    }
                    tokens_generated += 1;

                    let chunk = ChatCompletionChunk {
                        id: chunk_id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created: now,
                        model: model_id.clone(),
                        choices: vec![ChunkChoice {
                            index: 0,
                            delta: Delta {
                                role: None,
                                content: Some(tok.text),
                            },
                            finish_reason: None,
                        }],
                    };
                    if let Ok(json) = serde_json::to_string(&chunk) {
                        if tx.send(Ok(Event::default().data(json))).await.is_err() {
                            break;
                        }
                    }
                }
                Ok(Ok(None)) => break,
                Ok(Err(_)) | Err(_) => break,
            }
        }

        // Send final chunk with finish_reason
        let final_chunk = ChatCompletionChunk {
            id: chunk_id,
            object: "chat.completion.chunk".to_string(),
            created: now,
            model: model_id,
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
        };
        if let Ok(json) = serde_json::to_string(&final_chunk) {
            let _ = tx.send(Ok(Event::default().data(json))).await;
        }

        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
    });

    Ok(Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::default()))
}

/// Streaming chat completions via the serving engine.
async fn serving_chat_completions_stream(
    state: Arc<AppState>,
    engine: crabinfer_core::serving::engine_loop::EngineHandle,
    req: ChatCompletionRequest,
) -> Result<
    Sse<axum::response::sse::KeepAliveStream<ReceiverStream<Result<Event, std::convert::Infallible>>>>,
    ServerError,
> {
    use crabinfer_core::serving::sequence::SamplingParams;

    let architecture = state.model_info.architecture.clone();
    let prompt = apply_chat_template(&architecture, &req.messages);
    let model_id = state.model_id.clone();

    let prompt_tokens = engine
        .encode(&prompt)
        .map_err(|e| ServerError::internal(format!("tokenization failed: {e}")))?;

    let max_tokens = (req.max_tokens.unwrap_or(256) as usize).min(MAX_TOKENS_CAP);
    let temperature = req.temperature.unwrap_or(0.7);
    let top_p = req.top_p.unwrap_or(0.9);

    let params = SamplingParams {
        temperature,
        top_p,
        max_tokens,
        ..SamplingParams::default()
    };

    let mut rx = engine.submit(prompt_tokens, params).map_err(|e| {
        use crabinfer_core::serving::engine_loop::EngineError;
        match e {
            EngineError::Overloaded => ServerError::too_many_requests("server is overloaded, try again later"),
            _ => ServerError::internal(format!("engine error: {e}")),
        }
    })?;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let chunk_id = format!("chatcmpl-{}", now);

    let (tx, sse_rx) = tokio::sync::mpsc::channel(32);

    tokio::spawn(async move {
        // Send initial chunk with role
        let initial_chunk = ChatCompletionChunk {
            id: chunk_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: now,
            model: model_id.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
        };
        if let Ok(json) = serde_json::to_string(&initial_chunk) {
            let _ = tx.send(Ok(Event::default().data(json))).await;
        }

        // Stream tokens with timeout
        let deadline = tokio::time::Instant::now() + Duration::from_secs(REQUEST_TIMEOUT_SECS);
        let mut finish_reason_str = None;
        loop {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(tok)) => {
                    let text = engine.decode(&[tok.token_id]).unwrap_or_default();
                    let is_done = tok.finish_reason.is_some();
                    if let Some(reason) = tok.finish_reason {
                        finish_reason_str = Some(finish_reason_to_openai(reason).to_string());
                    }

                    if !text.is_empty() {
                        let chunk = ChatCompletionChunk {
                            id: chunk_id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created: now,
                            model: model_id.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: Delta {
                                    role: None,
                                    content: Some(text),
                                },
                                finish_reason: None,
                            }],
                        };
                        if let Ok(json) = serde_json::to_string(&chunk) {
                            if tx.send(Ok(Event::default().data(json))).await.is_err() {
                                break;
                            }
                        }
                    }

                    if is_done {
                        break;
                    }
                }
                Ok(None) => break, // Channel closed
                Err(_) => {
                    tracing::warn!("Streaming request timed out after {REQUEST_TIMEOUT_SECS}s");
                    finish_reason_str = Some("length".to_string());
                    break;
                }
            }
        }

        // Send final chunk with finish_reason
        let final_chunk = ChatCompletionChunk {
            id: chunk_id,
            object: "chat.completion.chunk".to_string(),
            created: now,
            model: model_id,
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: None,
                },
                finish_reason: Some(finish_reason_str.unwrap_or_else(|| "stop".to_string())),
            }],
        };
        if let Ok(json) = serde_json::to_string(&final_chunk) {
            let _ = tx.send(Ok(Event::default().data(json))).await;
        }

        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
    });

    Ok(Sse::new(ReceiverStream::new(sse_rx)).keep_alive(KeepAlive::default()))
}

/// Map FinishReason to OpenAI finish_reason string.
fn finish_reason_to_openai(reason: crabinfer_core::serving::sequence::FinishReason) -> &'static str {
    use crabinfer_core::serving::sequence::FinishReason;
    match reason {
        FinishReason::EndOfSequence => "stop",
        FinishReason::MaxTokens => "length",
        FinishReason::Stop => "stop",
        FinishReason::Cancelled => "stop",
        FinishReason::Preempted => "stop",
    }
}

/// Approximate token count from messages (text.len() / 4)
fn estimate_tokens(messages: &[crate::types::common::ChatMessage]) -> u32 {
    let total_chars: usize = messages.iter().map(|m| m.content.len() + m.role.len()).sum();
    (total_chars / 4).max(1) as u32
}
