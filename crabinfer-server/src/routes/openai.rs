use crate::chat_template::apply_chat_template;
use crate::error::ServerError;
use crate::state::AppState;
use crate::types::openai::*;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::Json;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_stream::wrappers::ReceiverStream;

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

    // Acquire inference lock
    let _lock = state.inference_lock.lock().await;

    let architecture = &state.model_info.architecture;
    let prompt = apply_chat_template(architecture, &req.messages);
    let max_tokens = req.max_tokens.unwrap_or(256);
    let temperature = req.temperature.unwrap_or(0.7);

    // Run inference on blocking thread
    let engine = Arc::clone(&state.engine);
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

/// Streaming variant of chat completions
async fn chat_completions_stream(
    state: Arc<AppState>,
    req: ChatCompletionRequest,
) -> Result<
    Sse<axum::response::sse::KeepAliveStream<ReceiverStream<Result<Event, std::convert::Infallible>>>>,
    ServerError,
> {
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

    tokio::spawn(async move {
        // Acquire inference lock inside the spawned task
        let _lock = state.inference_lock.lock().await;

        let engine = Arc::clone(&state.engine);
        let prompt_clone = prompt.clone();

        // Reset engine and start streaming
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
                            break; // Client disconnected
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

        // Send [DONE]
        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
    });

    Ok(Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::default()))
}

/// Approximate token count from messages (text.len() / 4)
fn estimate_tokens(messages: &[crate::types::common::ChatMessage]) -> u32 {
    let total_chars: usize = messages.iter().map(|m| m.content.len() + m.role.len()).sum();
    (total_chars / 4).max(1) as u32
}

use axum::response::IntoResponse;
