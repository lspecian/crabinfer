use crate::chat_template::apply_chat_template;
use crate::error::ServerError;
use crate::state::AppState;
use crate::types::anthropic::*;
use crate::types::common::ChatMessage;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::Json;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_stream::wrappers::ReceiverStream;

/// POST /v1/messages
pub async fn messages(
    State(state): State<Arc<AppState>>,
    Json(req): Json<MessagesRequest>,
) -> Result<axum::response::Response, ServerError> {
    state.metrics.inc_request();

    if req.stream == Some(true) {
        return Ok(messages_stream(state, req).await?.into_response());
    }

    let _lock = state.inference_lock.lock().await;

    // Prepend system message if provided
    let mut messages = Vec::new();
    if let Some(ref system) = req.system {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: system.clone(),
        });
    }
    messages.extend(req.messages.iter().cloned());

    let architecture = &state.model_info.architecture;
    let prompt = apply_chat_template(architecture, &messages);
    let temperature = req.temperature.unwrap_or(0.7);

    let engine = Arc::clone(&state.engine);
    let max_tokens = req.max_tokens;
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

    let input_tokens = estimate_tokens_anthropic(&messages);
    let output_tokens = (result.len() / 4).max(1) as u32;
    state.metrics.inc_success();
    state
        .metrics
        .add_tokens(input_tokens as u64, output_tokens as u64);

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let response = MessagesResponse {
        id: format!("msg_{}", now),
        type_field: "message".to_string(),
        role: "assistant".to_string(),
        content: vec![ContentBlock {
            type_field: "text".to_string(),
            text: result,
        }],
        model: state.model_id.clone(),
        stop_reason: "end_turn".to_string(),
        usage: AnthropicUsage {
            input_tokens,
            output_tokens,
        },
    };

    Ok(Json(response).into_response())
}

/// Streaming variant of messages
async fn messages_stream(
    state: Arc<AppState>,
    req: MessagesRequest,
) -> Result<
    Sse<axum::response::sse::KeepAliveStream<ReceiverStream<Result<Event, std::convert::Infallible>>>>,
    ServerError,
> {
    let architecture = state.model_info.architecture.clone();
    let model_id = state.model_id.clone();

    // Prepend system message if provided
    let mut messages = Vec::new();
    if let Some(ref system) = req.system {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: system.clone(),
        });
    }
    messages.extend(req.messages.iter().cloned());

    let prompt = apply_chat_template(&architecture, &messages);
    let max_tokens = req.max_tokens;
    let input_tokens = estimate_tokens_anthropic(&messages);

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let msg_id = format!("msg_{}", now);

    let (tx, rx) = tokio::sync::mpsc::channel(32);

    tokio::spawn(async move {
        let _lock = state.inference_lock.lock().await;

        let engine = Arc::clone(&state.engine);

        // Reset engine
        let engine_reset = Arc::clone(&engine);
        let _ = tokio::task::spawn_blocking(move || engine_reset.reset()).await;

        // 1. message_start
        let start_event = MessageStartEvent {
            type_field: "message_start".to_string(),
            message: MessageStartBody {
                id: msg_id.clone(),
                type_field: "message".to_string(),
                role: "assistant".to_string(),
                content: vec![],
                model: model_id.clone(),
                stop_reason: None,
                usage: AnthropicUsage {
                    input_tokens,
                    output_tokens: 0,
                },
            },
        };
        if let Ok(json) = serde_json::to_string(&start_event) {
            let _ = tx
                .send(Ok(Event::default()
                    .event("message_start")
                    .data(json)))
                .await;
        }

        // 2. content_block_start
        let block_start = ContentBlockStartEvent {
            type_field: "content_block_start".to_string(),
            index: 0,
            content_block: ContentBlock {
                type_field: "text".to_string(),
                text: String::new(),
            },
        };
        if let Ok(json) = serde_json::to_string(&block_start) {
            let _ = tx
                .send(Ok(Event::default()
                    .event("content_block_start")
                    .data(json)))
                .await;
        }

        // 3. content_block_delta (stream tokens)
        let prompt_clone = prompt.clone();
        let mut output_tokens = 0u32;
        loop {
            if output_tokens >= max_tokens {
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
                    output_tokens += 1;

                    let delta_event = ContentBlockDeltaEvent {
                        type_field: "content_block_delta".to_string(),
                        index: 0,
                        delta: TextDelta {
                            type_field: "text_delta".to_string(),
                            text: tok.text,
                        },
                    };
                    if let Ok(json) = serde_json::to_string(&delta_event) {
                        if tx
                            .send(Ok(Event::default()
                                .event("content_block_delta")
                                .data(json)))
                            .await
                            .is_err()
                        {
                            break; // Client disconnected
                        }
                    }
                }
                Ok(Ok(None)) => break,
                Ok(Err(_)) | Err(_) => break,
            }
        }

        // 4. content_block_stop
        let block_stop = ContentBlockStopEvent {
            type_field: "content_block_stop".to_string(),
            index: 0,
        };
        if let Ok(json) = serde_json::to_string(&block_stop) {
            let _ = tx
                .send(Ok(Event::default()
                    .event("content_block_stop")
                    .data(json)))
                .await;
        }

        // 5. message_delta
        let msg_delta = MessageDeltaEvent {
            type_field: "message_delta".to_string(),
            delta: MessageDelta {
                stop_reason: "end_turn".to_string(),
            },
            usage: AnthropicUsage {
                input_tokens,
                output_tokens,
            },
        };
        if let Ok(json) = serde_json::to_string(&msg_delta) {
            let _ = tx
                .send(Ok(Event::default()
                    .event("message_delta")
                    .data(json)))
                .await;
        }

        // 6. message_stop
        let msg_stop = MessageStopEvent {
            type_field: "message_stop".to_string(),
        };
        if let Ok(json) = serde_json::to_string(&msg_stop) {
            let _ = tx
                .send(Ok(Event::default()
                    .event("message_stop")
                    .data(json)))
                .await;
        }
    });

    Ok(Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::default()))
}

fn estimate_tokens_anthropic(messages: &[ChatMessage]) -> u32 {
    let total_chars: usize = messages.iter().map(|m| m.content.len() + m.role.len()).sum();
    (total_chars / 4).max(1) as u32
}

use axum::response::IntoResponse;
