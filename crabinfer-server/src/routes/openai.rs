use crate::chat_template::apply_chat_template;
use crate::error::ServerError;
use crate::state::AppState;
use crate::types::openai::*;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use crabinfer_core::serving::guided::GuidedConstraint;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
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

    let architecture = resolve_architecture(&state);
    let prompt = apply_chat_template(&architecture, &req.messages);
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
        state.metrics.dec_running();
        ServerError::internal(format!("task join error: {}", e))
    })?
    .map_err(|e| {
        state.metrics.inc_error();
        state.metrics.dec_running();
        ServerError::from(e)
    })?;

    let prompt_tokens = estimate_tokens(&req.messages);
    let completion_tokens = (result.len() / 4) as u32;
    state.metrics.inc_success();
    state.metrics.dec_running();
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
                content: Some(result),
                tool_calls: None,
            },
            finish_reason: "stop".to_string(),
            logprobs: None,
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
    engine: &crabinfer_core::serving::worker_pool::WorkerPool,
    req: &ChatCompletionRequest,
) -> Result<axum::response::Response, ServerError> {
    use crabinfer_core::serving::sequence::SamplingParams;

    let architecture = resolve_architecture(&state);
    let messages = prepare_messages(&req.messages, &req.tools, &req.tool_choice, &req.response_format);
    let prompt = apply_chat_template(&architecture, &messages);
    tracing::debug!("Chat prompt ({} chars): {:?}", prompt.len(), &prompt[..prompt.len().min(200)]);

    let prompt_tokens = engine
        .encode(&prompt)
        .map_err(|e| ServerError::internal(format!("tokenization failed: {e}")))?;
    let prompt_token_count = prompt_tokens.len() as u32;
    tracing::debug!("Encoded {} tokens", prompt_token_count);

    let max_tokens = (req.max_tokens.unwrap_or(256) as usize).min(MAX_TOKENS_CAP);
    let temperature = req.temperature.unwrap_or(0.7);
    let top_p = req.top_p.unwrap_or(0.9);

    let want_logprobs = req.logprobs == Some(true);
    let top_logprobs_n = req.top_logprobs.unwrap_or(0).min(20) as usize;

    let priority = req.priority.unwrap_or(0);
    let guided_constraint = extract_guided_constraint(&req.response_format);

    // Parse optional LoRA adapter from model field (e.g., "llama:my-adapter")
    let (_base_model, lora_adapter) = crabinfer_core::serving::lora::parse_model_adapter(&req.model);
    let lora_adapter = lora_adapter.map(|s| s.to_string());

    let params = SamplingParams {
        temperature,
        top_p,
        max_tokens,
        logprobs: want_logprobs,
        top_logprobs: top_logprobs_n,
        priority,
        guided_constraint,
        lora_adapter,
        cache_salt: req.cache_salt.clone(),
        ..SamplingParams::default()
    };

    let request_start = Instant::now();

    let mut rx = engine.submit(prompt_tokens, params).map_err(|e| {
        use crabinfer_core::serving::engine_loop::EngineError;
        match e {
            EngineError::Overloaded => ServerError::too_many_requests("server is overloaded, try again later"),
            _ => ServerError::internal(format!("engine error: {e}")),
        }
    })?;

    // Collect all generated tokens with timeout
    let mut generated_ids: Vec<u32> = Vec::new();
    let mut token_logprobs: Vec<TokenLogprobInfo> = Vec::new();
    let mut finish_reason_str = "stop".to_string();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(REQUEST_TIMEOUT_SECS);
    let mut ttft_recorded = false;

    loop {
        match tokio::time::timeout_at(deadline, rx.recv()).await {
            Ok(Some(tok)) => {
                // Record time to first token
                if !ttft_recorded {
                    state.metrics.ttft.observe(request_start.elapsed().as_secs_f64());
                    ttft_recorded = true;
                }
                if want_logprobs {
                    let token_text = engine.decode(&[tok.token_id]).unwrap_or_default();
                    let top = tok.top_logprobs.as_ref().map(|entries| {
                        entries
                            .iter()
                            .map(|e| TopLogprobEntry {
                                token: engine.decode(&[e.token_id]).unwrap_or_default(),
                                logprob: e.logprob,
                            })
                            .collect()
                    });
                    token_logprobs.push(TokenLogprobInfo {
                        token: token_text,
                        logprob: tok.logprob.unwrap_or(0.0),
                        top_logprobs: top,
                    });
                }
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
    state.metrics.dec_running();
    state
        .metrics
        .add_tokens(prompt_token_count as u64, completion_tokens as u64);
    state.metrics.request_latency.observe(request_start.elapsed().as_secs_f64());

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let logprobs_obj = if want_logprobs {
        Some(ChoiceLogprobs {
            content: token_logprobs,
        })
    } else {
        None
    };

    // Post-process completion: tool calls or JSON mode
    let has_tools = req.tools.is_some();
    let is_json_mode = req
        .response_format
        .as_ref()
        .map(|f| f.type_field == "json_object" || f.type_field == "json_schema")
        .unwrap_or(false);

    let (content, tool_calls, final_finish_reason) = if has_tools {
        match parse_tool_calls_from_output(&completion_text) {
            Some(calls) => (None, Some(calls), "tool_calls".to_string()),
            None => (Some(maybe_clean_json(&completion_text, is_json_mode)), None, finish_reason_str),
        }
    } else {
        (Some(maybe_clean_json(&completion_text, is_json_mode)), None, finish_reason_str)
    };

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", now),
        object: "chat.completion".to_string(),
        created: now,
        model: state.model_id.clone(),
        choices: vec![Choice {
            index: 0,
            message: ChoiceMessage {
                role: "assistant".to_string(),
                content,
                tool_calls,
            },
            finish_reason: final_finish_reason,
            logprobs: logprobs_obj,
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

    let architecture = resolve_architecture(&state);
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
                    tool_calls: None,
                },
                finish_reason: None,
                logprobs: None,
            }],
            usage: None,
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
                                tool_calls: None,
                            },
                            finish_reason: None,
                            logprobs: None,
                        }],
                        usage: None,
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
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
                logprobs: None,
            }],
            usage: None,
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
    engine: crabinfer_core::serving::worker_pool::WorkerPool,
    req: ChatCompletionRequest,
) -> Result<
    Sse<axum::response::sse::KeepAliveStream<ReceiverStream<Result<Event, std::convert::Infallible>>>>,
    ServerError,
> {
    use crabinfer_core::serving::sequence::SamplingParams;

    let architecture = resolve_architecture(&state);
    let messages = prepare_messages(&req.messages, &req.tools, &req.tool_choice, &req.response_format);
    let prompt = apply_chat_template(&architecture, &messages);
    let model_id = state.model_id.clone();

    let prompt_tokens = engine
        .encode(&prompt)
        .map_err(|e| ServerError::internal(format!("tokenization failed: {e}")))?;
    let prompt_token_count = prompt_tokens.len() as u32;

    let max_tokens = (req.max_tokens.unwrap_or(256) as usize).min(MAX_TOKENS_CAP);
    let temperature = req.temperature.unwrap_or(0.7);
    let top_p = req.top_p.unwrap_or(0.9);

    let want_logprobs = req.logprobs == Some(true);
    let top_logprobs_n = req.top_logprobs.unwrap_or(0).min(20) as usize;
    let include_usage = req
        .stream_options
        .as_ref()
        .and_then(|o| o.include_usage)
        .unwrap_or(false);
    let priority = req.priority.unwrap_or(0);
    let guided_constraint = extract_guided_constraint(&req.response_format);

    // Parse optional LoRA adapter from model field (e.g., "llama:my-adapter")
    let (_base_model, lora_adapter) = crabinfer_core::serving::lora::parse_model_adapter(&req.model);
    let lora_adapter = lora_adapter.map(|s| s.to_string());

    let params = SamplingParams {
        temperature,
        top_p,
        max_tokens,
        logprobs: want_logprobs,
        top_logprobs: top_logprobs_n,
        priority,
        guided_constraint,
        lora_adapter,
        cache_salt: req.cache_salt.clone(),
        ..SamplingParams::default()
    };

    let request_start = Instant::now();

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
                    tool_calls: None,
                },
                finish_reason: None,
                logprobs: None,
            }],
            usage: None,
        };
        if let Ok(json) = serde_json::to_string(&initial_chunk) {
            let _ = tx.send(Ok(Event::default().data(json))).await;
        }

        // Stream tokens with timeout
        let deadline = tokio::time::Instant::now() + Duration::from_secs(REQUEST_TIMEOUT_SECS);
        let mut finish_reason_str = None;
        let mut completion_tokens: u32 = 0;
        let mut ttft_recorded = false;
        let mut last_token_time = request_start;
        loop {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(tok)) => {
                    let now_inst = Instant::now();
                    completion_tokens += 1;

                    // Record time to first token
                    if !ttft_recorded {
                        state.metrics.ttft.observe(request_start.elapsed().as_secs_f64());
                        ttft_recorded = true;
                    } else {
                        // Record inter-token latency
                        state.metrics.itl.observe(now_inst.duration_since(last_token_time).as_secs_f64());
                    }
                    last_token_time = now_inst;
                    let text = engine.decode(&[tok.token_id]).unwrap_or_default();
                    let is_done = tok.finish_reason.is_some();
                    if let Some(reason) = tok.finish_reason {
                        finish_reason_str = Some(finish_reason_to_openai(reason).to_string());
                    }

                    // Build per-token logprobs for this chunk
                    let chunk_logprobs = if want_logprobs {
                        let top = tok.top_logprobs.as_ref().map(|entries| {
                            entries
                                .iter()
                                .map(|e| TopLogprobEntry {
                                    token: engine.decode(&[e.token_id]).unwrap_or_default(),
                                    logprob: e.logprob,
                                })
                                .collect()
                        });
                        Some(ChoiceLogprobs {
                            content: vec![TokenLogprobInfo {
                                token: text.clone(),
                                logprob: tok.logprob.unwrap_or(0.0),
                                top_logprobs: top,
                            }],
                        })
                    } else {
                        None
                    };

                    if !text.is_empty() || chunk_logprobs.is_some() {
                        let chunk = ChatCompletionChunk {
                            id: chunk_id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created: now,
                            model: model_id.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: Delta {
                                    role: None,
                                    content: if text.is_empty() { None } else { Some(text) },
                                    tool_calls: None,
                                },
                                finish_reason: None,
                                logprobs: chunk_logprobs,
                            }],
                            usage: None,
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

        // Update metrics
        state.metrics.inc_success();
        state.metrics.dec_running();
        state
            .metrics
            .add_tokens(prompt_token_count as u64, completion_tokens as u64);
        state.metrics.request_latency.observe(request_start.elapsed().as_secs_f64());

        // Send final chunk with finish_reason
        let final_chunk = ChatCompletionChunk {
            id: chunk_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: now,
            model: model_id.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: None,
                    tool_calls: None,
                },
                finish_reason: Some(finish_reason_str.unwrap_or_else(|| "stop".to_string())),
                logprobs: None,
            }],
            usage: None,
        };
        if let Ok(json) = serde_json::to_string(&final_chunk) {
            let _ = tx.send(Ok(Event::default().data(json))).await;
        }

        // Send usage chunk if requested (OpenAI stream_options.include_usage)
        if include_usage {
            let usage_chunk = ChatCompletionChunk {
                id: chunk_id,
                object: "chat.completion.chunk".to_string(),
                created: now,
                model: model_id,
                choices: vec![],
                usage: Some(Usage {
                    prompt_tokens: prompt_token_count,
                    completion_tokens,
                    total_tokens: prompt_token_count + completion_tokens,
                }),
            };
            if let Ok(json) = serde_json::to_string(&usage_chunk) {
                let _ = tx.send(Ok(Event::default().data(json))).await;
            }
        }

        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
    });

    Ok(Sse::new(ReceiverStream::new(sse_rx)).keep_alive(KeepAlive::default()))
}

/// Map FinishReason to OpenAI finish_reason string.
pub fn finish_reason_to_openai(reason: crabinfer_core::serving::sequence::FinishReason) -> &'static str {
    use crabinfer_core::serving::sequence::FinishReason;
    match reason {
        FinishReason::EndOfSequence => "stop",
        FinishReason::MaxTokens => "length",
        FinishReason::Stop => "stop",
        FinishReason::ToolCalls => "tool_calls",
        FinishReason::Cancelled => "stop",
        FinishReason::Preempted => "stop",
    }
}

/// Resolve the chat template architecture string.
/// Uses the override from `--chat-template` if set, otherwise falls back to the model's architecture.
pub fn resolve_architecture(state: &AppState) -> String {
    state
        .chat_template
        .as_deref()
        .unwrap_or(&state.model_info.architecture)
        .to_string()
}

/// Approximate token count from messages (text.len() / 4)
fn estimate_tokens(messages: &[crate::types::common::ChatMessage]) -> u32 {
    let total_chars: usize = messages.iter().map(|m| m.content_str().len() + m.role.len()).sum();
    (total_chars / 4).max(1) as u32
}

// ---------------------------------------------------------------------------
// Message preparation (tool & response_format injection)
// ---------------------------------------------------------------------------

/// Prepare messages for template application.
///
/// - Injects tool definitions as a system message if tools are provided
/// - Injects response_format constraints as a system message
/// - Converts `tool` role messages into user messages
/// - Converts assistant tool_calls into visible `<tool_call>` text
fn prepare_messages(
    messages: &[crate::types::common::ChatMessage],
    tools: &Option<Vec<ToolDefinition>>,
    tool_choice: &Option<ToolChoice>,
    response_format: &Option<ResponseFormat>,
) -> Vec<crate::types::common::ChatMessage> {
    use crate::types::common::ChatMessage;

    let mut out: Vec<ChatMessage> = Vec::with_capacity(messages.len() + 2);

    // Inject tools system message if tools are provided
    if let Some(ref tools) = tools {
        if !tools.is_empty() {
            let tools_prompt = build_tools_system_prompt(tools, tool_choice);
            out.push(ChatMessage {
                role: "system".to_string(),
                content: Some(tools_prompt.into()),
                tool_call_id: None,
                tool_calls: None,
                name: None,
            });
        }
    }

    // Inject response_format constraint
    if let Some(ref fmt) = response_format {
        if let Some(prompt) = build_response_format_prompt(fmt) {
            out.push(ChatMessage {
                role: "system".to_string(),
                content: Some(prompt.into()),
                tool_call_id: None,
                tool_calls: None,
                name: None,
            });
        }
    }

    // Convert messages, handling special roles
    for msg in messages {
        match msg.role.as_str() {
            "tool" => {
                // Convert tool result into a user-visible message
                let tool_name = msg.name.as_deref().unwrap_or("tool");
                let tool_id = msg.tool_call_id.as_deref().unwrap_or("unknown");
                let result_text = msg.content_str();
                out.push(ChatMessage {
                    role: "user".to_string(),
                    content: Some(format!(
                        "[Tool result for {tool_name} (call_id: {tool_id})]:\n{result_text}"
                    ).into()),
                    tool_call_id: None,
                    tool_calls: None,
                    name: None,
                });
            }
            "assistant" if msg.tool_calls.is_some() => {
                // Convert assistant tool_calls into visible text
                let mut content = msg.content_str().to_string();
                if let Some(ref calls) = msg.tool_calls {
                    for call in calls {
                        content.push_str(&format!(
                            "\n<tool_call>\n{{\"name\": \"{}\", \"arguments\": {}}}\n</tool_call>",
                            call.function.name, call.function.arguments
                        ));
                    }
                }
                out.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: Some(content.into()),
                    tool_call_id: None,
                    tool_calls: None,
                    name: None,
                });
            }
            _ => {
                out.push(msg.clone());
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Tool call parsing
// ---------------------------------------------------------------------------

/// Build a system prompt snippet describing available tools.
///
/// This is injected into the prompt when `tools` are provided, so the model
/// knows the function signatures and can produce structured tool calls.
fn build_tools_system_prompt(tools: &[ToolDefinition], tool_choice: &Option<ToolChoice>) -> String {
    let mut prompt = String::from(
        "You have access to the following tools. To call a tool, respond with a JSON object \
         wrapped in <tool_call> tags. You may call multiple tools.\n\n\
         Format:\n\
         <tool_call>\n\
         {\"name\": \"function_name\", \"arguments\": {\"arg1\": \"value1\"}}\n\
         </tool_call>\n\n\
         Available tools:\n",
    );

    for tool in tools {
        if tool.type_field != "function" {
            continue;
        }
        prompt.push_str(&format!("\n### {}\n", tool.function.name));
        if let Some(ref desc) = tool.function.description {
            prompt.push_str(&format!("{}\n", desc));
        }
        if let Some(ref params) = tool.function.parameters {
            prompt.push_str(&format!(
                "Parameters: {}\n",
                serde_json::to_string(params).unwrap_or_default()
            ));
        }
    }

    // Add guidance based on tool_choice
    match tool_choice {
        Some(ToolChoice::String(s)) if s == "required" => {
            prompt.push_str("\nYou MUST call at least one tool in your response.\n");
        }
        Some(ToolChoice::String(s)) if s == "none" => {
            prompt.push_str("\nDo NOT call any tools. Respond with text only.\n");
        }
        Some(ToolChoice::Specific(spec)) => {
            prompt.push_str(&format!(
                "\nYou MUST call the tool named \"{}\".\n",
                spec.function.name
            ));
        }
        _ => {} // "auto" or None — model decides
    }

    prompt
}

// ---------------------------------------------------------------------------
// Response format (JSON mode / structured output)
// ---------------------------------------------------------------------------

/// Extract a guided decoding constraint from the request's response_format.
///
/// Returns `Some(GuidedConstraint)` for `json_schema` type (token-level DFA masking).
/// Returns `None` for `json_object` (prompt-only), `text`, or missing format.
/// The prompt-based injection (`build_response_format_prompt`) still runs alongside
/// the token-level constraint for belt-and-suspenders reliability.
fn extract_guided_constraint(response_format: &Option<ResponseFormat>) -> Option<GuidedConstraint> {
    let fmt = response_format.as_ref()?;
    match fmt.type_field.as_str() {
        "json_schema" => {
            let spec = fmt.json_schema.as_ref()?;
            let schema = spec.schema.as_ref()?;
            Some(GuidedConstraint::JsonSchema(schema.clone()))
        }
        _ => None, // "text", "json_object", unknown -- no token-level constraint
    }
}

/// Build a system prompt for response_format constraints.
///
/// Returns `None` for "text" (no constraint) or unrecognized types.
fn build_response_format_prompt(fmt: &ResponseFormat) -> Option<String> {
    match fmt.type_field.as_str() {
        "json_object" => Some(
            "You must respond with valid JSON only. Do not include any text outside the JSON object. \
             Do not wrap the response in markdown code blocks. Output a single JSON object."
                .to_string(),
        ),
        "json_schema" => {
            let spec = fmt.json_schema.as_ref()?;
            let mut prompt = format!(
                "You must respond with valid JSON that conforms to the \"{}\" schema.\n\
                 Do not include any text outside the JSON object. \
                 Do not wrap the response in markdown code blocks.\n",
                spec.name
            );
            if let Some(ref desc) = spec.description {
                prompt.push_str(&format!("Description: {desc}\n"));
            }
            if let Some(ref schema) = spec.schema {
                prompt.push_str(&format!(
                    "JSON Schema:\n{}\n",
                    serde_json::to_string_pretty(schema).unwrap_or_default()
                ));
            }
            if spec.strict == Some(true) {
                prompt.push_str("Your response MUST strictly match every field, type, and constraint in the schema.\n");
            }
            Some(prompt)
        }
        _ => None, // "text" or unknown — no constraint
    }
}

/// Attempt to extract valid JSON from model output.
///
/// Handles common model quirks:
/// - Leading/trailing whitespace
/// - Markdown code fences (```json ... ```)
/// - Text before/after the JSON
fn extract_json_from_output(text: &str) -> Option<serde_json::Value> {
    let trimmed = text.trim();

    // Try direct parse first
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
        return Some(v);
    }

    // Try stripping markdown code fences
    if let Some(content) = strip_code_fence(trimmed) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(content.trim()) {
            return Some(v);
        }
    }

    // Try finding the first { ... } or [ ... ] block
    if let Some(json_str) = extract_first_json_block(trimmed) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(json_str) {
            return Some(v);
        }
    }

    None
}

/// Strip ```json ... ``` or ``` ... ``` fences.
fn strip_code_fence(text: &str) -> Option<&str> {
    let start_markers = ["```json\n", "```json\r\n", "```\n", "```\r\n"];
    for marker in &start_markers {
        if let Some(rest) = text.strip_prefix(marker) {
            if let Some(end) = rest.rfind("```") {
                return Some(&rest[..end]);
            }
        }
    }
    None
}

/// Find the first balanced JSON object or array in text.
fn extract_first_json_block(text: &str) -> Option<&str> {
    let open_pos = text.find(|c| c == '{' || c == '[')?;
    let open_char = text.as_bytes()[open_pos];
    let close_char = if open_char == b'{' { b'}' } else { b']' };

    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape_next = false;

    for (i, &ch) in text.as_bytes()[open_pos..].iter().enumerate() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if ch == b'\\' && in_string {
            escape_next = true;
            continue;
        }
        if ch == b'"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        if ch == open_char {
            depth += 1;
        } else if ch == close_char {
            depth -= 1;
            if depth == 0 {
                return Some(&text[open_pos..open_pos + i + 1]);
            }
        }
    }

    None
}

/// If JSON mode is active, try to extract and re-serialize clean JSON.
/// Falls back to the original text if extraction fails.
fn maybe_clean_json(text: &str, is_json_mode: bool) -> String {
    if !is_json_mode {
        return text.to_string();
    }
    match extract_json_from_output(text) {
        Some(v) => serde_json::to_string(&v).unwrap_or_else(|_| text.to_string()),
        None => text.to_string(),
    }
}

/// Try to parse tool calls from model output text.
///
/// Looks for `<tool_call>...</tool_call>` blocks containing JSON with
/// `name` and `arguments` fields. Returns `None` if no tool calls found.
fn parse_tool_calls_from_output(text: &str) -> Option<Vec<ToolCall>> {
    let mut calls = Vec::new();
    let mut search_from = 0;

    loop {
        let start_tag = "<tool_call>";
        let end_tag = "</tool_call>";

        let start = match text[search_from..].find(start_tag) {
            Some(pos) => search_from + pos + start_tag.len(),
            None => break,
        };
        let end = match text[start..].find(end_tag) {
            Some(pos) => start + pos,
            None => break,
        };
        search_from = end + end_tag.len();

        let json_str = text[start..end].trim();
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str) {
            let name = parsed
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            let arguments = parsed
                .get("arguments")
                .map(|v| {
                    if v.is_string() {
                        v.as_str().unwrap_or("{}").to_string()
                    } else {
                        serde_json::to_string(v).unwrap_or_else(|_| "{}".to_string())
                    }
                })
                .unwrap_or_else(|| "{}".to_string());

            calls.push(ToolCall {
                id: format!("call_{}", generate_tool_call_id()),
                type_field: "function".to_string(),
                function: FunctionCall { name, arguments },
            });
        }
    }

    if calls.is_empty() {
        None
    } else {
        Some(calls)
    }
}

/// Generate a short random ID for tool calls.
fn generate_tool_call_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    format!("{:08x}", nanos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tool_calls_single() {
        let text = r#"I'll look that up for you.
<tool_call>
{"name": "get_weather", "arguments": {"location": "San Francisco", "unit": "celsius"}}
</tool_call>"#;

        let calls = parse_tool_calls_from_output(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].type_field, "function");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["location"], "San Francisco");
        assert_eq!(args["unit"], "celsius");
    }

    #[test]
    fn test_parse_tool_calls_multiple() {
        let text = r#"<tool_call>
{"name": "search", "arguments": {"query": "rust"}}
</tool_call>
<tool_call>
{"name": "calculate", "arguments": {"expression": "2+2"}}
</tool_call>"#;

        let calls = parse_tool_calls_from_output(text).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "search");
        assert_eq!(calls[1].function.name, "calculate");
    }

    #[test]
    fn test_parse_tool_calls_none() {
        let text = "Just a regular response with no tool calls.";
        assert!(parse_tool_calls_from_output(text).is_none());
    }

    #[test]
    fn test_parse_tool_calls_malformed_json() {
        let text = "<tool_call>\nnot valid json\n</tool_call>";
        assert!(parse_tool_calls_from_output(text).is_none());
    }

    #[test]
    fn test_parse_tool_calls_unclosed_tag() {
        let text = r#"<tool_call>{"name": "test", "arguments": {}}"#;
        assert!(parse_tool_calls_from_output(text).is_none());
    }

    #[test]
    fn test_build_tools_system_prompt_auto() {
        let tools = vec![ToolDefinition {
            type_field: "function".to_string(),
            function: FunctionDefinition {
                name: "get_weather".to_string(),
                description: Some("Get the current weather".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                })),
            },
        }];

        let prompt = build_tools_system_prompt(&tools, &None);
        assert!(prompt.contains("get_weather"));
        assert!(prompt.contains("Get the current weather"));
        assert!(prompt.contains("<tool_call>"));
        assert!(prompt.contains("location"));
    }

    #[test]
    fn test_build_tools_system_prompt_required() {
        let tools = vec![ToolDefinition {
            type_field: "function".to_string(),
            function: FunctionDefinition {
                name: "search".to_string(),
                description: None,
                parameters: None,
            },
        }];

        let choice = Some(ToolChoice::String("required".to_string()));
        let prompt = build_tools_system_prompt(&tools, &choice);
        assert!(prompt.contains("MUST call at least one tool"));
    }

    #[test]
    fn test_build_tools_system_prompt_none() {
        let tools = vec![ToolDefinition {
            type_field: "function".to_string(),
            function: FunctionDefinition {
                name: "search".to_string(),
                description: None,
                parameters: None,
            },
        }];

        let choice = Some(ToolChoice::String("none".to_string()));
        let prompt = build_tools_system_prompt(&tools, &choice);
        assert!(prompt.contains("Do NOT call any tools"));
    }

    #[test]
    fn test_build_tools_system_prompt_specific() {
        let tools = vec![ToolDefinition {
            type_field: "function".to_string(),
            function: FunctionDefinition {
                name: "my_func".to_string(),
                description: None,
                parameters: None,
            },
        }];

        let choice = Some(ToolChoice::Specific(ToolChoiceSpecific {
            type_field: "function".to_string(),
            function: ToolChoiceFunction {
                name: "my_func".to_string(),
            },
        }));
        let prompt = build_tools_system_prompt(&tools, &choice);
        assert!(prompt.contains("MUST call the tool named \"my_func\""));
    }

    #[test]
    fn test_parse_tool_calls_string_arguments() {
        // Some models return arguments as a JSON string rather than an object
        let text = r#"<tool_call>
{"name": "test", "arguments": "{\"key\": \"value\"}"}
</tool_call>"#;

        let calls = parse_tool_calls_from_output(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.arguments, r#"{"key": "value"}"#);
    }

    #[test]
    fn test_tool_call_id_format() {
        let id = generate_tool_call_id();
        assert_eq!(id.len(), 8);
        // Should be valid hex
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }

    // ── JSON mode / structured output tests ──

    #[test]
    fn test_extract_json_direct() {
        let text = r#"{"name": "Alice", "age": 30}"#;
        let v = extract_json_from_output(text).unwrap();
        assert_eq!(v["name"], "Alice");
        assert_eq!(v["age"], 30);
    }

    #[test]
    fn test_extract_json_with_whitespace() {
        let text = "  \n{\"key\": \"value\"}\n  ";
        let v = extract_json_from_output(text).unwrap();
        assert_eq!(v["key"], "value");
    }

    #[test]
    fn test_extract_json_from_code_fence() {
        let text = "```json\n{\"result\": 42}\n```";
        let v = extract_json_from_output(text).unwrap();
        assert_eq!(v["result"], 42);
    }

    #[test]
    fn test_extract_json_from_code_fence_no_lang() {
        let text = "```\n{\"result\": 42}\n```";
        let v = extract_json_from_output(text).unwrap();
        assert_eq!(v["result"], 42);
    }

    #[test]
    fn test_extract_json_from_surrounding_text() {
        let text = "Here is the result:\n{\"answer\": \"yes\"}\nDone.";
        let v = extract_json_from_output(text).unwrap();
        assert_eq!(v["answer"], "yes");
    }

    #[test]
    fn test_extract_json_array() {
        let text = "[1, 2, 3]";
        let v = extract_json_from_output(text).unwrap();
        assert!(v.is_array());
        assert_eq!(v.as_array().unwrap().len(), 3);
    }

    #[test]
    fn test_extract_json_nested() {
        let text = r#"Output: {"user": {"name": "Bob", "items": [1, 2]}}"#;
        let v = extract_json_from_output(text).unwrap();
        assert_eq!(v["user"]["name"], "Bob");
    }

    #[test]
    fn test_extract_json_no_json() {
        let text = "Just some regular text without any JSON.";
        assert!(extract_json_from_output(text).is_none());
    }

    #[test]
    fn test_maybe_clean_json_passthrough() {
        let text = "Hello world";
        assert_eq!(maybe_clean_json(text, false), "Hello world");
    }

    #[test]
    fn test_maybe_clean_json_active() {
        let text = "Result: {\"key\": \"value\"}  extra text";
        let cleaned = maybe_clean_json(text, true);
        assert_eq!(cleaned, r#"{"key":"value"}"#);
    }

    #[test]
    fn test_build_response_format_prompt_json_object() {
        let fmt = ResponseFormat {
            type_field: "json_object".to_string(),
            json_schema: None,
        };
        let prompt = build_response_format_prompt(&fmt).unwrap();
        assert!(prompt.contains("valid JSON"));
        assert!(prompt.contains("single JSON object"));
    }

    #[test]
    fn test_build_response_format_prompt_json_schema() {
        let fmt = ResponseFormat {
            type_field: "json_schema".to_string(),
            json_schema: Some(JsonSchemaSpec {
                name: "UserProfile".to_string(),
                description: Some("A user profile".to_string()),
                schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    },
                    "required": ["name", "age"]
                })),
                strict: Some(true),
            }),
        };
        let prompt = build_response_format_prompt(&fmt).unwrap();
        assert!(prompt.contains("UserProfile"));
        assert!(prompt.contains("A user profile"));
        assert!(prompt.contains("\"name\""));
        assert!(prompt.contains("strictly match"));
    }

    #[test]
    fn test_build_response_format_prompt_text() {
        let fmt = ResponseFormat {
            type_field: "text".to_string(),
            json_schema: None,
        };
        assert!(build_response_format_prompt(&fmt).is_none());
    }

    #[test]
    fn test_extract_json_with_string_containing_braces() {
        let text = r#"{"message": "Use {curly} braces"}"#;
        let v = extract_json_from_output(text).unwrap();
        assert_eq!(v["message"], "Use {curly} braces");
    }

    // ── Guided constraint extraction tests ──

    #[test]
    fn test_response_format_json_schema_creates_constraint() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        });
        let fmt = Some(ResponseFormat {
            type_field: "json_schema".to_string(),
            json_schema: Some(JsonSchemaSpec {
                name: "UserProfile".to_string(),
                description: Some("A user profile".to_string()),
                schema: Some(schema.clone()),
                strict: Some(true),
            }),
        });
        let constraint = extract_guided_constraint(&fmt);
        match constraint {
            Some(GuidedConstraint::JsonSchema(v)) => {
                assert_eq!(v, schema, "Schema value should match input");
            }
            other => panic!("Expected GuidedConstraint::JsonSchema, got {:?}", other),
        }
    }

    #[test]
    fn test_response_format_text_no_constraint() {
        let fmt = Some(ResponseFormat {
            type_field: "text".to_string(),
            json_schema: None,
        });
        assert!(
            extract_guided_constraint(&fmt).is_none(),
            "text format should not produce a constraint"
        );
    }

    #[test]
    fn test_response_format_json_object_no_constraint() {
        let fmt = Some(ResponseFormat {
            type_field: "json_object".to_string(),
            json_schema: None,
        });
        assert!(
            extract_guided_constraint(&fmt).is_none(),
            "json_object format should not produce a constraint (prompt-only)"
        );
    }

    #[test]
    fn test_response_format_none_no_constraint() {
        let fmt: Option<ResponseFormat> = None;
        assert!(
            extract_guided_constraint(&fmt).is_none(),
            "None response_format should not produce a constraint"
        );
    }

    #[test]
    fn test_response_format_json_schema_no_schema_value() {
        // json_schema type but no actual schema provided -- should return None
        let fmt = Some(ResponseFormat {
            type_field: "json_schema".to_string(),
            json_schema: Some(JsonSchemaSpec {
                name: "Empty".to_string(),
                description: None,
                schema: None,
                strict: None,
            }),
        });
        assert!(
            extract_guided_constraint(&fmt).is_none(),
            "json_schema without schema value should not produce a constraint"
        );
    }

    #[test]
    fn test_cache_salt_field_exists_on_sampling_params() {
        // Marker test: catches accidental field renames in crabinfer-core.
        // If this fails to compile, cache_salt was renamed and all four
        // propagation sites need updating.
        use crabinfer_core::serving::sequence::SamplingParams;
        let params = SamplingParams {
            cache_salt: Some("test-salt".to_string()),
            ..Default::default()
        };
        assert_eq!(params.cache_salt.as_deref(), Some("test-salt"));
    }
}
