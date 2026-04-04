//! Guided completions endpoint: POST /v1/guided/completions
//!
//! Accepts `constraint` (regex or json_schema) with configurable error behavior
//! via `strict_constraints` (default: true → HTTP 400 on invalid constraint).

use crate::chat_template::apply_chat_template;
use crate::error::ServerError;
use crate::state::AppState;
use crate::types::guided::{GuidedCompletionRequest, GuidedConstraintSpec};
use crate::types::openai::*;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::Json;
use crabinfer_core::serving::guided::GuidedConstraint;
use crabinfer_core::serving::sequence::SamplingParams;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio_stream::wrappers::ReceiverStream;

/// Hard cap on max_tokens to prevent runaway generation.
const MAX_TOKENS_CAP: usize = 16384;

/// Per-request timeout for token generation (seconds).
const REQUEST_TIMEOUT_SECS: u64 = 120;

/// Convert a `GuidedConstraintSpec` (API type) to `GuidedConstraint` (core type).
fn to_core_constraint(spec: &GuidedConstraintSpec) -> GuidedConstraint {
    match spec {
        GuidedConstraintSpec::Regex { pattern } => {
            GuidedConstraint::Regex(pattern.clone())
        }
        GuidedConstraintSpec::JsonSchema { json_schema } => {
            GuidedConstraint::JsonSchema(json_schema.clone())
        }
    }
}

/// POST /v1/guided/completions
///
/// Constrained generation endpoint. Accepts all standard chat completion fields
/// plus a required `constraint` field. When `strict_constraints` is `true`
/// (default), an invalid constraint returns HTTP 400. When `false`, a warning
/// is logged and generation proceeds unconstrained.
pub async fn guided_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GuidedCompletionRequest>,
) -> Result<axum::response::Response, ServerError> {
    state.metrics.inc_request();

    let engine = state
        .serving_engine
        .as_ref()
        .ok_or_else(|| ServerError::service_unavailable("serving engine not available"))?;

    let core_constraint = to_core_constraint(&req.constraint);

    // Validate constraint before submission (strict mode → HTTP 400 on failure)
    let validated_constraint = match engine.validate_constraint(&core_constraint) {
        Ok(()) => Some(core_constraint),
        Err(e) => {
            if req.strict_constraints {
                state.metrics.inc_error();
                state.metrics.dec_running();
                return Err(ServerError::bad_request(format!(
                    "Invalid guided constraint: {e}"
                )));
            } else {
                tracing::warn!(
                    "Guided constraint failed to compile: {e}; generating unconstrained"
                );
                None
            }
        }
    };

    if req.stream == Some(true) {
        return Ok(
            guided_completions_stream(
                Arc::clone(&state),
                engine.clone(),
                req,
                validated_constraint,
            )
            .await?
            .into_response(),
        );
    }

    // ── Non-streaming path ──
    let architecture = super::openai::resolve_architecture(&state);
    let messages = req.messages.clone();
    let prompt = apply_chat_template(&architecture, &messages);
    tracing::debug!(
        "Guided prompt ({} chars): {:?}",
        prompt.len(),
        &prompt[..prompt.len().min(200)]
    );

    let prompt_tokens = engine
        .encode(&prompt)
        .map_err(|e| ServerError::internal(format!("tokenization failed: {e}")))?;
    let prompt_token_count = prompt_tokens.len() as u32;

    let max_tokens = (req.max_tokens.unwrap_or(256) as usize).min(MAX_TOKENS_CAP);
    let temperature = req.temperature.unwrap_or(0.7);
    let top_p = req.top_p.unwrap_or(0.9);
    let want_logprobs = req.logprobs == Some(true);
    let top_logprobs_n = req.top_logprobs.unwrap_or(0).min(20) as usize;
    let priority = req.priority.unwrap_or(0);

    let (_base_model, lora_adapter) =
        crabinfer_core::serving::lora::parse_model_adapter(&req.model);
    let lora_adapter = lora_adapter.map(|s| s.to_string());

    let params = SamplingParams {
        temperature,
        top_p,
        max_tokens,
        logprobs: want_logprobs,
        top_logprobs: top_logprobs_n,
        priority,
        guided_constraint: validated_constraint,
        lora_adapter,
        ..SamplingParams::default()
    };

    let request_start = Instant::now();

    let mut rx = engine.submit(prompt_tokens, params).map_err(|e| {
        use crabinfer_core::serving::engine_loop::EngineError;
        match e {
            EngineError::Overloaded => {
                ServerError::too_many_requests("server is overloaded, try again later")
            }
            _ => ServerError::internal(format!("engine error: {e}")),
        }
    })?;

    let mut generated_ids: Vec<u32> = Vec::new();
    let mut token_logprobs: Vec<TokenLogprobInfo> = Vec::new();
    let mut finish_reason_str = "stop".to_string();
    let deadline =
        tokio::time::Instant::now() + Duration::from_secs(REQUEST_TIMEOUT_SECS);
    let mut ttft_recorded = false;

    loop {
        match tokio::time::timeout_at(deadline, rx.recv()).await {
            Ok(Some(tok)) => {
                if !ttft_recorded {
                    state
                        .metrics
                        .ttft
                        .observe(request_start.elapsed().as_secs_f64());
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
                    finish_reason_str =
                        super::openai::finish_reason_to_openai(reason).to_string();
                    break;
                }
            }
            Ok(None) => break,
            Err(_) => {
                finish_reason_str = "length".to_string();
                tracing::warn!(
                    "Guided request timed out after {}s ({} tokens generated)",
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
    state
        .metrics
        .request_latency
        .observe(request_start.elapsed().as_secs_f64());

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

    let response = ChatCompletionResponse {
        id: format!("guidedcmpl-{}", now),
        object: "chat.completion".to_string(),
        created: now,
        model: state.model_id.clone(),
        choices: vec![Choice {
            index: 0,
            message: ChoiceMessage {
                role: "assistant".to_string(),
                content: Some(completion_text),
                tool_calls: None,
            },
            finish_reason: finish_reason_str,
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

/// Streaming variant for `/v1/guided/completions`.
async fn guided_completions_stream(
    state: Arc<AppState>,
    engine: crabinfer_core::serving::worker_pool::WorkerPool,
    req: GuidedCompletionRequest,
    validated_constraint: Option<GuidedConstraint>,
) -> Result<
    Sse<
        axum::response::sse::KeepAliveStream<
            ReceiverStream<Result<Event, std::convert::Infallible>>,
        >,
    >,
    ServerError,
> {
    let architecture = super::openai::resolve_architecture(&state);
    let messages = req.messages.clone();
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

    let (_base_model, lora_adapter) =
        crabinfer_core::serving::lora::parse_model_adapter(&req.model);
    let lora_adapter = lora_adapter.map(|s| s.to_string());

    let params = SamplingParams {
        temperature,
        top_p,
        max_tokens,
        logprobs: want_logprobs,
        top_logprobs: top_logprobs_n,
        priority,
        guided_constraint: validated_constraint,
        lora_adapter,
        ..SamplingParams::default()
    };

    let request_start = Instant::now();

    let mut rx = engine.submit(prompt_tokens, params).map_err(|e| {
        use crabinfer_core::serving::engine_loop::EngineError;
        match e {
            EngineError::Overloaded => {
                ServerError::too_many_requests("server is overloaded, try again later")
            }
            _ => ServerError::internal(format!("engine error: {e}")),
        }
    })?;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let chunk_id = format!("guidedcmpl-{}", now);

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

        let deadline =
            tokio::time::Instant::now() + Duration::from_secs(REQUEST_TIMEOUT_SECS);
        let mut finish_reason_str = None;
        let mut completion_tokens: u32 = 0;
        let mut ttft_recorded = false;
        let mut last_token_time = request_start;

        loop {
            match tokio::time::timeout_at(deadline, rx.recv()).await {
                Ok(Some(tok)) => {
                    let now_inst = Instant::now();
                    completion_tokens += 1;

                    if !ttft_recorded {
                        state
                            .metrics
                            .ttft
                            .observe(request_start.elapsed().as_secs_f64());
                        ttft_recorded = true;
                    } else {
                        state
                            .metrics
                            .itl
                            .observe(now_inst.duration_since(last_token_time).as_secs_f64());
                    }
                    last_token_time = now_inst;
                    let text = engine.decode(&[tok.token_id]).unwrap_or_default();
                    let is_done = tok.finish_reason.is_some();
                    if let Some(reason) = tok.finish_reason {
                        finish_reason_str =
                            Some(super::openai::finish_reason_to_openai(reason).to_string());
                    }

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
                Ok(None) => break,
                Err(_) => {
                    tracing::warn!(
                        "Streaming guided request timed out after {REQUEST_TIMEOUT_SECS}s"
                    );
                    finish_reason_str = Some("length".to_string());
                    break;
                }
            }
        }

        state.metrics.inc_success();
        state.metrics.dec_running();
        state
            .metrics
            .add_tokens(prompt_token_count as u64, completion_tokens as u64);
        state
            .metrics
            .request_latency
            .observe(request_start.elapsed().as_secs_f64());

        // Final chunk with finish_reason
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
