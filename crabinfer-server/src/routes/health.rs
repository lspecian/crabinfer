use crate::state::AppState;
use axum::extract::State;
use axum::Json;
use serde_json::{json, Value};
use std::sync::Arc;
use std::sync::atomic::Ordering;

pub async fn health() -> Json<Value> {
    Json(json!({
        "status": "ok"
    }))
}

/// GET /metrics — Prometheus text exposition format
pub async fn metrics(State(state): State<Arc<AppState>>) -> String {
    let m = &state.metrics;
    let requests_total = m.requests_total.load(Ordering::Relaxed);
    let requests_success = m.requests_success.load(Ordering::Relaxed);
    let requests_error = m.requests_error.load(Ordering::Relaxed);
    let tokens_generated = m.tokens_generated.load(Ordering::Relaxed);
    let prompt_tokens = m.prompt_tokens.load(Ordering::Relaxed);

    format!(
        "# HELP crabinfer_requests_total Total inference requests received.\n\
         # TYPE crabinfer_requests_total counter\n\
         crabinfer_requests_total {requests_total}\n\
         # HELP crabinfer_requests_success Total successful responses.\n\
         # TYPE crabinfer_requests_success counter\n\
         crabinfer_requests_success {requests_success}\n\
         # HELP crabinfer_requests_error Total error responses.\n\
         # TYPE crabinfer_requests_error counter\n\
         crabinfer_requests_error {requests_error}\n\
         # HELP crabinfer_tokens_generated Total tokens generated.\n\
         # TYPE crabinfer_tokens_generated counter\n\
         crabinfer_tokens_generated {tokens_generated}\n\
         # HELP crabinfer_prompt_tokens Total prompt tokens processed.\n\
         # TYPE crabinfer_prompt_tokens counter\n\
         crabinfer_prompt_tokens {prompt_tokens}\n"
    )
}
