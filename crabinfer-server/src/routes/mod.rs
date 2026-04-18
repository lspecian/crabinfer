pub mod anthropic;
pub mod embeddings;
pub mod guided;
pub mod health;
pub mod openai;
pub mod tokenize;

use crate::state::AppState;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use axum::Router;
use std::sync::Arc;
use tower_http::cors::CorsLayer;

/// Maximum request body size: 1MB (prevents trivial DoS from huge payloads).
const MAX_REQUEST_BODY_BYTES: usize = 1024 * 1024;

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        // Health, readiness & metrics
        .route("/health", get(health::health))
        .route("/ready", get(health::ready))
        .route("/metrics", get(health::metrics))
        // OpenAI-compatible
        .route("/v1/models", get(openai::list_models))
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/completions", post(openai::completions))
        // Guided decoding
        .route("/v1/guided/completions", post(guided::guided_completions))
        .route("/v1/embeddings", post(embeddings::create_embeddings))
        // Token counting utilities
        .route("/v1/tokenize", post(tokenize::tokenize))
        .route("/v1/detokenize", post(tokenize::detokenize))
        // Anthropic-compatible
        .route("/v1/messages", post(anthropic::messages))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES))
        .layer(CorsLayer::permissive())
        .with_state(state)
}
