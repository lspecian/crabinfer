pub mod anthropic;
pub mod health;
pub mod openai;

use crate::state::AppState;
use axum::routing::{get, post};
use axum::Router;
use std::sync::Arc;
use tower_http::cors::CorsLayer;

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        // Health & metrics
        .route("/health", get(health::health))
        .route("/metrics", get(health::metrics))
        // OpenAI-compatible
        .route("/v1/models", get(openai::list_models))
        .route("/v1/chat/completions", post(openai::chat_completions))
        // Anthropic-compatible
        .route("/v1/messages", post(anthropic::messages))
        .layer(CorsLayer::permissive())
        .with_state(state)
}
