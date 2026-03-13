use crate::error::ServerError;
use crate::state::AppState;
use crate::types::embeddings::*;
use axum::extract::State;
use axum::Json;
use std::sync::Arc;

/// POST /v1/embeddings
///
/// OpenAI-compatible embeddings endpoint. Accepts single or batch string inputs
/// and returns embedding vectors.
pub async fn create_embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, ServerError> {
    state.metrics.inc_request();

    let engine = state
        .serving_engine
        .as_ref()
        .ok_or_else(|| ServerError::service_unavailable("Serving engine not available"))?;

    let texts = req.input.into_texts();

    if texts.is_empty() {
        state.metrics.dec_running();
        return Err(ServerError::bad_request("Input must not be empty"));
    }

    let (embeddings, token_counts) = engine.embed(texts).map_err(|e| {
        state.metrics.inc_error();
        state.metrics.dec_running();
        ServerError::internal(format!("Embedding failed: {e}"))
    })?;

    let prompt_tokens: u32 = token_counts.iter().sum();

    let data: Vec<EmbeddingObject> = embeddings
        .into_iter()
        .enumerate()
        .map(|(i, embedding)| EmbeddingObject {
            object: "embedding".to_string(),
            embedding,
            index: i,
        })
        .collect();

    state.metrics.dec_running();
    state.metrics.inc_success();
    state.metrics.add_tokens(prompt_tokens as u64, 0);

    Ok(Json(EmbeddingResponse {
        object: "list".to_string(),
        data,
        model: state.model_id.clone(),
        usage: EmbeddingUsage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    }))
}
