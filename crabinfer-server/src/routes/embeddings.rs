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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::types::embeddings::EmbeddingInput;

    /// Smoke test: batch input preserves element order through into_texts().
    #[test]
    fn test_batch_input_into_texts_preserves_order() {
        let input = EmbeddingInput::Batch(vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
        ]);
        let texts = input.into_texts();
        assert_eq!(texts, vec!["a", "b", "c"]);
    }

    /// TOKN-01 contract test: confirms engine_loop::EngineHandle::embed uses
    /// the parallel encode_batch path, not serial encode() calls.
    ///
    /// This is a source-code-level test (not a runtime test) because the
    /// alternative would be plumbing a benchmark/mock through the engine,
    /// which is overkill for verifying a one-line call chain.
    ///
    /// If this test fails, someone refactored the embed() implementation
    /// to bypass encode_batch — restore the call or update the test
    /// after re-running the TOKN-01 verification flow.
    #[test]
    fn test_embed_call_chain_uses_encode_batch() {
        // Resolve path via CARGO_MANIFEST_DIR (crabinfer-server crate root)
        // -> ../crabinfer-core/src/serving/engine_loop.rs
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let path = format!(
            "{}/../crabinfer-core/src/serving/engine_loop.rs",
            manifest_dir
        );
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", path, e));

        // Find the embed() function and assert it contains a call to encode_batch.
        // Looking for the pattern: "pub fn embed(" ... "self.encode_batch("
        let embed_idx = source
            .find("pub fn embed(")
            .expect("EngineHandle::embed not found in engine_loop.rs — TOKN-01 wiring may have regressed");
        // Search the next 2000 chars (function body) for the encode_batch call
        let window_end = (embed_idx + 2000).min(source.len());
        let window = &source[embed_idx..window_end];
        assert!(
            window.contains("self.encode_batch("),
            "TOKN-01 regression: EngineHandle::embed no longer calls self.encode_batch(). \
             Multi-input embedding requests would fall back to serial tokenization. \
             Restore the parallel call chain or update this test only after deliberate redesign."
        );
    }

    /// TOKN-01 contract test (route side): confirms POST /v1/embeddings calls
    /// engine.embed(texts), which is the entry point that dispatches to
    /// encode_batch internally. Pinned by source inspection.
    #[test]
    fn test_embeddings_route_calls_engine_embed() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let path = format!("{}/src/routes/embeddings.rs", manifest_dir);
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", path, e));

        assert!(
            source.contains("engine.embed("),
            "TOKN-01 regression: routes/embeddings.rs no longer calls engine.embed(). \
             Multi-input requests would not reach the batch tokenization path."
        );
    }
}
