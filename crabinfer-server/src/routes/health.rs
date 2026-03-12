use crate::state::AppState;
use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde_json::{json, Value};
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// GET /health — detailed server status.
///
/// Returns 200 with JSON including model info, KV cache usage, queue depth,
/// and uptime. Compatible with Kubernetes liveness probes.
pub async fn health(State(state): State<Arc<AppState>>) -> Json<Value> {
    let uptime_seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
        .saturating_sub(state.created_at);

    let mut response = json!({
        "status": "ok",
        "model": state.model_id,
        "uptime_seconds": uptime_seconds,
    });

    // Add serving engine metrics when available
    if let Some(ref engine) = state.serving_engine {
        let kv_used = engine.kv_blocks_used();
        let kv_total = engine.kv_blocks_total();
        let kv_usage = engine.kv_cache_usage();
        let in_flight = engine.in_flight_count();

        let obj = response.as_object_mut().unwrap();
        obj.insert("kv_cache_usage".into(), json!(kv_usage));
        obj.insert("kv_cache_blocks_used".into(), json!(kv_used));
        obj.insert("kv_cache_blocks_total".into(), json!(kv_total));
        obj.insert("queue_depth".into(), json!(in_flight));
    }

    Json(response)
}

/// GET /ready — Kubernetes readiness probe.
///
/// Returns 200 when the server is ready to accept requests.
/// Returns 503 if the engine is not yet loaded or has been shut down.
pub async fn ready(State(state): State<Arc<AppState>>) -> StatusCode {
    // Check if we have at least one engine available
    let has_engine = state.engine.is_some() || state.serving_engine.is_some();
    if has_engine {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

/// GET /metrics — Prometheus text exposition format
pub async fn metrics(State(state): State<Arc<AppState>>) -> String {
    let m = &state.metrics;
    let requests_total = m.requests_total.load(Ordering::Relaxed);
    let requests_success = m.requests_success.load(Ordering::Relaxed);
    let requests_error = m.requests_error.load(Ordering::Relaxed);
    let tokens_generated = m.tokens_generated.load(Ordering::Relaxed);
    let prompt_tokens = m.prompt_tokens.load(Ordering::Relaxed);
    let requests_running = m.requests_running.load(Ordering::Relaxed);

    let mut out = format!(
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
         crabinfer_prompt_tokens {prompt_tokens}\n\
         # HELP crabinfer_requests_running Number of currently running requests.\n\
         # TYPE crabinfer_requests_running gauge\n\
         crabinfer_requests_running {requests_running}\n"
    );

    // Latency histograms
    out.push_str(&m.request_latency.to_prometheus(
        "crabinfer_request_latency_seconds",
        "End-to-end request latency in seconds.",
    ));
    out.push_str(&m.ttft.to_prometheus(
        "crabinfer_time_to_first_token_seconds",
        "Time to first token in seconds.",
    ));
    out.push_str(&m.itl.to_prometheus(
        "crabinfer_inter_token_latency_seconds",
        "Inter-token latency in seconds.",
    ));

    // KV cache metrics (only when serving engine is active)
    if let Some(ref engine) = state.serving_engine {
        let kv_used = engine.kv_blocks_used();
        let kv_total = engine.kv_blocks_total();
        let kv_usage = engine.kv_cache_usage();
        let in_flight = engine.in_flight_count();
        out.push_str(&format!(
            "# HELP crabinfer_kv_cache_blocks_used Number of KV cache blocks in use.\n\
             # TYPE crabinfer_kv_cache_blocks_used gauge\n\
             crabinfer_kv_cache_blocks_used {kv_used}\n\
             # HELP crabinfer_kv_cache_blocks_total Total KV cache blocks allocated.\n\
             # TYPE crabinfer_kv_cache_blocks_total gauge\n\
             crabinfer_kv_cache_blocks_total {kv_total}\n\
             # HELP crabinfer_kv_cache_usage_ratio KV cache usage ratio (0.0-1.0).\n\
             # TYPE crabinfer_kv_cache_usage_ratio gauge\n\
             crabinfer_kv_cache_usage_ratio {kv_usage:.4}\n\
             # HELP crabinfer_in_flight_requests Number of in-flight inference requests.\n\
             # TYPE crabinfer_in_flight_requests gauge\n\
             crabinfer_in_flight_requests {in_flight}\n"
        ));
        let prefix_hit_rate = engine.prefix_cache_hit_rate();
        let num_waiting = engine.num_waiting();
        out.push_str(&format!(
            "# HELP crabinfer_prefix_cache_hit_rate Prefix cache hit rate (0.0-1.0).\n\
             # TYPE crabinfer_prefix_cache_hit_rate gauge\n\
             crabinfer_prefix_cache_hit_rate {prefix_hit_rate:.4}\n\
             # HELP crabinfer_requests_waiting Number of requests waiting in scheduler queue.\n\
             # TYPE crabinfer_requests_waiting gauge\n\
             crabinfer_requests_waiting {num_waiting}\n"
        ));
    }

    // Tokens per second (computed gauge)
    let uptime_seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
        .saturating_sub(state.created_at);
    if uptime_seconds > 0 {
        let tps = tokens_generated as f64 / uptime_seconds as f64;
        out.push_str(&format!(
            "# HELP crabinfer_tokens_per_second Average tokens generated per second.\n\
             # TYPE crabinfer_tokens_per_second gauge\n\
             crabinfer_tokens_per_second {tps:.2}\n"
        ));
    }

    out
}
