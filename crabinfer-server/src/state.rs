use crabinfer_core::engine::CrabInferEngine;
use crabinfer_core::ModelInfo;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Shared application state
pub struct AppState {
    pub engine: Arc<CrabInferEngine>,
    /// Serializes inference requests (engine supports one streaming session at a time)
    pub inference_lock: Mutex<()>,
    /// Cached model info (set after engine loads)
    pub model_info: ModelInfo,
    /// Model identifier for API responses
    pub model_id: String,
    /// Server start time (unix epoch seconds)
    pub created_at: u64,
    /// Metrics counters
    pub metrics: ServerMetrics,
}

/// Lightweight atomic counters for Prometheus metrics.
pub struct ServerMetrics {
    /// Total requests received (all endpoints).
    pub requests_total: AtomicU64,
    /// Total successful responses (2xx).
    pub requests_success: AtomicU64,
    /// Total error responses (4xx/5xx).
    pub requests_error: AtomicU64,
    /// Total tokens generated.
    pub tokens_generated: AtomicU64,
    /// Total prompt tokens processed.
    pub prompt_tokens: AtomicU64,
}

impl ServerMetrics {
    pub fn new() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            requests_success: AtomicU64::new(0),
            requests_error: AtomicU64::new(0),
            tokens_generated: AtomicU64::new(0),
            prompt_tokens: AtomicU64::new(0),
        }
    }

    pub fn inc_request(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_success(&self) {
        self.requests_success.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_error(&self) {
        self.requests_error.fetch_add(1, Ordering::Relaxed);
    }

    pub fn add_tokens(&self, prompt: u64, generated: u64) {
        self.prompt_tokens.fetch_add(prompt, Ordering::Relaxed);
        self.tokens_generated.fetch_add(generated, Ordering::Relaxed);
    }
}
