use crabinfer_core::engine::CrabInferEngine;
use crabinfer_core::serving::worker_pool::WorkerPool;
use crabinfer_core::ModelInfo;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Shared application state
pub struct AppState {
    /// Legacy single-request engine (used when `serving_engine` is None).
    pub engine: Option<Arc<CrabInferEngine>>,
    /// Serializes legacy inference requests.
    pub inference_lock: Mutex<()>,
    /// New PagedAttention serving engine (when enabled, routes prefer this).
    /// Wraps one or more EngineHandle workers behind a WorkerPool for
    /// round-robin request distribution.
    pub serving_engine: Option<WorkerPool>,
    /// Cached model info (set after engine loads)
    pub model_info: ModelInfo,
    /// Model identifier for API responses
    pub model_id: String,
    /// Server start time (unix epoch seconds)
    pub created_at: u64,
    /// Metrics counters
    pub metrics: ServerMetrics,
    /// Optional chat template override (architecture name or file path).
    pub chat_template: Option<String>,
}

// ─── Histogram ────────────────────────────────────────────────────────────

/// Lightweight histogram for latency tracking (no external deps).
///
/// Uses fixed Prometheus-style buckets. Thread-safe via atomics.
pub struct Histogram {
    /// Bucket upper bounds in seconds.
    bucket_bounds: &'static [f64],
    /// Cumulative counts per bucket + one for +Inf.
    bucket_counts: Vec<AtomicU64>,
    /// Sum of all observed values (stored as u64 microseconds).
    sum_us: AtomicU64,
    /// Total number of observations.
    count: AtomicU64,
}

/// Default latency buckets (in seconds): 5ms → 120s.
const LATENCY_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0,
];

/// Fine-grained buckets for inter-token latency (in seconds): 1ms → 500ms.
const ITL_BUCKETS: &[f64] = &[
    0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.2, 0.5,
];

impl Histogram {
    pub fn new(buckets: &'static [f64]) -> Self {
        let bucket_counts = (0..=buckets.len())
            .map(|_| AtomicU64::new(0))
            .collect();
        Self {
            bucket_bounds: buckets,
            bucket_counts,
            sum_us: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    /// Record a value in seconds.
    pub fn observe(&self, value_secs: f64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_us
            .fetch_add((value_secs * 1_000_000.0) as u64, Ordering::Relaxed);

        // Increment the first bucket where value <= bound
        for (i, &bound) in self.bucket_bounds.iter().enumerate() {
            if value_secs <= bound {
                self.bucket_counts[i].fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        // +Inf bucket
        self.bucket_counts[self.bucket_bounds.len()].fetch_add(1, Ordering::Relaxed);
    }

    /// Format as Prometheus histogram lines.
    pub fn to_prometheus(&self, name: &str, help: &str) -> String {
        let mut out = format!(
            "# HELP {name} {help}\n# TYPE {name} histogram\n"
        );
        let mut cumulative = 0u64;
        for (i, &bound) in self.bucket_bounds.iter().enumerate() {
            cumulative += self.bucket_counts[i].load(Ordering::Relaxed);
            out.push_str(&format!(
                "{name}_bucket{{le=\"{bound}\"}} {cumulative}\n"
            ));
        }
        cumulative += self.bucket_counts[self.bucket_bounds.len()].load(Ordering::Relaxed);
        out.push_str(&format!("{name}_bucket{{le=\"+Inf\"}} {cumulative}\n"));
        let sum = self.sum_us.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        let count = self.count.load(Ordering::Relaxed);
        out.push_str(&format!("{name}_sum {sum:.6}\n"));
        out.push_str(&format!("{name}_count {count}\n"));
        out
    }
}

// ─── ServerMetrics ────────────────────────────────────────────────────────

/// Lightweight atomic counters and histograms for Prometheus metrics.
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
    /// Currently running requests.
    pub requests_running: AtomicU64,

    /// End-to-end request latency (seconds).
    pub request_latency: Histogram,
    /// Time to first token (seconds).
    pub ttft: Histogram,
    /// Inter-token latency (seconds).
    pub itl: Histogram,
}

impl ServerMetrics {
    pub fn new() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            requests_success: AtomicU64::new(0),
            requests_error: AtomicU64::new(0),
            tokens_generated: AtomicU64::new(0),
            prompt_tokens: AtomicU64::new(0),
            requests_running: AtomicU64::new(0),
            request_latency: Histogram::new(LATENCY_BUCKETS),
            ttft: Histogram::new(LATENCY_BUCKETS),
            itl: Histogram::new(ITL_BUCKETS),
        }
    }

    pub fn inc_request(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.requests_running.fetch_add(1, Ordering::Relaxed);
    }

    pub fn dec_running(&self) {
        self.requests_running.fetch_sub(1, Ordering::Relaxed);
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
