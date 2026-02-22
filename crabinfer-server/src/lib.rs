pub mod chat_template;
pub mod error;
pub mod routes;
pub mod state;
pub mod types;

use crate::state::AppState;
use crabinfer_core::{EngineConfig, engine::CrabInferEngine};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Configuration for starting the inference server.
pub struct ServerConfig {
    pub model_path: String,
    pub host: String,
    pub port: u16,
    pub context_length: u32,
    pub cpu: bool,
    /// Advertise the server via Bonjour/mDNS (macOS only).
    pub advertise: bool,
}

/// Start the CrabInfer API server with the given configuration.
///
/// Loads the model, binds to `host:port`, and serves OpenAI + Anthropic
/// compatible endpoints until a SIGINT is received.
pub async fn run_server(config: ServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    tracing::info!("Loading model: {}", config.model_path);

    let engine_config = EngineConfig {
        model_path: config.model_path.clone(),
        max_tokens: 4096,
        temperature: 0.7,
        top_p: 0.9,
        context_length: config.context_length,
        use_metal: !config.cpu,
        memory_limit_bytes: 0,
        metallib_path: String::new(),
    };

    let engine = CrabInferEngine::new(engine_config)?;

    let model_info = engine.model_info()?;
    let model_id = model_info.model_name.clone();
    tracing::info!(
        "Model loaded: {} ({}, {})",
        model_id,
        model_info.architecture,
        model_info.quantization
    );

    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let state = Arc::new(AppState {
        engine: Arc::new(engine),
        inference_lock: tokio::sync::Mutex::new(()),
        model_info,
        model_id,
        created_at,
        metrics: state::ServerMetrics::new(),
    });

    let app = routes::create_router(state.clone());
    let addr = format!("{}:{}", config.host, config.port);
    tracing::info!("Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;

    tracing::info!("Server ready. Endpoints:");
    tracing::info!("  GET  http://{}/health", addr);
    tracing::info!("  GET  http://{}/v1/models", addr);
    tracing::info!("  POST http://{}/v1/chat/completions", addr);
    tracing::info!("  POST http://{}/v1/messages", addr);
    tracing::info!("  GET  http://{}/metrics", addr);

    // Bonjour/mDNS advertisement (macOS only)
    #[cfg(target_os = "macos")]
    let _advertise_child = if config.advertise {
        let service_name = format!("CrabInfer - {}", state.model_id);
        tracing::info!("Advertising via Bonjour: {} on port {}", service_name, config.port);
        match std::process::Command::new("dns-sd")
            .args(["-R", &service_name, "_crabinfer._tcp", "local", &config.port.to_string()])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
        {
            Ok(child) => Some(child),
            Err(e) => {
                tracing::warn!("Failed to start Bonjour advertisement: {}", e);
                None
            }
        }
    } else {
        None
    };

    #[cfg(not(target_os = "macos"))]
    if config.advertise {
        tracing::warn!("Bonjour advertisement is only supported on macOS");
    }

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    // Clean up Bonjour advertisement subprocess
    #[cfg(target_os = "macos")]
    if let Some(mut child) = _advertise_child {
        let _ = child.kill();
    }

    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install CTRL+C handler");
    tracing::info!("Shutdown signal received");
}
