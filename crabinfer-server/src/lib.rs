pub mod chat_template;
pub mod error;
pub mod routes;
pub mod state;
pub mod types;

use crate::state::AppState;
use crabinfer_core::serving::models::ModelRunner;
use crabinfer_core::{EngineConfig, ModelInfo, engine::CrabInferEngine};
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
    /// Use the new PagedAttention serving engine instead of the legacy engine.
    pub serving: bool,
    /// Path to a draft model GGUF for speculative decoding (optional).
    /// Only used when `serving` is true.
    pub draft_model_path: Option<String>,
    /// Number of draft tokens per speculative step (default: 4).
    pub num_draft_tokens: u32,
}

/// Start the CrabInfer API server with the given configuration.
///
/// Loads the model, binds to `host:port`, and serves OpenAI + Anthropic
/// compatible endpoints until a SIGINT is received.
pub async fn run_server(config: ServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    tracing::info!("Loading model: {}", config.model_path);

    let created_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let (engine, serving_engine, model_info, model_id) = if config.serving {
        load_serving_engine(&config)?
    } else {
        load_legacy_engine(&config)?
    };

    tracing::info!(
        "Model loaded: {} ({}, {})",
        model_id,
        model_info.architecture,
        model_info.quantization
    );

    let state = Arc::new(AppState {
        engine,
        inference_lock: tokio::sync::Mutex::new(()),
        serving_engine,
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

    if config.serving {
        tracing::info!("  Mode: PagedAttention serving engine (continuous batching)");
    } else {
        tracing::info!("  Mode: Legacy single-request engine");
    }

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
        .with_graceful_shutdown(shutdown_signal(state.clone()))
        .await?;

    // Clean up Bonjour advertisement subprocess
    #[cfg(target_os = "macos")]
    if let Some(mut child) = _advertise_child {
        let _ = child.kill();
    }

    Ok(())
}

/// Load the legacy single-request engine.
fn load_legacy_engine(
    config: &ServerConfig,
) -> Result<
    (
        Option<Arc<CrabInferEngine>>,
        Option<crabinfer_core::serving::engine_loop::EngineHandle>,
        ModelInfo,
        String,
    ),
    Box<dyn std::error::Error>,
> {
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

    Ok((Some(Arc::new(engine)), None, model_info, model_id))
}

/// Load the GGUF model directly and create the PagedAttention serving engine.
fn load_serving_engine(
    config: &ServerConfig,
) -> Result<
    (
        Option<Arc<CrabInferEngine>>,
        Option<crabinfer_core::serving::engine_loop::EngineHandle>,
        ModelInfo,
        String,
    ),
    Box<dyn std::error::Error>,
> {
    use candle_core::quantized::gguf_file;
    use crabinfer_core::serving::engine_loop::{EngineHandle, ServingEngineConfig};
    use crabinfer_core::serving::models::llama::LlamaModel;

    let model_path = std::path::Path::new(&config.model_path);

    // ── Select device ──
    let device = if config.cpu {
        candle_core::Device::Cpu
    } else {
        candle_core::Device::new_metal(0).unwrap_or(candle_core::Device::Cpu)
    };
    tracing::info!("Serving engine device: {:?}", device);

    // ── Open GGUF file and load model ──
    let mut file = std::fs::File::open(model_path)
        .map_err(|e| format!("Failed to open model file: {e}"))?;
    let ct = gguf_file::Content::read(&mut file)
        .map_err(|e| format!("Failed to read GGUF content: {e}"))?;

    // Extract model info and EOS token from GGUF metadata before loading weights
    let model_info = extract_model_info(&ct, model_path)?;
    let model_id = model_info.model_name.clone();

    // Read EOS token ID from GGUF metadata (most models include this)
    let gguf_eos = ct
        .metadata
        .get("tokenizer.ggml.eos_token_id")
        .and_then(|v| v.to_u32().ok());

    tracing::info!("Loading paged-attention model from GGUF...");
    let model = LlamaModel::from_gguf(ct, &mut file, &device)
        .map_err(|e| format!("Failed to load model: {e}"))?;

    let model_config = model.config().clone();

    // ── Load tokenizer ──
    let tokenizer = load_tokenizer(model_path)?;

    // ── Detect EOS token ──
    // Prefer GGUF metadata, fall back to common EOS token strings
    let eos_token_id = gguf_eos
        .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
        .or_else(|| tokenizer.token_to_id("<|end|>"))
        .or_else(|| tokenizer.token_to_id("</s>"))
        .or_else(|| tokenizer.token_to_id("<|eot_id|>"))
        .unwrap_or(2);
    tracing::info!("EOS token ID: {eos_token_id}");

    // ── Calculate KV cache blocks based on available memory ──
    let total_blocks = estimate_kv_cache_blocks(&model_config, &device);

    // If a draft model is specified, split the KV cache budget
    let (target_blocks, draft_info) = if config.draft_model_path.is_some() {
        // 85% for target, 15% for draft
        let target = (total_blocks as f64 * 0.85) as usize;
        let draft = total_blocks - target;
        (target.max(16), Some(draft.max(4)))
    } else {
        (total_blocks, None)
    };

    tracing::info!(
        "KV cache: {target_blocks} blocks ({} tokens capacity)",
        target_blocks * crabinfer_core::serving::kernels::BLOCK_SIZE,
    );

    // ── Create engine ──
    let engine_config = ServingEngineConfig {
        max_num_seqs: 64,
        max_num_batched_tokens: 2048,
        num_kv_cache_blocks: target_blocks,
        enable_prefix_cache: true,
    };

    // ── Load draft model for speculative decoding (if configured) ──
    let handle = if let (Some(draft_path), Some(draft_blocks)) =
        (&config.draft_model_path, draft_info)
    {
        use crabinfer_core::serving::speculative::{SpeculativeConfig, SpeculativeState};

        let draft_model_path = std::path::Path::new(draft_path);
        tracing::info!("Loading draft model for speculative decoding: {}", draft_path);

        let mut draft_file = std::fs::File::open(draft_model_path)
            .map_err(|e| format!("Failed to open draft model file: {e}"))?;
        let draft_ct = gguf_file::Content::read(&mut draft_file)
            .map_err(|e| format!("Failed to read draft GGUF content: {e}"))?;

        let draft_model = LlamaModel::from_gguf(draft_ct, &mut draft_file, &device)
            .map_err(|e| format!("Failed to load draft model: {e}"))?;

        tracing::info!(
            "Draft model loaded: {} layers, {} KV heads, {} head_size",
            draft_model.config().num_layers,
            draft_model.config().num_kv_heads,
            draft_model.config().head_size,
        );

        let k = config.num_draft_tokens.max(1) as usize;
        let spec_config = SpeculativeConfig {
            num_draft_tokens: k,
            adaptive: true,
            min_draft_tokens: 1,
            max_draft_tokens: 8,
        };

        let speculative = SpeculativeState::new(
            Box::new(draft_model),
            &device,
            draft_blocks,
            spec_config,
        )
        .map_err(|e| format!("Failed to create speculative state: {e}"))?;

        tracing::info!(
            "Speculative decoding: K={}, draft KV blocks={}",
            k,
            draft_blocks,
        );

        EngineHandle::start_with_draft(
            Box::new(model),
            tokenizer,
            eos_token_id,
            device,
            engine_config,
            speculative,
        )
        .map_err(|e| format!("Failed to start serving engine with speculative: {e}"))?
    } else {
        EngineHandle::start(
            Box::new(model),
            tokenizer,
            eos_token_id,
            device,
            engine_config,
        )
        .map_err(|e| format!("Failed to start serving engine: {e}"))?
    };

    tracing::info!("PagedAttention serving engine started");

    Ok((None, Some(handle), model_info, model_id))
}

/// Extract ModelInfo from GGUF metadata.
fn extract_model_info(
    ct: &candle_core::quantized::gguf_file::Content,
    model_path: &std::path::Path,
) -> Result<ModelInfo, Box<dyn std::error::Error>> {
    let md = &ct.metadata;

    // Model name: try GGUF metadata, fall back to filename
    let model_name = md
        .get("general.name")
        .and_then(|v| v.to_string().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            model_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string()
        });

    // Architecture
    let architecture = md
        .get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "llama".to_string());

    // Quantization type from the general.file_type or from tensor inspection
    let quantization = md
        .get("general.file_type")
        .and_then(|v| v.to_u32().ok())
        .map(|ft| gguf_file_type_name(ft))
        .unwrap_or_else(|| "unknown".to_string());

    // Parameter count (approximate from element counts)
    let parameter_count: u64 = ct
        .tensor_infos
        .values()
        .map(|t| t.shape.elem_count() as u64)
        .sum();

    // File size
    let file_size_bytes = std::fs::metadata(model_path)
        .map(|m| m.len())
        .unwrap_or(0);

    // Context length
    let context_length = md
        .get(&format!("{architecture}.context_length"))
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(4096);

    // Vocab size
    let vocab_size = md
        .get(&format!("{architecture}.embedding_length"))
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(0); // Will be overridden by actual embedding table size

    Ok(ModelInfo {
        model_name,
        architecture,
        parameter_count,
        active_parameter_count: parameter_count, // Dense model: active == total
        quantization,
        file_size_bytes,
        context_length,
        vocab_size,
        is_moe: false,
        expert_count: 0,
        expert_used_count: 0,
    })
}

/// Map GGUF file_type integer to a human-readable quantization name.
fn gguf_file_type_name(ft: u32) -> String {
    match ft {
        0 => "F32".to_string(),
        1 => "F16".to_string(),
        2 => "Q4_0".to_string(),
        3 => "Q4_1".to_string(),
        7 => "Q8_0".to_string(),
        8 => "Q8_1".to_string(),
        10 => "Q4_K_S".to_string(),
        11 => "Q4_K_M".to_string(),
        12 => "Q5_K_S".to_string(),
        13 => "Q5_K_M".to_string(),
        14 => "Q6_K".to_string(),
        15 => "Q2_K".to_string(),
        16 => "Q3_K_S".to_string(),
        17 => "Q3_K_M".to_string(),
        18 => "Q3_K_L".to_string(),
        _ => format!("type_{ft}"),
    }
}

/// Load tokenizer from a tokenizer.json file adjacent to the model.
fn load_tokenizer(
    model_path: &std::path::Path,
) -> Result<tokenizers::Tokenizer, Box<dyn std::error::Error>> {
    let model_dir = model_path
        .parent()
        .ok_or("invalid model path")?;

    let model_stem = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default();

    // Search order: model-specific dirs first (with suffix stripping), then adjacent, then parent
    let stem_dirs = model_stem_to_dirs(model_stem);
    let mut candidates: Vec<std::path::PathBuf> = stem_dirs
        .iter()
        .map(|d| model_dir.join(d).join("tokenizer.json"))
        .collect();
    candidates.push(model_dir.join("tokenizer.json"));
    if let Some(parent) = model_dir.parent() {
        candidates.push(parent.join("tokenizer.json"));
    }

    for path in &candidates {
        if path.exists() {
            tracing::info!("Loading tokenizer from: {}", path.display());
            let tokenizer = tokenizers::Tokenizer::from_file(path)
                .map_err(|e| format!("Failed to load tokenizer: {e}"))?;
            return Ok(tokenizer);
        }
    }

    Err(format!(
        "No tokenizer.json found. Searched: {:?}",
        candidates
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
    )
    .into())
}

/// Convert a model filename stem to candidate tokenizer directory names.
/// e.g. "qwen2.5-7b-instruct-q4_k_m" -> ["qwen2.5-7b-instruct", "qwen2.5-7b"]
fn model_stem_to_dirs(stem: &str) -> Vec<String> {
    let s = stem.to_lowercase();
    // Strip common quantization suffixes
    let s = s
        .trim_end_matches("-gguf")
        .trim_end_matches(".gguf");
    // Strip quantization like -q4_k_m, -q5_k_s, -q8_0, etc.
    let base = if let Some(idx) = s.rfind("-q") {
        s[..idx].to_string()
    } else if let Some(idx) = s.rfind("-f16") {
        s[..idx].to_string()
    } else if let Some(idx) = s.rfind("-f32") {
        s[..idx].to_string()
    } else {
        s.to_string()
    };

    let mut candidates = vec![base.clone()];
    // Also try stripping common instruction-tuning suffixes
    for suffix in ["-instruct", "-chat", "-it", "-hf"] {
        if let Some(stripped) = base.strip_suffix(suffix) {
            candidates.push(stripped.to_string());
        }
    }
    candidates
}

/// Estimate the number of KV cache blocks based on model size and available memory.
fn estimate_kv_cache_blocks(
    model_config: &crabinfer_core::serving::models::ModelConfig,
    _device: &candle_core::Device,
) -> usize {
    use crabinfer_core::serving::kernels::BLOCK_SIZE;

    // Each block stores BLOCK_SIZE tokens of KV data per layer per head
    // K: [num_kv_heads, head_size/x, BLOCK_SIZE, x] per layer
    // V: [num_kv_heads, head_size, BLOCK_SIZE] per layer
    // With F32: bytes_per_block = 4 * num_kv_heads * head_size * BLOCK_SIZE * 2 (K+V) * num_layers
    let bytes_per_block_per_layer =
        4 * model_config.num_kv_heads * model_config.head_size * BLOCK_SIZE * 2; // 4 bytes for F32, *2 for K+V
    let bytes_per_block = bytes_per_block_per_layer * model_config.num_layers;

    // Query system memory and allocate a reasonable fraction for KV cache.
    // On Apple Silicon unified memory systems, model weights + KV cache + OS
    // all share the same physical RAM. Reserve:
    //   - Model weights (roughly file_size, already loaded)
    //   - OS + other apps (~4GB)
    //   - KV cache gets up to 25% of total physical RAM, capped at 8GB
    let total_mem = get_total_system_memory();
    let kv_cache_budget = if total_mem > 0 {
        let quarter = total_mem / 4;
        let cap = 8usize * 1024 * 1024 * 1024; // 8GB max
        let budget = quarter.min(cap);
        tracing::info!(
            "System memory: {:.1}GB, KV cache budget: {:.1}GB",
            total_mem as f64 / (1024.0 * 1024.0 * 1024.0),
            budget as f64 / (1024.0 * 1024.0 * 1024.0),
        );
        budget
    } else {
        // Fallback: 2GB if we can't detect system memory
        tracing::warn!("Could not detect system memory, using 2GB KV cache budget");
        2usize * 1024 * 1024 * 1024
    };

    let num_blocks = kv_cache_budget / bytes_per_block.max(1);

    // Clamp to reasonable range
    num_blocks.clamp(16, 8192)
}

/// Query total physical memory (platform-specific).
fn get_total_system_memory() -> usize {
    #[cfg(target_os = "macos")]
    {
        use std::mem;
        let mut size: u64 = 0;
        let mut len = mem::size_of::<u64>();
        let mib = [libc::CTL_HW, libc::HW_MEMSIZE];
        let ret = unsafe {
            libc::sysctl(
                mib.as_ptr() as *mut _,
                2,
                &mut size as *mut u64 as *mut _,
                &mut len as *mut _,
                std::ptr::null_mut(),
                0,
            )
        };
        if ret == 0 { size as usize } else { 0 }
    }

    #[cfg(target_os = "linux")]
    {
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) } as usize;
        page_size * pages
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        0 // Unknown platform
    }
}

async fn shutdown_signal(state: Arc<AppState>) {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install CTRL+C handler");
    tracing::info!("Shutdown signal received, draining in-flight requests...");

    // Signal the engine to stop accepting new work
    if let Some(ref engine) = state.serving_engine {
        engine.shutdown();

        // Give in-flight requests up to 10 seconds to complete
        let drain_deadline = tokio::time::Instant::now()
            + std::time::Duration::from_secs(10);
        loop {
            if tokio::time::Instant::now() >= drain_deadline {
                tracing::warn!("Drain timeout reached, forcing shutdown");
                break;
            }
            // Engine loop will exit on its own when shutdown flag is set
            // and all sequences finish. We just wait here.
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        tracing::info!("Serving engine shut down");
    }
}
