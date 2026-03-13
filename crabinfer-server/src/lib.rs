pub mod chat_template;
pub mod config;
pub mod error;
pub mod routes;
pub mod state;
pub mod types;

use crate::state::AppState;
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
    /// Disable CUDA graphs and use eager execution (for debugging).
    pub enforce_eager: bool,
    /// Fraction of GPU memory to use for KV cache (0.0-1.0, default 0.90).
    pub gpu_memory_utilization: f64,
    /// Maximum concurrent sequences in the serving engine (default: 64).
    pub max_num_seqs: usize,
    /// Maximum tokens per scheduling step (default: 2048).
    pub max_num_batched_tokens: usize,
    /// Disable prefix caching.
    pub disable_prefix_cache: bool,
    /// Weight quantization method (none, int8). Default: none.
    pub quantization: String,
    /// KV cache data type (auto, fp16, bf16). Default: auto.
    pub kv_cache_dtype: String,
    /// Maximum model context length. None = use model default.
    pub max_model_len: Option<usize>,
    /// Chat template override. Architecture name (e.g., "chatml", "llama3")
    /// or path to a template file. None = auto-detect from model metadata.
    pub chat_template: Option<String>,
    /// CPU swap space for KV cache in GiB (0.0 = disabled). Default: 0.0.
    /// When > 0, preempted sequences have their KV cache blocks swapped to
    /// pinned CPU memory instead of being discarded.
    pub swap_space: f64,
    /// Number of inference workers (default: 1). Multiple workers share model
    /// weights but each has its own KV cache and scheduler. Requests are
    /// distributed round-robin across workers.
    pub workers: usize,
    /// Enable LoRA adapter serving.
    pub enable_lora: bool,
    /// Maximum number of LoRA adapters to keep in GPU memory simultaneously.
    pub max_loras: usize,
    /// Pre-registered LoRA adapter modules: `"name1=path1,name2=path2"`.
    pub lora_modules: Option<String>,
    /// Tokens per KV cache block. Must be a power of 2 between 8 and 64.
    /// Default: 16.
    pub block_size: usize,
    /// Number of GPUs for tensor parallelism (default: 1 = no TP).
    /// Model weights are sharded across GPUs using NCCL for communication.
    pub tensor_parallel_size: usize,
    /// Number of pipeline parallel stages (default: 1 = disabled).
    /// Model layers are partitioned across stages for pipeline parallelism.
    pub pipeline_parallel_stages: usize,
}

/// Start the CrabInfer API server with the given configuration.
///
/// Loads the model, binds to `host:port`, and serves OpenAI + Anthropic
/// compatible endpoints until a SIGINT is received.
pub async fn run_server(mut config: ServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    config::apply_env_overrides(&mut config);
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
        chat_template: config.chat_template.clone(),
    });

    let app = routes::create_router(state.clone());
    let addr = format!("{}:{}", config.host, config.port);
    tracing::info!("Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;

    tracing::info!("Server ready. Endpoints:");
    tracing::info!("  GET  http://{}/health", addr);
    tracing::info!("  GET  http://{}/ready", addr);
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
        Option<crabinfer_core::serving::worker_pool::WorkerPool>,
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
///
/// When `config.workers > 1`, multiple engine workers are created sharing the
/// same model weights (candle tensors are Arc-based internally) but each with
/// its own KV cache. KV cache blocks are split evenly across workers.
fn load_serving_engine(
    config: &ServerConfig,
) -> Result<
    (
        Option<Arc<CrabInferEngine>>,
        Option<crabinfer_core::serving::worker_pool::WorkerPool>,
        ModelInfo,
        String,
    ),
    Box<dyn std::error::Error>,
> {
    use candle_core::quantized::gguf_file;
    use crabinfer_core::serving::engine_loop::{EngineHandle, ServingEngineConfig};
    use crabinfer_core::serving::models::load_model_from_gguf;
    use crabinfer_core::serving::quantization::QuantizationMethod;
    use crabinfer_core::serving::worker_pool::WorkerPool;

    // ── Resolve HF repo ID to local cache if needed ──
    let (resolved_model_path, hf_repo_id) =
        if crabinfer_core::serving::hub_download::is_hf_repo_id(&config.model_path) {
            tracing::info!("Detected HuggingFace repo ID: {}", config.model_path);
            let repo_id = config.model_path.clone();
            let local_dir = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    crabinfer_core::serving::hub_download::ensure_model_cached(&repo_id).await
                })
            })
            .map_err(|e| format!("Failed to download model from HuggingFace Hub: {e}"))?;
            tracing::info!("Model cached at: {}", local_dir.display());
            (local_dir, Some(config.model_path.clone()))
        } else {
            (std::path::PathBuf::from(&config.model_path), None)
        };

    let model_path = resolved_model_path.as_path();
    let is_safetensors = crabinfer_core::serving::safetensors_loader::is_safetensors_dir(model_path);

    // ── Select device ──
    let device = if config.cpu {
        candle_core::Device::Cpu
    } else {
        // Try CUDA first, then Metal, then CPU
        #[cfg(feature = "cuda")]
        {
            candle_core::Device::new_cuda(0).unwrap_or_else(|_| {
                tracing::warn!("CUDA device not available, falling back to CPU");
                candle_core::Device::Cpu
            })
        }
        #[cfg(all(not(feature = "cuda"), feature = "metal"))]
        {
            candle_core::Device::new_metal(0).unwrap_or(candle_core::Device::Cpu)
        }
        #[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
        {
            candle_core::Device::Cpu
        }
    };
    tracing::info!("Serving engine device: {:?}", device);

    let quantization: QuantizationMethod = config
        .quantization
        .parse()
        .map_err(|e: String| format!("Invalid quantization method: {e}"))?;
    if quantization != QuantizationMethod::None {
        tracing::info!("Weight quantization: {quantization}");
    }

    let kv_cache_dtype: crabinfer_core::serving::quantization::KVCacheDType = config
        .kv_cache_dtype
        .parse()
        .map_err(|e: String| format!("Invalid KV cache dtype: {e}"))?;

    // ── Load model + tokenizer + model info (branching on format) ──
    let (model, model_info, model_id, tokenizer, eos_token_id) = if is_safetensors {
        tracing::info!("Detected HuggingFace safetensors directory");
        tracing::info!("Loading paged-attention model from safetensors...");

        let model = crabinfer_core::serving::safetensors_loader::load_model_from_safetensors(
            model_path, &device, quantization,
        )
        .map_err(|e| format!("Failed to load safetensors model: {e}"))?;

        let model_config = model.config();
        // Use the HF repo ID as model name if we downloaded from Hub,
        // otherwise fall back to the directory name.
        let model_name = hf_repo_id.clone().unwrap_or_else(|| {
            model_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string()
        });
        // Detect architecture from config.json model_type for chat template selection
        let architecture = {
            let config_path = model_path.join("config.json");
            if config_path.exists() {
                let config_text = std::fs::read_to_string(&config_path).unwrap_or_default();
                let config_json: serde_json::Value =
                    serde_json::from_str(&config_text).unwrap_or_default();
                let model_type = config_json
                    .get("model_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("llama");
                // Map HF model_type to our chat template architecture names
                match model_type {
                    "qwen2" | "qwen2_5" => "qwen2".to_string(),
                    "qwen3" | "qwen3_5" => "qwen2".to_string(), // Qwen3 uses ChatML like Qwen2
                    "phi3" | "phi" => "phi3".to_string(),
                    "gemma" | "gemma2" | "gemma3" => "gemma3".to_string(),
                    "mistral" => "mistral".to_string(),
                    _ => "llama".to_string(),
                }
            } else {
                "llama".to_string()
            }
        };

        let model_info = ModelInfo {
            model_name: model_name.clone(),
            architecture,
            parameter_count: 0, // Not easily available from safetensors without counting
            active_parameter_count: 0,
            quantization: quantization.to_string(),
            file_size_bytes: 0,
            context_length: model_config.max_seq_len as u32,
            vocab_size: model_config.vocab_size as u32,
            is_moe: false,
            expert_count: 0,
            expert_used_count: 0,
        };

        // Tokenizer: expect tokenizer.json inside the model directory
        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            tracing::info!("Loading tokenizer from: {}", tokenizer_path.display());
            tokenizers::Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| format!("Failed to load tokenizer: {e}"))?
        } else {
            return Err(format!(
                "No tokenizer.json found in {}", model_path.display()
            ).into());
        };

        // Detect EOS from tokenizer (no GGUF metadata available)
        let eos = tokenizer.token_to_id("<|endoftext|>")
            .or_else(|| tokenizer.token_to_id("<|end|>"))
            .or_else(|| tokenizer.token_to_id("</s>"))
            .or_else(|| tokenizer.token_to_id("<|eot_id|>"))
            .or_else(|| tokenizer.token_to_id("<|end_of_text|>"))
            .unwrap_or(2);

        (model, model_info, model_name, tokenizer, eos)
    } else {
        // ── GGUF path ──
        let mut file = std::fs::File::open(model_path)
            .map_err(|e| format!("Failed to open model file: {e}"))?;
        let ct = gguf_file::Content::read(&mut file)
            .map_err(|e| format!("Failed to read GGUF content: {e}"))?;

        let model_info = extract_model_info(&ct, model_path)?;
        // Use the HF repo ID if we downloaded from Hub, otherwise use GGUF metadata name
        let model_id = hf_repo_id.clone().unwrap_or_else(|| model_info.model_name.clone());

        let gguf_eos = ct
            .metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(|v| v.to_u32().ok());

        tracing::info!("Loading paged-attention model from GGUF...");
        let model = load_model_from_gguf(ct, &mut file, &device, quantization)
            .map_err(|e| format!("Failed to load model: {e}"))?;

        let tokenizer = load_tokenizer(model_path)?;

        let eos = gguf_eos
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .or_else(|| tokenizer.token_to_id("<|end|>"))
            .or_else(|| tokenizer.token_to_id("</s>"))
            .or_else(|| tokenizer.token_to_id("<|eot_id|>"))
            .unwrap_or(2);

        (model, model_info, model_id, tokenizer, eos)
    };

    let model_config = model.config().clone();
    tracing::info!("EOS token ID: {eos_token_id}");

    // ── Calculate KV cache blocks based on available memory ──
    let total_blocks = estimate_kv_cache_blocks(&model_config, &device, config.block_size);

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
        "KV cache: {target_blocks} blocks ({} tokens capacity, block_size={})",
        target_blocks * config.block_size,
        config.block_size,
    );

    // ── Parse LoRA modules ──
    let lora_modules = if let Some(ref lora_str) = config.lora_modules {
        crabinfer_core::serving::lora::parse_lora_modules(lora_str)
            .map_err(|e| format!("Failed to parse --lora-modules: {e}"))?
    } else {
        Vec::new()
    };

    if config.enable_lora {
        tracing::info!(
            "LoRA serving enabled: max_loras={}, registered adapters={}",
            config.max_loras,
            lora_modules.len(),
        );
        for (name, path) in &lora_modules {
            tracing::info!("  LoRA adapter: {} -> {}", name, path);
        }
    }

    // ── Create engine ──
    let engine_config = ServingEngineConfig {
        max_num_seqs: config.max_num_seqs,
        max_num_batched_tokens: config.max_num_batched_tokens,
        block_size: config.block_size,
        num_kv_cache_blocks: Some(target_blocks),
        enable_prefix_cache: !config.disable_prefix_cache,
        enforce_eager: config.enforce_eager,
        gpu_memory_utilization: config.gpu_memory_utilization,
        quantization,
        kv_cache_dtype,
        max_model_len: config.max_model_len,
        enable_lora: config.enable_lora,
        max_loras: config.max_loras,
        lora_modules,
        tensor_parallel_size: config.tensor_parallel_size,
        pipeline_parallel_stages: config.pipeline_parallel_stages,
        ..ServingEngineConfig::default()
    };

    if config.pipeline_parallel_stages > 1 {
        tracing::info!(
            "Pipeline parallelism: {} stages (model layers will be partitioned)",
            config.pipeline_parallel_stages,
        );
    }

    if config.tensor_parallel_size > 1 {
        tracing::info!(
            "Tensor parallelism: {} GPUs (model weights will be sharded)",
            config.tensor_parallel_size,
        );
    }

    // ── Prepare for multi-worker mode ──
    // If more than 1 worker is requested, keep a clone-capable reference
    // to the model and clone tokenizer/device/config so we can create
    // additional workers after the first. clone_model() is cheap: candle
    // tensors share storage via Arc.
    let num_workers = config.workers.max(1);
    let model_for_cloning: Option<Box<dyn crabinfer_core::serving::models::ModelRunner>> =
        if num_workers > 1 {
            Some(model.clone_model())
        } else {
            None
        };

    // Pre-clone values consumed by EngineHandle::start() that we need
    // again for additional workers.
    let extra_tokenizer = if num_workers > 1 { Some(tokenizer.clone()) } else { None };
    let extra_device = if num_workers > 1 { Some(device.clone()) } else { None };

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

        let draft_model = load_model_from_gguf(draft_ct, &mut draft_file, &device, quantization)
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
            shared_kv_cache: false,
        };

        let speculative = SpeculativeState::new(
            draft_model,
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
            model,
            tokenizer,
            eos_token_id,
            device,
            engine_config.clone(),
            speculative,
        )
        .map_err(|e| format!("Failed to start serving engine with speculative: {e}"))?
    } else {
        EngineHandle::start(
            model,
            tokenizer,
            eos_token_id,
            device,
            engine_config.clone(),
        )
        .map_err(|e| format!("Failed to start serving engine: {e}"))?
    };

    // ── Wrap in WorkerPool ──
    // For num_workers == 1, this is a thin wrapper with no overhead.
    // For num_workers > 1, additional workers are created by cloning the
    // model (sharing Arc-based weight tensors) and starting separate
    // EngineHandle instances. Each worker gets its own KV cache, scheduler,
    // and engine thread. Requests are distributed round-robin.
    let mut handles = vec![handle];

    if num_workers > 1 {
        tracing::info!(
            "Multi-worker mode: {} workers requested (model weight sharing via Arc tensors)",
            num_workers,
        );

        // Each worker gets its own KV cache allocation (same size as worker 0).
        // All workers share model weights via tensor Arc cloning.
        let blocks_per_worker = target_blocks;
        let model_template = model_for_cloning
            .expect("model_for_cloning must be Some when num_workers > 1");
        let tok = extra_tokenizer.expect("extra_tokenizer must be Some when num_workers > 1");
        let dev = extra_device.expect("extra_device must be Some when num_workers > 1");

        for worker_idx in 1..num_workers {
            let cloned_model = model_template.clone_model();
            let worker_config = ServingEngineConfig {
                num_kv_cache_blocks: Some(blocks_per_worker),
                ..engine_config.clone()
            };

            let worker_handle = EngineHandle::start(
                cloned_model,
                tok.clone(),
                eos_token_id,
                dev.clone(),
                worker_config,
            )
            .map_err(|e| format!("Failed to start worker {worker_idx}: {e}"))?;

            handles.push(worker_handle);
            tracing::info!("Worker {worker_idx} started ({blocks_per_worker} KV blocks)");
        }
    }

    let pool = WorkerPool::new(handles);

    tracing::info!(
        "PagedAttention serving engine started ({} worker{})",
        pool.num_workers(),
        if pool.num_workers() == 1 { "" } else { "s" },
    );

    Ok((None, Some(pool), model_info, model_id))
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
    block_size: usize,
) -> usize {
    // Each block stores block_size tokens of KV data per layer per head
    // K: [num_kv_heads, head_size/x, block_size, x] per layer
    // V: [num_kv_heads, head_size, block_size] per layer
    // With F32: bytes_per_block = 4 * num_kv_heads * head_size * block_size * 2 (K+V) * num_layers
    let bytes_per_block_per_layer =
        4 * model_config.num_kv_heads * model_config.head_size * block_size * 2; // 4 bytes for F32, *2 for K+V
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
    // Wait for SIGINT (Ctrl+C) or SIGTERM (Docker/K8s)
    let ctrl_c = tokio::signal::ctrl_c();

    #[cfg(unix)]
    {
        let mut sigterm =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("failed to install SIGTERM handler");
        tokio::select! {
            _ = ctrl_c => {},
            _ = sigterm.recv() => {},
        }
    }
    #[cfg(not(unix))]
    {
        ctrl_c.await.expect("failed to install CTRL+C handler");
    }

    tracing::info!("Shutdown signal received, draining in-flight requests...");

    // Signal the engine to stop accepting new work
    if let Some(ref engine) = state.serving_engine {
        engine.shutdown();

        // Wait for in-flight requests to complete (up to 30s)
        let drain_deadline = tokio::time::Instant::now()
            + std::time::Duration::from_secs(30);
        loop {
            let in_flight = engine.in_flight_count();
            if in_flight == 0 {
                tracing::info!("All in-flight requests completed");
                break;
            }
            if tokio::time::Instant::now() >= drain_deadline {
                tracing::warn!(
                    "Drain timeout reached with {} requests still in-flight, forcing shutdown",
                    in_flight
                );
                break;
            }
            tracing::debug!("Waiting for {} in-flight requests to drain...", in_flight);
            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        }
        tracing::info!("Serving engine shut down");
    }
}

#[cfg(test)]
mod tests {
    use crabinfer_core::serving::hub_download::is_hf_repo_id;

    /// Verify is_hf_repo_id returns true for HF repo IDs, triggering the download branch.
    #[test]
    fn test_hf_repo_id_triggers_download_branch() {
        // Verify is_hf_repo_id returns true for repo IDs
        assert!(is_hf_repo_id("meta-llama/Llama-3.1-8B-Instruct-GPTQ"));
        assert!(is_hf_repo_id("TheBloke/Mistral-7B-GPTQ"));
        // Verify is_hf_repo_id returns false for local paths
        assert!(!is_hf_repo_id("/models/llama"));
        assert!(!is_hf_repo_id("./models/llama"));
        assert!(!is_hf_repo_id("model.gguf"));
    }
}
