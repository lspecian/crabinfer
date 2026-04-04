use clap::Parser;
use crabinfer_server::config::{self, CliOverrides};
use std::path::Path;

#[derive(Parser)]
#[command(name = "crabinfer-server", about = "CrabInfer OpenAI/Anthropic-compatible API server")]
struct Cli {
    /// Path to a crabinfer.toml configuration file
    #[arg(long)]
    config: Option<String>,

    /// Path to a GGUF model file or HuggingFace safetensors directory
    #[arg(long)]
    model: Option<String>,

    /// Port to listen on
    #[arg(long)]
    port: Option<u16>,

    /// Host to bind to
    #[arg(long)]
    host: Option<String>,

    /// Context length (max tokens the model can see)
    #[arg(long)]
    context_length: Option<u32>,

    /// Disable Metal GPU acceleration (CPU only)
    #[arg(long)]
    cpu: bool,

    /// Advertise via Bonjour/mDNS for LAN discovery (macOS only)
    #[arg(long)]
    advertise: bool,

    /// Use the PagedAttention serving engine with continuous batching
    #[arg(long)]
    serving: bool,

    /// Path to a draft model GGUF for speculative decoding (requires --serving)
    #[arg(long)]
    draft_model: Option<String>,

    /// Number of draft tokens per speculative step
    #[arg(long)]
    num_draft_tokens: Option<u32>,

    /// Disable CUDA graphs and use eager execution (for debugging)
    #[arg(long)]
    enforce_eager: bool,

    /// Fraction of GPU memory to use for KV cache (0.0-1.0)
    #[arg(long)]
    gpu_memory_utilization: Option<f64>,

    /// Maximum concurrent sequences
    #[arg(long)]
    max_num_seqs: Option<usize>,

    /// Maximum tokens per scheduling step
    #[arg(long)]
    max_num_batched_tokens: Option<usize>,

    /// Disable prefix caching
    #[arg(long)]
    disable_prefix_cache: bool,

    /// Weight quantization method: none, int8 (W8A16), gptq, awq
    #[arg(long)]
    quantization: Option<String>,

    /// KV cache data type: auto, fp16, bf16
    #[arg(long)]
    kv_cache_dtype: Option<String>,

    /// Model weight dtype: auto, f32, f16, bf16 (default: auto)
    #[arg(long, help = "Model weight dtype: auto, f32, f16, bf16 (default: auto)")]
    dtype: Option<String>,

    /// Maximum model context length (overrides model default)
    #[arg(long)]
    max_model_len: Option<usize>,

    /// Chat template override: architecture name (chatml, llama3, phi3, gemma) or template file path
    #[arg(long)]
    chat_template: Option<String>,

    /// CPU swap space for KV cache in GiB (0 = disabled)
    #[arg(long)]
    swap_space: Option<f64>,

    /// Number of inference workers (default: 1)
    #[arg(long)]
    workers: Option<usize>,

    /// Enable LoRA adapter serving (requires --serving)
    #[arg(long)]
    enable_lora: bool,

    /// Maximum number of LoRA adapters to keep in GPU memory simultaneously (default: 4)
    #[arg(long)]
    max_loras: Option<usize>,

    /// Pre-register LoRA adapter modules: name1=path1,name2=path2
    #[arg(long)]
    lora_modules: Option<String>,

    /// Tokens per KV cache block (must be power of 2 between 8 and 64, default 16)
    #[arg(long)]
    block_size: Option<usize>,

    /// Number of GPUs for tensor parallelism (default: 1 = no TP)
    #[arg(long)]
    tensor_parallel_size: Option<usize>,

    /// Number of pipeline parallel stages (default: 1 = disabled)
    #[arg(long)]
    pipeline_parallel_stages: Option<usize>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    // 1. Load TOML config (explicit path or ./crabinfer.toml)
    let toml_cfg = match config::load_config(cli.config.as_deref().map(Path::new)) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("Config error: {e}");
            std::process::exit(1);
        }
    };

    // 2. Convert TOML to ServerConfig (fills defaults for None fields)
    let mut server_config = toml_cfg.to_server_config();

    // 3. Apply env var overrides (env > TOML)
    config::apply_env_overrides(&mut server_config);

    // 4. Apply CLI overrides (CLI > env > TOML)
    // Boolean flags: only override if the user actually passed them (true).
    // clap sets them to false by default, so we treat false as "not passed".
    let overrides = CliOverrides {
        model: cli.model,
        host: cli.host,
        port: cli.port,
        context_length: cli.context_length,
        cpu: if cli.cpu { Some(true) } else { None },
        advertise: if cli.advertise { Some(true) } else { None },
        serving: if cli.serving { Some(true) } else { None },
        draft_model: cli.draft_model,
        num_draft_tokens: cli.num_draft_tokens,
        enforce_eager: if cli.enforce_eager { Some(true) } else { None },
        gpu_memory_utilization: cli.gpu_memory_utilization,
        max_num_seqs: cli.max_num_seqs,
        max_num_batched_tokens: cli.max_num_batched_tokens,
        disable_prefix_cache: if cli.disable_prefix_cache { Some(true) } else { None },
        quantization: cli.quantization,
        kv_cache_dtype: cli.kv_cache_dtype,
        dtype: cli.dtype,
        max_model_len: cli.max_model_len,
        chat_template: cli.chat_template,
        swap_space: cli.swap_space,
        workers: cli.workers,
        enable_lora: if cli.enable_lora { Some(true) } else { None },
        max_loras: cli.max_loras,
        lora_modules: cli.lora_modules,
        block_size: cli.block_size,
        tensor_parallel_size: cli.tensor_parallel_size,
        pipeline_parallel_stages: cli.pipeline_parallel_stages,
    };
    config::apply_cli_overrides(&mut server_config, &overrides);

    // Validate: model path is required (from TOML, env, or CLI)
    if server_config.model_path.is_empty() {
        eprintln!("Error: --model is required (or set 'model' in crabinfer.toml)");
        std::process::exit(1);
    }

    if let Err(e) = crabinfer_server::run_server(server_config).await {
        eprintln!("Server error: {}", e);
        std::process::exit(1);
    }
}
