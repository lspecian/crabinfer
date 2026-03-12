use clap::Parser;
use crabinfer_server::{ServerConfig, run_server};

#[derive(Parser)]
#[command(name = "crabinfer-server", about = "CrabInfer OpenAI/Anthropic-compatible API server")]
struct Cli {
    /// Path to a GGUF model file or HuggingFace safetensors directory
    #[arg(long)]
    model: String,

    /// Port to listen on
    #[arg(long, default_value = "8080")]
    port: u16,

    /// Host to bind to
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Context length (max tokens the model can see)
    #[arg(long, default_value = "4096")]
    context_length: u32,

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

    /// Number of draft tokens per speculative step (default: 4)
    #[arg(long, default_value = "4")]
    num_draft_tokens: u32,

    /// Disable CUDA graphs and use eager execution (for debugging)
    #[arg(long)]
    enforce_eager: bool,

    /// Fraction of GPU memory to use for KV cache (0.0-1.0)
    #[arg(long, default_value = "0.90")]
    gpu_memory_utilization: f64,

    /// Maximum concurrent sequences (default: 64)
    #[arg(long, default_value = "64")]
    max_num_seqs: usize,

    /// Maximum tokens per scheduling step (default: 2048)
    #[arg(long, default_value = "2048")]
    max_num_batched_tokens: usize,

    /// Disable prefix caching
    #[arg(long)]
    disable_prefix_cache: bool,

    /// Weight quantization method: none, int8 (W8A16), gptq, awq
    #[arg(long, default_value = "none")]
    quantization: String,

    /// KV cache data type: auto, fp16, bf16
    #[arg(long, default_value = "auto")]
    kv_cache_dtype: String,

    /// Maximum model context length (overrides model default)
    #[arg(long)]
    max_model_len: Option<usize>,

    /// Chat template override: architecture name (chatml, llama3, phi3, gemma) or template file path
    #[arg(long)]
    chat_template: Option<String>,

    /// CPU swap space for KV cache in GiB (0 = disabled)
    #[arg(long, default_value = "0")]
    swap_space: f64,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    let config = ServerConfig {
        model_path: cli.model,
        host: cli.host,
        port: cli.port,
        context_length: cli.context_length,
        cpu: cli.cpu,
        advertise: cli.advertise,
        serving: cli.serving,
        draft_model_path: cli.draft_model,
        num_draft_tokens: cli.num_draft_tokens,
        enforce_eager: cli.enforce_eager,
        gpu_memory_utilization: cli.gpu_memory_utilization,
        max_num_seqs: cli.max_num_seqs,
        max_num_batched_tokens: cli.max_num_batched_tokens,
        disable_prefix_cache: cli.disable_prefix_cache,
        quantization: cli.quantization,
        kv_cache_dtype: cli.kv_cache_dtype,
        max_model_len: cli.max_model_len,
        chat_template: cli.chat_template,
        swap_space: cli.swap_space,
    };

    if let Err(e) = run_server(config).await {
        eprintln!("Server error: {}", e);
        std::process::exit(1);
    }
}
