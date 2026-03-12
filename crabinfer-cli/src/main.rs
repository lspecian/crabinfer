mod cmd_assistant;
mod cmd_auth;
mod cmd_bench;
mod cmd_chat;
mod cmd_info;
mod cmd_mcp;
mod cmd_models;
mod cmd_run;
mod cmd_serve;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "crabinfer", about = "Local LLM inference on Apple Silicon")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show device capabilities and recommended models
    Info,

    /// One-shot inference
    Run {
        /// Path to GGUF model file
        #[arg(long)]
        model: String,

        /// Input prompt
        #[arg(long)]
        prompt: String,

        /// Maximum tokens to generate
        #[arg(long, default_value = "128")]
        max_tokens: u32,

        /// Sampling temperature
        #[arg(long, default_value = "0.7")]
        temperature: f32,

        /// Top-p sampling
        #[arg(long, default_value = "0.9")]
        top_p: f32,

        /// Context window size
        #[arg(long, default_value = "4096")]
        context_length: u32,

        /// Force CPU-only (skip Metal)
        #[arg(long)]
        cpu: bool,

        /// Stream tokens as they are generated
        #[arg(long)]
        stream: bool,
    },

    /// Interactive chat REPL
    Chat {
        /// Path to GGUF model file
        #[arg(long)]
        model: String,

        /// Maximum tokens per response
        #[arg(long, default_value = "512")]
        max_tokens: u32,

        /// Sampling temperature
        #[arg(long, default_value = "0.7")]
        temperature: f32,

        /// Context window size
        #[arg(long, default_value = "4096")]
        context_length: u32,

        /// Force CPU-only (skip Metal)
        #[arg(long)]
        cpu: bool,
    },

    /// Run CPU vs Metal benchmarks
    Bench {
        /// Path to GGUF model file
        #[arg(long)]
        model: String,

        /// Benchmark prompt
        #[arg(long, default_value = "Explain the theory of relativity in simple terms.")]
        prompt: String,

        /// Maximum tokens to generate
        #[arg(long, default_value = "128")]
        max_tokens: u32,

        /// Sampling temperature
        #[arg(long, default_value = "0.7")]
        temperature: f32,

        /// Context window size
        #[arg(long, default_value = "4096")]
        context_length: u32,
    },

    /// Manage API keys for cloud providers
    Auth {
        #[command(subcommand)]
        action: AuthAction,
    },

    /// Manage model catalog and downloads
    Models {
        #[command(subcommand)]
        action: ModelsAction,
    },

    /// Interactive AI assistant with tool calling
    Assistant {
        /// Cloud provider: openai, anthropic, google, ollama, vllm
        #[arg(long, default_value = "openai")]
        provider: String,

        /// Model to use (e.g., gpt-4o, claude-sonnet-4-20250514)
        #[arg(long, default_value = "gpt-4o")]
        model: String,

        /// Maximum tokens per response
        #[arg(long, default_value = "2048")]
        max_tokens: u32,

        /// Sampling temperature
        #[arg(long, default_value = "0.7")]
        temperature: f32,

        /// Data directory for state persistence
        #[arg(long)]
        data_dir: Option<String>,

        /// Knowledge files to index for RAG
        #[arg(long)]
        knowledge: Vec<String>,
    },

    /// Manage MCP (Model Context Protocol) servers
    Mcp {
        #[command(subcommand)]
        action: McpAction,
    },

    /// Start OpenAI/Anthropic-compatible API server
    Serve {
        /// Path to GGUF model file or HuggingFace safetensors directory
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

        /// CPU swap space for KV cache in GiB (0 = disabled). Default: 0
        #[arg(long, default_value = "0")]
        swap_space: f64,
    },
}

#[derive(Subcommand)]
enum AuthAction {
    /// Store an API key for a provider (openai, anthropic, google)
    Set {
        /// Provider name
        provider: String,
    },
    /// List configured providers
    List,
    /// Remove a stored API key
    Remove {
        /// Provider name
        provider: String,
    },
    /// Validate a stored API key
    Test {
        /// Provider name
        provider: String,
    },
}

#[derive(Subcommand)]
enum ModelsAction {
    /// List available models from the catalog
    List {
        /// Only show models compatible with this device
        #[arg(long)]
        compatible: bool,

        /// Filter by category: general, code, reasoning
        #[arg(long)]
        category: Option<String>,
    },
    /// Show recommended models for this device
    Recommend,
    /// Download a model from HuggingFace
    Pull {
        /// Model ID from the catalog
        id: String,
    },
    /// List downloaded models
    Downloaded,
    /// Remove a downloaded model
    Remove {
        /// Model ID to remove
        id: String,
    },
    /// Show storage information
    Storage,
}

#[derive(Subcommand)]
enum McpAction {
    /// List configured MCP servers
    List,
    /// Add an MCP server
    Add {
        /// Unique server name
        #[arg(long)]
        name: String,
        /// Transport: stdio or http
        #[arg(long)]
        transport: String,
        /// For stdio: command to run. For http: URL
        #[arg(long)]
        command: String,
        /// Command-line arguments (stdio only)
        #[arg(long)]
        args: Vec<String>,
        /// Description
        #[arg(long, default_value = "")]
        description: String,
    },
    /// Remove an MCP server
    Remove {
        /// Server name to remove
        name: String,
    },
    /// Enable a disabled server
    Enable {
        /// Server name
        name: String,
    },
    /// Disable a server
    Disable {
        /// Server name
        name: String,
    },
    /// Test connection to an MCP server
    Test {
        /// Server name
        name: String,
    },
    /// Run CrabInfer as an MCP server (stdio)
    Serve,
}

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Info => cmd_info::run(),
        Commands::Run {
            model,
            prompt,
            max_tokens,
            temperature,
            top_p,
            context_length,
            cpu,
            stream,
        } => cmd_run::run(&model, &prompt, max_tokens, temperature, top_p, context_length, cpu, stream),
        Commands::Chat {
            model,
            max_tokens,
            temperature,
            context_length,
            cpu,
        } => cmd_chat::run(&model, max_tokens, temperature, context_length, cpu),
        Commands::Bench {
            model,
            prompt,
            max_tokens,
            temperature,
            context_length,
        } => cmd_bench::run(&model, &prompt, max_tokens, temperature, context_length),
        Commands::Auth { action } => match action {
            AuthAction::Set { provider } => cmd_auth::set(&provider),
            AuthAction::List => cmd_auth::list(),
            AuthAction::Remove { provider } => cmd_auth::remove(&provider),
            AuthAction::Test { provider } => cmd_auth::test(&provider),
        },
        Commands::Models { action } => match action {
            ModelsAction::List { compatible, category } => {
                cmd_models::list(compatible, category.as_deref())
            }
            ModelsAction::Recommend => cmd_models::recommend(),
            ModelsAction::Pull { id } => cmd_models::pull(&id),
            ModelsAction::Downloaded => cmd_models::downloaded(),
            ModelsAction::Remove { id } => cmd_models::remove(&id),
            ModelsAction::Storage => cmd_models::storage(),
        },
        Commands::Assistant {
            provider,
            model,
            max_tokens,
            temperature,
            data_dir,
            knowledge,
        } => cmd_assistant::run(
            &provider,
            &model,
            max_tokens,
            temperature,
            data_dir.as_deref(),
            &knowledge,
        ),
        Commands::Mcp { action } => match action {
            McpAction::List => cmd_mcp::list(),
            McpAction::Add {
                name,
                transport,
                command,
                args,
                description,
            } => cmd_mcp::add(&name, &transport, &command, &args, &description),
            McpAction::Remove { name } => cmd_mcp::remove(&name),
            McpAction::Enable { name } => cmd_mcp::toggle(&name, true),
            McpAction::Disable { name } => cmd_mcp::toggle(&name, false),
            McpAction::Test { name } => cmd_mcp::test_server(&name),
            McpAction::Serve => cmd_mcp::serve(),
        },
        Commands::Serve {
            model,
            port,
            host,
            context_length,
            cpu,
            advertise,
            serving,
            draft_model,
            num_draft_tokens,
            enforce_eager,
            gpu_memory_utilization,
            max_num_seqs,
            max_num_batched_tokens,
            disable_prefix_cache,
            quantization,
            kv_cache_dtype,
            max_model_len,
            chat_template,
            swap_space,
        } => cmd_serve::run(
            &model,
            &host,
            port,
            context_length,
            cpu,
            advertise,
            serving,
            draft_model,
            num_draft_tokens,
            enforce_eager,
            gpu_memory_utilization,
            max_num_seqs,
            max_num_batched_tokens,
            disable_prefix_cache,
            &quantization,
            &kv_cache_dtype,
            max_model_len,
            chat_template,
            swap_space,
        ),
    }
}
