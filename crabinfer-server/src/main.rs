use clap::Parser;
use crabinfer_server::{ServerConfig, run_server};

#[derive(Parser)]
#[command(name = "crabinfer-server", about = "CrabInfer OpenAI/Anthropic-compatible API server")]
struct Cli {
    /// Path to a GGUF model file
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
    };

    if let Err(e) = run_server(config).await {
        eprintln!("Server error: {}", e);
        std::process::exit(1);
    }
}
