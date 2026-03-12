use crabinfer_server::ServerConfig;

pub fn run(
    model: &str,
    host: &str,
    port: u16,
    context_length: u32,
    cpu: bool,
    advertise: bool,
    serving: bool,
    draft_model: Option<String>,
    num_draft_tokens: u32,
    enforce_eager: bool,
    gpu_memory_utilization: f64,
    max_num_seqs: usize,
    max_num_batched_tokens: usize,
    disable_prefix_cache: bool,
    quantization: &str,
    kv_cache_dtype: &str,
    max_model_len: Option<usize>,
    chat_template: Option<String>,
    swap_space: f64,
) {
    if draft_model.is_some() && !serving {
        eprintln!("Error: --draft-model requires --serving flag");
        std::process::exit(1);
    }

    let config = ServerConfig {
        model_path: model.to_string(),
        host: host.to_string(),
        port,
        context_length,
        cpu,
        advertise,
        serving,
        draft_model_path: draft_model,
        num_draft_tokens,
        enforce_eager,
        gpu_memory_utilization,
        max_num_seqs,
        max_num_batched_tokens,
        disable_prefix_cache,
        quantization: quantization.to_string(),
        kv_cache_dtype: kv_cache_dtype.to_string(),
        max_model_len,
        chat_template,
        swap_space,
    };

    let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| {
        eprintln!("Failed to create async runtime: {e}");
        std::process::exit(1);
    });

    if let Err(e) = rt.block_on(crabinfer_server::run_server(config)) {
        eprintln!("Server error: {e}");
        std::process::exit(1);
    }
}
