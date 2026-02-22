use crabinfer_server::ServerConfig;

pub fn run(model: &str, host: &str, port: u16, context_length: u32, cpu: bool, advertise: bool) {
    let config = ServerConfig {
        model_path: model.to_string(),
        host: host.to_string(),
        port,
        context_length,
        cpu,
        advertise,
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
