use crabinfer_server::config::{self, CliOverrides};
use std::path::Path;

pub fn run(
    config_path: Option<&str>,
    model: Option<&str>,
    host: Option<&str>,
    port: Option<u16>,
    context_length: Option<u32>,
    cpu: bool,
    advertise: bool,
    serving: bool,
    draft_model: Option<String>,
    num_draft_tokens: Option<u32>,
    enforce_eager: bool,
    gpu_memory_utilization: Option<f64>,
    max_num_seqs: Option<usize>,
    max_num_batched_tokens: Option<usize>,
    disable_prefix_cache: bool,
    quantization: Option<&str>,
    kv_cache_dtype: Option<&str>,
    max_model_len: Option<usize>,
    chat_template: Option<String>,
    swap_space: Option<f64>,
    workers: Option<usize>,
    enable_lora: bool,
    max_loras: Option<usize>,
    lora_modules: Option<String>,
    block_size: Option<usize>,
    tensor_parallel_size: Option<usize>,
    pipeline_parallel_stages: Option<usize>,
) {
    // 1. Load TOML config
    let toml_cfg = match config::load_config(config_path.map(Path::new)) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("Config error: {e}");
            std::process::exit(1);
        }
    };

    // 2. Convert TOML to ServerConfig (fills defaults)
    let mut server_config = toml_cfg.to_server_config();

    // 3. Apply env var overrides
    config::apply_env_overrides(&mut server_config);

    // 4. Apply CLI overrides
    let overrides = CliOverrides {
        model: model.map(|s| s.to_string()),
        host: host.map(|s| s.to_string()),
        port,
        context_length,
        cpu: if cpu { Some(true) } else { None },
        advertise: if advertise { Some(true) } else { None },
        serving: if serving { Some(true) } else { None },
        draft_model,
        num_draft_tokens,
        enforce_eager: if enforce_eager { Some(true) } else { None },
        gpu_memory_utilization,
        max_num_seqs,
        max_num_batched_tokens,
        disable_prefix_cache: if disable_prefix_cache { Some(true) } else { None },
        quantization: quantization.map(|s| s.to_string()),
        kv_cache_dtype: kv_cache_dtype.map(|s| s.to_string()),
        dtype: None,
        max_model_len,
        chat_template,
        swap_space,
        workers,
        enable_lora: if enable_lora { Some(true) } else { None },
        max_loras,
        lora_modules,
        block_size,
        tensor_parallel_size,
        pipeline_parallel_stages,
    };
    config::apply_cli_overrides(&mut server_config, &overrides);

    // Validate
    if server_config.model_path.is_empty() {
        eprintln!("Error: --model is required (or set 'model' in crabinfer.toml)");
        std::process::exit(1);
    }

    if server_config.draft_model_path.is_some() && !server_config.serving {
        eprintln!("Error: --draft-model requires --serving flag");
        std::process::exit(1);
    }

    if server_config.enable_lora && !server_config.serving {
        eprintln!("Error: --enable-lora requires --serving flag");
        std::process::exit(1);
    }

    // Validate block size: must be a power of 2 between 8 and 64
    let bs = server_config.block_size;
    if !bs.is_power_of_two() || bs < 8 || bs > 64 {
        eprintln!("Error: --block-size must be a power of 2 between 8 and 64 (got {bs})");
        std::process::exit(1);
    }

    // Validate tensor parallel size
    let tp = server_config.tensor_parallel_size;
    if tp == 0 {
        eprintln!("Error: --tensor-parallel-size must be >= 1 (got {tp})");
        std::process::exit(1);
    }
    if tp > 1 && !server_config.serving {
        eprintln!("Error: --tensor-parallel-size > 1 requires --serving flag");
        std::process::exit(1);
    }
    if tp > 1 && !tp.is_power_of_two() {
        eprintln!("Error: --tensor-parallel-size must be a power of 2 (got {tp})");
        std::process::exit(1);
    }

    // Validate pipeline parallel stages
    let pp = server_config.pipeline_parallel_stages;
    if pp == 0 {
        eprintln!("Error: --pipeline-parallel-stages must be >= 1 (got {pp})");
        std::process::exit(1);
    }
    if pp > 1 && !server_config.serving {
        eprintln!("Error: --pipeline-parallel-stages > 1 requires --serving flag");
        std::process::exit(1);
    }

    let rt = tokio::runtime::Runtime::new().unwrap_or_else(|e| {
        eprintln!("Failed to create async runtime: {e}");
        std::process::exit(1);
    });

    if let Err(e) = rt.block_on(crabinfer_server::run_server(server_config)) {
        eprintln!("Server error: {e}");
        std::process::exit(1);
    }
}
