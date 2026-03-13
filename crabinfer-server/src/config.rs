//! TOML configuration file support for CrabInfer.
//!
//! Precedence (highest to lowest): CLI flags > env vars > TOML file > defaults.
//!
//! Usage:
//! ```ignore
//! let cfg = load_config(None)?;                    // try ./crabinfer.toml
//! let mut server = cfg.to_server_config();         // fill defaults
//! apply_env_overrides(&mut server);                // env > TOML
//! apply_cli_overrides(&mut server, &cli_overrides); // CLI > env > TOML
//! ```

use crate::ServerConfig;
use serde::Deserialize;
use std::path::Path;

/// Deserialized representation of a `crabinfer.toml` file.
///
/// Every field is `Option<T>` so partial configs work naturally: omitted
/// keys stay `None` and get filled with defaults in [`to_server_config`].
#[derive(Debug, Deserialize, Default, Clone)]
#[serde(default)]
pub struct CrabInferConfig {
    pub model: Option<String>,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub context_length: Option<u32>,
    pub cpu: Option<bool>,
    pub advertise: Option<bool>,
    pub serving: Option<bool>,
    pub draft_model: Option<String>,
    pub num_draft_tokens: Option<u32>,
    pub enforce_eager: Option<bool>,
    pub gpu_memory_utilization: Option<f64>,
    pub max_num_seqs: Option<usize>,
    pub max_num_batched_tokens: Option<usize>,
    pub disable_prefix_cache: Option<bool>,
    pub quantization: Option<String>,
    pub kv_cache_dtype: Option<String>,
    pub max_model_len: Option<usize>,
    pub chat_template: Option<String>,
    pub swap_space: Option<f64>,
    pub workers: Option<usize>,
    pub enable_lora: Option<bool>,
    pub max_loras: Option<usize>,
    pub lora_modules: Option<String>,
    pub block_size: Option<usize>,
    pub tensor_parallel_size: Option<usize>,
    pub pipeline_parallel_stages: Option<usize>,
}

/// CLI override values. Each `Some` value means the user explicitly passed
/// the flag and should win over TOML + env.
#[derive(Debug, Default, Clone)]
pub struct CliOverrides {
    pub model: Option<String>,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub context_length: Option<u32>,
    pub cpu: Option<bool>,
    pub advertise: Option<bool>,
    pub serving: Option<bool>,
    pub draft_model: Option<String>,
    pub num_draft_tokens: Option<u32>,
    pub enforce_eager: Option<bool>,
    pub gpu_memory_utilization: Option<f64>,
    pub max_num_seqs: Option<usize>,
    pub max_num_batched_tokens: Option<usize>,
    pub disable_prefix_cache: Option<bool>,
    pub quantization: Option<String>,
    pub kv_cache_dtype: Option<String>,
    pub max_model_len: Option<usize>,
    pub chat_template: Option<String>,
    pub swap_space: Option<f64>,
    pub workers: Option<usize>,
    pub enable_lora: Option<bool>,
    pub max_loras: Option<usize>,
    pub lora_modules: Option<String>,
    pub block_size: Option<usize>,
    pub tensor_parallel_size: Option<usize>,
    pub pipeline_parallel_stages: Option<usize>,
}

impl CrabInferConfig {
    /// Merge CLI overrides into this config. For each `Some` value in
    /// `overrides`, the corresponding field is replaced.
    pub fn merge_cli_overrides(&mut self, overrides: &CliOverrides) {
        macro_rules! merge {
            ($field:ident) => {
                if overrides.$field.is_some() {
                    self.$field = overrides.$field.clone();
                }
            };
        }
        merge!(model);
        merge!(host);
        merge!(port);
        merge!(context_length);
        merge!(cpu);
        merge!(advertise);
        merge!(serving);
        merge!(draft_model);
        merge!(num_draft_tokens);
        merge!(enforce_eager);
        merge!(gpu_memory_utilization);
        merge!(max_num_seqs);
        merge!(max_num_batched_tokens);
        merge!(disable_prefix_cache);
        merge!(quantization);
        merge!(kv_cache_dtype);
        merge!(max_model_len);
        merge!(chat_template);
        merge!(swap_space);
        merge!(workers);
        merge!(enable_lora);
        merge!(max_loras);
        merge!(lora_modules);
        merge!(block_size);
        merge!(tensor_parallel_size);
        merge!(pipeline_parallel_stages);
    }

    /// Convert to a concrete [`ServerConfig`] filling in defaults for any
    /// `None` fields.
    pub fn to_server_config(&self) -> ServerConfig {
        ServerConfig {
            model_path: self.model.clone().unwrap_or_default(),
            host: self.host.clone().unwrap_or_else(|| "127.0.0.1".to_string()),
            port: self.port.unwrap_or(8080),
            context_length: self.context_length.unwrap_or(4096),
            cpu: self.cpu.unwrap_or(false),
            advertise: self.advertise.unwrap_or(false),
            serving: self.serving.unwrap_or(false),
            draft_model_path: self.draft_model.clone(),
            num_draft_tokens: self.num_draft_tokens.unwrap_or(4),
            enforce_eager: self.enforce_eager.unwrap_or(false),
            gpu_memory_utilization: self.gpu_memory_utilization.unwrap_or(0.90),
            max_num_seqs: self.max_num_seqs.unwrap_or(64),
            max_num_batched_tokens: self.max_num_batched_tokens.unwrap_or(2048),
            disable_prefix_cache: self.disable_prefix_cache.unwrap_or(false),
            quantization: self.quantization.clone().unwrap_or_else(|| "none".to_string()),
            kv_cache_dtype: self.kv_cache_dtype.clone().unwrap_or_else(|| "auto".to_string()),
            max_model_len: self.max_model_len,
            chat_template: self.chat_template.clone(),
            swap_space: self.swap_space.unwrap_or(0.0),
            workers: self.workers.unwrap_or(1),
            enable_lora: self.enable_lora.unwrap_or(false),
            max_loras: self.max_loras.unwrap_or(4),
            lora_modules: self.lora_modules.clone(),
            block_size: self.block_size.unwrap_or(16),
            tensor_parallel_size: self.tensor_parallel_size.unwrap_or(1),
            pipeline_parallel_stages: self.pipeline_parallel_stages.unwrap_or(1),
        }
    }
}

/// Load a [`CrabInferConfig`] from a TOML file.
///
/// - If `path` is `Some`, load from that exact path (error if missing).
/// - If `path` is `None`, try `./crabinfer.toml` in the current directory.
///   If the file doesn't exist, return a default config (no error).
pub fn load_config(path: Option<&Path>) -> Result<CrabInferConfig, Box<dyn std::error::Error>> {
    match path {
        Some(p) => {
            let contents = std::fs::read_to_string(p)
                .map_err(|e| format!("Failed to read config file {}: {e}", p.display()))?;
            let cfg: CrabInferConfig = toml::from_str(&contents)
                .map_err(|e| format!("Failed to parse config file {}: {e}", p.display()))?;
            Ok(cfg)
        }
        None => {
            let default_path = Path::new("crabinfer.toml");
            if default_path.exists() {
                let contents = std::fs::read_to_string(default_path)
                    .map_err(|e| format!("Failed to read crabinfer.toml: {e}"))?;
                let cfg: CrabInferConfig = toml::from_str(&contents)
                    .map_err(|e| format!("Failed to parse crabinfer.toml: {e}"))?;
                Ok(cfg)
            } else {
                Ok(CrabInferConfig::default())
            }
        }
    }
}

/// Apply environment variable overrides to a [`ServerConfig`].
///
/// Supported variables:
/// - `CRABINFER_HOST` -- host to bind to
/// - `CRABINFER_PORT` -- port number
/// - `CRABINFER_GPU_MEMORY_UTILIZATION` -- GPU memory fraction (0.0-1.0)
/// - `CRABINFER_MAX_NUM_SEQS` -- maximum concurrent sequences
/// - `CRABINFER_MAX_NUM_BATCHED_TOKENS` -- max tokens per scheduling step
/// - `CRABINFER_MAX_MODEL_LEN` -- override model context length
/// - `CRABINFER_QUANTIZATION` -- weight quantization method
/// - `CRABINFER_KV_CACHE_DTYPE` -- KV cache data type
/// - `CRABINFER_CHAT_TEMPLATE` -- chat template override
/// - `CRABINFER_ENFORCE_EAGER` -- set to "1" or "true" to disable CUDA graphs
/// - `CRABINFER_SWAP_SPACE` -- CPU swap space for KV cache in GiB (0 = disabled)
pub fn apply_env_overrides(config: &mut ServerConfig) {
    use std::env;

    if let Ok(v) = env::var("CRABINFER_HOST") {
        config.host = v;
    }
    if let Ok(v) = env::var("CRABINFER_PORT") {
        if let Ok(p) = v.parse() {
            config.port = p;
        }
    }
    if let Ok(v) = env::var("CRABINFER_GPU_MEMORY_UTILIZATION") {
        if let Ok(f) = v.parse() {
            config.gpu_memory_utilization = f;
        }
    }
    if let Ok(v) = env::var("CRABINFER_MAX_NUM_SEQS") {
        if let Ok(n) = v.parse() {
            config.max_num_seqs = n;
        }
    }
    if let Ok(v) = env::var("CRABINFER_MAX_NUM_BATCHED_TOKENS") {
        if let Ok(n) = v.parse() {
            config.max_num_batched_tokens = n;
        }
    }
    if let Ok(v) = env::var("CRABINFER_MAX_MODEL_LEN") {
        if let Ok(n) = v.parse() {
            config.max_model_len = Some(n);
        }
    }
    if let Ok(v) = env::var("CRABINFER_QUANTIZATION") {
        config.quantization = v;
    }
    if let Ok(v) = env::var("CRABINFER_KV_CACHE_DTYPE") {
        config.kv_cache_dtype = v;
    }
    if let Ok(v) = env::var("CRABINFER_CHAT_TEMPLATE") {
        config.chat_template = Some(v);
    }
    if let Ok(v) = env::var("CRABINFER_ENFORCE_EAGER") {
        config.enforce_eager = v == "1" || v.eq_ignore_ascii_case("true");
    }
    if let Ok(v) = env::var("CRABINFER_SWAP_SPACE") {
        if let Ok(f) = v.parse() {
            config.swap_space = f;
        }
    }
    if let Ok(v) = env::var("CRABINFER_WORKERS") {
        if let Ok(n) = v.parse() {
            config.workers = n;
        }
    }
    if let Ok(v) = env::var("CRABINFER_BLOCK_SIZE") {
        if let Ok(n) = v.parse() {
            config.block_size = n;
        }
    }
    if let Ok(v) = env::var("CRABINFER_TENSOR_PARALLEL_SIZE") {
        if let Ok(n) = v.parse() {
            config.tensor_parallel_size = n;
        }
    }
    if let Ok(v) = env::var("CRABINFER_PIPELINE_PARALLEL_STAGES") {
        if let Ok(n) = v.parse() {
            config.pipeline_parallel_stages = n;
        }
    }
}

/// Apply CLI overrides to a fully-resolved [`ServerConfig`].
///
/// Only fields where the user explicitly passed a CLI flag (`Some`) are
/// overwritten. This is the final step and gives CLI the highest precedence.
pub fn apply_cli_overrides(config: &mut ServerConfig, overrides: &CliOverrides) {
    if let Some(ref v) = overrides.model {
        config.model_path = v.clone();
    }
    if let Some(ref v) = overrides.host {
        config.host = v.clone();
    }
    if let Some(v) = overrides.port {
        config.port = v;
    }
    if let Some(v) = overrides.context_length {
        config.context_length = v;
    }
    if let Some(v) = overrides.cpu {
        config.cpu = v;
    }
    if let Some(v) = overrides.advertise {
        config.advertise = v;
    }
    if let Some(v) = overrides.serving {
        config.serving = v;
    }
    if overrides.draft_model.is_some() {
        config.draft_model_path = overrides.draft_model.clone();
    }
    if let Some(v) = overrides.num_draft_tokens {
        config.num_draft_tokens = v;
    }
    if let Some(v) = overrides.enforce_eager {
        config.enforce_eager = v;
    }
    if let Some(v) = overrides.gpu_memory_utilization {
        config.gpu_memory_utilization = v;
    }
    if let Some(v) = overrides.max_num_seqs {
        config.max_num_seqs = v;
    }
    if let Some(v) = overrides.max_num_batched_tokens {
        config.max_num_batched_tokens = v;
    }
    if let Some(v) = overrides.disable_prefix_cache {
        config.disable_prefix_cache = v;
    }
    if let Some(ref v) = overrides.quantization {
        config.quantization = v.clone();
    }
    if let Some(ref v) = overrides.kv_cache_dtype {
        config.kv_cache_dtype = v.clone();
    }
    if overrides.max_model_len.is_some() {
        config.max_model_len = overrides.max_model_len;
    }
    if overrides.chat_template.is_some() {
        config.chat_template = overrides.chat_template.clone();
    }
    if let Some(v) = overrides.swap_space {
        config.swap_space = v;
    }
    if let Some(v) = overrides.workers {
        config.workers = v;
    }
    if let Some(v) = overrides.enable_lora {
        config.enable_lora = v;
    }
    if let Some(v) = overrides.max_loras {
        config.max_loras = v;
    }
    if overrides.lora_modules.is_some() {
        config.lora_modules = overrides.lora_modules.clone();
    }
    if let Some(v) = overrides.block_size {
        config.block_size = v;
    }
    if let Some(v) = overrides.tensor_parallel_size {
        config.tensor_parallel_size = v;
    }
    if let Some(v) = overrides.pipeline_parallel_stages {
        config.pipeline_parallel_stages = v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_toml_deserialization() {
        let toml_str = r#"
model = "/models/llama-7b.gguf"
host = "0.0.0.0"
port = 9090
context_length = 8192
cpu = true
advertise = true
serving = true
draft_model = "/models/draft.gguf"
num_draft_tokens = 8
enforce_eager = true
gpu_memory_utilization = 0.75
max_num_seqs = 128
max_num_batched_tokens = 4096
disable_prefix_cache = true
quantization = "int8"
kv_cache_dtype = "fp16"
max_model_len = 16384
chat_template = "llama3"
swap_space = 2.0
"#;
        let cfg: CrabInferConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.model.as_deref(), Some("/models/llama-7b.gguf"));
        assert_eq!(cfg.host.as_deref(), Some("0.0.0.0"));
        assert_eq!(cfg.port, Some(9090));
        assert_eq!(cfg.context_length, Some(8192));
        assert_eq!(cfg.cpu, Some(true));
        assert_eq!(cfg.advertise, Some(true));
        assert_eq!(cfg.serving, Some(true));
        assert_eq!(cfg.draft_model.as_deref(), Some("/models/draft.gguf"));
        assert_eq!(cfg.num_draft_tokens, Some(8));
        assert_eq!(cfg.enforce_eager, Some(true));
        assert_eq!(cfg.gpu_memory_utilization, Some(0.75));
        assert_eq!(cfg.max_num_seqs, Some(128));
        assert_eq!(cfg.max_num_batched_tokens, Some(4096));
        assert_eq!(cfg.disable_prefix_cache, Some(true));
        assert_eq!(cfg.quantization.as_deref(), Some("int8"));
        assert_eq!(cfg.kv_cache_dtype.as_deref(), Some("fp16"));
        assert_eq!(cfg.max_model_len, Some(16384));
        assert_eq!(cfg.chat_template.as_deref(), Some("llama3"));
        assert_eq!(cfg.swap_space, Some(2.0));
    }

    #[test]
    fn test_partial_toml_uses_defaults() {
        let toml_str = r#"
model = "/models/tiny.gguf"
port = 3000
"#;
        let cfg: CrabInferConfig = toml::from_str(toml_str).unwrap();
        let sc = cfg.to_server_config();

        assert_eq!(sc.model_path, "/models/tiny.gguf");
        assert_eq!(sc.port, 3000);
        // Defaults for missing fields
        assert_eq!(sc.host, "127.0.0.1");
        assert_eq!(sc.context_length, 4096);
        assert!(!sc.cpu);
        assert!(!sc.advertise);
        assert!(!sc.serving);
        assert_eq!(sc.num_draft_tokens, 4);
        assert!(!sc.enforce_eager);
        assert!((sc.gpu_memory_utilization - 0.90).abs() < 0.001);
        assert_eq!(sc.max_num_seqs, 64);
        assert_eq!(sc.max_num_batched_tokens, 2048);
        assert!(!sc.disable_prefix_cache);
        assert_eq!(sc.quantization, "none");
        assert_eq!(sc.kv_cache_dtype, "auto");
        assert!(sc.max_model_len.is_none());
        assert!(sc.chat_template.is_none());
        assert!((sc.swap_space - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_merge_cli_overrides() {
        // Start with a TOML config
        let toml_str = r#"
model = "/models/llama.gguf"
host = "0.0.0.0"
port = 9090
quantization = "int8"
"#;
        let mut cfg: CrabInferConfig = toml::from_str(toml_str).unwrap();

        // CLI overrides only port and quantization
        let overrides = CliOverrides {
            port: Some(7070),
            quantization: Some("none".to_string()),
            ..Default::default()
        };
        cfg.merge_cli_overrides(&overrides);

        // port and quantization overridden by CLI
        assert_eq!(cfg.port, Some(7070));
        assert_eq!(cfg.quantization.as_deref(), Some("none"));
        // model and host kept from TOML
        assert_eq!(cfg.model.as_deref(), Some("/models/llama.gguf"));
        assert_eq!(cfg.host.as_deref(), Some("0.0.0.0"));
    }

    #[test]
    fn test_load_config_missing_explicit_path_errors() {
        let result = load_config(Some(Path::new("/nonexistent/crabinfer.toml")));
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_toml_produces_defaults() {
        let cfg: CrabInferConfig = toml::from_str("").unwrap();
        let sc = cfg.to_server_config();

        assert_eq!(sc.model_path, "");
        assert_eq!(sc.host, "127.0.0.1");
        assert_eq!(sc.port, 8080);
        assert_eq!(sc.context_length, 4096);
        assert!(!sc.cpu);
        assert!(!sc.serving);
        assert_eq!(sc.quantization, "none");
        assert_eq!(sc.kv_cache_dtype, "auto");
    }

    #[test]
    fn test_block_size_default() {
        let cfg = CrabInferConfig::default();
        let sc = cfg.to_server_config();
        assert_eq!(sc.block_size, 16);
    }

    #[test]
    fn test_block_size_from_toml() {
        let toml_str = r#"
model = "/models/test.gguf"
block_size = 32
"#;
        let cfg: CrabInferConfig = toml::from_str(toml_str).unwrap();
        let sc = cfg.to_server_config();
        assert_eq!(sc.block_size, 32);
    }

    #[test]
    fn test_block_size_cli_override() {
        let cfg = CrabInferConfig::default();
        let mut sc = cfg.to_server_config();
        assert_eq!(sc.block_size, 16);

        let overrides = CliOverrides {
            block_size: Some(8),
            ..Default::default()
        };
        apply_cli_overrides(&mut sc, &overrides);
        assert_eq!(sc.block_size, 8);
    }

    #[test]
    fn test_block_size_validation_helper() {
        // Test the validation logic that cmd_serve uses
        fn is_valid_block_size(bs: usize) -> bool {
            bs.is_power_of_two() && bs >= 8 && bs <= 64
        }

        assert!(is_valid_block_size(8));
        assert!(is_valid_block_size(16));
        assert!(is_valid_block_size(32));
        assert!(is_valid_block_size(64));
        assert!(!is_valid_block_size(4));   // too small
        assert!(!is_valid_block_size(128)); // too large
        assert!(!is_valid_block_size(12));  // not power of 2
        assert!(!is_valid_block_size(0));   // zero
        assert!(!is_valid_block_size(3));   // not power of 2
    }

    #[test]
    fn test_apply_cli_overrides_on_server_config() {
        let cfg = CrabInferConfig::default();
        let mut sc = cfg.to_server_config();

        let overrides = CliOverrides {
            host: Some("0.0.0.0".to_string()),
            port: Some(3000),
            model: Some("/models/test.gguf".to_string()),
            ..Default::default()
        };
        apply_cli_overrides(&mut sc, &overrides);

        assert_eq!(sc.host, "0.0.0.0");
        assert_eq!(sc.port, 3000);
        assert_eq!(sc.model_path, "/models/test.gguf");
        // Unchanged defaults
        assert_eq!(sc.context_length, 4096);
        assert!(!sc.cpu);
    }
}
