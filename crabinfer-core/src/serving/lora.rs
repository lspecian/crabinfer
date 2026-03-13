//! LoRA (Low-Rank Adaptation) adapter support for serving.
//!
//! Implements loading, caching, and applying LoRA adapters from HuggingFace format
//! (`adapter_model.safetensors` + `adapter_config.json`). Supports:
//!
//! - **D9.1**: Loading LoRA adapters with rank decomposition (A*B matrices)
//! - **D9.2**: Multi-LoRA serving with LRU eviction and per-request adapter selection
//! - **D9.3**: QLoRA — quantized base model + FP16 LoRA matrices
//!
//! # Architecture
//!
//! LoRA is applied additively after the base model forward pass:
//! ```text
//! output = base_forward(x) + scaling * (x @ lora_a^T @ lora_b^T)
//! ```
//!
//! where `scaling = lora_alpha / r` and `r` is the LoRA rank.
//!
//! Adapters are loaded lazily on first request and cached in GPU memory
//! with LRU eviction when `max_loras` is exceeded.

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};

// ─── LoRA configuration ──────────────────────────────────────────────────

/// LoRA adapter configuration, parsed from `adapter_config.json`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct LoraConfig {
    /// LoRA rank (dimension of the low-rank matrices).
    pub r: usize,
    /// Scaling factor alpha. The effective scaling is `lora_alpha / r`.
    pub lora_alpha: f32,
    /// Target module names to apply LoRA to.
    /// e.g., `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
    pub target_modules: Vec<String>,
    /// Dropout probability (applied during training, ignored at inference).
    #[serde(default)]
    pub lora_dropout: f32,
    /// Bias handling: "none", "all", or "lora_only".
    #[serde(default = "default_bias")]
    pub bias: String,
    /// Base model name/path (informational).
    #[serde(default)]
    pub base_model_name_or_path: Option<String>,
    /// Task type (informational, e.g., "CAUSAL_LM").
    #[serde(default)]
    pub task_type: Option<String>,
}

fn default_bias() -> String {
    "none".to_string()
}

impl LoraConfig {
    /// Load LoRA config from an `adapter_config.json` file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|e| {
            candle_core::Error::Msg(format!(
                "Failed to read adapter_config.json at {}: {e}",
                path.display()
            ))
        })?;
        let config: Self = serde_json::from_str(&text).map_err(|e| {
            candle_core::Error::Msg(format!(
                "Failed to parse adapter_config.json: {e}"
            ))
        })?;
        if config.r == 0 {
            return Err(candle_core::Error::Msg(
                "LoRA rank (r) must be > 0".to_string(),
            ));
        }
        Ok(config)
    }

    /// Compute the scaling factor: `lora_alpha / r`.
    pub fn scaling(&self) -> f32 {
        self.lora_alpha / self.r as f32
    }
}

// ─── LoRA weight pair ────────────────────────────────────────────────────

/// A single LoRA weight pair for one target module.
///
/// The low-rank decomposition stores two small matrices:
/// - `lora_a`: `[r, in_features]` — projects input to low-rank space
/// - `lora_b`: `[out_features, r]` — projects from low-rank space to output
///
/// The LoRA update is: `output += scaling * (input @ lora_a^T @ lora_b^T)`
/// Which is equivalent to: `output += scaling * (lora_b @ lora_a @ input^T)^T`
#[derive(Debug, Clone)]
pub struct LoraWeight {
    /// Down-projection matrix `[r, in_features]`.
    pub lora_a: Tensor,
    /// Up-projection matrix `[out_features, r]`.
    pub lora_b: Tensor,
}

impl LoraWeight {
    /// Memory usage in bytes for this weight pair.
    pub fn memory_bytes(&self) -> usize {
        let a_bytes = self.lora_a.elem_count() * self.lora_a.dtype().size_in_bytes();
        let b_bytes = self.lora_b.elem_count() * self.lora_b.dtype().size_in_bytes();
        a_bytes + b_bytes
    }
}

// ─── LoRA adapter ────────────────────────────────────────────────────────

/// A loaded LoRA adapter containing configuration and weight matrices.
///
/// Each adapter stores the A and B matrices for each target module across
/// all transformer layers. Memory footprint is tiny compared to the base model
/// (typically 1-10 MB for rank 8-64).
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    /// Human-readable adapter name (used in API requests).
    pub name: String,
    /// LoRA configuration.
    pub config: LoraConfig,
    /// Per-module weight pairs. Keys are fully qualified module names like
    /// `"model.layers.0.self_attn.q_proj"`.
    pub weights: HashMap<String, LoraWeight>,
    /// Precomputed scaling factor (`lora_alpha / r`).
    pub scaling: f32,
}

impl LoraAdapter {
    /// Total memory usage across all weight pairs.
    pub fn memory_bytes(&self) -> usize {
        self.weights.values().map(|w| w.memory_bytes()).sum()
    }

    /// Number of layers that have LoRA weights.
    pub fn num_target_modules(&self) -> usize {
        self.weights.len()
    }
}

// ─── LoRA manager ────────────────────────────────────────────────────────

/// Registry and cache of loaded LoRA adapters with LRU eviction.
///
/// The manager:
/// - Loads adapters from HuggingFace format on demand
/// - Caches loaded adapters in memory (GPU or CPU depending on device)
/// - Evicts least-recently-used adapters when `max_loras` is exceeded
/// - Provides thread-safe access via `Arc<LoraAdapter>`
pub struct LoraManager {
    /// Currently loaded adapters, keyed by adapter name.
    adapters: HashMap<String, Arc<LoraAdapter>>,
    /// Maximum number of concurrent adapters in memory.
    max_loras: usize,
    /// LRU tracking: front = least recently used, back = most recently used.
    lru_order: VecDeque<String>,
    /// Registered adapter paths for lazy loading.
    /// Maps adapter name -> filesystem path.
    registered_paths: HashMap<String, std::path::PathBuf>,
    /// Device for loading adapter weights.
    device: Device,
    /// Whether LoRA is enabled.
    enabled: bool,
}

impl LoraManager {
    /// Create a new LoRA manager.
    ///
    /// # Arguments
    /// - `max_loras`: maximum adapters to keep in memory simultaneously
    /// - `device`: target device for weight tensors
    pub fn new(max_loras: usize, device: Device) -> Self {
        Self {
            adapters: HashMap::new(),
            max_loras: max_loras.max(1),
            lru_order: VecDeque::new(),
            registered_paths: HashMap::new(),
            device,
            enabled: true,
        }
    }

    /// Create a disabled LoRA manager (no-op for all operations).
    pub fn disabled() -> Self {
        Self {
            adapters: HashMap::new(),
            max_loras: 0,
            lru_order: VecDeque::new(),
            registered_paths: HashMap::new(),
            device: Device::Cpu,
            enabled: false,
        }
    }

    /// Whether LoRA serving is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Register an adapter path for lazy loading.
    ///
    /// The adapter will be loaded into memory on first request.
    pub fn register_adapter(&mut self, name: &str, path: impl AsRef<Path>) {
        self.registered_paths
            .insert(name.to_string(), path.as_ref().to_path_buf());
    }

    /// Load an adapter from a HuggingFace adapter directory.
    ///
    /// The directory should contain:
    /// - `adapter_config.json` — LoRA configuration
    /// - `adapter_model.safetensors` — LoRA weight matrices
    ///
    /// If the adapter cache is full, the least-recently-used adapter is evicted.
    pub fn load_adapter(&mut self, name: &str, path: &Path) -> Result<()> {
        if !self.enabled {
            return Err(candle_core::Error::Msg(
                "LoRA is not enabled. Use --enable-lora to enable.".to_string(),
            ));
        }

        // If already loaded, just touch LRU
        if self.adapters.contains_key(name) {
            self.touch_lru(name);
            return Ok(());
        }

        // Evict if at capacity
        while self.adapters.len() >= self.max_loras {
            self.evict_lru()?;
        }

        // Load config
        let config_path = path.join("adapter_config.json");
        let config = LoraConfig::from_file(&config_path)?;
        let scaling = config.scaling();

        // Load weights from safetensors
        let weights = self.load_weights(path, &config)?;

        let adapter = Arc::new(LoraAdapter {
            name: name.to_string(),
            config,
            weights,
            scaling,
        });

        tracing::info!(
            "Loaded LoRA adapter '{}': rank={}, scaling={:.4}, modules={}, memory={:.2}MB",
            name,
            adapter.config.r,
            adapter.scaling,
            adapter.num_target_modules(),
            adapter.memory_bytes() as f64 / (1024.0 * 1024.0),
        );

        self.adapters.insert(name.to_string(), adapter);
        self.lru_order.push_back(name.to_string());

        Ok(())
    }

    /// Unload an adapter from memory.
    pub fn unload_adapter(&mut self, name: &str) -> Result<()> {
        if self.adapters.remove(name).is_some() {
            self.lru_order.retain(|n| n != name);
            tracing::info!("Unloaded LoRA adapter '{}'", name);
            Ok(())
        } else {
            Err(candle_core::Error::Msg(format!(
                "LoRA adapter '{}' is not loaded",
                name
            )))
        }
    }

    /// Get a loaded adapter by name, loading lazily from registered path if needed.
    pub fn get_adapter(&mut self, name: &str) -> Result<Arc<LoraAdapter>> {
        // Already loaded — touch LRU and return
        if self.adapters.contains_key(name) {
            self.touch_lru(name);
            return Ok(Arc::clone(self.adapters.get(name).unwrap()));
        }

        // Try lazy loading from registered path
        if let Some(path) = self.registered_paths.get(name).cloned() {
            self.load_adapter(name, &path)?;
            return self
                .adapters
                .get(name)
                .map(Arc::clone)
                .ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "Failed to load adapter '{}' from {}",
                        name,
                        path.display()
                    ))
                });
        }

        Err(candle_core::Error::Msg(format!(
            "LoRA adapter '{}' not found. Register it with --lora-modules name=path",
            name
        )))
    }

    /// Get a loaded adapter by name without modifying LRU state.
    /// Returns None if not loaded (does not trigger lazy loading).
    pub fn peek_adapter(&self, name: &str) -> Option<Arc<LoraAdapter>> {
        self.adapters.get(name).map(Arc::clone)
    }

    /// List all loaded adapter names.
    pub fn loaded_adapters(&self) -> Vec<&str> {
        self.adapters.keys().map(|s| s.as_str()).collect()
    }

    /// List all registered adapter names (loaded and unloaded).
    pub fn registered_adapters(&self) -> Vec<&str> {
        self.registered_paths.keys().map(|s| s.as_str()).collect()
    }

    /// Number of currently loaded adapters.
    pub fn num_loaded(&self) -> usize {
        self.adapters.len()
    }

    /// Total memory used by all loaded adapters.
    pub fn total_memory_bytes(&self) -> usize {
        self.adapters.values().map(|a| a.memory_bytes()).sum()
    }

    // ── Internal helpers ──

    /// Move an adapter to the back of the LRU queue (most recently used).
    fn touch_lru(&mut self, name: &str) {
        self.lru_order.retain(|n| n != name);
        self.lru_order.push_back(name.to_string());
    }

    /// Evict the least-recently-used adapter.
    fn evict_lru(&mut self) -> Result<()> {
        if let Some(evicted_name) = self.lru_order.pop_front() {
            if let Some(adapter) = self.adapters.remove(&evicted_name) {
                tracing::info!(
                    "Evicted LoRA adapter '{}' (freed {:.2}MB)",
                    evicted_name,
                    adapter.memory_bytes() as f64 / (1024.0 * 1024.0),
                );
            }
            Ok(())
        } else {
            Err(candle_core::Error::Msg(
                "Cannot evict: no adapters loaded".to_string(),
            ))
        }
    }

    /// Load LoRA weight matrices from a safetensors file.
    ///
    /// HuggingFace LoRA adapter weight naming convention:
    /// ```text
    /// base_model.model.model.layers.{i}.self_attn.{q,k,v,o}_proj.lora_A.weight
    /// base_model.model.model.layers.{i}.self_attn.{q,k,v,o}_proj.lora_B.weight
    /// base_model.model.model.layers.{i}.mlp.{gate,up,down}_proj.lora_A.weight
    /// base_model.model.model.layers.{i}.mlp.{gate,up,down}_proj.lora_B.weight
    /// ```
    fn load_weights(
        &self,
        adapter_dir: &Path,
        config: &LoraConfig,
    ) -> Result<HashMap<String, LoraWeight>> {
        let st_path = adapter_dir.join("adapter_model.safetensors");
        if !st_path.exists() {
            return Err(candle_core::Error::Msg(format!(
                "adapter_model.safetensors not found in {}",
                adapter_dir.display()
            )));
        }

        // Load safetensors
        let tensors =
            unsafe { candle_core::safetensors::MmapedSafetensors::new(&st_path)? };

        // Collect all tensor names
        let tensor_names: Vec<String> = tensors.tensors().into_iter().map(|(n, _)| n).collect();

        // Group by module: find pairs of lora_A and lora_B
        let mut weights = HashMap::new();

        // Find all lora_A tensors and match with lora_B
        for a_name in &tensor_names {
            if !a_name.contains("lora_A") {
                continue;
            }

            // Derive the B tensor name
            let b_name = a_name.replace("lora_A", "lora_B");
            if !tensor_names.contains(&b_name) {
                tracing::warn!(
                    "LoRA weight {} has no matching lora_B, skipping",
                    a_name
                );
                continue;
            }

            // Extract the module name (strip prefix and .lora_A.weight suffix)
            let module_name = extract_module_name(a_name);

            // Check if this module is in the target_modules list
            let module_short = module_name
                .rsplit('.')
                .next()
                .unwrap_or(&module_name);
            if !config
                .target_modules
                .iter()
                .any(|t| module_short == t.as_str())
            {
                tracing::debug!(
                    "Skipping non-target module: {} (short: {})",
                    module_name,
                    module_short,
                );
                continue;
            }

            let lora_a = tensors
                .load(a_name, &self.device)?
                .to_dtype(DType::F32)?;
            let lora_b = tensors
                .load(&b_name, &self.device)?
                .to_dtype(DType::F32)?;

            // Validate shapes
            let a_dims = lora_a.dims();
            let b_dims = lora_b.dims();
            if a_dims.len() != 2 || b_dims.len() != 2 {
                return Err(candle_core::Error::Msg(format!(
                    "LoRA weight tensors must be 2D, got A={:?} B={:?} for module {}",
                    a_dims, b_dims, module_name
                )));
            }
            if a_dims[0] != config.r || b_dims[1] != config.r {
                return Err(candle_core::Error::Msg(format!(
                    "LoRA rank mismatch for module {}: config.r={}, A.shape={:?}, B.shape={:?}",
                    module_name, config.r, a_dims, b_dims
                )));
            }

            weights.insert(
                module_name,
                LoraWeight { lora_a, lora_b },
            );
        }

        if weights.is_empty() {
            return Err(candle_core::Error::Msg(format!(
                "No LoRA weights found in {}. Check target_modules: {:?}",
                adapter_dir.display(),
                config.target_modules,
            )));
        }

        tracing::debug!("Loaded {} LoRA weight pairs", weights.len());
        Ok(weights)
    }
}

// ─── LoRA application ────────────────────────────────────────────────────

/// Apply a LoRA adapter's update to the base model output for a specific module.
///
/// Computes: `output = base_output + scaling * (input @ lora_a^T @ lora_b^T)`
///
/// This is a pure tensor operation that works on any device (CPU, Metal, CUDA).
///
/// # Arguments
/// - `base_output`: output from the base model's linear layer `[batch, out_features]`
/// - `input`: input to the linear layer `[batch, in_features]`
/// - `adapter`: the LoRA adapter containing weight matrices
/// - `module_name`: fully qualified module name (e.g., `"model.layers.0.self_attn.q_proj"`)
///
/// # Returns
/// The adjusted output tensor `[batch, out_features]`, or the base_output unchanged
/// if no LoRA weights exist for this module.
pub fn apply_lora(
    base_output: &Tensor,
    input: &Tensor,
    adapter: &LoraAdapter,
    module_name: &str,
) -> Result<Tensor> {
    let lora_weight = match adapter.weights.get(module_name) {
        Some(w) => w,
        None => return Ok(base_output.clone()),
    };

    // LoRA forward: input @ A^T @ B^T
    // input:  [batch, in_features]
    // lora_a: [r, in_features]  -> A^T: [in_features, r]
    // lora_b: [out_features, r] -> B^T: [r, out_features]
    //
    // Step 1: low_rank = input @ A^T = [batch, r]
    // Step 2: lora_out = low_rank @ B^T = [batch, out_features]

    let low_rank = input.matmul(&lora_weight.lora_a.t()?)?;
    let lora_out = low_rank.matmul(&lora_weight.lora_b.t()?)?;

    // Scale and add to base output
    let scaled = (lora_out * adapter.scaling as f64)?;
    base_output + &scaled
}

/// Apply LoRA for a batch where different sequences may use different adapters.
///
/// For multi-LoRA serving, each sequence in the batch may specify a different
/// adapter (or no adapter). This function handles the per-sequence dispatch.
///
/// # Arguments
/// - `base_output`: `[total_tokens, out_features]` from the base linear layer
/// - `input`: `[total_tokens, in_features]` input to the linear layer
/// - `adapters`: per-sequence adapter (None = no LoRA for that sequence)
/// - `seq_token_counts`: number of tokens per sequence in the batch
/// - `module_name`: which module this is for
///
/// # Returns
/// Adjusted output tensor with per-sequence LoRA applied.
pub fn apply_lora_batched(
    base_output: &Tensor,
    input: &Tensor,
    adapters: &[Option<Arc<LoraAdapter>>],
    seq_token_counts: &[usize],
    module_name: &str,
) -> Result<Tensor> {
    debug_assert_eq!(adapters.len(), seq_token_counts.len());

    // Fast path: if no sequence has a LoRA adapter, skip entirely
    if adapters.iter().all(|a| a.is_none()) {
        return Ok(base_output.clone());
    }

    // Fast path: if all sequences use the same adapter, apply once
    let first_adapter = adapters.iter().find_map(|a| a.as_ref());
    if let Some(first) = first_adapter {
        let all_same = adapters.iter().all(|a| match a {
            Some(a) => Arc::ptr_eq(a, first),
            None => false,
        });
        if all_same {
            return apply_lora(base_output, input, first, module_name);
        }
    }

    // Slow path: per-sequence LoRA application
    // Split the batch by sequence, apply LoRA per-sequence, then concatenate
    let mut offset = 0usize;
    let mut output_parts: Vec<Tensor> = Vec::with_capacity(adapters.len());

    for (seq_idx, adapter_opt) in adapters.iter().enumerate() {
        let num_tokens = seq_token_counts[seq_idx];
        let seq_base = base_output.narrow(0, offset, num_tokens)?;

        let seq_output = if let Some(adapter) = adapter_opt {
            let seq_input = input.narrow(0, offset, num_tokens)?;
            apply_lora(&seq_base, &seq_input, adapter, module_name)?
        } else {
            seq_base
        };

        output_parts.push(seq_output);
        offset += num_tokens;
    }

    Tensor::cat(&output_parts, 0)
}

// ─── Helpers ─────────────────────────────────────────────────────────────

/// Extract the canonical module name from a HuggingFace LoRA weight tensor name.
///
/// HuggingFace format: `base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`
/// Canonical output:   `model.layers.0.self_attn.q_proj`
///
/// Also handles PEFT format without the `base_model.model.` prefix.
fn extract_module_name(tensor_name: &str) -> String {
    let name = tensor_name
        .strip_suffix(".weight")
        .unwrap_or(tensor_name);

    // Remove .lora_A or .lora_B suffix
    let name = name
        .strip_suffix(".lora_A")
        .or_else(|| name.strip_suffix(".lora_B"))
        .unwrap_or(name);

    // Strip common HuggingFace prefixes
    let name = name
        .strip_prefix("base_model.model.")
        .unwrap_or(name);

    name.to_string()
}

/// Parse a model field to extract base model and optional adapter name.
///
/// Format: `"base-model:adapter-name"` or just `"base-model"`.
///
/// # Examples
/// ```ignore
/// parse_model_adapter("llama-3-8b:my-lora-v2") == ("llama-3-8b", Some("my-lora-v2"))
/// parse_model_adapter("llama-3-8b") == ("llama-3-8b", None)
/// ```
pub fn parse_model_adapter(model_field: &str) -> (&str, Option<&str>) {
    if let Some(idx) = model_field.rfind(':') {
        // Avoid splitting on Windows paths or URLs (e.g., "C:\path" or "http://...")
        let after_colon = &model_field[idx + 1..];
        let before_colon = &model_field[..idx];
        // Only split if the part after colon looks like an adapter name (non-empty, no slashes)
        if !after_colon.is_empty()
            && !after_colon.contains('/')
            && !after_colon.contains('\\')
            && !before_colon.ends_with('/')
            && !before_colon.ends_with('\\')
        {
            return (before_colon, Some(after_colon));
        }
    }
    (model_field, None)
}

/// Parse `--lora-modules` flag value: `"name1=path1,name2=path2"`.
pub fn parse_lora_modules(value: &str) -> Result<Vec<(String, String)>> {
    let mut modules = Vec::new();
    for entry in value.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let parts: Vec<&str> = entry.splitn(2, '=').collect();
        if parts.len() != 2 {
            return Err(candle_core::Error::Msg(format!(
                "Invalid --lora-modules entry '{}'. Expected format: name=path",
                entry
            )));
        }
        modules.push((parts[0].trim().to_string(), parts[1].trim().to_string()));
    }
    Ok(modules)
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    // ── Config tests ──

    #[test]
    fn test_lora_config_parse() {
        let json = r#"{
            "r": 16,
            "lora_alpha": 32.0,
            "target_modules": ["q_proj", "k_proj", "v_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }"#;
        let config: LoraConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.r, 16);
        assert_eq!(config.lora_alpha, 32.0);
        assert_eq!(config.target_modules.len(), 3);
        assert_eq!(config.lora_dropout, 0.05);
        assert_eq!(config.bias, "none");
        assert_eq!(config.task_type.as_deref(), Some("CAUSAL_LM"));
    }

    #[test]
    fn test_lora_config_defaults() {
        let json = r#"{
            "r": 8,
            "lora_alpha": 16.0,
            "target_modules": ["q_proj", "v_proj"]
        }"#;
        let config: LoraConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.lora_dropout, 0.0);
        assert_eq!(config.bias, "none");
        assert!(config.base_model_name_or_path.is_none());
    }

    #[test]
    fn test_lora_config_scaling() {
        let config = LoraConfig {
            r: 8,
            lora_alpha: 16.0,
            target_modules: vec![],
            lora_dropout: 0.0,
            bias: "none".to_string(),
            base_model_name_or_path: None,
            task_type: None,
        };
        assert!((config.scaling() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_lora_config_scaling_rank_64() {
        let config = LoraConfig {
            r: 64,
            lora_alpha: 128.0,
            target_modules: vec![],
            lora_dropout: 0.0,
            bias: "none".to_string(),
            base_model_name_or_path: None,
            task_type: None,
        };
        assert!((config.scaling() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_lora_config_from_file_missing() {
        let result = LoraConfig::from_file(Path::new("/nonexistent/adapter_config.json"));
        assert!(result.is_err());
    }

    #[test]
    fn test_lora_config_from_file_valid() {
        let tmp = std::env::temp_dir().join("crabinfer_test_lora_cfg");
        std::fs::create_dir_all(&tmp).unwrap();
        let cfg_path = tmp.join("adapter_config.json");
        std::fs::write(
            &cfg_path,
            r#"{"r": 16, "lora_alpha": 32.0, "target_modules": ["q_proj"]}"#,
        )
        .unwrap();
        let config = LoraConfig::from_file(&cfg_path).unwrap();
        assert_eq!(config.r, 16);
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    #[test]
    fn test_lora_config_from_file_zero_rank() {
        let tmp = std::env::temp_dir().join("crabinfer_test_lora_cfg_zero");
        std::fs::create_dir_all(&tmp).unwrap();
        let cfg_path = tmp.join("adapter_config.json");
        std::fs::write(
            &cfg_path,
            r#"{"r": 0, "lora_alpha": 0.0, "target_modules": ["q_proj"]}"#,
        )
        .unwrap();
        let result = LoraConfig::from_file(&cfg_path);
        assert!(result.is_err());
        std::fs::remove_dir_all(&tmp).unwrap();
    }

    // ── Weight tests ──

    #[test]
    fn test_lora_weight_memory() {
        let lora_a = Tensor::zeros((8, 4096), DType::F32, &Device::Cpu).unwrap();
        let lora_b = Tensor::zeros((4096, 8), DType::F32, &Device::Cpu).unwrap();
        let w = LoraWeight { lora_a, lora_b };
        // 8*4096*4 + 4096*8*4 = 131072 + 131072 = 262144 bytes
        assert_eq!(w.memory_bytes(), 262144);
    }

    // ── Apply LoRA tests ──

    #[test]
    fn test_apply_lora_basic() {
        let dev = &Device::Cpu;
        let batch = 4;
        let in_features = 64;
        let out_features = 64;
        let rank = 8;

        let input = Tensor::randn(0f32, 1.0, (batch, in_features), dev).unwrap();
        let base_output = Tensor::zeros((batch, out_features), DType::F32, dev).unwrap();

        let lora_a = Tensor::randn(0f32, 0.01, (rank, in_features), dev).unwrap();
        let lora_b = Tensor::randn(0f32, 0.01, (out_features, rank), dev).unwrap();

        let mut weights = HashMap::new();
        weights.insert(
            "model.layers.0.self_attn.q_proj".to_string(),
            LoraWeight {
                lora_a: lora_a.clone(),
                lora_b: lora_b.clone(),
            },
        );

        let adapter = LoraAdapter {
            name: "test".to_string(),
            config: LoraConfig {
                r: rank,
                lora_alpha: 16.0,
                target_modules: vec!["q_proj".to_string()],
                lora_dropout: 0.0,
                bias: "none".to_string(),
                base_model_name_or_path: None,
                task_type: None,
            },
            weights,
            scaling: 2.0, // 16.0 / 8
        };

        let result = apply_lora(
            &base_output,
            &input,
            &adapter,
            "model.layers.0.self_attn.q_proj",
        )
        .unwrap();

        assert_eq!(result.dims(), &[batch, out_features]);

        // Verify it's not all zeros (LoRA was applied)
        let sum: f32 = result
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(sum > 0.0, "LoRA output should not be all zeros");
    }

    #[test]
    fn test_apply_lora_missing_module() {
        let dev = &Device::Cpu;
        let base_output = Tensor::ones((2, 16), DType::F32, dev).unwrap();
        let input = Tensor::ones((2, 16), DType::F32, dev).unwrap();

        let adapter = LoraAdapter {
            name: "test".to_string(),
            config: LoraConfig {
                r: 4,
                lora_alpha: 8.0,
                target_modules: vec!["q_proj".to_string()],
                lora_dropout: 0.0,
                bias: "none".to_string(),
                base_model_name_or_path: None,
                task_type: None,
            },
            weights: HashMap::new(),
            scaling: 2.0,
        };

        // Should return base_output unchanged when module is not in weights
        let result = apply_lora(
            &base_output,
            &input,
            &adapter,
            "model.layers.0.self_attn.v_proj",
        )
        .unwrap();

        let diff: f32 = (&result - &base_output)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(diff < 1e-6, "output should match base_output exactly");
    }

    #[test]
    fn test_apply_lora_math_correctness() {
        // Verify the math: output = base + scaling * (input @ A^T @ B^T)
        let dev = &Device::Cpu;

        let input = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], dev).unwrap();
        let base_output = Tensor::zeros((1, 3), DType::F32, dev).unwrap();

        // A: [2, 4] (rank=2, in_features=4)
        let lora_a = Tensor::new(
            &[[1.0f32, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dev,
        )
        .unwrap();
        // B: [3, 2] (out_features=3, rank=2)
        let lora_b = Tensor::new(
            &[[1.0f32, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dev,
        )
        .unwrap();

        let mut weights = HashMap::new();
        weights.insert(
            "test_module".to_string(),
            LoraWeight {
                lora_a,
                lora_b,
            },
        );

        let adapter = LoraAdapter {
            name: "test".to_string(),
            config: LoraConfig {
                r: 2,
                lora_alpha: 2.0,
                target_modules: vec![],
                lora_dropout: 0.0,
                bias: "none".to_string(),
                base_model_name_or_path: None,
                task_type: None,
            },
            weights,
            scaling: 1.0, // 2.0 / 2
        };

        let result = apply_lora(&base_output, &input, &adapter, "test_module").unwrap();

        // Manual calculation:
        // input @ A^T = [1,2,3,4] @ [[1,0],[0,1],[0,0],[0,0]] = [1, 2]
        // low_rank @ B^T = [1,2] @ [[1,0,1],[0,1,1]] = [1, 2, 3]
        // scaled = 1.0 * [1, 2, 3] = [1, 2, 3]
        // output = [0,0,0] + [1,2,3] = [1, 2, 3]
        let data: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            (data[0] - 1.0).abs() < 1e-5,
            "expected 1.0, got {}",
            data[0]
        );
        assert!(
            (data[1] - 2.0).abs() < 1e-5,
            "expected 2.0, got {}",
            data[1]
        );
        assert!(
            (data[2] - 3.0).abs() < 1e-5,
            "expected 3.0, got {}",
            data[2]
        );
    }

    #[test]
    fn test_apply_lora_scaling() {
        // Verify scaling factor is applied correctly
        let dev = &Device::Cpu;

        let input = Tensor::ones((1, 4), DType::F32, dev).unwrap();
        let base_output = Tensor::zeros((1, 4), DType::F32, dev).unwrap();

        // Identity-like A and B with rank 1
        let lora_a = Tensor::ones((1, 4), DType::F32, dev).unwrap();
        let lora_b = Tensor::ones((4, 1), DType::F32, dev).unwrap();

        let mut weights = HashMap::new();
        weights.insert(
            "m".to_string(),
            LoraWeight { lora_a, lora_b },
        );

        // scaling = 0.5
        let adapter = LoraAdapter {
            name: "test".to_string(),
            config: LoraConfig {
                r: 1,
                lora_alpha: 0.5,
                target_modules: vec![],
                lora_dropout: 0.0,
                bias: "none".to_string(),
                base_model_name_or_path: None,
                task_type: None,
            },
            weights,
            scaling: 0.5,
        };

        let result = apply_lora(&base_output, &input, &adapter, "m").unwrap();
        // input @ A^T = [1,1,1,1] @ [1,1,1,1]^T = [4]  (shape [1,1])
        // low_rank @ B^T = [4] @ [1,1,1,1] = [4,4,4,4]
        // scaled = 0.5 * [4,4,4,4] = [2,2,2,2]
        let data: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        for &v in &data {
            assert!(
                (v - 2.0).abs() < 1e-5,
                "expected 2.0, got {}",
                v
            );
        }
    }

    // ── Batched LoRA tests ──

    #[test]
    fn test_apply_lora_batched_no_adapters() {
        let dev = &Device::Cpu;
        let base = Tensor::ones((6, 4), DType::F32, dev).unwrap();
        let input = Tensor::ones((6, 4), DType::F32, dev).unwrap();

        let result = apply_lora_batched(
            &base,
            &input,
            &[None, None],
            &[3, 3],
            "module",
        )
        .unwrap();

        // Should return base unchanged
        let diff: f32 = (&result - &base)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(diff < 1e-6);
    }

    #[test]
    fn test_apply_lora_batched_mixed() {
        let dev = &Device::Cpu;
        let in_features = 4;
        let out_features = 4;
        let rank = 2;

        // 2 sequences: seq0 has 2 tokens, seq1 has 3 tokens
        let total_tokens = 5;
        let base = Tensor::zeros((total_tokens, out_features), DType::F32, dev).unwrap();
        let input = Tensor::ones((total_tokens, in_features), DType::F32, dev).unwrap();

        let lora_a = Tensor::ones((rank, in_features), DType::F32, dev).unwrap();
        let lora_b = Tensor::ones((out_features, rank), DType::F32, dev).unwrap();

        let mut weights = HashMap::new();
        weights.insert(
            "m".to_string(),
            LoraWeight {
                lora_a,
                lora_b,
            },
        );

        let adapter = Arc::new(LoraAdapter {
            name: "test".to_string(),
            config: LoraConfig {
                r: rank,
                lora_alpha: rank as f32,
                target_modules: vec![],
                lora_dropout: 0.0,
                bias: "none".to_string(),
                base_model_name_or_path: None,
                task_type: None,
            },
            weights,
            scaling: 1.0,
        });

        // seq0 uses adapter, seq1 does not
        let adapters = vec![Some(adapter), None];

        let result = apply_lora_batched(
            &base,
            &input,
            &adapters,
            &[2, 3],
            "m",
        )
        .unwrap();

        assert_eq!(result.dims(), &[total_tokens, out_features]);

        // Seq0 (tokens 0-1) should have LoRA applied
        let seq0_data: Vec<f32> = result
            .narrow(0, 0, 2)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for &v in &seq0_data {
            assert!(v.abs() > 0.1, "seq0 should have LoRA applied, got {}", v);
        }

        // Seq1 (tokens 2-4) should be zeros (no LoRA)
        let seq1_data: Vec<f32> = result
            .narrow(0, 2, 3)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for &v in &seq1_data {
            assert!(
                v.abs() < 1e-6,
                "seq1 should have no LoRA, got {}",
                v
            );
        }
    }

    // ── Manager tests ──

    #[test]
    fn test_lora_manager_new() {
        let mgr = LoraManager::new(4, Device::Cpu);
        assert!(mgr.is_enabled());
        assert_eq!(mgr.num_loaded(), 0);
        assert_eq!(mgr.total_memory_bytes(), 0);
    }

    #[test]
    fn test_lora_manager_disabled() {
        let mgr = LoraManager::disabled();
        assert!(!mgr.is_enabled());
    }

    #[test]
    fn test_lora_manager_register_and_list() {
        let mut mgr = LoraManager::new(4, Device::Cpu);
        mgr.register_adapter("adapter1", "/path/to/adapter1");
        mgr.register_adapter("adapter2", "/path/to/adapter2");

        let registered = mgr.registered_adapters();
        assert_eq!(registered.len(), 2);
        assert!(registered.contains(&"adapter1"));
        assert!(registered.contains(&"adapter2"));
    }

    #[test]
    fn test_lora_manager_lru_eviction() {
        let mut mgr = LoraManager::new(2, Device::Cpu);

        // Manually insert adapters to test LRU
        let make_adapter = |name: &str| -> Arc<LoraAdapter> {
            Arc::new(LoraAdapter {
                name: name.to_string(),
                config: LoraConfig {
                    r: 4,
                    lora_alpha: 8.0,
                    target_modules: vec![],
                    lora_dropout: 0.0,
                    bias: "none".to_string(),
                    base_model_name_or_path: None,
                    task_type: None,
                },
                weights: HashMap::new(),
                scaling: 2.0,
            })
        };

        mgr.adapters.insert("a".to_string(), make_adapter("a"));
        mgr.lru_order.push_back("a".to_string());
        mgr.adapters.insert("b".to_string(), make_adapter("b"));
        mgr.lru_order.push_back("b".to_string());

        assert_eq!(mgr.num_loaded(), 2);

        // Evict LRU (should remove "a")
        mgr.evict_lru().unwrap();
        assert_eq!(mgr.num_loaded(), 1);
        assert!(mgr.peek_adapter("a").is_none());
        assert!(mgr.peek_adapter("b").is_some());
    }

    #[test]
    fn test_lora_manager_touch_lru() {
        let mut mgr = LoraManager::new(3, Device::Cpu);

        mgr.lru_order.push_back("a".to_string());
        mgr.lru_order.push_back("b".to_string());
        mgr.lru_order.push_back("c".to_string());

        // Touch "a" — should move it to back
        mgr.touch_lru("a");

        assert_eq!(mgr.lru_order[0], "b");
        assert_eq!(mgr.lru_order[1], "c");
        assert_eq!(mgr.lru_order[2], "a");
    }

    #[test]
    fn test_lora_manager_unload() {
        let mut mgr = LoraManager::new(4, Device::Cpu);

        mgr.adapters.insert(
            "test".to_string(),
            Arc::new(LoraAdapter {
                name: "test".to_string(),
                config: LoraConfig {
                    r: 4,
                    lora_alpha: 8.0,
                    target_modules: vec![],
                    lora_dropout: 0.0,
                    bias: "none".to_string(),
                    base_model_name_or_path: None,
                    task_type: None,
                },
                weights: HashMap::new(),
                scaling: 2.0,
            }),
        );
        mgr.lru_order.push_back("test".to_string());

        mgr.unload_adapter("test").unwrap();
        assert_eq!(mgr.num_loaded(), 0);
        assert!(mgr.lru_order.is_empty());
    }

    #[test]
    fn test_lora_manager_unload_not_found() {
        let mut mgr = LoraManager::new(4, Device::Cpu);
        assert!(mgr.unload_adapter("nonexistent").is_err());
    }

    #[test]
    fn test_lora_manager_get_not_registered() {
        let mut mgr = LoraManager::new(4, Device::Cpu);
        assert!(mgr.get_adapter("nonexistent").is_err());
    }

    #[test]
    fn test_lora_manager_disabled_load_fails() {
        let mut mgr = LoraManager::disabled();
        let result = mgr.load_adapter("test", Path::new("/tmp"));
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not enabled"));
    }

    // ── Module name extraction ──

    #[test]
    fn test_extract_module_name_hf_format() {
        assert_eq!(
            extract_module_name(
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
            ),
            "model.layers.0.self_attn.q_proj"
        );
    }

    #[test]
    fn test_extract_module_name_peft_format() {
        assert_eq!(
            extract_module_name("model.layers.5.mlp.gate_proj.lora_B.weight"),
            "model.layers.5.mlp.gate_proj"
        );
    }

    #[test]
    fn test_extract_module_name_no_prefix() {
        assert_eq!(
            extract_module_name("layers.0.self_attn.v_proj.lora_A.weight"),
            "layers.0.self_attn.v_proj"
        );
    }

    // ── Model adapter parsing ──

    #[test]
    fn test_parse_model_adapter_with_adapter() {
        let (base, adapter) = parse_model_adapter("llama-3-8b:my-lora-v2");
        assert_eq!(base, "llama-3-8b");
        assert_eq!(adapter, Some("my-lora-v2"));
    }

    #[test]
    fn test_parse_model_adapter_no_adapter() {
        let (base, adapter) = parse_model_adapter("llama-3-8b");
        assert_eq!(base, "llama-3-8b");
        assert_eq!(adapter, None);
    }

    #[test]
    fn test_parse_model_adapter_windows_path() {
        // Should not split on Windows drive letter colon
        let (base, adapter) = parse_model_adapter("C:\\models\\llama");
        assert_eq!(base, "C:\\models\\llama");
        assert_eq!(adapter, None);
    }

    #[test]
    fn test_parse_model_adapter_empty_after_colon() {
        let (base, adapter) = parse_model_adapter("model:");
        assert_eq!(base, "model:");
        assert_eq!(adapter, None);
    }

    // ── Parse lora modules ──

    #[test]
    fn test_parse_lora_modules_single() {
        let result = parse_lora_modules("my-lora=/path/to/adapter").unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, "my-lora");
        assert_eq!(result[0].1, "/path/to/adapter");
    }

    #[test]
    fn test_parse_lora_modules_multiple() {
        let result = parse_lora_modules("lora1=/path1,lora2=/path2").unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, "lora1");
        assert_eq!(result[1].0, "lora2");
    }

    #[test]
    fn test_parse_lora_modules_empty() {
        let result = parse_lora_modules("").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_lora_modules_invalid() {
        let result = parse_lora_modules("invalid-no-equals");
        assert!(result.is_err());
    }

    // ── Adapter memory ──

    #[test]
    fn test_adapter_memory_bytes() {
        let dev = &Device::Cpu;
        let mut weights = HashMap::new();
        // 2 modules, each with rank-4 [4, 64] A and [64, 4] B
        for i in 0..2 {
            let lora_a = Tensor::zeros((4, 64), DType::F32, dev).unwrap();
            let lora_b = Tensor::zeros((64, 4), DType::F32, dev).unwrap();
            weights.insert(format!("module_{i}"), LoraWeight { lora_a, lora_b });
        }

        let adapter = LoraAdapter {
            name: "test".to_string(),
            config: LoraConfig {
                r: 4,
                lora_alpha: 8.0,
                target_modules: vec![],
                lora_dropout: 0.0,
                bias: "none".to_string(),
                base_model_name_or_path: None,
                task_type: None,
            },
            weights,
            scaling: 2.0,
        };

        // Each module: (4*64 + 64*4) * 4 bytes = 2048 bytes
        // 2 modules: 4096 bytes
        assert_eq!(adapter.memory_bytes(), 4096);
        assert_eq!(adapter.num_target_modules(), 2);
    }

    // ── Load adapter integration test (with real safetensors) ──

    #[test]
    fn test_load_adapter_missing_dir() {
        let mut mgr = LoraManager::new(4, Device::Cpu);
        let result = mgr.load_adapter("test", Path::new("/nonexistent/adapter"));
        assert!(result.is_err());
    }

    #[test]
    fn test_load_adapter_missing_safetensors() {
        let tmp = std::env::temp_dir().join("crabinfer_test_lora_no_st");
        std::fs::create_dir_all(&tmp).unwrap();

        // Create config but no safetensors
        std::fs::write(
            tmp.join("adapter_config.json"),
            r#"{"r": 8, "lora_alpha": 16.0, "target_modules": ["q_proj"]}"#,
        )
        .unwrap();

        let mut mgr = LoraManager::new(4, Device::Cpu);
        let result = mgr.load_adapter("test", &tmp);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("safetensors"),
            "error should mention safetensors, got: {err}"
        );

        std::fs::remove_dir_all(&tmp).unwrap();
    }
}
