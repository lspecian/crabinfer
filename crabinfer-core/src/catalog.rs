//! Curated model catalog with device-aware filtering.
//!
//! Provides a static, embedded catalog of known-good GGUF models from
//! HuggingFace, filterable by device capability, category, and size.
//! The catalog is compiled into the binary — no network access needed.

use crate::memory::MemoryPressureManager;
use crate::DeviceInfo;

// ---------------------------------------------------------------------------
// Types (UniFFI-exported)
// ---------------------------------------------------------------------------

/// Model use-case category.
#[derive(Debug, Clone, PartialEq, Eq, Hash, uniffi::Enum)]
pub enum ModelCategory {
    /// General-purpose chat and instruction following.
    General,
    /// Optimized for code generation and understanding.
    Code,
    /// Optimized for chain-of-thought reasoning.
    Reasoning,
}

/// A model in the curated catalog.
#[derive(Debug, Clone, uniffi::Record)]
pub struct CatalogEntry {
    /// Unique short identifier, e.g. "qwen3-8b-q4km".
    pub id: String,
    /// Human-readable name, e.g. "Qwen3 8B Q4_K_M".
    pub name: String,
    /// HuggingFace repo containing the GGUF file.
    pub hf_repo: String,
    /// GGUF filename within the repo.
    pub gguf_file: String,
    /// HuggingFace repo containing tokenizer.json.
    pub tokenizer_repo: String,
    /// GGUF file size in bytes (approximate download size).
    pub size_bytes: u64,
    /// Total parameter count in billions (all experts for MoE).
    pub param_count_b: f32,
    /// Active parameter count in billions (per token; equals total for dense).
    pub active_param_count_b: f32,
    /// Quantization format, e.g. "Q4_K_M".
    pub quant: String,
    /// Model architecture, e.g. "qwen3", "phi3", "gemma3".
    pub architecture: String,
    /// Minimum device RAM in GB (rough guide; actual filtering uses memory estimator).
    pub min_ram_gb: u32,
    /// Use-case category.
    pub category: ModelCategory,
    /// Whether this is a Mixture-of-Experts model.
    pub is_moe: bool,
    /// SHA256 hash of the GGUF file (empty string if unknown).
    pub sha256: String,
    /// Whether HuggingFace requires authentication for download.
    pub requires_auth: bool,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Return the full catalog.
pub fn all() -> Vec<CatalogEntry> {
    catalog_data()
}

/// Look up a single entry by ID.
pub fn get(id: &str) -> Option<CatalogEntry> {
    catalog_data().into_iter().find(|e| e.id == id)
}

/// Return models that fit on the given device (using calibrated memory estimator).
pub fn for_device(device: &DeviceInfo) -> Vec<CatalogEntry> {
    catalog_data()
        .into_iter()
        .filter(|e| fits_device(e, device))
        .collect()
}

/// Filter by category.
pub fn by_category(category: &ModelCategory) -> Vec<CatalogEntry> {
    catalog_data()
        .into_iter()
        .filter(|e| e.category == *category)
        .collect()
}

/// Return the best (largest fitting) model per category for this device.
///
/// Returns at most one model per category (General, Code, Reasoning).
/// "Best" = largest `param_count_b` that fits, with file size as tiebreaker.
pub fn recommended(device: &DeviceInfo) -> Vec<CatalogEntry> {
    let compatible = for_device(device);
    let categories = [
        ModelCategory::General,
        ModelCategory::Code,
        ModelCategory::Reasoning,
    ];
    let mut result = Vec::new();
    for cat in &categories {
        if let Some(best) = compatible
            .iter()
            .filter(|e| e.category == *cat)
            .max_by(|a, b| {
                a.param_count_b
                    .partial_cmp(&b.param_count_b)
                    .unwrap()
                    .then(a.size_bytes.cmp(&b.size_bytes))
            })
        {
            result.push(best.clone());
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Filtering logic
// ---------------------------------------------------------------------------

/// Check if a model fits on a device using the calibrated memory estimator.
///
/// Uses the same `estimate_model_memory_moe` function as the engine's
/// pre-flight check, with a platform-appropriate memory budget.
fn fits_device(entry: &CatalogEntry, device: &DeviceInfo) -> bool {
    let usable_fraction: f64 = if cfg!(target_os = "ios") {
        0.55
    } else {
        0.60
    };
    let budget = (device.total_memory_bytes as f64 * usable_fraction) as u64;

    let estimated = MemoryPressureManager::estimate_model_memory_moe(
        entry.param_count_b,
        entry.active_param_count_b,
        &entry.quant,
        device.recommended_context_length,
    );

    estimated <= budget
}

// ---------------------------------------------------------------------------
// Static catalog data
// ---------------------------------------------------------------------------

fn catalog_data() -> Vec<CatalogEntry> {
    vec![
        // -- Tiny (iPhone 4-6 GB) --
        CatalogEntry {
            id: "smollm2-360m-q8".into(),
            name: "SmolLM2 360M Q8_0".into(),
            hf_repo: "bartowski/SmolLM2-360M-Instruct-GGUF".into(),
            gguf_file: "SmolLM2-360M-Instruct-Q8_0.gguf".into(),
            tokenizer_repo: "HuggingFaceTB/SmolLM2-360M-Instruct".into(),
            size_bytes: 386_000_000,
            param_count_b: 0.36,
            active_param_count_b: 0.36,
            quant: "Q8_0".into(),
            architecture: "llama".into(),
            min_ram_gb: 4,
            category: ModelCategory::General,
            is_moe: false,
            sha256: String::new(),
            requires_auth: false,
        },
        CatalogEntry {
            id: "qwen3-0.6b-q4km".into(),
            name: "Qwen3 0.6B Q4_K_M".into(),
            hf_repo: "unsloth/Qwen3-0.6B-GGUF".into(),
            gguf_file: "Qwen3-0.6B-Q4_K_M.gguf".into(),
            tokenizer_repo: "Qwen/Qwen3-0.6B".into(),
            size_bytes: 430_000_000,
            param_count_b: 0.6,
            active_param_count_b: 0.6,
            quant: "Q4_K_M".into(),
            architecture: "qwen3".into(),
            min_ram_gb: 4,
            category: ModelCategory::General,
            is_moe: false,
            sha256: String::new(),
            requires_auth: false,
        },
        // -- Small (iPhone Pro 8 GB / entry Mac) --
        CatalogEntry {
            id: "qwen3-1.7b-q4km".into(),
            name: "Qwen3 1.7B Q4_K_M".into(),
            hf_repo: "unsloth/Qwen3-1.7B-GGUF".into(),
            gguf_file: "Qwen3-1.7B-Q4_K_M.gguf".into(),
            tokenizer_repo: "Qwen/Qwen3-1.7B".into(),
            size_bytes: 1_100_000_000,
            param_count_b: 1.7,
            active_param_count_b: 1.7,
            quant: "Q4_K_M".into(),
            architecture: "qwen3".into(),
            min_ram_gb: 4,
            category: ModelCategory::General,
            is_moe: false,
            sha256: String::new(),
            requires_auth: false,
        },
        // -- Medium (Mac 8 GB) --
        CatalogEntry {
            id: "phi4-mini-q4km".into(),
            name: "Phi-4 Mini Instruct 3.8B Q4_K_M".into(),
            hf_repo: "bartowski/microsoft_Phi-4-mini-instruct-GGUF".into(),
            gguf_file: "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf".into(),
            tokenizer_repo: "microsoft/Phi-4-mini-instruct".into(),
            size_bytes: 2_491_874_688,
            param_count_b: 3.8,
            active_param_count_b: 3.8,
            quant: "Q4_K_M".into(),
            architecture: "phi3".into(),
            min_ram_gb: 8,
            category: ModelCategory::General,
            is_moe: false,
            sha256: String::new(),
            requires_auth: false,
        },
        CatalogEntry {
            id: "qwen3-4b-q4km".into(),
            name: "Qwen3 4B Q4_K_M".into(),
            hf_repo: "unsloth/Qwen3-4B-GGUF".into(),
            gguf_file: "Qwen3-4B-Q4_K_M.gguf".into(),
            tokenizer_repo: "Qwen/Qwen3-4B".into(),
            size_bytes: 2_600_000_000,
            param_count_b: 4.0,
            active_param_count_b: 4.0,
            quant: "Q4_K_M".into(),
            architecture: "qwen3".into(),
            min_ram_gb: 8,
            category: ModelCategory::General,
            is_moe: false,
            sha256: String::new(),
            requires_auth: false,
        },
        CatalogEntry {
            id: "gemma3-4b-q4km".into(),
            name: "Gemma 3 4B IT Q4_K_M".into(),
            hf_repo: "unsloth/gemma-3-4b-it-GGUF".into(),
            gguf_file: "gemma-3-4b-it-Q4_K_M.gguf".into(),
            tokenizer_repo: "google/gemma-3-4b-it".into(),
            size_bytes: 2_700_000_000,
            param_count_b: 4.0,
            active_param_count_b: 4.0,
            quant: "Q4_K_M".into(),
            architecture: "gemma3".into(),
            min_ram_gb: 8,
            category: ModelCategory::General,
            is_moe: false,
            sha256: String::new(),
            requires_auth: true,
        },
        // -- Large (Mac 16 GB) --
        CatalogEntry {
            id: "qwen3-8b-q4km".into(),
            name: "Qwen3 8B Q4_K_M".into(),
            hf_repo: "unsloth/Qwen3-8B-GGUF".into(),
            gguf_file: "Qwen3-8B-Q4_K_M.gguf".into(),
            tokenizer_repo: "Qwen/Qwen3-8B".into(),
            size_bytes: 4_900_000_000,
            param_count_b: 8.0,
            active_param_count_b: 8.0,
            quant: "Q4_K_M".into(),
            architecture: "qwen3".into(),
            min_ram_gb: 16,
            category: ModelCategory::General,
            is_moe: false,
            sha256: String::new(),
            requires_auth: false,
        },
        CatalogEntry {
            id: "mistral-7b-q6k".into(),
            name: "Mistral 7B Instruct v0.3 Q6_K".into(),
            hf_repo: "bartowski/Mistral-7B-Instruct-v0.3-GGUF".into(),
            gguf_file: "Mistral-7B-Instruct-v0.3-Q6_K.gguf".into(),
            tokenizer_repo: "mistralai/Mistral-7B-Instruct-v0.3".into(),
            size_bytes: 5_500_000_000,
            param_count_b: 7.0,
            active_param_count_b: 7.0,
            quant: "Q6_K".into(),
            architecture: "mistral".into(),
            min_ram_gb: 16,
            category: ModelCategory::General,
            is_moe: false,
            sha256: String::new(),
            requires_auth: false,
        },
        // -- XL (Mac 32 GB) --
        CatalogEntry {
            id: "gemma3-12b-q4km".into(),
            name: "Gemma 3 12B IT Q4_K_M".into(),
            hf_repo: "unsloth/gemma-3-12b-it-GGUF".into(),
            gguf_file: "gemma-3-12b-it-Q4_K_M.gguf".into(),
            tokenizer_repo: "google/gemma-3-12b-it".into(),
            size_bytes: 7_300_000_000,
            param_count_b: 12.0,
            active_param_count_b: 12.0,
            quant: "Q4_K_M".into(),
            architecture: "gemma3".into(),
            min_ram_gb: 32,
            category: ModelCategory::General,
            is_moe: false,
            sha256: String::new(),
            requires_auth: true,
        },
        CatalogEntry {
            id: "qwen3-14b-q6k".into(),
            name: "Qwen3 14B Q6_K".into(),
            hf_repo: "unsloth/Qwen3-14B-GGUF".into(),
            gguf_file: "Qwen3-14B-Q6_K.gguf".into(),
            tokenizer_repo: "Qwen/Qwen3-14B".into(),
            size_bytes: 11_500_000_000,
            param_count_b: 14.0,
            active_param_count_b: 14.0,
            quant: "Q6_K".into(),
            architecture: "qwen3".into(),
            min_ram_gb: 32,
            category: ModelCategory::General,
            is_moe: false,
            sha256: String::new(),
            requires_auth: false,
        },
        // -- XXL (Mac 64 GB) --
        CatalogEntry {
            id: "gemma3-27b-q4km".into(),
            name: "Gemma 3 27B IT Q4_K_M".into(),
            hf_repo: "unsloth/gemma-3-27b-it-GGUF".into(),
            gguf_file: "gemma-3-27b-it-Q4_K_M.gguf".into(),
            tokenizer_repo: "google/gemma-3-27b-it".into(),
            size_bytes: 15_700_000_000,
            param_count_b: 27.0,
            active_param_count_b: 27.0,
            quant: "Q4_K_M".into(),
            architecture: "gemma3".into(),
            min_ram_gb: 64,
            category: ModelCategory::General,
            is_moe: false,
            sha256: String::new(),
            requires_auth: true,
        },
        CatalogEntry {
            id: "qwen3-32b-q4km".into(),
            name: "Qwen3 32B Q4_K_M".into(),
            hf_repo: "unsloth/Qwen3-32B-GGUF".into(),
            gguf_file: "Qwen3-32B-Q4_K_M.gguf".into(),
            tokenizer_repo: "Qwen/Qwen3-32B".into(),
            size_bytes: 18_800_000_000,
            param_count_b: 32.0,
            active_param_count_b: 32.0,
            quant: "Q4_K_M".into(),
            architecture: "qwen3".into(),
            min_ram_gb: 64,
            category: ModelCategory::General,
            is_moe: false,
            sha256: String::new(),
            requires_auth: false,
        },
        // -- Flagship (Mac 128+ GB) --
        CatalogEntry {
            id: "llama31-70b-q4km".into(),
            name: "Llama 3.1 70B Instruct Q4_K_M".into(),
            hf_repo: "bartowski/Meta-Llama-3.1-70B-Instruct-GGUF".into(),
            gguf_file: "Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf".into(),
            tokenizer_repo: "meta-llama/Llama-3.1-70B-Instruct".into(),
            size_bytes: 40_500_000_000,
            param_count_b: 70.0,
            active_param_count_b: 70.0,
            quant: "Q4_K_M".into(),
            architecture: "llama".into(),
            min_ram_gb: 128,
            category: ModelCategory::General,
            is_moe: false,
            sha256: String::new(),
            requires_auth: true,
        },
        // -- Code --
        CatalogEntry {
            id: "qwen25-coder-7b-q4km".into(),
            name: "Qwen2.5 Coder 7B Instruct Q4_K_M".into(),
            hf_repo: "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF".into(),
            gguf_file: "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf".into(),
            tokenizer_repo: "Qwen/Qwen2.5-Coder-7B-Instruct".into(),
            size_bytes: 4_683_074_336,
            param_count_b: 7.0,
            active_param_count_b: 7.0,
            quant: "Q4_K_M".into(),
            architecture: "qwen2".into(),
            min_ram_gb: 16,
            category: ModelCategory::Code,
            is_moe: false,
            sha256: String::new(),
            requires_auth: false,
        },
        CatalogEntry {
            id: "qwen3-coder-30b-a3b-q4km".into(),
            name: "Qwen3 Coder 30B-A3B Q4_K_M".into(),
            hf_repo: "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF".into(),
            gguf_file: "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf".into(),
            tokenizer_repo: "Qwen/Qwen3-Coder-30B-A3B-Instruct".into(),
            size_bytes: 17_700_000_000,
            param_count_b: 30.0,
            active_param_count_b: 3.0,
            quant: "Q4_K_M".into(),
            architecture: "qwen3moe".into(),
            min_ram_gb: 64,
            category: ModelCategory::Code,
            is_moe: true,
            sha256: String::new(),
            requires_auth: false,
        },
        CatalogEntry {
            id: "qwen3-coder-30b-a3b-q2k".into(),
            name: "Qwen3 Coder 30B-A3B Q2_K".into(),
            hf_repo: "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF".into(),
            gguf_file: "Qwen3-Coder-30B-A3B-Instruct-Q2_K.gguf".into(),
            tokenizer_repo: "Qwen/Qwen3-Coder-30B-A3B-Instruct".into(),
            size_bytes: 11_300_000_000,
            param_count_b: 30.0,
            active_param_count_b: 3.0,
            quant: "Q2_K".into(),
            architecture: "qwen3moe".into(),
            min_ram_gb: 32,
            category: ModelCategory::Code,
            is_moe: true,
            sha256: String::new(),
            requires_auth: false,
        },
        // -- Reasoning --
        CatalogEntry {
            id: "phi4-mini-reasoning-q4km".into(),
            name: "Phi-4 Mini Reasoning 3.8B Q4_K_M".into(),
            hf_repo: "bartowski/microsoft_Phi-4-mini-reasoning-GGUF".into(),
            gguf_file: "microsoft_Phi-4-mini-reasoning-Q4_K_M.gguf".into(),
            tokenizer_repo: "microsoft/Phi-4-mini-reasoning".into(),
            size_bytes: 2_491_874_848,
            param_count_b: 3.8,
            active_param_count_b: 3.8,
            quant: "Q4_K_M".into(),
            architecture: "phi3".into(),
            min_ram_gb: 8,
            category: ModelCategory::Reasoning,
            is_moe: false,
            sha256: String::new(),
            requires_auth: false,
        },
        CatalogEntry {
            id: "deepseek-r1-0528-8b-q4km".into(),
            name: "DeepSeek R1 0528 Qwen3 8B Q4_K_M".into(),
            hf_repo: "unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF".into(),
            gguf_file: "DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf".into(),
            tokenizer_repo: "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B".into(),
            size_bytes: 5_027_785_216,
            param_count_b: 8.2,
            active_param_count_b: 8.2,
            quant: "Q4_K_M".into(),
            architecture: "qwen3".into(),
            min_ram_gb: 16,
            category: ModelCategory::Reasoning,
            is_moe: false,
            sha256: String::new(),
            requires_auth: false,
        },
        CatalogEntry {
            id: "deepseek-r1-14b-q4km".into(),
            name: "DeepSeek R1 Distill Qwen 14B Q4_K_M".into(),
            hf_repo: "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF".into(),
            gguf_file: "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf".into(),
            tokenizer_repo: "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B".into(),
            size_bytes: 8_700_000_000,
            param_count_b: 14.0,
            active_param_count_b: 14.0,
            quant: "Q4_K_M".into(),
            architecture: "qwen2".into(),
            min_ram_gb: 32,
            category: ModelCategory::Reasoning,
            is_moe: false,
            sha256: String::new(),
            requires_auth: false,
        },
        CatalogEntry {
            id: "phi4-reasoning-plus-q4km".into(),
            name: "Phi-4 Reasoning Plus 14B Q4_K_M".into(),
            hf_repo: "bartowski/microsoft_Phi-4-reasoning-plus-GGUF".into(),
            gguf_file: "microsoft_Phi-4-reasoning-plus-Q4_K_M.gguf".into(),
            tokenizer_repo: "microsoft/Phi-4-reasoning-plus".into(),
            size_bytes: 9_053_116_416,
            param_count_b: 14.0,
            active_param_count_b: 14.0,
            quant: "Q4_K_M".into(),
            architecture: "phi3".into(),
            min_ram_gb: 32,
            category: ModelCategory::Reasoning,
            is_moe: false,
            sha256: String::new(),
            requires_auth: false,
        },
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Build a mock DeviceInfo for testing.
    fn mock_device(total_gb: u64, ctx: u32) -> DeviceInfo {
        let gb = 1024 * 1024 * 1024;
        DeviceInfo {
            device_model: "Test".into(),
            total_memory_bytes: total_gb * gb,
            available_memory_bytes: total_gb * gb / 2,
            has_metal_gpu: true,
            has_neural_engine: true,
            recommended_quant: "Q4_K_M".into(),
            max_model_size_b: 0,
            max_model_file_size_bytes: total_gb * gb, // generous for tests
            chip_name: "Test".into(),
            chip_variant: String::new(),
            recommended_context_length: ctx,
        }
    }

    #[test]
    fn test_catalog_not_empty() {
        let catalog = all();
        assert!(
            catalog.len() >= 20,
            "Catalog should have at least 20 models, got {}",
            catalog.len()
        );
    }

    #[test]
    fn test_catalog_unique_ids() {
        let catalog = all();
        let ids: HashSet<&str> = catalog.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(
            ids.len(),
            catalog.len(),
            "All catalog IDs must be unique"
        );
    }

    #[test]
    fn test_catalog_valid_entries() {
        for entry in all() {
            assert!(!entry.id.is_empty(), "ID must not be empty");
            assert!(!entry.name.is_empty(), "Name must not be empty");
            assert!(!entry.hf_repo.is_empty(), "hf_repo must not be empty");
            assert!(!entry.gguf_file.is_empty(), "gguf_file must not be empty");
            assert!(
                entry.gguf_file.ends_with(".gguf"),
                "{}: gguf_file must end with .gguf",
                entry.id
            );
            assert!(!entry.tokenizer_repo.is_empty(), "tokenizer_repo must not be empty");
            assert!(entry.size_bytes > 0, "{}: size_bytes must be > 0", entry.id);
            assert!(
                entry.param_count_b > 0.0,
                "{}: param_count_b must be > 0",
                entry.id
            );
            assert!(
                entry.active_param_count_b > 0.0,
                "{}: active_param_count_b must be > 0",
                entry.id
            );
            assert!(
                entry.active_param_count_b <= entry.param_count_b,
                "{}: active params must be <= total params",
                entry.id
            );
            assert!(!entry.quant.is_empty(), "{}: quant must not be empty", entry.id);
            assert!(entry.min_ram_gb >= 4, "{}: min_ram_gb must be >= 4", entry.id);
            if entry.is_moe {
                assert!(
                    entry.active_param_count_b < entry.param_count_b,
                    "{}: MoE model must have active < total params",
                    entry.id
                );
            }
        }
    }

    #[test]
    fn test_get_existing() {
        let entry = get("qwen3-8b-q4km");
        assert!(entry.is_some());
        let entry = entry.unwrap();
        assert_eq!(entry.name, "Qwen3 8B Q4_K_M");
        assert_eq!(entry.param_count_b, 8.0);
        assert_eq!(entry.category, ModelCategory::General);
    }

    #[test]
    fn test_get_nonexistent() {
        assert!(get("nonexistent-model").is_none());
    }

    #[test]
    fn test_by_category() {
        let general = by_category(&ModelCategory::General);
        let code = by_category(&ModelCategory::Code);
        let reasoning = by_category(&ModelCategory::Reasoning);

        assert!(general.len() >= 10, "Should have >= 10 general models");
        assert!(code.len() >= 3, "Should have >= 3 code models");
        assert!(reasoning.len() >= 4, "Should have >= 4 reasoning models");

        // Sum should equal total catalog
        assert_eq!(
            general.len() + code.len() + reasoning.len(),
            all().len(),
            "Categories should cover entire catalog"
        );
    }

    #[test]
    fn test_for_device_8gb() {
        // 8 GB Mac: budget = 8 * 0.60 = 4.8 GiB
        // Should include small models (up to ~4B), exclude 8B+
        let device = mock_device(8, 4096);
        let compatible = for_device(&device);

        let ids: Vec<&str> = compatible.iter().map(|e| e.id.as_str()).collect();
        assert!(
            ids.contains(&"smollm2-360m-q8"),
            "360M should fit on 8GB"
        );
        assert!(
            ids.contains(&"qwen3-4b-q4km"),
            "4B Q4 should fit on 8GB"
        );
        assert!(
            !ids.contains(&"qwen3-8b-q4km"),
            "8B Q4 should NOT fit on 8GB"
        );
        assert!(
            !ids.contains(&"llama31-70b-q4km"),
            "70B should NOT fit on 8GB"
        );
    }

    #[test]
    fn test_for_device_128gb() {
        // 128 GB Mac: budget = 128 * 0.60 = 76.8 GiB
        // Should include all or nearly all models
        let device = mock_device(128, 32768);
        let compatible = for_device(&device);

        assert!(
            compatible.len() >= all().len() - 1,
            "128GB should fit most models, got {}/{}",
            compatible.len(),
            all().len()
        );

        let ids: Vec<&str> = compatible.iter().map(|e| e.id.as_str()).collect();
        assert!(ids.contains(&"qwen3-32b-q4km"), "32B should fit on 128GB");
    }

    #[test]
    fn test_recommended_at_most_one_per_category() {
        let device = mock_device(32, 16384);
        let recs = recommended(&device);

        // At most 3 entries (one per category)
        assert!(recs.len() <= 3);

        // No duplicate categories
        let cats: Vec<&ModelCategory> = recs.iter().map(|e| &e.category).collect();
        let unique_cats: HashSet<&&ModelCategory> = cats.iter().collect();
        assert_eq!(cats.len(), unique_cats.len(), "No duplicate categories");
    }

    #[test]
    fn test_recommended_picks_largest() {
        let device = mock_device(16, 8192);
        let recs = recommended(&device);

        // Should have a General recommendation
        let general = recs.iter().find(|e| e.category == ModelCategory::General);
        assert!(general.is_some(), "Should recommend a General model for 16GB");

        let gen = general.unwrap();
        // The recommended general model should be the largest that fits
        let all_general_compatible: Vec<_> = for_device(&device)
            .into_iter()
            .filter(|e| e.category == ModelCategory::General)
            .collect();
        for m in &all_general_compatible {
            assert!(
                m.param_count_b <= gen.param_count_b
                    || (m.param_count_b == gen.param_count_b && m.size_bytes <= gen.size_bytes),
                "Recommended {} ({}B) should be >= {} ({}B)",
                gen.id,
                gen.param_count_b,
                m.id,
                m.param_count_b
            );
        }
    }
}
