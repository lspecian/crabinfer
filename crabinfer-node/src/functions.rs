//! Top-level functions exported to Node.js.

use napi_derive::napi;

use crate::error::to_napi_error;
use crate::types::{CatalogEntry, DeviceInfo, ModelInfo};

/// Get the CrabInfer version.
#[napi]
pub fn version() -> String {
    crabinfer_core::version()
}

/// Detect the current device's capabilities.
#[napi]
pub fn detect_device() -> DeviceInfo {
    crabinfer_core::detect_device().into()
}

/// Estimate peak memory usage for a GGUF model before loading.
/// Returns estimated bytes as i64 (safe for JS number up to 2^53).
#[napi]
pub fn estimate_model_memory(model_path: String, context_length: u32) -> napi::Result<i64> {
    crabinfer_core::estimate_model_memory(model_path, context_length)
        .map(|v| v as i64)
        .map_err(to_napi_error)
}

/// Return the full curated model catalog.
#[napi]
pub fn model_catalog() -> Vec<CatalogEntry> {
    crabinfer_core::model_catalog()
        .into_iter()
        .map(Into::into)
        .collect()
}

/// Return catalog models that fit on the current device.
#[napi]
pub fn models_for_device(device: DeviceInfo) -> Vec<CatalogEntry> {
    let core_device = crabinfer_core::DeviceInfo {
        device_model: device.device_model,
        total_memory_bytes: device.total_memory_bytes as u64,
        available_memory_bytes: device.available_memory_bytes as u64,
        has_metal_gpu: device.has_metal_gpu,
        has_neural_engine: device.has_neural_engine,
        recommended_quant: device.recommended_quant,
        max_model_size_b: device.max_model_size_b,
        max_model_file_size_bytes: device.max_model_file_size_bytes as u64,
        chip_name: device.chip_name,
        chip_variant: device.chip_variant,
        recommended_context_length: device.recommended_context_length,
    };
    crabinfer_core::models_for_device(core_device)
        .into_iter()
        .map(Into::into)
        .collect()
}

/// Return recommended models for the current device.
#[napi]
pub fn recommended_models(device: DeviceInfo) -> Vec<CatalogEntry> {
    let core_device = crabinfer_core::DeviceInfo {
        device_model: device.device_model,
        total_memory_bytes: device.total_memory_bytes as u64,
        available_memory_bytes: device.available_memory_bytes as u64,
        has_metal_gpu: device.has_metal_gpu,
        has_neural_engine: device.has_neural_engine,
        recommended_quant: device.recommended_quant,
        max_model_size_b: device.max_model_size_b,
        max_model_file_size_bytes: device.max_model_file_size_bytes as u64,
        chip_name: device.chip_name,
        chip_variant: device.chip_variant,
        recommended_context_length: device.recommended_context_length,
    };
    crabinfer_core::recommended_models(core_device)
        .into_iter()
        .map(Into::into)
        .collect()
}

/// Look up a single catalog entry by ID.
#[napi]
pub fn catalog_entry(id: String) -> Option<CatalogEntry> {
    crabinfer_core::catalog_entry(id).map(Into::into)
}

/// Peek at GGUF model metadata without loading weights.
/// Returns architecture, parameter count, quantization, etc. from the file header only.
#[napi]
pub fn peek_model_metadata(model_path: String, context_length: u32) -> napi::Result<ModelInfo> {
    let (info, _overhead) = crabinfer_core::engine::peek_model_metadata(&model_path, context_length)
        .map_err(to_napi_error)?;
    Ok(info.into())
}

// ---------------------------------------------------------------------------
// Credential management
// ---------------------------------------------------------------------------

/// Store an API key for a cloud provider.
#[cfg(feature = "providers")]
#[napi]
pub fn set_credential(provider: String, api_key: String) {
    crabinfer_core::set_credential(provider, api_key);
}

/// Retrieve the stored API key for a provider.
#[cfg(feature = "providers")]
#[napi]
pub fn get_credential(provider: String) -> Option<String> {
    crabinfer_core::get_credential(provider)
}

/// Remove a stored API key.
#[cfg(feature = "providers")]
#[napi]
pub fn remove_credential(provider: String) {
    crabinfer_core::remove_credential(provider);
}

/// Check if a key is stored for the given provider.
#[cfg(feature = "providers")]
#[napi]
pub fn has_credential(provider: String) -> bool {
    crabinfer_core::has_credential(provider)
}

/// List provider names that have stored keys.
#[cfg(feature = "providers")]
#[napi]
pub fn configured_providers() -> Vec<String> {
    crabinfer_core::configured_providers()
}

/// Validate a stored API key by making a lightweight API call.
#[cfg(feature = "providers")]
#[napi]
pub fn validate_credential(provider: String) -> napi::Result<bool> {
    crabinfer_core::validate_credential(provider).map_err(to_napi_error)
}
