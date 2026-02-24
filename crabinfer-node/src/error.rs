//! Error mapping from CrabInferError to napi::Error.

use crabinfer_core::CrabInferError;
use napi::Status;

/// Convert a CrabInferError into a napi::Error with a descriptive code prefix.
pub fn to_napi_error(e: CrabInferError) -> napi::Error {
    let (code, message) = match &e {
        CrabInferError::ModelNotFound => ("MODEL_NOT_FOUND", e.to_string()),
        CrabInferError::ModelLoadFailed { .. } => ("MODEL_LOAD_FAILED", e.to_string()),
        CrabInferError::OutOfMemory { .. } => ("OUT_OF_MEMORY", e.to_string()),
        CrabInferError::MetalNotAvailable => ("METAL_NOT_AVAILABLE", e.to_string()),
        CrabInferError::TokenizationFailed => ("TOKENIZATION_FAILED", e.to_string()),
        CrabInferError::InferenceFailed => ("INFERENCE_FAILED", e.to_string()),
        CrabInferError::ContextOverflow => ("CONTEXT_OVERFLOW", e.to_string()),
        CrabInferError::InvalidConfig => ("INVALID_CONFIG", e.to_string()),
        CrabInferError::DeviceNotSupported => ("DEVICE_NOT_SUPPORTED", e.to_string()),
        CrabInferError::ModelTooLarge { .. } => ("MODEL_TOO_LARGE", e.to_string()),
        CrabInferError::NetworkError { .. } => ("NETWORK_ERROR", e.to_string()),
        CrabInferError::ApiError { .. } => ("API_ERROR", e.to_string()),
        CrabInferError::AuthenticationFailed { .. } => ("AUTH_FAILED", e.to_string()),
        CrabInferError::RateLimited { .. } => ("RATE_LIMITED", e.to_string()),
        CrabInferError::ProviderNotAvailable { .. } => ("PROVIDER_NOT_AVAILABLE", e.to_string()),
        CrabInferError::DownloadFailed { .. } => ("DOWNLOAD_FAILED", e.to_string()),
        CrabInferError::IntegrityCheckFailed { .. } => ("INTEGRITY_CHECK_FAILED", e.to_string()),
        CrabInferError::StorageError { .. } => ("STORAGE_ERROR", e.to_string()),
    };
    napi::Error::new(Status::GenericFailure, format!("[{}] {}", code, message))
}
