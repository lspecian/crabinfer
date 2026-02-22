//! Candle backend implementation.
//!
//! This module provides [`CandleBackend`], which implements
//! [`Backend`](crate::backend::Backend) using the Candle ML framework with
//! Metal GPU acceleration for Apple Silicon.

use crate::backend::{Backend, BackendDevice, LoadedModel};
use crate::{CrabInferError, ModelInfo};

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_gemma3::ModelWeights as Gemma3ModelWeights;
use candle_transformers::models::quantized_llama::ModelWeights as LlamaModelWeights;
use candle_transformers::models::quantized_phi3::ModelWeights as Phi3ModelWeights;
use candle_transformers::models::quantized_qwen2::ModelWeights as Qwen2ModelWeights;
use candle_transformers::models::quantized_qwen3::ModelWeights as Qwen3ModelWeights;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Mutex, MutexGuard};
use std::time::Instant;
use tokenizers::Tokenizer;

// ---------------------------------------------------------------------------
// Safe stderr logging macro
// ---------------------------------------------------------------------------

/// Safe stderr logging that never panics.
///
/// On iOS, stderr is not always writable (e.g. when not attached to Xcode
/// debugger, or on background threads). Rust's `eprintln!` panics when the
/// write fails, which is fatal inside `catch_unwind`. This macro uses `write!`
/// and silently drops failures.
macro_rules! log_debug {
    ($($arg:tt)*) => {{
        use std::io::Write;
        let _ = writeln!(std::io::stderr(), $($arg)*);
    }};
}

// ---------------------------------------------------------------------------
// Poison-tolerant Mutex locking
// ---------------------------------------------------------------------------

/// Extension trait for Mutex that recovers from poisoned state.
///
/// When a panic occurs (e.g. Candle's Metal shader compilation panics with
/// XPC_ERROR_CONNECTION_INTERRUPTED), any Mutex held during the panic becomes
/// "poisoned". Standard `.lock().unwrap()` would then panic on the *next*
/// lock attempt, causing a cascade of failures. This trait recovers the inner
/// value instead, breaking the cascade.
trait MutexExt<T> {
    fn lock_recover(&self) -> MutexGuard<'_, T>;
}

impl<T> MutexExt<T> for Mutex<T> {
    fn lock_recover(&self) -> MutexGuard<'_, T> {
        self.lock().unwrap_or_else(|poisoned| {
            log_debug!("[CrabInfer-Candle] Recovered poisoned Mutex (prior panic was caught)");
            poisoned.into_inner()
        })
    }
}

// ---------------------------------------------------------------------------
// Multi-architecture model wrapper (internal to this backend)
// ---------------------------------------------------------------------------

/// Supported GGUF model architectures.
///
/// Each variant wraps the quantized model weights from `candle_transformers`.
enum Model {
    Phi3(Phi3ModelWeights),
    Qwen2(Qwen2ModelWeights),
    Qwen3(Qwen3ModelWeights),
    Llama(LlamaModelWeights),
    Gemma3(Gemma3ModelWeights),
}

impl Model {
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> candle_core::Result<Tensor> {
        match self {
            Model::Phi3(m) => m.forward(x, index_pos),
            Model::Qwen2(m) => m.forward(x, index_pos),
            Model::Qwen3(m) => m.forward(x, index_pos),
            Model::Llama(m) => m.forward(x, index_pos),
            Model::Gemma3(m) => m.forward(x, index_pos),
        }
    }

    /// Explicitly clear KV caches.
    ///
    /// Phi3, Qwen2, Llama, and Gemma3 auto-reset their KV cache when
    /// `forward()` is called with `index_pos == 0`, but Qwen3 uses
    /// `ConcatKvCache` which only appends -- it must be cleared explicitly.
    fn clear_kv_cache(&mut self) {
        match self {
            Model::Qwen3(m) => m.clear_kv_cache(),
            // Other architectures auto-reset on index_pos==0
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// CandleBackend
// ---------------------------------------------------------------------------

/// Candle-based inference backend with Metal GPU acceleration.
///
/// Holds a persistent Metal device that is reused across load/unload cycles
/// to avoid recreating command queues, recompiling shaders, and leaking
/// Metal framework allocations.
pub struct CandleBackend {
    /// Persistent Metal device -- created once and reused across model loads.
    metal_device: Mutex<Option<Device>>,
}

impl CandleBackend {
    /// Create a new Candle backend.
    pub fn new() -> Self {
        Self {
            metal_device: Mutex::new(None),
        }
    }
}

impl Default for CandleBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CandleBackend {
    fn name(&self) -> &str {
        "candle"
    }

    fn load_model(
        &self,
        model_path: &str,
        device: BackendDevice,
        metallib_path: Option<&str>,
    ) -> Result<Box<dyn LoadedModel>, CrabInferError> {
        log_debug!("[CrabInfer-Candle] load_model called: {}", model_path);

        // Check if file exists
        if !std::path::Path::new(model_path).exists() {
            log_debug!("[CrabInfer-Candle] ERROR: model file not found");
            return Err(CrabInferError::ModelNotFound);
        }

        // Get file size
        let file_size = std::fs::metadata(model_path)
            .map_err(|e| CrabInferError::ModelLoadFailed {
                reason: format!("cannot read metadata: {}", e),
            })?
            .len();
        let file_size_mb = file_size / (1024 * 1024);
        log_debug!("[CrabInfer-Candle] Model file size: {} MB", file_size_mb);

        let use_metal = matches!(device, BackendDevice::Metal);

        // Reuse the persistent Metal device if available, otherwise create one.
        let cached_device = self.metal_device.lock_recover().clone();

        let (weights, info, candle_device) = load_model_weights(
            model_path,
            file_size,
            // Use a large default context length; the engine will override
            // with its configured value.
            4096,
            use_metal,
            metallib_path,
            cached_device,
        )?;

        // Cache the device for future loads
        *self.metal_device.lock_recover() = Some(candle_device.clone());

        // Load tokenizer from a file next to the model (tokenizer.json)
        let tokenizer = load_tokenizer(model_path)?;

        // Resolve EOS token ID -- try common EOS markers
        let eos_token_id = tokenizer
            .token_to_id("<|endoftext|>")
            .or_else(|| tokenizer.token_to_id("<|end|>"))
            .or_else(|| tokenizer.token_to_id("</s>"))
            .or_else(|| tokenizer.token_to_id("<|im_end|>"))
            .or_else(|| tokenizer.token_to_id("<end_of_turn>"))
            .unwrap_or(2); // fallback

        log_debug!(
            "[CrabInfer-Candle] Model loaded on {:?}, EOS token id: {}",
            candle_device,
            eos_token_id
        );

        Ok(Box::new(CandleLoadedModel {
            weights,
            tokenizer,
            device: candle_device,
            eos_token_id,
            info,
        }))
    }

    fn metal_allocated_bytes(&self) -> Option<usize> {
        #[cfg(feature = "metal")]
        {
            let guard = self.metal_device.lock_recover();
            if let Some(ref device) = *guard {
                if let Ok(metal) = device.as_metal_device() {
                    return Some(metal.current_allocated_size());
                }
            }
        }
        None
    }

    fn release_gpu_buffers(&self) -> Result<(), CrabInferError> {
        #[cfg(feature = "metal")]
        {
            let guard = self.metal_device.lock_recover();
            if let Some(ref device) = *guard {
                if let Ok(metal) = device.as_metal_device() {
                    metal.release_unused_buffers().map_err(|e| {
                        CrabInferError::ModelLoadFailed {
                            reason: format!("failed to release Metal buffers: {:?}", e),
                        }
                    })?;
                }
            }
        }
        Ok(())
    }

    fn wait_until_completed(&self) -> Result<(), CrabInferError> {
        #[cfg(feature = "metal")]
        {
            let guard = self.metal_device.lock_recover();
            if let Some(ref device) = *guard {
                if let Ok(metal) = device.as_metal_device() {
                    let _ = metal.wait_until_completed();
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CandleLoadedModel
// ---------------------------------------------------------------------------

/// A model loaded into Candle, ready for inference.
pub struct CandleLoadedModel {
    weights: Model,
    tokenizer: Tokenizer,
    device: Device,
    eos_token_id: u32,
    info: ModelInfo,
}

impl LoadedModel for CandleLoadedModel {
    fn forward(&mut self, tokens: &[u32], position: usize) -> Result<Vec<f32>, CrabInferError> {
        let input = Tensor::new(tokens, &self.device)
            .map_err(|e| {
                log_debug!("[CrabInfer-Candle] Failed to create input tensor: {}", e);
                CrabInferError::InferenceFailed
            })?
            .unsqueeze(0)
            .map_err(|_| CrabInferError::InferenceFailed)?;

        let logits = self
            .weights
            .forward(&input, position)
            .map_err(|e| {
                log_debug!("[CrabInfer-Candle] Forward pass failed: {}", e);
                CrabInferError::InferenceFailed
            })?;

        // Squeeze the batch dimension and convert to Vec<f32>
        let logits = logits
            .squeeze(0)
            .map_err(|_| CrabInferError::InferenceFailed)?;

        logits
            .to_vec1::<f32>()
            .map_err(|_| CrabInferError::InferenceFailed)
    }

    fn clear_kv_cache(&mut self) {
        self.weights.clear_kv_cache();
    }

    fn model_info(&self) -> ModelInfo {
        self.info.clone()
    }

    fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}

// ---------------------------------------------------------------------------
// Model loading with multi-architecture support
// ---------------------------------------------------------------------------

/// Load model weights, trying Metal first and falling back to CPU if Metal
/// ops are unsupported.
///
/// Detects architecture from GGUF metadata and loads the correct model type.
///
/// For Metal: wraps loading + a warmup forward pass in `catch_unwind` to catch
/// panics from Candle's Metal shader compilation (e.g. XPC_ERROR_CONNECTION_INTERRUPTED
/// on iOS). If the warmup panics, we fall back to CPU automatically.
fn load_model_weights(
    model_path: &str,
    file_size: u64,
    default_context_length: u32,
    use_metal: bool,
    metallib_path: Option<&str>,
    cached_device: Option<Device>,
) -> Result<(Model, ModelInfo, Device), CrabInferError> {
    let devices_to_try: Vec<Device> = if use_metal {
        #[cfg(not(feature = "metal"))]
        {
            let _ = (metallib_path, &cached_device);
            log_debug!("[CrabInfer-Candle] Metal requested but not compiled in, using CPU");
            vec![Device::Cpu]
        }
        #[cfg(feature = "metal")]
        // Reuse cached Metal device if available
        if let Some(ref dev) = cached_device {
            if matches!(dev, Device::Metal(_)) {
                log_debug!("[CrabInfer-Candle] Reusing cached Metal device");
                vec![dev.clone(), Device::Cpu]
            } else {
                log_debug!(
                    "[CrabInfer-Candle] Cached device is CPU, creating Metal device..."
                );
                match Device::new_metal(0) {
                    Ok(metal) => {
                        if let Some(dir) = metallib_path {
                            if let Ok(metal_dev) = metal.as_metal_device() {
                                log_debug!("[CrabInfer-Candle] Setting metallib dir: {}", dir);
                                metal_dev.set_metallib_dir(dir);
                            }
                        }
                        vec![metal, Device::Cpu]
                    }
                    Err(e) => {
                        log_debug!(
                            "[CrabInfer-Candle] Metal device FAILED: {}, falling back to CPU",
                            e
                        );
                        vec![Device::Cpu]
                    }
                }
            }
        } else {
            log_debug!("[CrabInfer-Candle] Metal requested, creating Metal device...");
            match Device::new_metal(0) {
                Ok(metal) => {
                    if let Some(dir) = metallib_path {
                        if let Ok(metal_dev) = metal.as_metal_device() {
                            log_debug!("[CrabInfer-Candle] Setting metallib dir: {}", dir);
                            metal_dev.set_metallib_dir(dir);
                        }
                    } else {
                        log_debug!(
                            "[CrabInfer-Candle] WARNING: No metallib_path provided \
                             -- Metal will use runtime XPC compilation (may fail on iOS)"
                        );
                    }
                    log_debug!("[CrabInfer-Candle] Metal device created OK");
                    vec![metal, Device::Cpu]
                }
                Err(e) => {
                    log_debug!(
                        "[CrabInfer-Candle] Metal device FAILED: {}, falling back to CPU",
                        e
                    );
                    vec![Device::Cpu]
                }
            }
        }
    } else {
        log_debug!("[CrabInfer-Candle] CPU-only mode requested");
        vec![Device::Cpu]
    };

    let is_reused_device = cached_device.is_some();
    let mut last_error = String::from("no devices available");

    for device in devices_to_try {
        log_debug!("[CrabInfer-Candle] Trying device: {:?}", device);
        tracing::info!("Trying device: {:?}", device);

        let mut file = std::fs::File::open(model_path).map_err(|e| {
            CrabInferError::ModelLoadFailed {
                reason: format!("cannot open file: {}", e),
            }
        })?;

        let content = gguf_file::Content::read(&mut file).map_err(|e| {
            tracing::error!("Failed to parse GGUF: {}", e);
            CrabInferError::ModelLoadFailed {
                reason: format!("invalid GGUF: {}", e),
            }
        })?;

        let info = extract_model_info(&content, model_path, file_size, default_context_length);
        tracing::info!(
            "Model: {} | arch: {} | quant: {} | vocab: {} | ctx: {}",
            info.model_name,
            info.architecture,
            info.quantization,
            info.vocab_size,
            info.context_length
        );

        let arch = info.architecture.to_lowercase();
        log_debug!(
            "[CrabInfer-Candle] Loading weights for arch '{}' on {:?}...",
            arch,
            device
        );
        let weight_start = Instant::now();

        let is_metal = matches!(&device, Device::Metal(_));

        let model_result: Result<Model, candle_core::Error> = if is_metal {
            // Wrap Metal loading + warmup in catch_unwind as a safety net.
            //
            // Our Candle fork now returns errors instead of panicking on
            // Metal shader compilation failures, and supports pre-compiled
            // metallib loading. But we keep catch_unwind as defense-in-depth
            // in case other Metal code paths still panic.
            let model_path_owned = model_path.to_string();
            let result = catch_unwind(AssertUnwindSafe(
                || -> candle_core::Result<Model> {
                    // Qwen3 GGUFs from different sources may use either "qwen3.*" or
                    // "qwen2.*" metadata keys. Probe for the actual key prefix to pick
                    // the right loader (Qwen3 vs Qwen2).
                    let qwen3_has_native_keys =
                        content.metadata.contains_key("qwen3.attention.head_count");

                    let mut weights = match arch.as_str() {
                        "phi3" => {
                            Phi3ModelWeights::from_gguf(false, content, &mut file, &device)
                                .map(Model::Phi3)?
                        }
                        "qwen2" => Qwen2ModelWeights::from_gguf(content, &mut file, &device)
                            .map(Model::Qwen2)?,
                        "qwen3" if qwen3_has_native_keys => {
                            log_debug!(
                                "[CrabInfer-Candle] Using Qwen3 loader (qwen3.* metadata keys found)"
                            );
                            Qwen3ModelWeights::from_gguf(content, &mut file, &device)
                                .map(Model::Qwen3)?
                        }
                        "qwen3" => {
                            log_debug!(
                                "[CrabInfer-Candle] Using Qwen2 loader for qwen3 arch \
                                 (qwen2.* metadata keys)"
                            );
                            Qwen2ModelWeights::from_gguf(content, &mut file, &device)
                                .map(Model::Qwen2)?
                        }
                        "gemma3" | "gemma2" | "gemma" => {
                            Gemma3ModelWeights::from_gguf(content, &mut file, &device)
                                .map(Model::Gemma3)?
                        }
                        "llama" => LlamaModelWeights::from_gguf(content, &mut file, &device)
                            .map(Model::Llama)?,
                        _other => {
                            tracing::warn!(
                                "Unknown architecture '{}', trying Llama-compatible loader",
                                _other
                            );
                            let mut file2 = std::fs::File::open(&model_path_owned).map_err(
                                |e| candle_core::Error::Msg(format!("Reopen failed: {}", e)),
                            )?;
                            let content2 = gguf_file::Content::read(&mut file2)?;
                            LlamaModelWeights::from_gguf(content2, &mut file2, &device)
                                .map(Model::Llama)?
                        }
                    };

                    // Warmup: trigger Metal shader compilation NOW during load_model()
                    // rather than letting it panic during the first next_token() call.
                    let warmup_start = Instant::now();

                    if is_reused_device {
                        // Shaders already compiled on the cached device -- skip tensor warmup,
                        // just run a forward pass to populate KV cache shapes for new weights.
                        log_debug!(
                            "[CrabInfer-Candle] Reused device, skipping shader warmup..."
                        );
                    } else {
                        // Step 1: Basic tensor ops (compiles fill.metal, binary.metal, cast.metal)
                        log_debug!(
                            "[CrabInfer-Candle] Metal warmup 1/3: basic tensor ops..."
                        );
                        let a =
                            Tensor::zeros((2, 2), candle_core::DType::F32, &device)?;
                        let b =
                            Tensor::ones((2, 2), candle_core::DType::F32, &device)?;
                        let _c = (&a + &b)?;
                        log_debug!(
                            "[CrabInfer-Candle] Metal warmup 1/3: OK ({:.1}s)",
                            warmup_start.elapsed().as_secs_f64()
                        );

                        // Step 2: Unary + reduce ops (compiles unary.metal, reduce.metal)
                        log_debug!(
                            "[CrabInfer-Candle] Metal warmup 2/3: unary + reduce ops..."
                        );
                        let _d = b.sqrt()?;
                        let _e = a.sum_all()?;
                        log_debug!(
                            "[CrabInfer-Candle] Metal warmup 2/3: OK ({:.1}s)",
                            warmup_start.elapsed().as_secs_f64()
                        );
                    }

                    // Forward pass warmup -- always run for new weights (compiles
                    // quantized.metal kernels specific to this model's tensor shapes)
                    log_debug!(
                        "[CrabInfer-Candle] Metal warmup: quantized forward pass..."
                    );
                    let dummy = Tensor::new(&[1u32], &device)?.unsqueeze(0)?;
                    weights.forward(&dummy, 0)?;
                    weights.clear_kv_cache();
                    log_debug!(
                        "[CrabInfer-Candle] Metal warmup complete! ({:.1}s total)",
                        warmup_start.elapsed().as_secs_f64()
                    );

                    Ok(weights)
                },
            ));

            match result {
                Ok(inner) => inner,
                Err(panic_payload) => {
                    let msg = extract_panic_message(&panic_payload);
                    log_debug!("[CrabInfer-Candle] Metal PANICKED: {}", msg);
                    Err(candle_core::Error::Msg(format!(
                        "Metal shader compilation panic: {}",
                        msg
                    )))
                }
            }
        } else {
            // CPU path: no catch_unwind needed (no Metal shader compilation)
            let qwen3_has_native_keys_cpu =
                content.metadata.contains_key("qwen3.attention.head_count");
            match arch.as_str() {
                "phi3" => Phi3ModelWeights::from_gguf(false, content, &mut file, &device)
                    .map(Model::Phi3),
                "qwen2" => Qwen2ModelWeights::from_gguf(content, &mut file, &device)
                    .map(Model::Qwen2),
                "qwen3" if qwen3_has_native_keys_cpu => {
                    Qwen3ModelWeights::from_gguf(content, &mut file, &device).map(Model::Qwen3)
                }
                "qwen3" => Qwen2ModelWeights::from_gguf(content, &mut file, &device)
                    .map(Model::Qwen2),
                "gemma3" | "gemma2" | "gemma" => {
                    Gemma3ModelWeights::from_gguf(content, &mut file, &device).map(Model::Gemma3)
                }
                "llama" => LlamaModelWeights::from_gguf(content, &mut file, &device)
                    .map(Model::Llama),
                other => {
                    tracing::warn!(
                        "Unknown architecture '{}', trying Llama-compatible loader",
                        other
                    );
                    // Re-read content since we consumed it
                    let mut file2 = std::fs::File::open(model_path).map_err(|e| {
                        CrabInferError::ModelLoadFailed {
                            reason: format!("reopen: {}", e),
                        }
                    })?;
                    let content2 = gguf_file::Content::read(&mut file2).map_err(|e| {
                        CrabInferError::ModelLoadFailed {
                            reason: format!("GGUF parse: {}", e),
                        }
                    })?;
                    LlamaModelWeights::from_gguf(content2, &mut file2, &device).map(Model::Llama)
                }
            }
        };

        let weight_time = weight_start.elapsed();
        match model_result {
            Ok(weights) => {
                log_debug!(
                    "[CrabInfer-Candle] Weights loaded on {:?} in {:.2}s",
                    device,
                    weight_time.as_secs_f64()
                );
                tracing::info!("Model loaded on {:?}", device);
                return Ok((weights, info, device));
            }
            Err(e) => {
                let msg = format!("{:?}: {}", device, e);
                log_debug!(
                    "[CrabInfer-Candle] FAILED on {} (after {:.2}s)",
                    msg,
                    weight_time.as_secs_f64()
                );
                tracing::warn!(
                    "Failed to load on {:?}: {}, trying next device...",
                    device,
                    e
                );
                last_error = msg;
                continue;
            }
        }
    }

    tracing::error!("Failed to load model on any device: {}", last_error);
    Err(CrabInferError::ModelLoadFailed {
        reason: last_error,
    })
}

// ---------------------------------------------------------------------------
// Panic message extraction
// ---------------------------------------------------------------------------

/// Extract a human-readable message from a panic payload.
fn extract_panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else {
        "unknown panic".to_string()
    }
}

// ---------------------------------------------------------------------------
// GGUF metadata extraction
// ---------------------------------------------------------------------------

/// Extract ModelInfo from GGUF metadata.
fn extract_model_info(
    content: &gguf_file::Content,
    model_path: &str,
    file_size: u64,
    default_context_length: u32,
) -> ModelInfo {
    let md = &content.metadata;

    // Model name: try general.name, fall back to filename
    let model_name = md
        .get("general.name")
        .and_then(|v| v.to_string().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            std::path::Path::new(model_path)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "unknown".to_string())
        });

    // Architecture: general.architecture
    let architecture = md
        .get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "unknown".to_string());

    // Context length from metadata
    let context_length =
        get_metadata_u32(md, &format!("{}.context_length", architecture))
            .unwrap_or(default_context_length);

    // Vocab size from embedding tensor shape
    let vocab_size = content
        .tensor_infos
        .get("token_embd.weight")
        .map(|t| t.shape.dims()[0] as u32)
        .unwrap_or(0);

    // Determine dominant quantization from tensor types
    let quantization = detect_quantization(&content.tensor_infos);

    // Estimate parameter count from tensor shapes
    let parameter_count: u64 = content
        .tensor_infos
        .values()
        .map(|t| t.shape.elem_count() as u64)
        .sum();

    // Detect MoE (Mixture of Experts) from GGUF metadata.
    // Standard keys: <arch>.expert_count, <arch>.expert_used_count
    let expert_count =
        get_metadata_u32(md, &format!("{}.expert_count", architecture)).unwrap_or(0);
    let expert_used_count =
        get_metadata_u32(md, &format!("{}.expert_used_count", architecture)).unwrap_or(0);
    let is_moe = expert_count > 0 && expert_used_count > 0;

    // Active parameter count: for MoE, roughly (shared_params + expert_fraction).
    let active_parameter_count = if is_moe && expert_count > expert_used_count {
        let shared_fraction = 0.20;
        let expert_fraction =
            (1.0 - shared_fraction) * (expert_used_count as f64 / expert_count as f64);
        ((shared_fraction + expert_fraction) * parameter_count as f64) as u64
    } else {
        parameter_count
    };

    ModelInfo {
        model_name,
        architecture,
        parameter_count,
        quantization,
        file_size_bytes: file_size,
        context_length,
        vocab_size,
        is_moe,
        expert_count,
        expert_used_count,
        active_parameter_count,
    }
}

/// Read a u32 from GGUF metadata, handling various integer types.
fn get_metadata_u32(
    md: &std::collections::HashMap<String, gguf_file::Value>,
    key: &str,
) -> Option<u32> {
    md.get(key).and_then(|v| match v {
        gguf_file::Value::U32(n) => Some(*n),
        gguf_file::Value::U64(n) => Some(*n as u32),
        gguf_file::Value::I32(n) => Some(*n as u32),
        _ => None,
    })
}

/// Detect the dominant quantization type from tensor infos.
fn detect_quantization(
    tensor_infos: &std::collections::HashMap<String, gguf_file::TensorInfo>,
) -> String {
    use candle_core::quantized::GgmlDType;
    use std::collections::HashMap;

    let mut counts: HashMap<GgmlDType, usize> = HashMap::new();
    for info in tensor_infos.values() {
        *counts.entry(info.ggml_dtype).or_default() += 1;
    }

    // Find the most common quantization type (excluding F32/F16 which are used for norms)
    let dominant = counts
        .iter()
        .filter(|(dtype, _)| !matches!(dtype, GgmlDType::F32 | GgmlDType::F16))
        .max_by_key(|(_, count)| *count)
        .map(|(dtype, _)| *dtype);

    match dominant {
        Some(GgmlDType::Q2K) => "Q2_K".to_string(),
        Some(GgmlDType::Q3K) => "Q3_K".to_string(),
        Some(GgmlDType::Q4K) => "Q4_K".to_string(),
        Some(GgmlDType::Q4_0) => "Q4_0".to_string(),
        Some(GgmlDType::Q4_1) => "Q4_1".to_string(),
        Some(GgmlDType::Q5K) => "Q5_K".to_string(),
        Some(GgmlDType::Q5_0) => "Q5_0".to_string(),
        Some(GgmlDType::Q5_1) => "Q5_1".to_string(),
        Some(GgmlDType::Q6K) => "Q6_K".to_string(),
        Some(GgmlDType::Q8_0) => "Q8_0".to_string(),
        Some(GgmlDType::Q8_1) => "Q8_1".to_string(),
        Some(GgmlDType::Q8K) => "Q8_K".to_string(),
        Some(GgmlDType::BF16) => "BF16".to_string(),
        Some(GgmlDType::F16) => "F16".to_string(),
        Some(GgmlDType::F32) => "F32".to_string(),
        None => "unknown".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Tokenizer loading
// ---------------------------------------------------------------------------

/// Load tokenizer from a tokenizer.json file adjacent to the model.
fn load_tokenizer(model_path: &str) -> Result<Tokenizer, CrabInferError> {
    let model_dir = std::path::Path::new(model_path)
        .parent()
        .ok_or(CrabInferError::ModelLoadFailed {
            reason: "invalid model path".into(),
        })?;

    let model_stem = std::path::Path::new(model_path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();

    // Search order (model-specific first to avoid wrong tokenizer):
    // 1. <model-stem-dir>/tokenizer.json (e.g. qwen2.5-7b/tokenizer.json)
    // 2. tokenizer.json next to the model file (works when only one model in dir)
    // 3. One level up: ../tokenizer.json
    let stem_dir = model_stem_to_dir(&model_stem);
    let candidates = vec![
        model_dir.join(&stem_dir).join("tokenizer.json"),
        model_dir.join("tokenizer.json"),
        model_dir
            .parent()
            .map(|p| p.join("tokenizer.json"))
            .unwrap_or_default(),
    ];

    for path in &candidates {
        if path.exists() {
            tracing::info!("Loading tokenizer from: {}", path.display());
            return Tokenizer::from_file(path).map_err(|e| {
                tracing::error!("Failed to load tokenizer: {}", e);
                CrabInferError::TokenizationFailed
            });
        }
    }

    tracing::error!(
        "No tokenizer.json found. Searched: {:?}",
        candidates
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
    );
    Err(CrabInferError::TokenizationFailed)
}

/// Convert a model filename stem to a likely tokenizer directory name.
/// e.g. "qwen2.5-7b-instruct-q4_k_m" -> "qwen2.5-7b"
fn model_stem_to_dir(stem: &str) -> String {
    // Strip common quantization suffixes
    let s = stem.to_lowercase();
    for suffix in &[
        "-q4_k_m", "-q4_k_s", "-q5_k_m", "-q5_k_s", "-q4_0", "-q8_0", "-q6_k", "-q3_k_m",
        "-q2_k",
    ] {
        if let Some(prefix) = s.strip_suffix(suffix) {
            // Also strip "-instruct", "-chat" etc. for the directory
            let prefix = prefix
                .strip_suffix("-instruct")
                .or_else(|| prefix.strip_suffix("-chat"))
                .unwrap_or(prefix);
            return prefix.to_string();
        }
    }
    s
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_name() {
        let backend = CandleBackend::new();
        assert_eq!(backend.name(), "candle");
    }

    #[test]
    fn test_model_not_found() {
        let backend = CandleBackend::new();
        let result = backend.load_model("/nonexistent/model.gguf", BackendDevice::Cpu, None);
        assert!(matches!(result, Err(CrabInferError::ModelNotFound)));
    }

    #[test]
    fn test_metal_allocated_bytes_no_device() {
        let backend = CandleBackend::new();
        // No Metal device initialized yet, should return None
        assert_eq!(backend.metal_allocated_bytes(), None);
    }

    #[test]
    fn test_release_gpu_buffers_no_device() {
        let backend = CandleBackend::new();
        // Should succeed even without a device
        assert!(backend.release_gpu_buffers().is_ok());
    }

    #[test]
    fn test_wait_until_completed_no_device() {
        let backend = CandleBackend::new();
        // Should succeed even without a device
        assert!(backend.wait_until_completed().is_ok());
    }

    #[test]
    fn test_model_stem_to_dir() {
        assert_eq!(
            model_stem_to_dir("qwen2.5-7b-instruct-q4_k_m"),
            "qwen2.5-7b"
        );
        assert_eq!(
            model_stem_to_dir("Phi-3-mini-4k-instruct-q4_k_m"),
            "phi-3-mini-4k"
        );
        assert_eq!(model_stem_to_dir("llama-7b-chat-q4_0"), "llama-7b");
    }

    #[test]
    fn test_backend_device_equality() {
        assert_eq!(BackendDevice::Cpu, BackendDevice::Cpu);
        assert_eq!(BackendDevice::Metal, BackendDevice::Metal);
        assert_ne!(BackendDevice::Cpu, BackendDevice::Metal);
    }
}
