/// Core inference engine
/// Wraps Candle for model loading and token generation
/// with iOS-specific memory management and device awareness

/// Safe stderr logging that never panics.
///
/// On iOS, stderr is not always writable (e.g. when not attached to Xcode
/// debugger, or on background threads). Rust's `log_debug!` panics when the
/// write fails, which is fatal inside `catch_unwind` — it gets caught and
/// reported as the actual error.  This macro uses `write!` and silently
/// drops failures.
macro_rules! log_debug {
    ($($arg:tt)*) => {{
        use std::io::Write;
        let _ = writeln!(std::io::stderr(), $($arg)*);
    }};
}

use crate::{
    CrabInferError, EngineConfig, GenerationStats, MemoryPressure,
    ModelInfo, TokenOutput, memory::MemoryPressureManager,
};
use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_gemma3::ModelWeights as Gemma3ModelWeights;
use candle_transformers::models::quantized_llama::ModelWeights as LlamaModelWeights;
use candle_transformers::models::quantized_phi3::ModelWeights as Phi3ModelWeights;
use candle_transformers::models::quantized_qwen2::ModelWeights as Qwen2ModelWeights;
use candle_transformers::models::quantized_qwen3::ModelWeights as Qwen3ModelWeights;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

// ---------------------------------------------------------------------------
// Multi-architecture model wrapper
// ---------------------------------------------------------------------------

/// Supported GGUF model architectures
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
    /// `ConcatKvCache` which only appends — it must be cleared explicitly.
    fn clear_kv_cache(&mut self) {
        match self {
            Model::Qwen3(m) => m.clear_kv_cache(),
            // Other architectures auto-reset on index_pos==0
            _ => {}
        }
    }

    /// Whether this architecture requires sequential (token-by-token) prefill
    /// instead of batch prefill on Metal.
    ///
    /// Qwen3's attention uses flatten-after-transpose for QK-norm, which
    /// triggers a Metal NaN bug when `seq_len == num_kv_heads` (typically 8).
    /// Sequential prefill avoids this entirely since each call has seq_len=1.
    fn needs_sequential_prefill(&self, device: &Device) -> bool {
        matches!((self, device), (Model::Qwen3(_), Device::Metal(_)))
    }
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
            log_debug!("[CrabInfer] Recovered poisoned Mutex (prior panic was caught)");
            poisoned.into_inner()
        })
    }
}

// ---------------------------------------------------------------------------
// Streaming state for next_token()
// ---------------------------------------------------------------------------

/// Persistent state for incremental token generation via next_token()
struct StreamingState {
    /// Tokens from the initial prompt (set on first call)
    prompt_tokens: Vec<u32>,
    /// Tokens generated so far
    generated_tokens: Vec<u32>,
    /// Current position in the KV cache (prompt_len + generated so far)
    position: usize,
    /// Logits processor (maintains sampling state / seed)
    logits_processor: LogitsProcessor,
    /// When generation started
    start_time: Instant,
    /// Time to first token (set after first generation step)
    first_token_time: Option<Duration>,
}

// ---------------------------------------------------------------------------
// Loaded model state
// ---------------------------------------------------------------------------

/// Loaded model state — holds the Candle model weights and tokenizer
struct LoadedModel {
    weights: Model,
    tokenizer: Tokenizer,
    device: Device,
    eos_token_id: u32,
}

// ---------------------------------------------------------------------------
// Public engine
// ---------------------------------------------------------------------------

/// RAII guard that clears the `generating` flag on drop, ensuring it is
/// reset even on early returns or errors during inference.
struct GenerationGuard<'a>(&'a AtomicBool);

impl Drop for GenerationGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// The main inference engine
pub struct CrabInferEngine {
    config: EngineConfig,
    memory_manager: Mutex<MemoryPressureManager>,
    model: Mutex<Option<LoadedModel>>,
    model_info: Mutex<Option<ModelInfo>>,
    last_stats: Mutex<Option<GenerationStats>>,
    streaming: Mutex<Option<StreamingState>>,
    /// Persistent Metal device — created once and reused across load/unload
    /// cycles. This avoids recreating command queues, recompiling shaders,
    /// and leaking Metal framework allocations on each load.
    metal_device: Mutex<Option<Device>>,
    /// Prevents concurrent `next_token()` / `complete()` calls from
    /// corrupting KV cache state.
    generating: AtomicBool,
}

impl CrabInferEngine {
    pub fn new(config: EngineConfig) -> Result<Self, CrabInferError> {
        // Validate config
        if config.temperature < 0.0 || config.temperature > 2.0 {
            return Err(CrabInferError::InvalidConfig);
        }
        if config.top_p < 0.0 || config.top_p > 1.0 {
            return Err(CrabInferError::InvalidConfig);
        }

        // Set up memory manager
        let memory_limit = if config.memory_limit_bytes > 0 {
            config.memory_limit_bytes
        } else {
            // Auto-detect: use 70% of available memory
            let device = crate::device::detect_device();
            (device.available_memory_bytes as f64 * 0.70) as u64
        };

        let memory_manager = MemoryPressureManager::new(memory_limit);

        let engine = Self {
            config,
            memory_manager: Mutex::new(memory_manager),
            model: Mutex::new(None),
            model_info: Mutex::new(None),
            last_stats: Mutex::new(None),
            streaming: Mutex::new(None),
            metal_device: Mutex::new(None),
            generating: AtomicBool::new(false),
        };

        // Auto-load model if path provided
        if !engine.config.model_path.is_empty() {
            engine.load_model(engine.config.model_path.clone())?;
        }

        Ok(engine)
    }

    pub fn load_model(&self, model_path: String) -> Result<(), CrabInferError> {
        log_debug!("[CrabInfer] load_model called: {}", model_path);

        // Check if file exists
        if !std::path::Path::new(&model_path).exists() {
            log_debug!("[CrabInfer] ERROR: model file not found");
            return Err(CrabInferError::ModelNotFound);
        }

        // Check file size
        let file_size = std::fs::metadata(&model_path)
            .map_err(|e| CrabInferError::ModelLoadFailed { reason: format!("cannot read metadata: {}", e) })?
            .len();
        let file_size_mb = file_size / (1024 * 1024);
        log_debug!("[CrabInfer] Model file size: {} MB", file_size_mb);

        if file_size < 100 * 1024 * 1024 {
            tracing::warn!(
                "Model file is small ({} MB), may not be a valid GGUF",
                file_size_mb
            );
        }

        // Model size enforcement: reject files larger than the device tier limit
        // to prevent jetsam kills on iOS.
        let max_file_size = crate::device::max_safe_model_file_size();
        let max_file_size_mb = max_file_size / (1024 * 1024);
        if file_size > max_file_size {
            log_debug!(
                "[CrabInfer] REFUSING: model {} MB exceeds device limit {} MB",
                file_size_mb, max_file_size_mb
            );
            return Err(CrabInferError::ModelTooLarge {
                file_size_mb,
                max_allowed_mb: max_file_size_mb,
            });
        }

        // Pre-flight memory check: read GGUF header to get model metadata,
        // then use the calibrated multi-component estimator (weights + KV cache
        // + activations + embedding overhead × 1.35 safety) instead of the old
        // crude file_size × 1.2 heuristic.
        let available_mb = {
            #[cfg(any(target_os = "ios", target_os = "macos"))]
            {
                extern "C" { fn os_proc_available_memory() -> u64; }
                (unsafe { os_proc_available_memory() }) / (1024 * 1024)
            }
            #[cfg(not(any(target_os = "ios", target_os = "macos")))]
            { 0u64 } // Skip check on other platforms
        };

        if available_mb > 0 {
            // If Metal buffers already hold data >= model size, we can expect
            // buffer reuse (memcpy into existing Metal allocations). Only need
            // overhead for tokenizer + GGUF parsing + temporaries.
            let metal_mb = Self::metal_allocated_mb(&self.metal_device);
            let estimated_need_mb = if metal_mb >= file_size_mb {
                log_debug!(
                    "[CrabInfer] Warm reload: Metal already has {} MB (>= model {} MB), expecting buffer reuse",
                    metal_mb, file_size_mb
                );
                150 // ~150 MB overhead for tokenizer + GGUF parsing + temps
            } else {
                // Cold load: use proper estimator with actual model metadata.
                // peek_model_metadata reads only the GGUF header (~ms).
                match peek_model_metadata(&model_path, self.config.context_length) {
                    Ok((info, emb_overhead)) => {
                        let total_b = info.parameter_count as f64 / 1e9;
                        let active_b = info.active_parameter_count as f64 / 1e9;
                        let estimated = MemoryPressureManager::estimate_model_memory_corrected(
                            total_b as f32,
                            active_b as f32,
                            &info.quantization,
                            self.config.context_length,
                            emb_overhead,
                        );
                        log_debug!(
                            "[CrabInfer] Calibrated estimate: {} MB ({}B params, {}, ctx {}, emb overhead {} MB)",
                            estimated / (1024 * 1024),
                            total_b,
                            info.quantization,
                            self.config.context_length,
                            emb_overhead / (1024 * 1024),
                        );
                        (estimated / (1024 * 1024)) as u64
                    }
                    Err(_) => {
                        // Fallback if GGUF header can't be read (shouldn't happen
                        // since we already checked the file exists and size).
                        (file_size_mb as f64 * 1.3) as u64
                    }
                }
            };
            log_debug!(
                "[CrabInfer] Memory check: available={} MB, estimated need={} MB",
                available_mb, estimated_need_mb
            );
            if estimated_need_mb > available_mb {
                let reason = format!(
                    "Model needs ~{} MB but only {} MB available. Use a smaller model or quantization.",
                    estimated_need_mb, available_mb
                );
                log_debug!("[CrabInfer] REFUSING to load: {}", reason);
                return Err(CrabInferError::OutOfMemory { reason });
            }
        }

        // Check memory pressure before loading
        let mut mem = self.memory_manager.lock_recover();
        let pressure = mem.check_pressure();
        match pressure {
            MemoryPressure::Critical | MemoryPressure::Terminal => {
                log_debug!("[CrabInfer] ERROR: memory pressure too high to load ({:?})", pressure);
                return Err(CrabInferError::OutOfMemory {
                    reason: format!("Memory pressure is {:?} — free some memory first", pressure),
                });
            }
            _ => {
                log_debug!("[CrabInfer] Memory pressure: {:?}", pressure);
            }
        }
        drop(mem);

        tracing::info!("Loading model from: {}", model_path);
        let load_start = Instant::now();

        // Resolve metallib path (empty string = none)
        log_debug!(
            "[CrabInfer] metallib_path config value: '{}' (len={})",
            self.config.metallib_path,
            self.config.metallib_path.len()
        );
        let metallib_path = if self.config.metallib_path.is_empty() {
            log_debug!("[CrabInfer] metallib_path is EMPTY, will use runtime shader compilation");
            None
        } else {
            let path = std::path::Path::new(&self.config.metallib_path);
            let exists = path.exists();
            log_debug!(
                "[CrabInfer] metallib_path: '{}' (exists={})",
                self.config.metallib_path,
                exists
            );
            if exists {
                if let Ok(entries) = std::fs::read_dir(path) {
                    let files: Vec<String> = entries
                        .filter_map(|e| e.ok())
                        .map(|e| e.file_name().to_string_lossy().into_owned())
                        .collect();
                    log_debug!("[CrabInfer] metallib dir contents ({} files): {:?}", files.len(), files);
                }
            } else {
                // Try listing the parent dir to see what's actually there
                if let Some(parent) = path.parent() {
                    log_debug!("[CrabInfer] metallib dir NOT found, listing parent: {}", parent.display());
                    if let Ok(entries) = std::fs::read_dir(parent) {
                        let files: Vec<String> = entries
                            .filter_map(|e| e.ok())
                            .map(|e| e.file_name().to_string_lossy().into_owned())
                            .collect();
                        log_debug!("[CrabInfer] parent dir contents: {:?}", files);
                    }
                }
            }
            Some(self.config.metallib_path.as_str())
        };

        // Reuse the persistent Metal device if available, otherwise create one.
        // This avoids recreating command queues and recompiling shaders each load.
        let cached_device = self.metal_device.lock_recover().clone();

        // Spawn weight loading on a separate thread with a timeout to prevent
        // corrupt GGUF files from hanging the engine indefinitely.
        const LOADING_TIMEOUT_SECS: u64 = 30;
        let (tx, rx) = std::sync::mpsc::channel();
        let load_path = model_path.clone();
        let load_ctx = self.config.context_length;
        let load_metal = self.config.use_metal;
        let load_metallib = metallib_path.map(|s| s.to_string());
        std::thread::spawn(move || {
            let result = load_model_weights(
                &load_path,
                file_size,
                load_ctx,
                load_metal,
                load_metallib.as_deref(),
                cached_device,
            );
            let _ = tx.send(result);
        });

        let (weights, info, device) = rx
            .recv_timeout(Duration::from_secs(LOADING_TIMEOUT_SECS))
            .map_err(|_| CrabInferError::ModelLoadFailed {
                reason: format!("loading timed out after {}s", LOADING_TIMEOUT_SECS),
            })??;

        // Cache the device for future loads
        *self.metal_device.lock_recover() = Some(device.clone());

        // Load tokenizer from a file next to the model (tokenizer.json)
        let tokenizer = load_tokenizer(&model_path)?;

        // Resolve EOS token ID — try common EOS markers
        let eos_token_id = tokenizer
            .token_to_id("<|endoftext|>")
            .or_else(|| tokenizer.token_to_id("<|end|>"))
            .or_else(|| tokenizer.token_to_id("</s>"))
            .or_else(|| tokenizer.token_to_id("<|im_end|>"))
            .or_else(|| tokenizer.token_to_id("<end_of_turn>"))
            .unwrap_or(2); // fallback

        let load_time = load_start.elapsed();
        log_debug!("[CrabInfer] Model loaded in {:.2}s on {:?}", load_time.as_secs_f64(), device);
        log_debug!("[CrabInfer] EOS token id: {}", eos_token_id);
        tracing::info!("Model loaded in {:.2}s on {:?}", load_time.as_secs_f64(), device);

        *self.model.lock_recover() = Some(LoadedModel {
            weights,
            tokenizer,
            device,
            eos_token_id,
        });
        *self.model_info.lock_recover() = Some(info);
        // Reset streaming state when loading a new model
        *self.streaming.lock_recover() = None;

        Ok(())
    }

    pub fn model_info(&self) -> Result<ModelInfo, CrabInferError> {
        self.model_info
            .lock()
            .unwrap()
            .clone()
            .ok_or(CrabInferError::ModelLoadFailed { reason: "no model loaded".into() })
    }

    pub fn complete(
        &self,
        prompt: String,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<String, CrabInferError> {
        // Prevent concurrent generation calls from corrupting KV cache state
        if self.generating.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
            return Err(CrabInferError::InferenceFailed);
        }
        let _gen_guard = GenerationGuard(&self.generating);

        let mut model_guard = self.model.lock_recover();
        let loaded = model_guard.as_mut().ok_or(CrabInferError::ModelLoadFailed { reason: "no model loaded".into() })?;

        let start = Instant::now();

        // Tokenize prompt
        let encoding = loaded.tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| {
                tracing::error!("Tokenization failed: {}", e);
                CrabInferError::TokenizationFailed
            })?;
        let prompt_tokens = encoding.get_ids();

        if prompt_tokens.is_empty() {
            return Err(CrabInferError::TokenizationFailed);
        }

        // Check context length
        if prompt_tokens.len() as u32 >= self.config.context_length {
            return Err(CrabInferError::ContextOverflow);
        }

        // Set up logits processor for sampling
        let sampling = if temperature < 1e-7 {
            Sampling::ArgMax
        } else {
            let top_p = self.config.top_p as f64;
            if top_p > 0.0 && top_p < 1.0 {
                Sampling::TopP { p: top_p, temperature: temperature as f64 }
            } else {
                Sampling::All { temperature: temperature as f64 }
            }
        };
        let mut logits_processor = LogitsProcessor::from_sampling(42, sampling);

        // Process prompt (prefill)
        let input = Tensor::new(prompt_tokens, &loaded.device)
            .map_err(|_| CrabInferError::InferenceFailed)?
            .unsqueeze(0)
            .map_err(|_| CrabInferError::InferenceFailed)?;
        let logits = loaded.weights
            .forward(&input, 0)
            .map_err(|e| {
                log_debug!("[CrabInfer] complete(): forward FAILED: {}", e);
                CrabInferError::InferenceFailed
            })?;
        let logits = logits.squeeze(0).map_err(|_| CrabInferError::InferenceFailed)?;

        // Candle+Metal NaN workaround: Qwen3's attention produces all-NaN logits
        // at specific sequence lengths (seq_len == num_kv_heads, typically 8).
        // Detect this and retry with sequential (token-by-token) prefill.
        let logits = if loaded.weights.needs_sequential_prefill(&loaded.device) && has_nan(&logits) {
            log_debug!("[CrabInfer] NaN in batch prefill, retrying sequential (Qwen3+Metal workaround)");
            loaded.weights.clear_kv_cache();
            let mut last_logits = None;
            for (i, &token) in prompt_tokens.iter().enumerate() {
                let tok_input = Tensor::new(&[token], &loaded.device)
                    .map_err(|_| CrabInferError::InferenceFailed)?
                    .unsqueeze(0)
                    .map_err(|_| CrabInferError::InferenceFailed)?;
                last_logits = Some(loaded.weights
                    .forward(&tok_input, i)
                    .map_err(|e| {
                        log_debug!("[CrabInfer] sequential prefill FAILED at {}: {}", i, e);
                        CrabInferError::InferenceFailed
                    })?);
            }
            last_logits.unwrap().squeeze(0).map_err(|_| CrabInferError::InferenceFailed)?
        } else {
            logits
        };
        let first_token_time = start.elapsed();

        // Sample first token
        let mut next_token = logits_processor
            .sample(&logits)
            .map_err(|_| CrabInferError::InferenceFailed)?;

        let mut all_tokens: Vec<u32> = vec![next_token];
        let mut generated_count: u32 = 1;

        // Autoregressive generation loop
        let max = max_tokens.min(
            self.config.context_length.saturating_sub(prompt_tokens.len() as u32)
        );

        for i in 0..max.saturating_sub(1) {
            // Check for EOS
            if next_token == loaded.eos_token_id {
                break;
            }

            // Check memory pressure periodically (every 32 tokens)
            if i % 32 == 0 {
                let mut mem = self.memory_manager.lock_recover();
                let pressure = mem.check_pressure();
                match pressure {
                    MemoryPressure::Terminal => {
                        tracing::error!("Terminal memory pressure, stopping generation");
                        break;
                    }
                    MemoryPressure::Critical => {
                        tracing::warn!("Critical memory pressure, stopping generation");
                        break;
                    }
                    _ => {}
                }
            }

            // Forward pass with single token
            let input = Tensor::new(&[next_token], &loaded.device)
                .map_err(|_| CrabInferError::InferenceFailed)?
                .unsqueeze(0)
                .map_err(|_| CrabInferError::InferenceFailed)?;

            let pos = prompt_tokens.len() + i as usize;
            let logits = loaded.weights
                .forward(&input, pos)
                .map_err(|e| {
                    tracing::error!("Forward pass failed at token {}: {}", i, e);
                    CrabInferError::InferenceFailed
                })?;

            let logits = logits.squeeze(0).map_err(|_| CrabInferError::InferenceFailed)?;

            // Apply repeat penalty
            let logits = candle_transformers::utils::apply_repeat_penalty(
                &logits,
                1.1,
                &all_tokens,
            ).map_err(|_| CrabInferError::InferenceFailed)?;

            next_token = logits_processor
                .sample(&logits)
                .map_err(|_| CrabInferError::InferenceFailed)?;

            all_tokens.push(next_token);
            generated_count += 1;
        }

        let elapsed = start.elapsed();

        // Decode output tokens to text
        let output = loaded.tokenizer
            .decode(&all_tokens, true)
            .map_err(|e| {
                tracing::error!("Detokenization failed: {}", e);
                CrabInferError::InferenceFailed
            })?;

        // Store stats
        let tokens_per_second = if elapsed.as_secs_f64() > 0.0 {
            generated_count as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };

        *self.last_stats.lock_recover() = Some(GenerationStats {
            tokens_generated: generated_count,
            tokens_per_second,
            time_to_first_token_ms: first_token_time.as_secs_f64() * 1000.0,
            total_time_ms: elapsed.as_secs_f64() * 1000.0,
            peak_memory_bytes: self.memory_manager.lock_recover().peak_usage(),
            compute_backend: format!("{:?}", loaded.device),
        });

        Ok(output)
    }

    /// Generate the next token in a streaming fashion.
    ///
    /// First call: provide the prompt text. The engine tokenizes it, runs
    /// prefill, and returns the first generated token.
    ///
    /// Subsequent calls: pass the *same* prompt string (it is ignored after
    /// the first call). The engine advances one token using the KV cache.
    ///
    /// Returns `None` when EOS is reached or generation should stop.
    /// Call `reset()` to start a new generation.
    pub fn next_token(&self, prompt: String) -> Result<Option<TokenOutput>, CrabInferError> {
        // Prevent concurrent generation calls from corrupting KV cache state
        if self.generating.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
            return Err(CrabInferError::InferenceFailed);
        }
        let _gen_guard = GenerationGuard(&self.generating);

        let mut model_guard = self.model.lock_recover();
        let loaded = model_guard.as_mut().ok_or(CrabInferError::ModelLoadFailed { reason: "no model loaded".into() })?;

        // Check memory pressure
        {
            let mut mem = self.memory_manager.lock_recover();
            let pressure = mem.check_pressure();
            if matches!(pressure, MemoryPressure::Terminal) {
                log_debug!("[CrabInfer] Terminal memory pressure, stopping");
                return Ok(None);
            }
        }

        let mut streaming_guard = self.streaming.lock_recover();

        // ---- First call: prefill ----
        if streaming_guard.is_none() {
            let encoding = loaded.tokenizer
                .encode(prompt.as_str(), true)
                .map_err(|e| {
                    log_debug!("[CrabInfer] Tokenization failed: {}", e);
                    CrabInferError::TokenizationFailed
                })?;
            let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
            log_debug!("[CrabInfer] Prefill: {} prompt tokens on {:?}", prompt_tokens.len(), loaded.device);

            if prompt_tokens.is_empty() {
                log_debug!("[CrabInfer] Empty prompt tokens, returning None");
                return Ok(None);
            }

            if prompt_tokens.len() as u32 >= self.config.context_length {
                log_debug!("[CrabInfer] Context overflow: {} >= {}", prompt_tokens.len(), self.config.context_length);
                return Err(CrabInferError::ContextOverflow);
            }

            let sampling = if self.config.temperature < 1e-7 {
                Sampling::ArgMax
            } else {
                let top_p = self.config.top_p as f64;
                if top_p > 0.0 && top_p < 1.0 {
                    Sampling::TopP { p: top_p, temperature: self.config.temperature as f64 }
                } else {
                    Sampling::All { temperature: self.config.temperature as f64 }
                }
            };
            let logits_processor = LogitsProcessor::from_sampling(42, sampling);

            let start_time = Instant::now();

            log_debug!("[CrabInfer] Starting prefill forward pass...");
            let input = Tensor::new(prompt_tokens.as_slice(), &loaded.device)
                .map_err(|e| {
                    log_debug!("[CrabInfer] Failed to create input tensor: {}", e);
                    CrabInferError::InferenceFailed
                })?
                .unsqueeze(0)
                .map_err(|_| CrabInferError::InferenceFailed)?;
            let logits = loaded.weights
                .forward(&input, 0)
                .map_err(|e| {
                    log_debug!("[CrabInfer] Prefill forward FAILED: {}", e);
                    CrabInferError::InferenceFailed
                })?;
            let logits = logits.squeeze(0).map_err(|_| CrabInferError::InferenceFailed)?;

            // NaN workaround — same as complete()
            let logits = if loaded.weights.needs_sequential_prefill(&loaded.device) && has_nan(&logits) {
                log_debug!("[CrabInfer] NaN in prefill, retrying sequential");
                loaded.weights.clear_kv_cache();
                let mut last_logits = None;
                for (i, &token) in prompt_tokens.iter().enumerate() {
                    let tok = Tensor::new(&[token], &loaded.device)
                        .map_err(|_| CrabInferError::InferenceFailed)?
                        .unsqueeze(0)
                        .map_err(|_| CrabInferError::InferenceFailed)?;
                    last_logits = Some(loaded.weights
                        .forward(&tok, i)
                        .map_err(|e| {
                            log_debug!("[CrabInfer] Sequential prefill FAILED at {}: {}", i, e);
                            CrabInferError::InferenceFailed
                        })?);
                }
                last_logits.unwrap().squeeze(0).map_err(|_| CrabInferError::InferenceFailed)?
            } else {
                logits
            };

            let prefill_time = start_time.elapsed();
            log_debug!("[CrabInfer] Prefill done in {:.2}s", prefill_time.as_secs_f64());
            let first_token_time = Some(prefill_time);

            let mut state = StreamingState {
                position: prompt_tokens.len(),
                prompt_tokens,
                generated_tokens: Vec::new(),
                logits_processor,
                start_time,
                first_token_time,
            };

            // Sample first token
            let token_id = state.logits_processor
                .sample(&logits)
                .map_err(|_| CrabInferError::InferenceFailed)?;

            log_debug!("[CrabInfer] First token: id={} eos={}", token_id, token_id == loaded.eos_token_id);

            if token_id == loaded.eos_token_id {
                log_debug!("[CrabInfer] EOS on first token, stopping");
                self.store_streaming_stats(&state, &loaded.device);
                return Ok(None);
            }

            state.generated_tokens.push(token_id);

            // Decode with full context so the tokenizer preserves whitespace correctly.
            // Single-token decode strips leading spaces (SentencePiece treats it as first token).
            let text = loaded.tokenizer
                .decode(&state.generated_tokens, true)
                .map_err(|_| CrabInferError::InferenceFailed)?;

            log_debug!("[CrabInfer] First token text: {:?}", text);

            let probability = token_probability(&logits, token_id);

            *streaming_guard = Some(state);

            return Ok(Some(TokenOutput {
                text,
                token_id,
                probability,
                is_end_of_sequence: false,
            }));
        }

        // ---- Subsequent calls: generate one more token ----
        let state = streaming_guard.as_mut().unwrap();
        let gen_count = state.generated_tokens.len();

        // Check context overflow
        let total_len = state.position + gen_count;
        if total_len as u32 >= self.config.context_length {
            log_debug!("[CrabInfer] Context overflow at token {}, stopping", gen_count);
            self.store_streaming_stats(state, &loaded.device);
            *streaming_guard = None;
            return Ok(None);
        }

        let last_token = *state.generated_tokens.last().unwrap();
        let index_pos = state.position + gen_count - 1;

        // Single-token forward pass (KV cache handles the history)
        if gen_count <= 3 || gen_count % 50 == 0 {
            log_debug!("[CrabInfer] Token #{}: forward at pos {} (last_token={})", gen_count, index_pos, last_token);
        }

        let input = Tensor::new(&[last_token], &loaded.device)
            .map_err(|_| CrabInferError::InferenceFailed)?
            .unsqueeze(0)
            .map_err(|_| CrabInferError::InferenceFailed)?;

        let logits = loaded.weights
            .forward(&input, index_pos)
            .map_err(|e| {
                log_debug!("[CrabInfer] Forward FAILED at token #{}: {}", gen_count, e);
                CrabInferError::InferenceFailed
            })?;

        let logits = logits.squeeze(0).map_err(|_| CrabInferError::InferenceFailed)?;

        // Apply repeat penalty
        let all_tokens: Vec<u32> = state.prompt_tokens.iter()
            .chain(state.generated_tokens.iter())
            .copied()
            .collect();
        let logits = candle_transformers::utils::apply_repeat_penalty(
            &logits, 1.1, &all_tokens,
        ).map_err(|_| CrabInferError::InferenceFailed)?;

        let token_id = state.logits_processor
            .sample(&logits)
            .map_err(|_| CrabInferError::InferenceFailed)?;

        if gen_count <= 5 || gen_count % 50 == 0 {
            log_debug!("[CrabInfer] Token #{}: id={} eos={}", gen_count + 1, token_id, token_id == loaded.eos_token_id);
        }

        // Check EOS
        if token_id == loaded.eos_token_id {
            log_debug!("[CrabInfer] EOS at token #{}", gen_count + 1);
            self.store_streaming_stats(state, &loaded.device);
            *streaming_guard = None;
            return Ok(Some(TokenOutput {
                text: String::new(),
                token_id,
                probability: token_probability(&logits, token_id),
                is_end_of_sequence: true,
            }));
        }

        // Decode the full sequence before and after adding the new token.
        // This preserves whitespace that the tokenizer strips when decoding
        // a single token in isolation (SentencePiece/BPE leading-space issue).
        let prev_text = loaded.tokenizer
            .decode(&state.generated_tokens, true)
            .unwrap_or_default();

        state.generated_tokens.push(token_id);

        let full_text = loaded.tokenizer
            .decode(&state.generated_tokens, true)
            .map_err(|_| CrabInferError::InferenceFailed)?;

        let text = if full_text.len() >= prev_text.len() {
            full_text[prev_text.len()..].to_string()
        } else {
            // Fallback: decode single token (edge case where BPE merges change output)
            loaded.tokenizer.decode(&[token_id], true).unwrap_or_default()
        };

        if gen_count <= 5 {
            log_debug!("[CrabInfer] Token #{} text: {:?}", gen_count + 1, text);
        }

        let probability = token_probability(&logits, token_id);

        Ok(Some(TokenOutput {
            text,
            token_id,
            probability,
            is_end_of_sequence: false,
        }))
    }

    /// Store stats from a finished streaming session
    fn store_streaming_stats(&self, state: &StreamingState, device: &Device) {
        let elapsed = state.start_time.elapsed();
        let count = state.generated_tokens.len() as u32;
        let tps = if elapsed.as_secs_f64() > 0.0 {
            count as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        let ttft = state.first_token_time
            .map(|d| d.as_secs_f64() * 1000.0)
            .unwrap_or(0.0);

        *self.last_stats.lock_recover() = Some(GenerationStats {
            tokens_generated: count,
            tokens_per_second: tps,
            time_to_first_token_ms: ttft,
            total_time_ms: elapsed.as_secs_f64() * 1000.0,
            peak_memory_bytes: self.memory_manager.lock_recover().peak_usage(),
            compute_backend: format!("{:?}", device),
        });
    }

    pub fn reset(&self) {
        // Save stats from the active streaming session before clearing it
        if let Some(state) = self.streaming.lock_recover().as_ref() {
            if let Some(loaded) = self.model.lock_recover().as_ref() {
                self.store_streaming_stats(state, &loaded.device);
            }
        }
        *self.streaming.lock_recover() = None;
        // Clear KV caches for models that don't auto-reset (e.g. Qwen3)
        if let Some(loaded) = self.model.lock_recover().as_mut() {
            loaded.weights.clear_kv_cache();
        }
    }

    pub fn last_stats(&self) -> Option<GenerationStats> {
        self.last_stats.lock_recover().clone()
    }

    pub fn memory_pressure(&self) -> MemoryPressure {
        self.memory_manager.lock_recover().check_pressure()
    }

    /// Register a closure to be called when memory pressure level changes.
    pub fn set_pressure_callback<F>(&self, callback: F)
    where
        F: Fn(MemoryPressure, MemoryPressure) + Send + Sync + 'static,
    {
        self.memory_manager.lock_recover().set_pressure_callback(callback);
    }

    /// Progressive memory reduction based on current memory pressure.
    ///
    /// - **Warning**: clears KV cache (resets streaming state)
    /// - **Critical**: clears KV cache + forces system memory reclaim
    /// - **Terminal**: unloads the model entirely
    /// - **Normal**: no action
    ///
    /// Swift should call this in response to `MemoryPressureListener` callbacks
    /// or iOS memory warning notifications.
    pub fn reduce_memory(&self) {
        let pressure = self.memory_manager.lock_recover().check_pressure();
        match pressure {
            MemoryPressure::Normal => {
                tracing::debug!("reduce_memory: Normal pressure, no action");
            }
            MemoryPressure::Warning => {
                tracing::info!("reduce_memory: Warning pressure, clearing KV cache");
                self.reset();
            }
            MemoryPressure::Critical => {
                tracing::warn!("reduce_memory: Critical pressure, clearing KV cache + reclaiming memory");
                self.reset();
                Self::force_memory_reclaim();
            }
            MemoryPressure::Terminal => {
                tracing::error!("reduce_memory: Terminal pressure, unloading model");
                self.unload_model();
            }
        }
    }

    pub fn unload_model(&self) {
        // Take model out of the mutex, then drop the lock before dropping
        // the model. This ensures heavy Drop work (Candle tensors, Metal
        // buffers) doesn't hold the mutex.
        let old_model = self.model.lock_recover().take();
        *self.model_info.lock_recover() = None;
        *self.streaming.lock_recover() = None;
        *self.last_stats.lock_recover() = None;

        if let Some(loaded) = old_model {
            // Drop the model weights + tokenizer (heavy), but keep the device
            // alive in self.metal_device for reuse on next load.
            drop(loaded);
            log_debug!("[CrabInfer] Model weights dropped");
        }

        // Always flush the Metal buffer pool, even if no model was loaded.
        // The stress test's partial unload drops model tensors but deliberately
        // keeps buffers in the pool for reuse.  When unload_model() is called
        // afterwards (or when loading a different model), we must flush the
        // pool to avoid a stale 1 GB+ Metal footprint that would cause a
        // jetsam kill on the next load.
        #[cfg(feature = "metal")]
        if let Some(ref device) = *self.metal_device.lock_recover() {
            if let Ok(metal) = device.as_metal_device() {
                let _ = metal.wait_until_completed();
                if let Err(e) = metal.release_unused_buffers() {
                    log_debug!("[CrabInfer] Warning: failed to release Metal buffers: {:?}", e);
                }
            }
        }

        // Force the system allocator to return freed pages to the OS.
        Self::force_memory_reclaim();

        log_debug!("[CrabInfer] Model unloaded, Metal buffers released");
    }

    pub fn is_model_loaded(&self) -> bool {
        self.model.lock_recover().is_some()
    }

    /// Get current Metal GPU memory allocation in bytes.
    ///
    /// Returns 0 if no Metal device is active or on non-Metal platforms.
    /// Swift can call this after `load_model()` to display Metal usage in the UI.
    pub fn metal_allocated_bytes(&self) -> u64 {
        #[cfg(feature = "metal")]
        if let Some(ref device) = *self.metal_device.lock_recover() {
            if let Ok(metal) = device.as_metal_device() {
                return metal.current_allocated_size() as u64;
            }
        }
        0
    }

    /// Get current memory usage from the memory manager (for benchmarking)
    pub fn memory_usage_bytes(&self) -> u64 {
        self.memory_manager.lock_recover().peak_usage()
    }

    /// Get Metal device allocated size in MB (0 if no device).
    fn metal_allocated_mb(metal_device: &Mutex<Option<Device>>) -> u64 {
        let _ = metal_device; // Used only when Metal feature is enabled
        #[cfg(feature = "metal")]
        if let Some(ref device) = *metal_device.lock_recover() {
            if let Ok(metal) = device.as_metal_device() {
                return (metal.current_allocated_size() / (1024 * 1024)) as u64;
            }
        }
        0
    }

    /// Ask the system allocator to return freed pages to the OS.
    ///
    /// After dropping large allocations (model weights, Metal buffers), the
    /// C allocator keeps freed virtual pages mapped as "dirty" for potential
    /// reuse. This inflates RSS even though the memory is logically free.
    /// `malloc_zone_pressure_relief` walks all malloc zones and returns
    /// clean pages to the kernel, reducing RSS to reflect actual usage.
    fn force_memory_reclaim() {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            extern "C" {
                fn malloc_zone_pressure_relief(
                    zone: *mut std::ffi::c_void,
                    goal: usize,
                ) -> usize;
            }
            let freed = unsafe { malloc_zone_pressure_relief(std::ptr::null_mut(), 0) };
            log_debug!("[CrabInfer] malloc_zone_pressure_relief freed {} bytes", freed);
        }
    }
}

// Stress test lives in a separate file for clarity.
#[path = "stress.rs"]
mod stress;

// ---------------------------------------------------------------------------
// Model loading with multi-architecture support
// ---------------------------------------------------------------------------

/// Load model weights, trying Metal first and falling back to CPU if Metal ops are unsupported.
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
            let _ = metallib_path;
            let _ = &cached_device;
            log_debug!("[CrabInfer] Metal requested but not compiled in, using CPU");
            vec![Device::Cpu]
        }
        #[cfg(feature = "metal")]
        // Reuse cached Metal device if available
        if let Some(ref dev) = cached_device {
            if matches!(dev, Device::Metal(_)) {
                log_debug!("[CrabInfer] Reusing cached Metal device");
                vec![dev.clone(), Device::Cpu]
            } else {
                log_debug!("[CrabInfer] Cached device is CPU, creating Metal device...");
                match Device::new_metal(0) {
                    Ok(metal) => {
                        if let Some(dir) = metallib_path {
                            if let Ok(metal_dev) = metal.as_metal_device() {
                                log_debug!("[CrabInfer] Setting metallib dir: {}", dir);
                                metal_dev.set_metallib_dir(dir);
                            }
                        }
                        vec![metal, Device::Cpu]
                    }
                    Err(e) => {
                        log_debug!("[CrabInfer] Metal device FAILED: {}, falling back to CPU", e);
                        vec![Device::Cpu]
                    }
                }
            }
        } else {
            log_debug!("[CrabInfer] Metal requested, creating Metal device...");
            match Device::new_metal(0) {
                Ok(metal) => {
                    if let Some(dir) = metallib_path {
                        if let Ok(metal_dev) = metal.as_metal_device() {
                            log_debug!("[CrabInfer] Setting metallib dir: {}", dir);
                            metal_dev.set_metallib_dir(dir);
                        }
                    } else {
                        log_debug!("[CrabInfer] WARNING: No metallib_path provided — Metal will use runtime XPC compilation (may fail on iOS)");
                    }
                    log_debug!("[CrabInfer] Metal device created OK");
                    vec![metal, Device::Cpu]
                }
                Err(e) => {
                    log_debug!("[CrabInfer] Metal device FAILED: {}, falling back to CPU", e);
                    vec![Device::Cpu]
                }
            }
        }
    } else {
        log_debug!("[CrabInfer] CPU-only mode requested");
        vec![Device::Cpu]
    };

    let is_reused_device = cached_device.is_some();
    let mut last_error = String::from("no devices available");
    for device in devices_to_try {
        log_debug!("[CrabInfer] Trying device: {:?}", device);
        tracing::info!("Trying device: {:?}", device);

        let mut file = std::fs::File::open(model_path)
            .map_err(|e| CrabInferError::ModelLoadFailed { reason: format!("cannot open file: {}", e) })?;

        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| {
                tracing::error!("Failed to parse GGUF: {}", e);
                CrabInferError::ModelLoadFailed { reason: format!("invalid GGUF: {}", e) }
            })?;

        let info = extract_model_info(&content, model_path, file_size, default_context_length);
        tracing::info!(
            "Model: {} | arch: {} | quant: {} | vocab: {} | ctx: {}",
            info.model_name, info.architecture, info.quantization,
            info.vocab_size, info.context_length
        );

        let arch = info.architecture.to_lowercase();
        log_debug!("[CrabInfer] Loading weights for arch '{}' on {:?}...", arch, device);
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
            let result = catch_unwind(AssertUnwindSafe(|| -> candle_core::Result<Model> {
                // Qwen3 GGUFs from different sources may use either "qwen3.*" or
                // "qwen2.*" metadata keys. Probe for the actual key prefix to pick
                // the right loader (Qwen3 vs Qwen2). The Qwen3 loader expects
                // "qwen3.attention.head_count"; the Qwen2 loader expects
                // "qwen2.attention.head_count".
                let qwen3_has_native_keys = content.metadata.contains_key("qwen3.attention.head_count");

                let mut weights = match arch.as_str() {
                    "phi3" => Phi3ModelWeights::from_gguf(false, content, &mut file, &device)
                        .map(Model::Phi3)?,
                    "qwen2" => Qwen2ModelWeights::from_gguf(content, &mut file, &device)
                        .map(Model::Qwen2)?,
                    "qwen3" if qwen3_has_native_keys => {
                        log_debug!("[CrabInfer] Using Qwen3 loader (qwen3.* metadata keys found)");
                        Qwen3ModelWeights::from_gguf(content, &mut file, &device)
                            .map(Model::Qwen3)?
                    }
                    "qwen3" => {
                        log_debug!("[CrabInfer] Using Qwen2 loader for qwen3 arch (qwen2.* metadata keys)");
                        Qwen2ModelWeights::from_gguf(content, &mut file, &device)
                            .map(Model::Qwen2)?
                    }
                    "gemma3" | "gemma2" | "gemma" => Gemma3ModelWeights::from_gguf(content, &mut file, &device)
                        .map(Model::Gemma3)?,
                    "llama" => LlamaModelWeights::from_gguf(content, &mut file, &device)
                        .map(Model::Llama)?,
                    _other => {
                        tracing::warn!(
                            "Unknown architecture '{}', trying Llama-compatible loader", _other
                        );
                        let mut file2 = std::fs::File::open(&model_path_owned)
                            .map_err(|e| candle_core::Error::Msg(format!("Reopen failed: {}", e)))?;
                        let content2 = gguf_file::Content::read(&mut file2)?;
                        LlamaModelWeights::from_gguf(content2, &mut file2, &device)
                            .map(Model::Llama)?
                    }
                };

                // Warmup: trigger Metal shader compilation NOW during load_model()
                // rather than letting it panic during the first next_token() call.
                //
                // Candle compiles Metal shaders lazily from ~16K lines of MSL source
                // via XPC to com.apple.MTLCompilerService. On iOS this can fail with
                // XPC_ERROR_CONNECTION_INTERRUPTED. We do incremental warmup to
                // identify exactly which shader compilation fails.
                //
                // Phi3, Llama, Qwen2, and Gemma3 auto-reset their KV cache when
                // forward() is called with index_pos==0. Qwen3 uses ConcatKvCache
                // which only appends, so we explicitly clear after warmup.
                let warmup_start = Instant::now();

                if is_reused_device {
                    // Shaders already compiled on the cached device — skip tensor warmup,
                    // just run a forward pass to populate KV cache shapes for new weights.
                    log_debug!("[CrabInfer] Reused device, skipping shader warmup...");
                } else {
                    // Step 1: Basic tensor ops (compiles fill.metal, binary.metal, cast.metal)
                    log_debug!("[CrabInfer] Metal warmup 1/3: basic tensor ops...");
                    let a = Tensor::zeros((2, 2), candle_core::DType::F32, &device)?;
                    let b = Tensor::ones((2, 2), candle_core::DType::F32, &device)?;
                    let _c = (&a + &b)?;
                    log_debug!("[CrabInfer] Metal warmup 1/3: OK ({:.1}s)", warmup_start.elapsed().as_secs_f64());

                    // Step 2: Unary + reduce ops (compiles unary.metal, reduce.metal)
                    log_debug!("[CrabInfer] Metal warmup 2/3: unary + reduce ops...");
                    let _d = b.sqrt()?;
                    let _e = a.sum_all()?;
                    log_debug!("[CrabInfer] Metal warmup 2/3: OK ({:.1}s)", warmup_start.elapsed().as_secs_f64());
                }

                // Forward pass warmup — always run for new weights (compiles
                // quantized.metal kernels specific to this model's tensor shapes)
                log_debug!("[CrabInfer] Metal warmup: quantized forward pass...");
                let dummy = Tensor::new(&[1u32], &device)?.unsqueeze(0)?;
                weights.forward(&dummy, 0)?;
                weights.clear_kv_cache();
                log_debug!("[CrabInfer] Metal warmup complete! ({:.1}s total)", warmup_start.elapsed().as_secs_f64());

                Ok(weights)
            }));

            match result {
                Ok(inner) => inner,
                Err(panic_payload) => {
                    let msg = extract_panic_message(&panic_payload);
                    log_debug!("[CrabInfer] Metal PANICKED: {}", msg);
                    Err(candle_core::Error::Msg(format!(
                        "Metal shader compilation panic: {}", msg
                    )))
                }
            }
        } else {
            // CPU path: no catch_unwind needed (no Metal shader compilation)
            let qwen3_has_native_keys_cpu = content.metadata.contains_key("qwen3.attention.head_count");
            match arch.as_str() {
                "phi3" => Phi3ModelWeights::from_gguf(false, content, &mut file, &device)
                    .map(Model::Phi3),
                "qwen2" => Qwen2ModelWeights::from_gguf(content, &mut file, &device)
                    .map(Model::Qwen2),
                "qwen3" if qwen3_has_native_keys_cpu => Qwen3ModelWeights::from_gguf(content, &mut file, &device)
                    .map(Model::Qwen3),
                "qwen3" => Qwen2ModelWeights::from_gguf(content, &mut file, &device)
                    .map(Model::Qwen2),
                "gemma3" | "gemma2" | "gemma" => Gemma3ModelWeights::from_gguf(content, &mut file, &device)
                    .map(Model::Gemma3),
                "llama" => LlamaModelWeights::from_gguf(content, &mut file, &device)
                    .map(Model::Llama),
                other => {
                    tracing::warn!(
                        "Unknown architecture '{}', trying Llama-compatible loader", other
                    );
                    // Re-read content since we consumed it
                    let mut file2 = std::fs::File::open(model_path)
                        .map_err(|e| CrabInferError::ModelLoadFailed { reason: format!("reopen: {}", e) })?;
                    let content2 = gguf_file::Content::read(&mut file2)
                        .map_err(|e| CrabInferError::ModelLoadFailed { reason: format!("GGUF parse: {}", e) })?;
                    LlamaModelWeights::from_gguf(content2, &mut file2, &device)
                        .map(Model::Llama)
                }
            }
        };

        let weight_time = weight_start.elapsed();
        match model_result {
            Ok(weights) => {
                log_debug!("[CrabInfer] Weights loaded on {:?} in {:.2}s", device, weight_time.as_secs_f64());
                tracing::info!("Model loaded on {:?}", device);
                return Ok((weights, info, device));
            }
            Err(e) => {
                let msg = format!("{:?}: {}", device, e);
                log_debug!("[CrabInfer] FAILED on {} (after {:.2}s)", msg, weight_time.as_secs_f64());
                tracing::warn!("Failed to load on {:?}: {}, trying next device...", device, e);
                last_error = msg;
                continue;
            }
        }
    }

    tracing::error!("Failed to load model on any device: {}", last_error);
    Err(CrabInferError::ModelLoadFailed { reason: last_error })
}

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

/// Extract ModelInfo from GGUF metadata
fn extract_model_info(
    content: &gguf_file::Content,
    model_path: &str,
    file_size: u64,
    default_context_length: u32,
) -> ModelInfo {
    let md = &content.metadata;

    // Model name: try general.name, fall back to filename
    let model_name = md.get("general.name")
        .and_then(|v| v.to_string().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            std::path::Path::new(model_path)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "unknown".to_string())
        });

    // Architecture: general.architecture
    let architecture = md.get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "unknown".to_string());

    // Context length from metadata
    let context_length = get_metadata_u32(md, &format!("{}.context_length", architecture))
        .unwrap_or(default_context_length);

    // Vocab size from embedding tensor shape
    let vocab_size = content.tensor_infos
        .get("token_embd.weight")
        .map(|t| t.shape.dims()[0] as u32)
        .unwrap_or(0);

    // Determine dominant quantization from tensor types
    let quantization = detect_quantization(&content.tensor_infos);

    // Estimate parameter count from tensor shapes
    let parameter_count: u64 = content.tensor_infos
        .values()
        .map(|t| t.shape.elem_count() as u64)
        .sum();

    // Detect MoE (Mixture of Experts) from GGUF metadata.
    // Standard keys: <arch>.expert_count, <arch>.expert_used_count
    let expert_count = get_metadata_u32(md, &format!("{}.expert_count", architecture))
        .unwrap_or(0);
    let expert_used_count = get_metadata_u32(md, &format!("{}.expert_used_count", architecture))
        .unwrap_or(0);
    let is_moe = expert_count > 0 && expert_used_count > 0;

    // Active parameter count: for MoE, roughly (shared_params + expert_fraction).
    // A good approximation: active_params ≈ total_params * used / total_experts
    // for the expert layers, plus shared layers (embeddings, norms, etc.)
    // which are ~15-25% of total params. We use a conservative estimate.
    let active_parameter_count = if is_moe && expert_count > expert_used_count {
        // Shared layers (embeddings, output head, layer norms) are ~20% of total
        let shared_fraction = 0.20;
        let expert_fraction = (1.0 - shared_fraction)
            * (expert_used_count as f64 / expert_count as f64);
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

/// Read a u32 from GGUF metadata, handling various integer types
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

/// Detect the dominant quantization type from tensor infos
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
    let dominant = counts.iter()
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
// GGUF metadata peek (no weight loading)
// ---------------------------------------------------------------------------

/// Compute extra memory from F16/F32 embedding and output tensors.
///
/// GGUF models store `token_embd.weight` and `output.weight` (or `lm_head.weight`)
/// in F16 regardless of the dominant quantization. For large-vocab models (150K+),
/// this adds significant memory beyond what the bits-per-param formula predicts.
///
/// Returns the excess bytes: (actual F16/F32 size) - (what the quant formula would predict).
fn compute_embedding_overhead(
    tensor_infos: &std::collections::HashMap<String, gguf_file::TensorInfo>,
    dominant_quant: &str,
) -> u64 {
    use candle_core::quantized::GgmlDType;

    let bits_per_param_quant: f64 = match dominant_quant {
        "Q2_K" => 2.5,
        "Q3_K" | "Q3_K_S" | "Q3_K_M" | "Q3_K_L" => 3.5,
        "Q4_0" | "Q4_1" | "Q4_K" | "Q4_K_S" | "Q4_K_M" => 4.5,
        "Q5_0" | "Q5_1" | "Q5_K" | "Q5_K_S" | "Q5_K_M" => 5.5,
        "Q6_K" => 6.5,
        "Q8_0" | "Q8_K" => 8.5,
        "F16" | "BF16" => 16.0,
        "F32" => 32.0,
        _ => 4.5,
    };

    // Tensors stored in F16/F32 regardless of quant
    let embedding_names = ["token_embd.weight", "output.weight", "lm_head.weight"];

    let mut overhead: u64 = 0;
    for name in &embedding_names {
        if let Some(info) = tensor_infos.get(*name) {
            let elem_count = info.shape.elem_count() as f64;
            let actual_bits: f64 = match info.ggml_dtype {
                GgmlDType::F16 | GgmlDType::BF16 => 16.0,
                GgmlDType::F32 => 32.0,
                _ => continue, // Already quantized, no overhead
            };
            let actual_bytes = (elem_count * actual_bits / 8.0) as u64;
            let predicted_bytes = (elem_count * bits_per_param_quant / 8.0) as u64;
            overhead += actual_bytes.saturating_sub(predicted_bytes);
        }
    }

    overhead
}

/// Read GGUF header only (no weight loading) and return model metadata
/// plus the embedding overhead for memory estimation.
///
/// This is cheap (~ms) — `Content::read()` parses tensor names/shapes/dtypes
/// but does not read the multi-GB weight data.
pub fn peek_model_metadata(
    model_path: &str,
    default_context_length: u32,
) -> Result<(ModelInfo, u64), CrabInferError> {
    let mut file = std::fs::File::open(model_path).map_err(|e| {
        CrabInferError::ModelLoadFailed {
            reason: format!("cannot open: {}", e),
        }
    })?;
    let file_size = file.metadata().map(|m| m.len()).unwrap_or(0);
    let content = gguf_file::Content::read(&mut file).map_err(|e| {
        CrabInferError::ModelLoadFailed {
            reason: format!("invalid GGUF: {}", e),
        }
    })?;
    let info = extract_model_info(&content, model_path, file_size, default_context_length);
    let emb_overhead = compute_embedding_overhead(&content.tensor_infos, &info.quantization);
    Ok((info, emb_overhead))
}

// ---------------------------------------------------------------------------
// Tokenizer loading
// ---------------------------------------------------------------------------

/// Load tokenizer from a tokenizer.json file adjacent to the model
fn load_tokenizer(model_path: &str) -> Result<Tokenizer, CrabInferError> {
    let model_dir = std::path::Path::new(model_path)
        .parent()
        .ok_or(CrabInferError::ModelLoadFailed { reason: "invalid model path".into() })?;

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
        model_dir.parent().map(|p| p.join("tokenizer.json")).unwrap_or_default(),
    ];

    for path in &candidates {
        if path.exists() {
            tracing::info!("Loading tokenizer from: {}", path.display());
            return Tokenizer::from_file(path)
                .map_err(|e| {
                    tracing::error!("Failed to load tokenizer: {}", e);
                    CrabInferError::TokenizationFailed
                });
        }
    }

    tracing::error!(
        "No tokenizer.json found. Searched: {:?}",
        candidates.iter().map(|p| p.display().to_string()).collect::<Vec<_>>()
    );
    Err(CrabInferError::TokenizationFailed)
}

/// Convert a model filename stem to a likely tokenizer directory name.
/// e.g. "qwen2.5-7b-instruct-q4_k_m" → "qwen2.5-7b"
fn model_stem_to_dir(stem: &str) -> String {
    // Strip common quantization suffixes
    let s = stem.to_lowercase();
    for suffix in &["-q4_k_m", "-q4_k_s", "-q5_k_m", "-q5_k_s", "-q4_0", "-q8_0", "-q6_k", "-q3_k_m", "-q2_k"] {
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
// Helpers
// ---------------------------------------------------------------------------

/// Check if a 1-D logits tensor contains any NaN values.
///
/// Used to detect the Candle+Metal NaN bug in Qwen3 attention when
/// `seq_len == num_kv_heads`.  Only samples the first 32 values for
/// speed — if the bug triggers, ALL values are NaN.
fn has_nan(logits: &Tensor) -> bool {
    logits
        .flatten_all()
        .and_then(|f| f.narrow(0, 0, 32.min(f.elem_count()))?.to_vec1::<f32>())
        .map(|v| v.iter().any(|x| x.is_nan()))
        .unwrap_or(false)
}

/// Compute probability of a token from logits (softmax)
fn token_probability(logits: &Tensor, token_id: u32) -> f32 {
    candle_nn::ops::softmax_last_dim(logits)
        .and_then(|p| p.to_vec1::<f32>())
        .map(|p| p.get(token_id as usize).copied().unwrap_or(0.0))
        .unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> EngineConfig {
        EngineConfig {
            model_path: String::new(),
            max_tokens: 100,
            temperature: 0.7,
            top_p: 0.9,
            context_length: 2048,
            use_metal: false,
            memory_limit_bytes: 2 * 1024 * 1024 * 1024,
            metallib_path: String::new(),
        }
    }

    #[test]
    fn test_engine_creation() {
        let engine = CrabInferEngine::new(test_config());
        assert!(engine.is_ok());
    }

    #[test]
    fn test_invalid_temperature() {
        let mut config = test_config();
        config.temperature = 5.0;
        let engine = CrabInferEngine::new(config);
        assert!(matches!(engine, Err(CrabInferError::InvalidConfig)));
    }

    #[test]
    fn test_model_not_found() {
        let engine = CrabInferEngine::new(test_config()).unwrap();
        let result = engine.load_model("/nonexistent/model.gguf".to_string());
        assert!(matches!(result, Err(CrabInferError::ModelNotFound)));
    }

    #[test]
    fn test_complete_without_model() {
        let engine = CrabInferEngine::new(test_config()).unwrap();
        let result = engine.complete("hello".to_string(), 10, 0.7);
        assert!(matches!(result, Err(CrabInferError::ModelLoadFailed { .. })));
    }

    #[test]
    fn test_model_stem_to_dir() {
        assert_eq!(model_stem_to_dir("qwen2.5-7b-instruct-q4_k_m"), "qwen2.5-7b");
        assert_eq!(model_stem_to_dir("Phi-3-mini-4k-instruct-q4_k_m"), "phi-3-mini-4k");
        assert_eq!(model_stem_to_dir("llama-7b-chat-q4_0"), "llama-7b");
    }

    #[test]
    fn test_generation_guard() {
        let flag = AtomicBool::new(false);
        assert!(!flag.load(Ordering::Relaxed));
        {
            flag.store(true, Ordering::Release);
            let _guard = GenerationGuard(&flag);
            assert!(flag.load(Ordering::Relaxed));
        }
        // Guard dropped — flag should be cleared
        assert!(!flag.load(Ordering::Relaxed));
    }

    #[test]
    fn test_concurrent_call_rejected() {
        let engine = CrabInferEngine::new(test_config()).unwrap();
        // Simulate generating flag being set (as if another call is in progress)
        engine.generating.store(true, Ordering::Release);
        let result = engine.complete("hello".to_string(), 10, 0.7);
        assert!(matches!(result, Err(CrabInferError::InferenceFailed)));
        // Also check next_token
        let result = engine.next_token("hello".to_string());
        assert!(matches!(result, Err(CrabInferError::InferenceFailed)));
        // Clean up
        engine.generating.store(false, Ordering::Release);
    }

    #[test]
    fn test_reduce_memory_no_panic() {
        let engine = CrabInferEngine::new(test_config()).unwrap();
        // Should not panic even with no model loaded
        engine.reduce_memory();
    }

    #[test]
    fn test_metal_allocated_bytes_no_model() {
        let engine = CrabInferEngine::new(test_config()).unwrap();
        // No Metal device active — should return 0
        assert_eq!(engine.metal_allocated_bytes(), 0);
    }
}
