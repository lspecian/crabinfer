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

/// The main inference engine
pub struct CrabInferEngine {
    config: EngineConfig,
    memory_manager: Mutex<MemoryPressureManager>,
    model: Mutex<Option<LoadedModel>>,
    model_info: Mutex<Option<ModelInfo>>,
    last_stats: Mutex<Option<GenerationStats>>,
    streaming: Mutex<Option<StreamingState>>,
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

        // Pre-flight memory check: compare model size against available memory.
        // On iOS, both Metal (shared storage) and CPU buffers consume app RAM.
        // Loading a model needs ~1.3x the file size (weights + KV cache + overhead).
        // If this exceeds available memory, iOS will kill the app (EXC_RESOURCE).
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
            let estimated_need_mb = file_size_mb as f64 * 1.4; // model + KV cache + overhead
            log_debug!(
                "[CrabInfer] Memory check: available={} MB, estimated need={:.0} MB",
                available_mb, estimated_need_mb
            );
            if estimated_need_mb > available_mb as f64 {
                log_debug!(
                    "[CrabInfer] REFUSING to load: model needs ~{:.0} MB but only {} MB available. \
                     Use a smaller quantization (Q2_K) or smaller model.",
                    estimated_need_mb, available_mb
                );
                return Err(CrabInferError::OutOfMemory);
            }
        }

        // Check memory pressure before loading
        let mut mem = self.memory_manager.lock_recover();
        let pressure = mem.check_pressure();
        match pressure {
            MemoryPressure::Critical | MemoryPressure::Terminal => {
                log_debug!("[CrabInfer] ERROR: memory pressure too high to load ({:?})", pressure);
                return Err(CrabInferError::OutOfMemory);
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

        // Try loading with Metal first, fall back to CPU if it fails
        let (weights, info, device) = load_model_weights(
            &model_path,
            file_size,
            self.config.context_length,
            self.config.use_metal,
            metallib_path,
        )?;

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

        // Process prompt (prefill) — feed all prompt tokens at once
        let input = Tensor::new(prompt_tokens, &loaded.device)
            .map_err(|_| CrabInferError::InferenceFailed)?
            .unsqueeze(0)
            .map_err(|_| CrabInferError::InferenceFailed)?;

        let logits = loaded.weights
            .forward(&input, 0)
            .map_err(|e| {
                tracing::error!("Forward pass failed: {}", e);
                CrabInferError::InferenceFailed
            })?;

        let logits = logits.squeeze(0).map_err(|_| CrabInferError::InferenceFailed)?;
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

            let logits = loaded.weights
                .forward(&input, prompt_tokens.len() + i as usize)
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

            // Prefill: feed all prompt tokens
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

            let prefill_time = start_time.elapsed();
            log_debug!("[CrabInfer] Prefill done in {:.2}s", prefill_time.as_secs_f64());

            let logits = logits.squeeze(0).map_err(|_| CrabInferError::InferenceFailed)?;
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

    pub fn reduce_memory(&self) {
        tracing::info!("Memory reduction requested");
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
            // Save the device handle before dropping the model weights,
            // since we need it to flush the Metal buffer pool afterwards.
            let device = loaded.device.clone();

            // Explicitly drop fields in order: weights (heavy), then tokenizer.
            // `drop(loaded)` would do the same, but being explicit documents intent.
            drop(loaded);

            // Flush Metal buffer pool — without this, Arc<Buffer> entries with
            // strong_count==1 stay cached in the device's buffer pool forever,
            // leaking GPU memory across load/unload cycles.
            if let Ok(metal) = device.as_metal_device() {
                // Wait for any in-flight GPU work to complete first
                let _ = metal.wait_until_completed();
                // Now release buffers whose only reference is the pool itself
                if let Err(e) = metal.release_unused_buffers() {
                    log_debug!("[CrabInfer] Warning: failed to release Metal buffers: {:?}", e);
                }
            }

            log_debug!("[CrabInfer] Model unloaded, Metal buffers released");
        } else {
            log_debug!("[CrabInfer] unload_model called but no model was loaded");
        }
    }

    pub fn is_model_loaded(&self) -> bool {
        self.model.lock_recover().is_some()
    }

    /// Get current memory usage from the memory manager (for benchmarking)
    pub fn memory_usage_bytes(&self) -> u64 {
        self.memory_manager.lock_recover().peak_usage()
    }

    /// Run a load/unload stress test to detect memory leaks.
    ///
    /// Loads the model, generates `tokens_per_cycle` tokens, unloads,
    /// and repeats for `cycles` iterations. Returns a log of RSS after
    /// each cycle so the caller can check for monotonic growth.
    pub fn stress_test(
        &self,
        model_path: String,
        cycles: u32,
        tokens_per_cycle: u32,
    ) -> Result<Vec<String>, CrabInferError> {
        let mut log: Vec<String> = Vec::new();

        let baseline_rss = Self::resident_memory_mb();
        log.push(format!("Baseline RSS: {} MB", baseline_rss));
        log_debug!("[CrabInfer-StressTest] Baseline RSS: {} MB", baseline_rss);

        for i in 0..cycles {
            log_debug!("[CrabInfer-StressTest] === Cycle {}/{} ===", i + 1, cycles);

            // Load
            self.load_model(model_path.clone())?;
            let after_load = Self::resident_memory_mb();

            // Generate tokens
            self.reset();
            let prompt = "Hello world".to_string();
            for t in 0..tokens_per_cycle {
                match self.next_token(prompt.clone())? {
                    Some(tok) if tok.is_end_of_sequence => {
                        log_debug!("[CrabInfer-StressTest] EOS at token {}", t + 1);
                        break;
                    }
                    None => break,
                    _ => {}
                }
            }
            let after_gen = Self::resident_memory_mb();

            // Unload
            self.unload_model();
            let after_unload = Self::resident_memory_mb();

            let entry = format!(
                "Cycle {}: load={} MB, gen={} MB, unload={} MB (delta from baseline: +{} MB)",
                i + 1, after_load, after_gen, after_unload,
                after_unload.saturating_sub(baseline_rss)
            );
            log_debug!("[CrabInfer-StressTest] {}", entry);
            log.push(entry);
        }

        let final_rss = Self::resident_memory_mb();
        let leak = final_rss.saturating_sub(baseline_rss);
        log.push(format!("Final RSS: {} MB (leaked: {} MB)", final_rss, leak));
        log_debug!(
            "[CrabInfer-StressTest] Final RSS: {} MB, leaked: {} MB",
            final_rss, leak
        );

        Ok(log)
    }

    /// Get current process resident memory in MB (for stress test logging).
    fn resident_memory_mb() -> u64 {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            use std::mem;

            #[repr(C)]
            struct TaskBasicInfo {
                virtual_size: u64,
                resident_size: u64,
                resident_size_max: u64,
                user_time: [u32; 2],
                system_time: [u32; 2],
                policy: i32,
                suspend_count: i32,
            }

            extern "C" {
                fn mach_task_self() -> u32;
                fn task_info(
                    target_task: u32,
                    flavor: u32,
                    task_info: *mut TaskBasicInfo,
                    task_info_count: *mut u32,
                ) -> i32;
            }

            const MACH_TASK_BASIC_INFO: u32 = 20;

            unsafe {
                let mut info: TaskBasicInfo = mem::zeroed();
                let mut count = (mem::size_of::<TaskBasicInfo>() / mem::size_of::<u32>()) as u32;
                let result = task_info(
                    mach_task_self(),
                    MACH_TASK_BASIC_INFO,
                    &mut info as *mut _,
                    &mut count,
                );
                if result == 0 {
                    return info.resident_size / (1024 * 1024);
                }
            }
        }

        0
    }
}

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
) -> Result<(Model, ModelInfo, Device), CrabInferError> {
    let devices_to_try: Vec<Device> = if use_metal {
        log_debug!("[CrabInfer] Metal requested, creating Metal device...");
        match Device::new_metal(0) {
            Ok(metal) => {
                // Configure pre-compiled metallib directory if provided
                if let Some(dir) = metallib_path {
                    if let Ok(metal_dev) = metal.as_metal_device() {
                        log_debug!("[CrabInfer] Setting metallib dir: {}", dir);
                        metal_dev.set_metallib_dir(dir);
                    }
                } else {
                    log_debug!("[CrabInfer] WARNING: No metallib_path provided — Metal will use runtime XPC compilation (may fail on iOS)");
                }
                log_debug!("[CrabInfer] Metal device created OK");
                // CPU fallback is allowed since the pre-flight memory check
                // above already verified the model fits in available memory.
                vec![metal, Device::Cpu]
            }
            Err(e) => {
                log_debug!("[CrabInfer] Metal device FAILED: {}, falling back to CPU", e);
                vec![Device::Cpu]
            }
        }
    } else {
        log_debug!("[CrabInfer] CPU-only mode requested");
        vec![Device::Cpu]
    };

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

                // Step 3: Full forward pass (compiles quantized.metal — 7741 lines,
                // the heaviest compilation and most likely to trigger XPC failure)
                log_debug!("[CrabInfer] Metal warmup 3/3: quantized forward pass (this compiles ~8K lines of Metal shaders)...");
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

    ModelInfo {
        model_name,
        architecture,
        parameter_count,
        quantization,
        file_size_bytes: file_size,
        context_length,
        vocab_size,
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
}
