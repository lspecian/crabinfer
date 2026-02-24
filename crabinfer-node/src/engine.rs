//! CrabInferEngine class — local on-device inference with async operations.

use std::sync::Arc;

use napi::bindgen_prelude::{AsyncTask, Env, Task};
use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi::JsFunction;
use napi_derive::napi;

use crate::error::to_napi_error;
use crate::types::{EngineConfig, GenerationStats, ModelInfo, TokenOutput};
use crate::enums::MemoryPressure;

// ---------------------------------------------------------------------------
// AsyncTask implementations for heavy operations
// ---------------------------------------------------------------------------

pub struct LoadModelTask {
    engine: Arc<crabinfer_core::engine::CrabInferEngine>,
    model_path: String,
}

#[napi]
impl Task for LoadModelTask {
    type Output = ();
    type JsValue = ();

    fn compute(&mut self) -> napi::Result<Self::Output> {
        self.engine
            .load_model(self.model_path.clone())
            .map_err(to_napi_error)
    }

    fn resolve(&mut self, _env: Env, _output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(())
    }
}

pub struct CompleteTask {
    engine: Arc<crabinfer_core::engine::CrabInferEngine>,
    prompt: String,
    max_tokens: u32,
    temperature: f32,
}

#[napi]
impl Task for CompleteTask {
    type Output = String;
    type JsValue = String;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        self.engine
            .complete(self.prompt.clone(), self.max_tokens, self.temperature)
            .map_err(to_napi_error)
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(output)
    }
}

pub struct StressTestTask {
    engine: Arc<crabinfer_core::engine::CrabInferEngine>,
    model_path: String,
    cycles: u32,
    tokens_per_cycle: u32,
}

#[napi]
impl Task for StressTestTask {
    type Output = Vec<String>;
    type JsValue = Vec<String>;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        self.engine
            .stress_test(self.model_path.clone(), self.cycles, self.tokens_per_cycle)
            .map_err(to_napi_error)
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// CrabInferEngine class
// ---------------------------------------------------------------------------

/// Local inference engine — loads GGUF models and runs on-device inference.
#[napi]
pub struct CrabInferEngine {
    inner: Arc<crabinfer_core::engine::CrabInferEngine>,
}

#[napi]
impl CrabInferEngine {
    /// Create a new engine with the given configuration.
    #[napi(constructor)]
    pub fn new(config: EngineConfig) -> napi::Result<Self> {
        let core_config: crabinfer_core::EngineConfig = config.into();
        let inner =
            crabinfer_core::engine::CrabInferEngine::new(core_config).map_err(to_napi_error)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Load a GGUF model file (async — runs on libuv thread pool).
    #[napi(ts_return_type = "Promise<void>")]
    pub fn load_model(&self, model_path: String) -> AsyncTask<LoadModelTask> {
        AsyncTask::new(LoadModelTask {
            engine: self.inner.clone(),
            model_path,
        })
    }

    /// Get loaded model info (sync — fast).
    #[napi]
    pub fn model_info(&self) -> napi::Result<ModelInfo> {
        self.inner.model_info().map(Into::into).map_err(to_napi_error)
    }

    /// Generate a complete response (async — runs on libuv thread pool).
    #[napi(ts_return_type = "Promise<string>")]
    pub fn complete(
        &self,
        prompt: String,
        max_tokens: u32,
        temperature: Option<f64>,
    ) -> AsyncTask<CompleteTask> {
        AsyncTask::new(CompleteTask {
            engine: self.inner.clone(),
            prompt,
            max_tokens,
            temperature: temperature.unwrap_or(0.7) as f32,
        })
    }

    /// Get the next token (sync — used for manual streaming loops).
    #[napi]
    pub fn next_token(&self, prompt: String) -> napi::Result<Option<TokenOutput>> {
        self.inner
            .next_token(prompt)
            .map(|opt| opt.map(Into::into))
            .map_err(to_napi_error)
    }

    /// Reset the engine state.
    #[napi]
    pub fn reset(&self) {
        self.inner.reset();
    }

    /// Get generation stats from the last complete() call.
    #[napi]
    pub fn last_stats(&self) -> Option<GenerationStats> {
        self.inner.last_stats().map(Into::into)
    }

    /// Get current memory pressure level (sync — fast).
    #[napi]
    pub fn memory_pressure(&self) -> MemoryPressure {
        self.inner.memory_pressure().into()
    }

    /// Reduce memory usage by clearing caches.
    #[napi]
    pub fn reduce_memory(&self) {
        self.inner.reduce_memory();
    }

    /// Unload the current model.
    #[napi]
    pub fn unload_model(&self) {
        self.inner.unload_model();
    }

    /// Check if a model is currently loaded.
    #[napi]
    pub fn is_model_loaded(&self) -> bool {
        self.inner.is_model_loaded()
    }

    /// Get current Metal GPU memory allocation in bytes.
    #[napi]
    pub fn metal_allocated_bytes(&self) -> i64 {
        self.inner.metal_allocated_bytes() as i64
    }

    /// Register a callback for memory pressure changes.
    /// The callback receives (oldLevel: string, newLevel: string).
    #[napi(ts_args_type = "callback: (oldLevel: string, newLevel: string) => void")]
    pub fn on_memory_pressure(&self, callback: JsFunction) -> napi::Result<()> {
        let tsfn: ThreadsafeFunction<(String, String), ErrorStrategy::Fatal> =
            callback.create_threadsafe_function(0, |ctx: napi::threadsafe_function::ThreadSafeCallContext<(String, String)>| {
                Ok(vec![
                    ctx.env.create_string(&ctx.value.0)?.into_unknown(),
                    ctx.env.create_string(&ctx.value.1)?.into_unknown(),
                ])
            })?;

        self.inner.set_pressure_callback(move |old, new| {
            let old_str = format!("{:?}", old);
            let new_str = format!("{:?}", new);
            tsfn.call((old_str, new_str), ThreadsafeFunctionCallMode::NonBlocking);
        });

        Ok(())
    }

    /// Run a load/unload stress test (async — long running).
    #[napi(ts_return_type = "Promise<string[]>")]
    pub fn stress_test(
        &self,
        model_path: String,
        cycles: u32,
        tokens_per_cycle: u32,
    ) -> AsyncTask<StressTestTask> {
        AsyncTask::new(StressTestTask {
            engine: self.inner.clone(),
            model_path,
            cycles,
            tokens_per_cycle,
        })
    }
}
