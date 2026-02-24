//! ModelDownloadManager class — download models from HuggingFace with progress.

#[cfg(feature = "providers")]
use napi::bindgen_prelude::{AsyncTask, Env, Task};
#[cfg(feature = "providers")]
use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode};
#[cfg(feature = "providers")]
use napi::JsFunction;
#[cfg(feature = "providers")]
use napi_derive::napi;

#[cfg(feature = "providers")]
use crate::error::to_napi_error;
#[cfg(feature = "providers")]
use crate::types::{CatalogEntry, DownloadedModel};

// ---------------------------------------------------------------------------
// Progress listener bridging JS callback to Rust trait
// ---------------------------------------------------------------------------

#[cfg(feature = "providers")]
struct JsProgressListener {
    tsfn: ThreadsafeFunction<(u64, u64, String), ErrorStrategy::Fatal>,
}

#[cfg(feature = "providers")]
impl crabinfer_core::download::DownloadProgressListener for JsProgressListener {
    fn on_progress(&self, bytes_downloaded: u64, bytes_total: u64, phase: String) {
        self.tsfn.call(
            (bytes_downloaded, bytes_total, phase),
            ThreadsafeFunctionCallMode::NonBlocking,
        );
    }
}

// ---------------------------------------------------------------------------
// Download task (runs on libuv thread pool)
// ---------------------------------------------------------------------------

#[cfg(feature = "providers")]
pub struct DownloadTask {
    manager: std::sync::Arc<crabinfer_core::download::ModelDownloadManager>,
    entry: crabinfer_core::catalog::CatalogEntry,
    listener: Option<Box<dyn crabinfer_core::download::DownloadProgressListener>>,
}

#[cfg(feature = "providers")]
#[napi]
impl Task for DownloadTask {
    type Output = crabinfer_core::download::DownloadedModel;
    type JsValue = DownloadedModel;

    fn compute(&mut self) -> napi::Result<Self::Output> {
        self.manager
            .download(self.entry.clone(), self.listener.take().map(|l| l as Box<_>))
            .map_err(to_napi_error)
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> napi::Result<Self::JsValue> {
        Ok(output.into())
    }
}

// ---------------------------------------------------------------------------
// ModelDownloadManager class
// ---------------------------------------------------------------------------

/// Manages model downloads from HuggingFace with resume and integrity checks.
#[cfg(feature = "providers")]
#[napi]
pub struct ModelDownloadManager {
    inner: std::sync::Arc<crabinfer_core::download::ModelDownloadManager>,
}

#[cfg(feature = "providers")]
#[napi]
impl ModelDownloadManager {
    /// Create a download manager with the default storage directory.
    #[napi(constructor)]
    pub fn new() -> napi::Result<Self> {
        let inner = crabinfer_core::download::ModelDownloadManager::new().map_err(to_napi_error)?;
        Ok(Self {
            inner: std::sync::Arc::new(inner),
        })
    }

    /// Create a download manager with a custom storage directory.
    #[napi(factory)]
    pub fn with_directory(directory: String) -> napi::Result<Self> {
        let inner = crabinfer_core::download::ModelDownloadManager::with_directory(directory)
            .map_err(to_napi_error)?;
        Ok(Self {
            inner: std::sync::Arc::new(inner),
        })
    }

    /// Download a model (async with optional progress callback).
    #[napi(
        ts_args_type = "entry: CatalogEntry, onProgress?: (bytesDownloaded: number, bytesTotal: number, phase: string) => void",
        ts_return_type = "Promise<DownloadedModel>"
    )]
    pub fn download(
        &self,
        entry: CatalogEntry,
        on_progress: Option<JsFunction>,
    ) -> napi::Result<AsyncTask<DownloadTask>> {
        let core_entry: crabinfer_core::catalog::CatalogEntry = entry.into();

        let listener: Option<Box<dyn crabinfer_core::download::DownloadProgressListener>> =
            match on_progress {
                Some(callback) => {
                    let tsfn: ThreadsafeFunction<(u64, u64, String), ErrorStrategy::Fatal> =
                        callback.create_threadsafe_function(
                            0,
                            |ctx: napi::threadsafe_function::ThreadSafeCallContext<(u64, u64, String)>| {
                                Ok(vec![
                                    ctx.env.create_int64(ctx.value.0 as i64)?.into_unknown(),
                                    ctx.env.create_int64(ctx.value.1 as i64)?.into_unknown(),
                                    ctx.env.create_string(&ctx.value.2)?.into_unknown(),
                                ])
                            },
                        )?;
                    Some(Box::new(JsProgressListener { tsfn }))
                }
                None => None,
            };

        Ok(AsyncTask::new(DownloadTask {
            manager: self.inner.clone(),
            entry: core_entry,
            listener,
        }))
    }

    /// Check if a model is downloaded by catalog ID.
    #[napi]
    pub fn is_downloaded(&self, catalog_id: String) -> bool {
        self.inner.is_downloaded(catalog_id)
    }

    /// Get the model file path for a downloaded model.
    #[napi]
    pub fn model_path(&self, catalog_id: String) -> Option<String> {
        self.inner.model_path(catalog_id)
    }

    /// Get the tokenizer path for a downloaded model.
    #[napi]
    pub fn tokenizer_path(&self, catalog_id: String) -> Option<String> {
        self.inner.tokenizer_path(catalog_id)
    }

    /// List all downloaded models.
    #[napi]
    pub fn list_downloaded(&self) -> Vec<DownloadedModel> {
        self.inner
            .list_downloaded()
            .into_iter()
            .map(Into::into)
            .collect()
    }

    /// Delete a downloaded model by catalog ID.
    #[napi]
    pub fn delete(&self, catalog_id: String) -> napi::Result<()> {
        self.inner.delete(catalog_id).map_err(to_napi_error)
    }

    /// Total disk usage of all downloaded models in bytes.
    #[napi]
    pub fn total_disk_usage(&self) -> i64 {
        self.inner.total_disk_usage() as i64
    }

    /// Get the storage directory path.
    #[napi]
    pub fn storage_directory(&self) -> String {
        self.inner.storage_directory()
    }
}
