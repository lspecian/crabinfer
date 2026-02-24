//! Node.js bindings for A6 (Knowledge Layer / RAG).
//!
//! Exposes KnowledgeBase to JS with TF-IDF embedding as default
//! and optional OpenAI embeddings when API key is provided.

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::error::to_napi_error;

/// A search result from the knowledge base.
#[napi(object)]
pub struct JsSearchResult {
    pub text: String,
    pub source: String,
    pub score: f64,
    pub chunk_index: u32,
}

/// RAG knowledge base — add documents, query for relevant context.
#[napi]
pub struct KnowledgeBase {
    inner: crabinfer_core::knowledge::KnowledgeBase,
}

#[napi]
impl KnowledgeBase {
    /// Create a knowledge base with TF-IDF embeddings (offline, no API key needed).
    #[napi(constructor)]
    pub fn new() -> Self {
        let embedder = Box::new(crabinfer_core::embedding::TfIdfEmbedder::default());
        Self {
            inner: crabinfer_core::knowledge::KnowledgeBase::new(embedder),
        }
    }

    /// Create a knowledge base with OpenAI embeddings.
    #[cfg(feature = "providers")]
    #[napi(factory)]
    pub fn with_openai(api_key: String) -> Self {
        let embedder = Box::new(crabinfer_core::embedding::OpenAIEmbedder::new(&api_key));
        Self {
            inner: crabinfer_core::knowledge::KnowledgeBase::new(embedder),
        }
    }

    /// Set the file path for persisting the vector store.
    #[napi]
    pub fn set_persist_path(&mut self, path: String) {
        // Reconstruct with persist path
        // Since KnowledgeBase uses builder pattern on construction, we store the path
        // and use it on save
        let embedder = Box::new(crabinfer_core::embedding::TfIdfEmbedder::default());
        let kb = crabinfer_core::knowledge::KnowledgeBase::new(embedder)
            .with_persist_path(&path);
        self.inner = kb;
    }

    /// Add a text document to the knowledge base.
    /// Returns the number of chunks created.
    #[napi]
    pub fn add_text(&mut self, source: String, text: String) -> Result<u32> {
        self.inner
            .add_text(&source, &text)
            .map(|n| n as u32)
            .map_err(to_napi_error)
    }

    /// Add a file to the knowledge base.
    /// Returns the number of chunks created.
    #[napi]
    pub fn add_file(&mut self, path: String) -> Result<u32> {
        self.inner
            .add_file(&path)
            .map(|n| n as u32)
            .map_err(to_napi_error)
    }

    /// Remove a document and all its chunks.
    #[napi]
    pub fn remove_document(&mut self, source: String) {
        self.inner.remove_document(&source);
    }

    /// Query for relevant chunks.
    #[napi]
    pub fn query(&self, text: String, top_k: u32) -> Result<Vec<JsSearchResult>> {
        let results = self
            .inner
            .query_with_sources(&text, top_k as usize)
            .map_err(to_napi_error)?;

        Ok(results
            .into_iter()
            .map(|(text, source, score)| JsSearchResult {
                text,
                source,
                score: score as f64,
                chunk_index: 0,
            })
            .collect())
    }

    /// Query and return only text strings (for prompt injection).
    #[napi]
    pub fn query_for_prompt(&self, text: String, top_k: u32) -> Result<Vec<String>> {
        self.inner
            .query_for_prompt(&text, top_k as usize)
            .map_err(to_napi_error)
    }

    /// Get the number of indexed documents.
    #[napi(getter)]
    pub fn document_count(&self) -> u32 {
        self.inner.document_count() as u32
    }

    /// Get the number of stored chunks.
    #[napi(getter)]
    pub fn chunk_count(&self) -> u32 {
        self.inner.chunk_count() as u32
    }

    /// List all indexed document sources.
    #[napi]
    pub fn document_sources(&self) -> Vec<String> {
        self.inner.document_sources().to_vec()
    }

    /// Get the embedding provider name.
    #[napi(getter)]
    pub fn embedder_name(&self) -> String {
        self.inner.embedder_name().to_string()
    }

    /// Save the vector store to disk.
    #[napi]
    pub fn save(&self) -> Result<()> {
        self.inner.save().map_err(to_napi_error)
    }

    /// Clear all documents.
    #[napi]
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}
