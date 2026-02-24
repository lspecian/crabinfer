//! Knowledge Base — RAG (Retrieval-Augmented Generation) pipeline.
//!
//! Combines document chunking, embedding, and vector search into a simple
//! API for adding documents and retrieving relevant context for completions.
//!
//! ```rust,no_run
//! use crabinfer_core::knowledge::KnowledgeBase;
//! use crabinfer_core::embedding::TfIdfEmbedder;
//!
//! let mut kb = KnowledgeBase::new(Box::new(TfIdfEmbedder::default()));
//! kb.add_text("doc1", "Rust is a systems programming language...");
//! let results = kb.query("What is Rust?", 3).unwrap();
//! ```

use crate::chunker::TextChunker;
use crate::embedding::EmbeddingProvider;
use crate::vectorstore::{SearchResult, VectorStore};
use crate::CrabInferError;

/// A knowledge base for RAG (Retrieval-Augmented Generation).
///
/// Documents are chunked, embedded, and stored in an in-process vector store.
/// At query time, relevant chunks are retrieved by semantic similarity and
/// can be injected into the system prompt.
pub struct KnowledgeBase {
    chunker: TextChunker,
    embedder: Box<dyn EmbeddingProvider>,
    store: VectorStore,
    /// Track document sources for management.
    document_sources: Vec<String>,
}

impl KnowledgeBase {
    /// Create a new knowledge base with the given embedding provider.
    pub fn new(embedder: Box<dyn EmbeddingProvider>) -> Self {
        let dimension = embedder.dimension();
        Self {
            chunker: TextChunker::default(),
            embedder,
            store: VectorStore::new(dimension),
            document_sources: Vec::new(),
        }
    }

    /// Use a custom chunker configuration.
    pub fn with_chunker(mut self, chunk_size: usize, overlap: usize) -> Self {
        self.chunker = TextChunker::new(chunk_size, overlap);
        self
    }

    /// Set a file path for persisting the vector store.
    pub fn with_persist_path(mut self, path: &str) -> Self {
        self.store = self.store.with_persist_path(path);
        self
    }

    /// Add a text document to the knowledge base.
    ///
    /// The text is chunked, embedded, and stored. If a document with the same
    /// source name already exists, it is replaced.
    pub fn add_text(&mut self, source: &str, text: &str) -> Result<usize, CrabInferError> {
        // Remove existing chunks for this source
        self.store.remove_by_source(source);
        self.document_sources.retain(|s| s != source);

        // Chunk the document
        let chunks = self.chunker.chunk(text, source);
        if chunks.is_empty() {
            return Ok(0);
        }

        // Embed all chunks
        let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
        let embeddings = self.embedder.embed(&texts)?;

        // Store chunks with embeddings
        let chunk_count = chunks.len();
        for (i, (chunk, embedding)) in chunks.into_iter().zip(embeddings).enumerate() {
            let id = format!("{}::{}", source, i);
            self.store.add(&id, embedding, &chunk.text, chunk.metadata);
        }

        self.document_sources.push(source.to_string());
        Ok(chunk_count)
    }

    /// Add a file to the knowledge base.
    ///
    /// Reads the file contents and indexes them. Supports plain text and
    /// markdown files.
    pub fn add_file(&mut self, path: &str) -> Result<usize, CrabInferError> {
        let content = std::fs::read_to_string(path).map_err(|e| CrabInferError::StorageError {
            reason: format!("Failed to read file '{}': {}", path, e),
        })?;

        self.add_text(path, &content)
    }

    /// Remove a document and all its chunks from the knowledge base.
    pub fn remove_document(&mut self, source: &str) {
        self.store.remove_by_source(source);
        self.document_sources.retain(|s| s != source);
    }

    /// Query the knowledge base for relevant chunks.
    ///
    /// Returns the top-k most semantically similar chunks to the query text.
    pub fn query(&self, text: &str, top_k: usize) -> Result<Vec<SearchResult>, CrabInferError> {
        if self.store.is_empty() {
            return Ok(Vec::new());
        }

        // Embed the query
        let query_embeddings = self.embedder.embed(&[text.to_string()])?;
        let query_embedding = query_embeddings.into_iter().next().ok_or(
            CrabInferError::InferenceFailed,
        )?;

        // Search
        Ok(self.store.search(&query_embedding, top_k))
    }

    /// Query and format results as strings for injection into a system prompt.
    ///
    /// Returns a vec of chunk texts, ready to pass to `SystemPrompt::build_with_context()`.
    pub fn query_for_prompt(
        &self,
        text: &str,
        top_k: usize,
    ) -> Result<Vec<String>, CrabInferError> {
        let results = self.query(text, top_k)?;
        Ok(results.into_iter().map(|r| r.text).collect())
    }

    /// Query with source attribution.
    ///
    /// Returns results with both text and source metadata, useful for
    /// displaying citations to the user.
    pub fn query_with_sources(
        &self,
        text: &str,
        top_k: usize,
    ) -> Result<Vec<(String, String, f32)>, CrabInferError> {
        let results = self.query(text, top_k)?;
        Ok(results
            .into_iter()
            .map(|r| (r.text, r.metadata.source, r.score))
            .collect())
    }

    /// Get the number of indexed documents.
    pub fn document_count(&self) -> usize {
        self.document_sources.len()
    }

    /// Get the number of stored chunks.
    pub fn chunk_count(&self) -> usize {
        self.store.len()
    }

    /// List all indexed document sources.
    pub fn document_sources(&self) -> &[String] {
        &self.document_sources
    }

    /// Get the name of the embedding provider.
    pub fn embedder_name(&self) -> &str {
        self.embedder.name()
    }

    /// Save the vector store to disk.
    pub fn save(&self) -> Result<(), CrabInferError> {
        self.store.save()
    }

    /// Clear all documents and chunks.
    pub fn clear(&mut self) {
        self.store.clear();
        self.document_sources.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::TfIdfEmbedder;

    fn make_kb() -> KnowledgeBase {
        let mut embedder = TfIdfEmbedder::new(128);
        // Pre-fit with some corpus
        embedder.fit(&[
            "Rust is a systems programming language focused on safety and performance".to_string(),
            "Python is widely used for data science and machine learning".to_string(),
            "JavaScript powers the web and runs in browsers".to_string(),
        ]);
        KnowledgeBase::new(Box::new(embedder))
    }

    #[test]
    fn test_add_and_query() {
        let mut kb = make_kb();
        let count = kb
            .add_text(
                "rust-intro",
                "Rust is a systems programming language. It provides memory safety without garbage collection. Rust's ownership system ensures thread safety at compile time.",
            )
            .unwrap();
        assert!(count >= 1);

        let results = kb.query("What is Rust?", 3).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].text.contains("Rust"));
    }

    #[test]
    fn test_add_replaces_existing() {
        let mut kb = make_kb();
        kb.add_text("doc", "Original content about Rust").unwrap();
        let count_before = kb.chunk_count();

        kb.add_text("doc", "Updated content about Python").unwrap();
        // Should not accumulate chunks
        assert!(kb.chunk_count() <= count_before + 1);
    }

    #[test]
    fn test_remove_document() {
        let mut kb = make_kb();
        kb.add_text("doc1", "Content about Rust").unwrap();
        kb.add_text("doc2", "Content about Python").unwrap();
        assert_eq!(kb.document_count(), 2);

        kb.remove_document("doc1");
        assert_eq!(kb.document_count(), 1);
        assert_eq!(kb.document_sources(), &["doc2"]);
    }

    #[test]
    fn test_query_empty_kb() {
        let kb = make_kb();
        let results = kb.query("anything", 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_query_for_prompt() {
        let mut kb = make_kb();
        kb.add_text("doc", "Rust provides memory safety through ownership").unwrap();

        let prompt_chunks = kb.query_for_prompt("Tell me about Rust memory", 2).unwrap();
        assert!(!prompt_chunks.is_empty());
        assert!(prompt_chunks[0].contains("Rust") || prompt_chunks[0].contains("memory"));
    }

    #[test]
    fn test_query_with_sources() {
        let mut kb = make_kb();
        kb.add_text("rust-guide.md", "Rust ownership and borrowing explained").unwrap();

        let results = kb.query_with_sources("ownership", 2).unwrap();
        assert!(!results.is_empty());
        let (text, source, score) = &results[0];
        assert!(!text.is_empty());
        assert_eq!(source, "rust-guide.md");
        assert!(*score > 0.0);
    }

    #[test]
    fn test_clear() {
        let mut kb = make_kb();
        kb.add_text("doc", "Some content").unwrap();
        kb.clear();
        assert_eq!(kb.document_count(), 0);
        assert_eq!(kb.chunk_count(), 0);
    }

    #[test]
    fn test_custom_chunker() {
        let embedder = TfIdfEmbedder::new(128);
        let mut kb = KnowledgeBase::new(Box::new(embedder))
            .with_chunker(50, 10); // Small chunks for testing

        let long_text = "word ".repeat(100); // 500 chars
        let count = kb.add_text("long-doc", &long_text).unwrap();
        assert!(count > 1, "Long text should produce multiple small chunks");
    }
}
