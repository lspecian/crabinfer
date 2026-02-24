//! In-Process Vector Store — stores and searches embedding vectors.
//!
//! A simple but effective vector store that runs entirely in-process
//! with no external dependencies. Supports cosine similarity search,
//! persistence to disk, and incremental updates.

use crate::chunker::ChunkMetadata;
use crate::CrabInferError;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A search result with similarity score.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub text: String,
    pub score: f32,
    pub metadata: ChunkMetadata,
}

/// A stored vector entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredVector {
    pub id: String,
    pub embedding: Vec<f32>,
    pub text: String,
    pub metadata: ChunkMetadata,
}

/// In-process vector store with cosine similarity search.
#[derive(Debug, Serialize, Deserialize)]
pub struct VectorStore {
    vectors: Vec<StoredVector>,
    dimension: usize,
    #[serde(skip)]
    persist_path: Option<String>,
}

impl VectorStore {
    /// Create a new empty vector store with the given embedding dimension.
    pub fn new(dimension: usize) -> Self {
        Self {
            vectors: Vec::new(),
            dimension,
            persist_path: None,
        }
    }

    /// Set a file path for persistence.
    pub fn with_persist_path(mut self, path: &str) -> Self {
        self.persist_path = Some(path.to_string());
        self
    }

    /// Add a vector to the store.
    pub fn add(
        &mut self,
        id: &str,
        embedding: Vec<f32>,
        text: &str,
        metadata: ChunkMetadata,
    ) {
        self.vectors.push(StoredVector {
            id: id.to_string(),
            embedding,
            text: text.to_string(),
            metadata,
        });
    }

    /// Search for the top-k most similar vectors to the query.
    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Vec<SearchResult> {
        if self.vectors.is_empty() || query_embedding.is_empty() {
            return Vec::new();
        }

        let mut scores: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, cosine_similarity(query_embedding, &v.embedding)))
            .collect();

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scores
            .into_iter()
            .take(top_k)
            .filter(|(_, score)| *score > 0.0) // Filter out zero-similarity
            .map(|(i, score)| {
                let v = &self.vectors[i];
                SearchResult {
                    text: v.text.clone(),
                    score,
                    metadata: v.metadata.clone(),
                }
            })
            .collect()
    }

    /// Remove all vectors from a given source document.
    pub fn remove_by_source(&mut self, source: &str) {
        self.vectors.retain(|v| v.metadata.source != source);
    }

    /// Remove a vector by ID.
    pub fn remove_by_id(&mut self, id: &str) {
        self.vectors.retain(|v| v.id != id);
    }

    /// Get the number of stored vectors.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Get all unique source identifiers.
    pub fn sources(&self) -> Vec<String> {
        let mut sources: Vec<String> = self
            .vectors
            .iter()
            .map(|v| v.metadata.source.clone())
            .collect();
        sources.sort();
        sources.dedup();
        sources
    }

    /// Get the embedding dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Clear all vectors.
    pub fn clear(&mut self) {
        self.vectors.clear();
    }

    /// Save the store to disk (JSON).
    pub fn save(&self) -> Result<(), CrabInferError> {
        let path = self.persist_path.as_ref().ok_or(CrabInferError::StorageError {
            reason: "No persist path set".to_string(),
        })?;

        let json = serde_json::to_string(self).map_err(|e| CrabInferError::StorageError {
            reason: format!("Serialization failed: {}", e),
        })?;

        if let Some(parent) = Path::new(path).parent() {
            std::fs::create_dir_all(parent).map_err(|e| CrabInferError::StorageError {
                reason: format!("Failed to create directory: {}", e),
            })?;
        }

        std::fs::write(path, json).map_err(|e| CrabInferError::StorageError {
            reason: format!("Failed to write file: {}", e),
        })?;

        Ok(())
    }

    /// Load a store from a JSON file.
    pub fn load(path: &str) -> Result<Self, CrabInferError> {
        let json = std::fs::read_to_string(path).map_err(|e| CrabInferError::StorageError {
            reason: format!("Failed to read file: {}", e),
        })?;

        let mut store: Self =
            serde_json::from_str(&json).map_err(|e| CrabInferError::StorageError {
                reason: format!("Deserialization failed: {}", e),
            })?;

        store.persist_path = Some(path.to_string());
        Ok(store)
    }
}

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_metadata(source: &str, idx: usize) -> ChunkMetadata {
        ChunkMetadata {
            source: source.to_string(),
            chunk_index: idx,
            start_offset: 0,
        }
    }

    #[test]
    fn test_add_and_search() {
        let mut store = VectorStore::new(3);
        store.add("v1", vec![1.0, 0.0, 0.0], "doc about rust", make_metadata("rust.md", 0));
        store.add("v2", vec![0.0, 1.0, 0.0], "doc about python", make_metadata("python.md", 0));
        store.add("v3", vec![0.9, 0.1, 0.0], "another rust doc", make_metadata("rust2.md", 0));

        // Search for something similar to rust
        let results = store.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].text, "doc about rust"); // Exact match
        assert!(results[0].score > results[1].score);
    }

    #[test]
    fn test_remove_by_source() {
        let mut store = VectorStore::new(3);
        store.add("v1", vec![1.0, 0.0, 0.0], "chunk 1", make_metadata("doc.md", 0));
        store.add("v2", vec![0.0, 1.0, 0.0], "chunk 2", make_metadata("doc.md", 1));
        store.add("v3", vec![0.0, 0.0, 1.0], "chunk 3", make_metadata("other.md", 0));

        store.remove_by_source("doc.md");
        assert_eq!(store.len(), 1);
        assert_eq!(store.vectors[0].text, "chunk 3");
    }

    #[test]
    fn test_sources() {
        let mut store = VectorStore::new(3);
        store.add("v1", vec![1.0, 0.0, 0.0], "a", make_metadata("a.md", 0));
        store.add("v2", vec![0.0, 1.0, 0.0], "b", make_metadata("b.md", 0));
        store.add("v3", vec![0.0, 0.0, 1.0], "c", make_metadata("a.md", 1));

        let sources = store.sources();
        assert_eq!(sources, vec!["a.md", "b.md"]);
    }

    #[test]
    fn test_empty_search() {
        let store = VectorStore::new(3);
        let results = store.search(&[1.0, 0.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_cosine_similarity() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 0.001);
        assert!((cosine_similarity(&[1.0, 0.0], &[0.0, 1.0])).abs() < 0.001);
        assert!((cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_persistence() {
        let dir = std::env::temp_dir().join("crabinfer-test-vecstore");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.json");
        let path_str = path.to_str().unwrap();

        // Save
        let mut store = VectorStore::new(3).with_persist_path(path_str);
        store.add("v1", vec![1.0, 0.0, 0.0], "test doc", make_metadata("test.md", 0));
        store.save().unwrap();

        // Load
        let loaded = VectorStore::load(path_str).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.dimension(), 3);

        // Cleanup
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }
}
