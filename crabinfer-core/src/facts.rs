//! Persistent Memory Store — key-value facts the model should always know.
//!
//! Stores facts like "User's name is Luis" or "User prefers concise answers"
//! that get automatically injected into the system prompt via the PromptBuilder.

use crate::CrabInferError;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A single fact about the user or context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub key: String,
    pub value: String,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

/// Persistent store for user/context facts.
///
/// Facts are key-value pairs that get injected into the system prompt
/// automatically. Keys are unique — adding a fact with an existing key
/// updates the value.
///
/// ```rust
/// use crabinfer_core::facts::MemoryStore;
///
/// let mut store = MemoryStore::new();
/// store.add_fact("name", "Luis");
/// store.add_fact("language", "Rust");
///
/// let context = store.as_prompt_context();
/// // Returns: ["name: Luis", "language: Rust"]
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStore {
    facts: Vec<Fact>,
    #[serde(skip)]
    persist_path: Option<String>,
}

impl MemoryStore {
    /// Create an empty memory store.
    pub fn new() -> Self {
        Self {
            facts: Vec::new(),
            persist_path: None,
        }
    }

    /// Set a file path for JSON persistence.
    pub fn with_persist_path(mut self, path: &str) -> Self {
        self.persist_path = Some(path.to_string());
        self
    }

    fn now_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Add or update a fact. If the key already exists, the value is updated.
    pub fn add_fact(&mut self, key: &str, value: &str) {
        let now = Self::now_ms();
        if let Some(existing) = self.facts.iter_mut().find(|f| f.key == key) {
            existing.value = value.to_string();
            existing.updated_at_ms = now;
        } else {
            self.facts.push(Fact {
                key: key.to_string(),
                value: value.to_string(),
                created_at_ms: now,
                updated_at_ms: now,
            });
        }
    }

    /// Remove a fact by key. Returns true if the fact existed.
    pub fn remove_fact(&mut self, key: &str) -> bool {
        let len_before = self.facts.len();
        self.facts.retain(|f| f.key != key);
        self.facts.len() < len_before
    }

    /// Get a fact by key.
    pub fn get_fact(&self, key: &str) -> Option<&Fact> {
        self.facts.iter().find(|f| f.key == key)
    }

    /// Get the value of a fact by key.
    pub fn get_value(&self, key: &str) -> Option<&str> {
        self.get_fact(key).map(|f| f.value.as_str())
    }

    /// Check if a fact exists.
    pub fn has_fact(&self, key: &str) -> bool {
        self.facts.iter().any(|f| f.key == key)
    }

    /// Get all facts.
    pub fn all_facts(&self) -> &[Fact] {
        &self.facts
    }

    /// Get the number of stored facts.
    pub fn count(&self) -> usize {
        self.facts.len()
    }

    /// Clear all facts.
    pub fn clear(&mut self) {
        self.facts.clear();
    }

    /// Format all facts as strings for injection into the system prompt.
    /// Returns a vec of strings like ["name: Luis", "language: Rust"].
    pub fn as_prompt_context(&self) -> Vec<String> {
        self.facts
            .iter()
            .map(|f| format!("{}: {}", f.key, f.value))
            .collect()
    }

    /// Save the store to disk (JSON).
    pub fn save(&self) -> Result<(), CrabInferError> {
        let path = self.persist_path.as_ref().ok_or(CrabInferError::StorageError {
            reason: "No persist path set".to_string(),
        })?;

        let json = serde_json::to_string_pretty(self).map_err(|e| CrabInferError::StorageError {
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

impl Default for MemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_get() {
        let mut store = MemoryStore::new();
        store.add_fact("name", "Luis");

        assert_eq!(store.get_value("name"), Some("Luis"));
        assert_eq!(store.count(), 1);
    }

    #[test]
    fn test_update_fact() {
        let mut store = MemoryStore::new();
        store.add_fact("language", "Python");
        store.add_fact("language", "Rust");

        assert_eq!(store.get_value("language"), Some("Rust"));
        assert_eq!(store.count(), 1); // Should not duplicate
    }

    #[test]
    fn test_remove_fact() {
        let mut store = MemoryStore::new();
        store.add_fact("name", "Luis");
        store.add_fact("language", "Rust");

        assert!(store.remove_fact("name"));
        assert!(!store.has_fact("name"));
        assert_eq!(store.count(), 1);

        assert!(!store.remove_fact("nonexistent"));
    }

    #[test]
    fn test_as_prompt_context() {
        let mut store = MemoryStore::new();
        store.add_fact("name", "Luis");
        store.add_fact("preference", "concise answers");

        let context = store.as_prompt_context();
        assert_eq!(context.len(), 2);
        assert!(context.contains(&"name: Luis".to_string()));
        assert!(context.contains(&"preference: concise answers".to_string()));
    }

    #[test]
    fn test_clear() {
        let mut store = MemoryStore::new();
        store.add_fact("a", "1");
        store.add_fact("b", "2");
        store.clear();
        assert_eq!(store.count(), 0);
    }

    #[test]
    fn test_json_persistence() {
        let dir = std::env::temp_dir().join("crabinfer-test-facts");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test-facts.json");
        let path_str = path.to_str().unwrap();

        // Save
        let mut store = MemoryStore::new().with_persist_path(path_str);
        store.add_fact("name", "Luis");
        store.add_fact("language", "Rust");
        store.save().unwrap();

        // Load
        let loaded = MemoryStore::load(path_str).unwrap();
        assert_eq!(loaded.count(), 2);
        assert_eq!(loaded.get_value("name"), Some("Luis"));

        // Cleanup
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }
}
