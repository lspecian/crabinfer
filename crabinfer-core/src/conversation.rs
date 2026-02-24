//! Conversation Memory — manages chat history with windowing and persistence.
//!
//! Handles message history for multi-turn conversations with configurable
//! window size, token-budget truncation, and optional JSON persistence.

use crate::provider::ChatMessage;
use crate::CrabInferError;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A single conversation session with message history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationMemory {
    /// Unique conversation ID.
    id: String,
    /// All messages in this conversation.
    messages: Vec<StoredMessage>,
    /// Maximum number of messages to keep (0 = unlimited).
    max_messages: usize,
    /// Maximum token budget for messages (0 = unlimited).
    /// When exceeded, oldest non-system messages are removed.
    token_budget: u32,
    /// Optional file path for persistence.
    #[serde(skip)]
    persist_path: Option<String>,
    /// Summary of older messages that were truncated.
    summary: Option<String>,
}

/// A message with metadata for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredMessage {
    pub role: String,
    pub content: String,
    pub timestamp_ms: u64,
}

impl StoredMessage {
    pub fn to_chat_message(&self) -> ChatMessage {
        ChatMessage {
            role: self.role.clone(),
            content: self.content.clone(),
        }
    }

    fn estimate_tokens(&self) -> u32 {
        // Rough estimate: 1 token ≈ 4 chars
        ((self.role.len() + self.content.len()) / 4).max(1) as u32
    }
}

impl ConversationMemory {
    /// Create a new conversation with the given ID.
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            messages: Vec::new(),
            max_messages: 0,
            token_budget: 0,
            persist_path: None,
            summary: None,
        }
    }

    /// Set the maximum number of messages to retain.
    pub fn with_max_messages(mut self, max: usize) -> Self {
        self.max_messages = max;
        self
    }

    /// Set the token budget for the conversation window.
    pub fn with_token_budget(mut self, budget: u32) -> Self {
        self.token_budget = budget;
        self
    }

    /// Set a file path for JSON persistence.
    pub fn with_persist_path(mut self, path: &str) -> Self {
        self.persist_path = Some(path.to_string());
        self
    }

    /// Get the conversation ID.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Add a message to the conversation.
    pub fn add_message(&mut self, role: &str, content: &str) {
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        self.messages.push(StoredMessage {
            role: role.to_string(),
            content: content.to_string(),
            timestamp_ms,
        });

        self.enforce_limits();
    }

    /// Add a user message.
    pub fn add_user_message(&mut self, content: &str) {
        self.add_message("user", content);
    }

    /// Add an assistant message.
    pub fn add_assistant_message(&mut self, content: &str) {
        self.add_message("assistant", content);
    }

    /// Get all messages as ChatMessage for use in completions.
    pub fn messages(&self) -> Vec<ChatMessage> {
        let mut result = Vec::new();

        // If there's a summary, prepend it as a system message
        if let Some(ref summary) = self.summary {
            result.push(ChatMessage {
                role: "system".to_string(),
                content: format!("Summary of earlier conversation: {}", summary),
            });
        }

        for msg in &self.messages {
            result.push(msg.to_chat_message());
        }

        result
    }

    /// Get the N most recent messages.
    pub fn recent_messages(&self, count: usize) -> Vec<ChatMessage> {
        let start = self.messages.len().saturating_sub(count);
        self.messages[start..]
            .iter()
            .map(|m| m.to_chat_message())
            .collect()
    }

    /// Get the number of stored messages.
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Estimate the total token count of all messages.
    pub fn estimate_tokens(&self) -> u32 {
        let msg_tokens: u32 = self.messages.iter().map(|m| m.estimate_tokens()).sum();
        let summary_tokens = self
            .summary
            .as_ref()
            .map(|s| (s.len() / 4) as u32)
            .unwrap_or(0);
        msg_tokens + summary_tokens
    }

    /// Clear all messages (keeps conversation ID and settings).
    pub fn clear(&mut self) {
        self.messages.clear();
        self.summary = None;
    }

    /// Set a summary of older truncated messages.
    pub fn set_summary(&mut self, summary: &str) {
        self.summary = Some(summary.to_string());
    }

    /// Get the current summary, if any.
    pub fn summary(&self) -> Option<&str> {
        self.summary.as_deref()
    }

    /// Save the conversation to disk (JSON).
    pub fn save(&self) -> Result<(), CrabInferError> {
        let path = self.persist_path.as_ref().ok_or(CrabInferError::StorageError {
            reason: "No persist path set".to_string(),
        })?;

        let json = serde_json::to_string_pretty(self).map_err(|e| CrabInferError::StorageError {
            reason: format!("Serialization failed: {}", e),
        })?;

        // Ensure parent directory exists
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

    /// Load a conversation from a JSON file.
    pub fn load(path: &str) -> Result<Self, CrabInferError> {
        let json = std::fs::read_to_string(path).map_err(|e| CrabInferError::StorageError {
            reason: format!("Failed to read file: {}", e),
        })?;

        let mut conv: Self =
            serde_json::from_str(&json).map_err(|e| CrabInferError::StorageError {
                reason: format!("Deserialization failed: {}", e),
            })?;

        conv.persist_path = Some(path.to_string());
        Ok(conv)
    }

    /// Enforce max_messages and token_budget limits by removing oldest messages.
    fn enforce_limits(&mut self) {
        // Enforce message count limit
        if self.max_messages > 0 && self.messages.len() > self.max_messages {
            let excess = self.messages.len() - self.max_messages;
            self.messages.drain(..excess);
        }

        // Enforce token budget
        if self.token_budget > 0 {
            while self.estimate_tokens() > self.token_budget && self.messages.len() > 1 {
                // Don't remove the last message (it's the most recent)
                // Skip system messages at the beginning
                let remove_idx = self
                    .messages
                    .iter()
                    .position(|m| m.role != "system")
                    .unwrap_or(0);
                self.messages.remove(remove_idx);
            }
        }
    }
}

/// Manages multiple conversation sessions.
pub struct ConversationStore {
    conversations: Vec<ConversationMemory>,
    storage_dir: Option<String>,
}

impl ConversationStore {
    /// Create a new conversation store.
    pub fn new() -> Self {
        Self {
            conversations: Vec::new(),
            storage_dir: None,
        }
    }

    /// Set the directory for persisting conversations.
    pub fn with_storage_dir(mut self, dir: &str) -> Self {
        self.storage_dir = Some(dir.to_string());
        self
    }

    /// Create a new conversation and add it to the store.
    pub fn create_conversation(&mut self, id: &str) -> &mut ConversationMemory {
        let mut conv = ConversationMemory::new(id);
        if let Some(ref dir) = self.storage_dir {
            let path = format!("{}/{}.json", dir, id);
            conv = conv.with_persist_path(&path);
        }
        self.conversations.push(conv);
        self.conversations.last_mut().unwrap()
    }

    /// Get a conversation by ID.
    pub fn get(&self, id: &str) -> Option<&ConversationMemory> {
        self.conversations.iter().find(|c| c.id() == id)
    }

    /// Get a mutable reference to a conversation by ID.
    pub fn get_mut(&mut self, id: &str) -> Option<&mut ConversationMemory> {
        self.conversations.iter_mut().find(|c| c.id() == id)
    }

    /// List all conversation IDs.
    pub fn list_ids(&self) -> Vec<String> {
        self.conversations.iter().map(|c| c.id().to_string()).collect()
    }

    /// Delete a conversation by ID.
    pub fn delete(&mut self, id: &str) -> bool {
        if let Some(pos) = self.conversations.iter().position(|c| c.id() == id) {
            let conv = self.conversations.remove(pos);
            // Delete the persisted file if it exists
            if let Some(ref path) = conv.persist_path {
                let _ = std::fs::remove_file(path);
            }
            true
        } else {
            false
        }
    }

    /// Save all conversations to disk.
    pub fn save_all(&self) -> Result<(), CrabInferError> {
        for conv in &self.conversations {
            if conv.persist_path.is_some() {
                conv.save()?;
            }
        }
        Ok(())
    }

    /// Load all conversations from the storage directory.
    pub fn load_all(&mut self) -> Result<(), CrabInferError> {
        let dir = self.storage_dir.as_ref().ok_or(CrabInferError::StorageError {
            reason: "No storage directory set".to_string(),
        })?;

        let entries = std::fs::read_dir(dir).map_err(|e| CrabInferError::StorageError {
            reason: format!("Failed to read directory: {}", e),
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| CrabInferError::StorageError {
                reason: format!("Failed to read entry: {}", e),
            })?;
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "json") {
                if let Ok(conv) = ConversationMemory::load(path.to_str().unwrap_or("")) {
                    self.conversations.push(conv);
                }
            }
        }

        Ok(())
    }

    /// Get the number of conversations.
    pub fn count(&self) -> usize {
        self.conversations.len()
    }
}

impl Default for ConversationMemory {
    fn default() -> Self {
        Self::new("default")
    }
}

impl Default for ConversationStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_conversation() {
        let conv = ConversationMemory::new("test-1");
        assert_eq!(conv.id(), "test-1");
        assert_eq!(conv.message_count(), 0);
    }

    #[test]
    fn test_add_messages() {
        let mut conv = ConversationMemory::new("test");
        conv.add_user_message("Hello");
        conv.add_assistant_message("Hi there!");
        assert_eq!(conv.message_count(), 2);

        let messages = conv.messages();
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content, "Hello");
        assert_eq!(messages[1].role, "assistant");
        assert_eq!(messages[1].content, "Hi there!");
    }

    #[test]
    fn test_max_messages_limit() {
        let mut conv = ConversationMemory::new("test").with_max_messages(3);
        conv.add_user_message("1");
        conv.add_assistant_message("2");
        conv.add_user_message("3");
        conv.add_assistant_message("4");

        assert_eq!(conv.message_count(), 3);
        // Oldest message should be removed
        let messages = conv.messages();
        assert_eq!(messages[0].content, "2");
    }

    #[test]
    fn test_recent_messages() {
        let mut conv = ConversationMemory::new("test");
        conv.add_user_message("1");
        conv.add_assistant_message("2");
        conv.add_user_message("3");

        let recent = conv.recent_messages(2);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].content, "2");
        assert_eq!(recent[1].content, "3");
    }

    #[test]
    fn test_summary_injection() {
        let mut conv = ConversationMemory::new("test");
        conv.set_summary("User asked about Rust earlier.");
        conv.add_user_message("Tell me more.");

        let messages = conv.messages();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert!(messages[0].content.contains("earlier conversation"));
        assert_eq!(messages[1].role, "user");
    }

    #[test]
    fn test_clear() {
        let mut conv = ConversationMemory::new("test");
        conv.add_user_message("Hello");
        conv.set_summary("Some context");
        conv.clear();

        assert_eq!(conv.message_count(), 0);
        assert!(conv.summary().is_none());
    }

    #[test]
    fn test_json_persistence() {
        let dir = std::env::temp_dir().join("crabinfer-test-conv");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test-conv.json");
        let path_str = path.to_str().unwrap();

        // Save
        let mut conv = ConversationMemory::new("persist-test").with_persist_path(path_str);
        conv.add_user_message("Hello");
        conv.add_assistant_message("World");
        conv.save().unwrap();

        // Load
        let loaded = ConversationMemory::load(path_str).unwrap();
        assert_eq!(loaded.id(), "persist-test");
        assert_eq!(loaded.message_count(), 2);

        // Cleanup
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_conversation_store() {
        let mut store = ConversationStore::new();
        store.create_conversation("conv-1");
        store.create_conversation("conv-2");

        assert_eq!(store.count(), 2);
        assert_eq!(store.list_ids(), vec!["conv-1", "conv-2"]);
        assert!(store.get("conv-1").is_some());
        assert!(store.get("missing").is_none());

        assert!(store.delete("conv-1"));
        assert_eq!(store.count(), 1);
        assert!(!store.delete("conv-1")); // Already deleted
    }

    #[test]
    fn test_token_budget() {
        let mut conv = ConversationMemory::new("test").with_token_budget(10);
        // Each message is ~5-10 tokens. Adding many should trigger trimming.
        for i in 0..20 {
            conv.add_message("user", &format!("This is message number {}", i));
        }
        // Should have trimmed to fit within ~10 tokens
        assert!(conv.estimate_tokens() <= 10 || conv.message_count() <= 2);
    }
}
