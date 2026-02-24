//! Node.js bindings for A5 (Memory Layer) and A7 (System Prompt).
//!
//! Exposes ConversationMemory, MemoryStore, and SystemPrompt to JS.

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::error::to_napi_error;

// ─── SystemPrompt ────────────────────────────────────────────────────────────

/// Composable system prompt builder.
#[napi]
pub struct SystemPrompt {
    inner: crabinfer_core::prompt::SystemPrompt,
}

#[napi]
impl SystemPrompt {
    /// Create an empty system prompt.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: crabinfer_core::prompt::SystemPrompt::new(),
        }
    }

    /// Set the identity section.
    #[napi]
    pub fn identity(&mut self, identity: String) -> &Self {
        let inner = std::mem::take(&mut self.inner);
        self.inner = inner.identity(&identity);
        self
    }

    /// Add an instruction.
    #[napi]
    pub fn instruction(&mut self, instruction: String) -> &Self {
        let inner = std::mem::take(&mut self.inner);
        self.inner = inner.instruction(&instruction);
        self
    }

    /// Set the output format.
    #[napi]
    pub fn output_format(&mut self, format: String) -> &Self {
        let inner = std::mem::take(&mut self.inner);
        self.inner = inner.output_format(&format);
        self
    }

    /// Set the token budget.
    #[napi]
    pub fn token_budget(&mut self, budget: u32) -> &Self {
        let inner = std::mem::take(&mut self.inner);
        self.inner = inner.token_budget(budget);
        self
    }

    /// Render the system prompt.
    #[napi]
    pub fn build(&self) -> String {
        self.inner.build()
    }

    /// Render with injected facts and knowledge context.
    #[napi]
    pub fn build_with_context(&self, facts: Vec<String>, knowledge_chunks: Vec<String>) -> String {
        self.inner.build_with_context(&facts, &knowledge_chunks)
    }

    /// Estimate token count.
    #[napi]
    pub fn estimate_tokens(&self) -> u32 {
        self.inner.estimate_tokens()
    }

    /// Save to JSON string.
    #[napi]
    pub fn to_json(&self) -> Result<String> {
        self.inner.to_json().map_err(|e| {
            napi::Error::new(Status::GenericFailure, format!("JSON error: {}", e))
        })
    }

    /// Load from JSON string.
    #[napi(factory)]
    pub fn from_json(json: String) -> Result<Self> {
        let inner = crabinfer_core::prompt::SystemPrompt::from_json(&json).map_err(|e| {
            napi::Error::new(Status::GenericFailure, format!("JSON error: {}", e))
        })?;
        Ok(Self { inner })
    }

    /// Template: coding assistant.
    #[napi(factory)]
    pub fn coding_assistant() -> Self {
        Self {
            inner: crabinfer_core::prompt::SystemPrompt::coding_assistant(),
        }
    }

    /// Template: document Q&A.
    #[napi(factory)]
    pub fn document_qa() -> Self {
        Self {
            inner: crabinfer_core::prompt::SystemPrompt::document_qa(),
        }
    }

    /// Template: conversational.
    #[napi(factory)]
    pub fn conversational() -> Self {
        Self {
            inner: crabinfer_core::prompt::SystemPrompt::conversational(),
        }
    }
}

// ─── ConversationMemory ──────────────────────────────────────────────────────

/// Conversation memory for multi-turn chat history.
#[napi]
pub struct ConversationMemory {
    inner: crabinfer_core::conversation::ConversationMemory,
}

#[napi]
impl ConversationMemory {
    /// Create a new conversation with the given ID.
    #[napi(constructor)]
    pub fn new(id: String) -> Self {
        Self {
            inner: crabinfer_core::conversation::ConversationMemory::new(&id),
        }
    }

    /// Set the maximum number of messages.
    #[napi]
    pub fn with_max_messages(&mut self, max: u32) {
        let inner = std::mem::take(&mut self.inner);
        self.inner = inner.with_max_messages(max as usize);
    }

    /// Set the token budget.
    #[napi]
    pub fn with_token_budget(&mut self, budget: u32) {
        let inner = std::mem::take(&mut self.inner);
        self.inner = inner.with_token_budget(budget);
    }

    /// Set the file path for persistence.
    #[napi]
    pub fn with_persist_path(&mut self, path: String) {
        let inner = std::mem::take(&mut self.inner);
        self.inner = inner.with_persist_path(&path);
    }

    /// Get the conversation ID.
    #[napi(getter)]
    pub fn id(&self) -> String {
        self.inner.id().to_string()
    }

    /// Add a user message.
    #[napi]
    pub fn add_user_message(&mut self, content: String) {
        self.inner.add_user_message(&content);
    }

    /// Add an assistant message.
    #[napi]
    pub fn add_assistant_message(&mut self, content: String) {
        self.inner.add_assistant_message(&content);
    }

    /// Add a message with a specific role.
    #[napi]
    pub fn add_message(&mut self, role: String, content: String) {
        self.inner.add_message(&role, &content);
    }

    /// Get all messages as an array of {role, content}.
    #[napi]
    pub fn messages(&self) -> Vec<JsChatMessage> {
        self.inner
            .messages()
            .into_iter()
            .map(|m| JsChatMessage {
                role: m.role,
                content: m.content,
            })
            .collect()
    }

    /// Get the N most recent messages.
    #[napi]
    pub fn recent_messages(&self, count: u32) -> Vec<JsChatMessage> {
        self.inner
            .recent_messages(count as usize)
            .into_iter()
            .map(|m| JsChatMessage {
                role: m.role,
                content: m.content,
            })
            .collect()
    }

    /// Get the number of messages.
    #[napi(getter)]
    pub fn message_count(&self) -> u32 {
        self.inner.message_count() as u32
    }

    /// Estimate total token count.
    #[napi]
    pub fn estimate_tokens(&self) -> u32 {
        self.inner.estimate_tokens()
    }

    /// Clear all messages.
    #[napi]
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Set a summary of older messages.
    #[napi]
    pub fn set_summary(&mut self, summary: String) {
        self.inner.set_summary(&summary);
    }

    /// Save to disk.
    #[napi]
    pub fn save(&self) -> Result<()> {
        self.inner.save().map_err(to_napi_error)
    }

    /// Load from a JSON file.
    #[napi(factory)]
    pub fn load(path: String) -> Result<Self> {
        let inner = crabinfer_core::conversation::ConversationMemory::load(&path)
            .map_err(to_napi_error)?;
        Ok(Self { inner })
    }
}

/// A chat message (role + content) for JS.
#[napi(object)]
pub struct JsChatMessage {
    pub role: String,
    pub content: String,
}

// ─── MemoryStore ─────────────────────────────────────────────────────────────

/// Persistent store for user/context facts.
#[napi]
pub struct MemoryStore {
    inner: crabinfer_core::facts::MemoryStore,
}

#[napi]
impl MemoryStore {
    /// Create a new empty store.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            inner: crabinfer_core::facts::MemoryStore::new(),
        }
    }

    /// Set the file path for persistence.
    #[napi]
    pub fn with_persist_path(&mut self, path: String) {
        let inner = std::mem::take(&mut self.inner);
        self.inner = inner.with_persist_path(&path);
    }

    /// Add or update a fact.
    #[napi]
    pub fn add_fact(&mut self, key: String, value: String) {
        self.inner.add_fact(&key, &value);
    }

    /// Remove a fact. Returns true if it existed.
    #[napi]
    pub fn remove_fact(&mut self, key: String) -> bool {
        self.inner.remove_fact(&key)
    }

    /// Get the value of a fact.
    #[napi]
    pub fn get_value(&self, key: String) -> Option<String> {
        self.inner.get_value(&key).map(|s| s.to_string())
    }

    /// Check if a fact exists.
    #[napi]
    pub fn has_fact(&self, key: String) -> bool {
        self.inner.has_fact(&key)
    }

    /// Get the number of facts.
    #[napi(getter)]
    pub fn count(&self) -> u32 {
        self.inner.count() as u32
    }

    /// Get all facts as strings for prompt injection.
    #[napi]
    pub fn as_prompt_context(&self) -> Vec<String> {
        self.inner.as_prompt_context()
    }

    /// Clear all facts.
    #[napi]
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// Save to disk.
    #[napi]
    pub fn save(&self) -> Result<()> {
        self.inner.save().map_err(to_napi_error)
    }

    /// Load from a JSON file.
    #[napi(factory)]
    pub fn load(path: String) -> Result<Self> {
        let inner = crabinfer_core::facts::MemoryStore::load(&path).map_err(to_napi_error)?;
        Ok(Self { inner })
    }
}

