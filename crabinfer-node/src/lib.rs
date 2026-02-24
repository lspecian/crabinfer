//! CrabInfer Node.js binding via napi-rs.
//!
//! Provides the same API surface as the Swift SDK:
//! - CrabInferEngine: local on-device inference
//! - CrabInferProvider: unified local/cloud provider
//! - CrabInferRouter: smart routing between local, self-hosted, and cloud
//! - CrabInferVllm: vLLM-specific features
//! - ModelDownloadManager: HuggingFace model downloads
//! - TokenStream: async iterator for streaming tokens
//! - SystemPrompt: composable prompt builder (A7)
//! - ConversationMemory: multi-turn chat history (A5)
//! - MemoryStore: persistent user facts (A5)
//! - KnowledgeBase: RAG pipeline (A6)

pub mod enums;
pub mod error;
pub mod types;
pub mod functions;
pub mod engine;
pub mod stream;
pub mod provider;
pub mod router;
pub mod vllm;
pub mod download;
pub mod memory;
pub mod knowledge;
pub mod agent;
