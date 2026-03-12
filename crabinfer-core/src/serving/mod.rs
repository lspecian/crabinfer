//! Serving infrastructure for PagedAttention-based continuous batching.
//!
//! This module implements the core components for a high-performance inference server:
//! - Block-based KV cache management with prefix caching
//! - Token-budget continuous batching scheduler
//! - Metal paged attention kernel dispatch
//! - Model runner trait for paged-attention-native model implementations

pub mod block;
pub mod block_pool;
pub mod cuda_graphs;
pub mod engine_loop;
pub mod gpu_memory;
pub mod kernels;
pub mod kv_cache;
pub mod models;
pub mod ngram_draft;
pub mod quantization;
pub mod safetensors_loader;
pub mod scheduler;
pub mod sequence;
pub mod speculative;
pub mod swap;
