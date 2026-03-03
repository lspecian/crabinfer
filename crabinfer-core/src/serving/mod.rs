//! Serving infrastructure for PagedAttention-based continuous batching.
//!
//! This module implements the core components for a high-performance inference server:
//! - Block-based KV cache management with prefix caching
//! - Token-budget continuous batching scheduler
//! - Metal paged attention kernel dispatch
//! - Model runner trait for paged-attention-native model implementations

pub mod block;
pub mod block_pool;
pub mod engine_loop;
pub mod kernels;
pub mod kv_cache;
pub mod models;
pub mod scheduler;
pub mod sequence;
pub mod speculative;
