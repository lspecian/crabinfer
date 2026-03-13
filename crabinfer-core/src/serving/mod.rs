//! Serving infrastructure for PagedAttention-based continuous batching.
//!
//! This module implements the core components for a high-performance inference server:
//! - Block-based KV cache management with prefix caching
//! - Token-budget continuous batching scheduler
//! - Metal paged attention kernel dispatch
//! - Model runner trait for paged-attention-native model implementations

pub mod arena;
pub mod beam_search;
pub mod block;
pub mod block_pool;
pub mod cuda_graphs;
pub mod disaggregated;
pub mod engine_loop;
pub mod fp8;
pub mod gpu_memory;
pub mod hub_download;
pub mod kernels;
pub mod kv_cache;
pub mod lora;
pub mod models;
pub mod ngram_draft;
pub mod quantization;
pub mod safetensors_loader;
pub mod scheduler;
pub mod sequence;
pub mod speculative;
pub mod guided;
pub mod swap;
pub mod expert_parallel;
pub mod nccl;
pub mod pipeline_parallel;
pub mod tensor_parallel;
pub mod tokenizer_cache;
pub mod worker_pool;

#[cfg(test)]
mod hub_download_tests;
#[cfg(test)]
mod safetensors_loader_tests;
#[cfg(test)]
mod quantization_tests;
