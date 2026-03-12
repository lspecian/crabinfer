//! Metal backend wrapping the existing `metal_dispatch` module.
//!
//! Zero-overhead delegation — each method calls the corresponding
//! free function in `metal_dispatch.rs`.

use candle_core::{DType, Device, Result, Tensor};

use super::backend::{KernelBackend, PagedAttentionConfig};
use super::metal_dispatch;
use super::HeadSize;

/// Metal GPU backend for Apple Silicon.
pub struct MetalBackend;

impl MetalBackend {
    pub fn new() -> Self {
        Self
    }
}

impl KernelBackend for MetalBackend {
    fn name(&self) -> &'static str {
        "metal"
    }

    fn paged_attention(
        &self,
        output: &Tensor,
        query: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        block_tables: &Tensor,
        context_lens: &Tensor,
        config: &PagedAttentionConfig,
    ) -> Result<()> {
        let head_size = HeadSize::from_usize(config.head_size).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "unsupported head size {} for Metal paged attention (supported: 64, 128)",
                config.head_size
            ))
        })?;

        let metal_config = metal_dispatch::PagedAttentionConfig {
            head_size,
            num_kv_heads: config.num_kv_heads,
            scale: config.scale,
            max_context_len: config.max_context_len,
        };

        metal_dispatch::paged_attention(
            output,
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
            &metal_config,
        )
    }

    fn reshape_and_cache(
        &self,
        key: &Tensor,
        value: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        slot_mapping: &Tensor,
    ) -> Result<()> {
        metal_dispatch::reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
    }

    fn copy_blocks(
        &self,
        key_cache: &Tensor,
        value_cache: &Tensor,
        block_mapping: &Tensor,
        numel_per_block: usize,
    ) -> Result<()> {
        metal_dispatch::copy_blocks(key_cache, value_cache, block_mapping, numel_per_block)
    }

    fn allocate_kv_caches(
        &self,
        num_layers: usize,
        num_blocks: usize,
        num_kv_heads: usize,
        head_size: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        metal_dispatch::allocate_kv_caches(num_layers, num_blocks, num_kv_heads, head_size, dtype, device)
    }
}
