//! Metal kernel dispatch for paged attention operations.
//!
//! Compiles and caches Metal compute pipelines at first use.
//! Provides three main operations:
//! - `paged_attention`: batched attention over paged KV cache
//! - `reshape_and_cache`: write new K/V tokens into the paged cache
//! - `copy_blocks`: copy KV cache blocks for prefix sharing

#[cfg(feature = "metal")]
pub(crate) mod metal_dispatch;

#[cfg(feature = "metal")]
pub use metal_dispatch::*;

/// Supported attention head sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadSize {
    H64 = 64,
    H128 = 128,
}

impl HeadSize {
    pub fn from_usize(n: usize) -> Option<Self> {
        match n {
            64 => Some(Self::H64),
            128 => Some(Self::H128),
            _ => None,
        }
    }

    pub fn value(&self) -> usize {
        *self as usize
    }
}

/// KV cache block size.
pub const BLOCK_SIZE: usize = 16;

/// Partition size for V2 paged attention.
pub const PARTITION_SIZE: usize = 512;

/// Number of threads per threadgroup (fixed for all kernels).
pub const NUM_THREADS: usize = 256;
