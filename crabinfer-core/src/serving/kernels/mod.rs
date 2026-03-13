//! Kernel dispatch for paged attention operations.
//!
//! Provides a `KernelBackend` trait abstraction over GPU-specific
//! implementations (Metal, CUDA) and a CPU fallback. The engine loop
//! calls `detect_backend()` at startup to select the best available backend.
//!
//! Three core operations:
//! - `paged_attention`: batched attention over paged KV cache
//! - `reshape_and_cache`: write new K/V tokens into the paged cache
//! - `copy_blocks`: copy KV cache blocks for prefix sharing

pub mod backend;
pub mod cpu_backend;

#[cfg(feature = "metal")]
pub(crate) mod metal_dispatch;

#[cfg(feature = "metal")]
pub mod metal_backend;

#[cfg(feature = "cuda")]
pub mod cuda_backend;

// Re-exports
pub use backend::{KernelBackend, PagedAttentionConfig};
pub use cpu_backend::CpuBackend;

#[cfg(feature = "metal")]
pub use metal_backend::MetalBackend;
#[cfg(feature = "metal")]
pub use metal_dispatch::allocate_kv_caches;

#[cfg(feature = "cuda")]
pub use cuda_backend::CudaBackend;

/// Supported attention head sizes (used by Metal dispatch).
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

/// Detect and return the best available kernel backend for this platform.
///
/// If `device` is provided and is a CUDA device, the backend will share
/// that CUDA context so kernels operate on the same device as model tensors.
///
/// Selection order:
/// 1. CUDA (if `cuda` feature enabled and NVIDIA GPU available)
/// 2. Metal (if `metal` feature enabled — Apple Silicon)
/// 3. CPU (always available, used for CI and testing)
pub fn detect_backend(device: Option<&candle_core::Device>) -> Box<dyn KernelBackend> {
    #[cfg(feature = "cuda")]
    {
        match CudaBackend::new_with_device(device, 0) {
            Ok(backend) => {
                tracing::info!("Kernel backend: CUDA (device 0)");
                return Box::new(backend);
            }
            Err(e) => {
                tracing::warn!("CUDA backend unavailable: {e}, falling back");
            }
        }
    }

    // Silence unused variable warning on non-CUDA builds
    let _ = device;

    #[cfg(feature = "metal")]
    {
        tracing::info!("Kernel backend: Metal");
        return Box::new(MetalBackend::new());
    }

    tracing::info!("Kernel backend: CPU (fallback)");
    Box::new(CpuBackend::new())
}
