//! GPU memory profiling and automatic KV cache block calculation.
//!
//! Determines the optimal number of KV cache blocks based on available
//! device memory, model weight size, and a user-specified utilization ratio.
//! Also provides system memory queries for both macOS and Linux.

use super::kernels::BLOCK_SIZE;
use super::models::ModelConfig;

/// Result of GPU memory profiling at engine startup.
#[derive(Debug, Clone)]
pub struct GpuMemoryProfile {
    /// Total device memory in bytes (GPU VRAM or system RAM for CPU/unified).
    pub total_memory: usize,
    /// Estimated memory consumed by model weights in bytes.
    pub weights_memory: usize,
    /// Estimated memory for activations (intermediate tensors) in bytes.
    pub activation_memory: usize,
    /// Available memory for KV cache after subtracting weights + activations.
    pub available_for_kv_cache: usize,
    /// Number of KV cache blocks that fit in available memory.
    pub num_blocks: usize,
    /// Bytes consumed by a single KV cache block (across all layers).
    pub bytes_per_block: usize,
}

/// Calculate the number of bytes per KV cache block for the given model config.
///
/// Each block stores `BLOCK_SIZE` tokens of KV data across all layers:
/// - K cache: `[num_kv_heads, head_size/x, BLOCK_SIZE, x]` per layer
/// - V cache: `[num_kv_heads, head_size, BLOCK_SIZE]` per layer
pub fn kv_cache_block_bytes(
    model_config: &ModelConfig,
    dtype_bytes: usize,
) -> usize {
    let per_block_per_layer =
        2 * model_config.num_kv_heads * model_config.head_size * BLOCK_SIZE * dtype_bytes;
    per_block_per_layer * model_config.num_layers
}

/// Estimate model weight memory from the model configuration.
///
/// This is a rough estimate based on parameter counts. The actual memory
/// may differ for quantized models, but this gives a reasonable upper bound
/// for memory planning purposes.
fn estimate_weights_memory(model_config: &ModelConfig) -> usize {
    // Per transformer layer:
    //   QKV projection: hidden * (num_heads + 2*num_kv_heads) * head_size
    //   Output projection: hidden * hidden
    //   Gate + Up + Down MLP: 3 * hidden * intermediate
    //   RMSNorm: 2 * hidden (attention + post-attention)
    let h = model_config.hidden_size;
    let i = model_config.intermediate_size;
    let qkv_params = h * (model_config.num_heads + 2 * model_config.num_kv_heads)
        * model_config.head_size
        / h; // = (num_heads + 2*num_kv_heads) * head_size
    let qkv_total = h * qkv_params;
    let o_proj = h * h;
    let mlp = 3 * h * i;
    let norms = 2 * h;
    let per_layer = qkv_total + o_proj + mlp + norms;

    // Embedding + final norm + lm_head
    let embedding = model_config.vocab_size * h;
    let lm_head = model_config.vocab_size * h;
    let final_norm = h;

    let total_params = model_config.num_layers * per_layer + embedding + lm_head + final_norm;

    // Assume 2 bytes per parameter (F16/BF16 — quantized models use less,
    // but we err on the side of overestimating for safety margin).
    total_params * 2
}

/// Estimate activation memory (intermediate tensors during forward pass).
///
/// The main consumers are:
/// - Input embeddings: `max_batch_tokens * hidden_size * dtype_bytes`
/// - Attention: `max_batch_tokens * num_heads * max_seq_len * dtype_bytes`
///   (but paged attention avoids the full materialization)
/// - MLP intermediates: `max_batch_tokens * intermediate_size * dtype_bytes`
///
/// We use a conservative estimate assuming peak batch size.
fn estimate_activation_memory(
    model_config: &ModelConfig,
    max_num_batched_tokens: usize,
) -> usize {
    let dtype_bytes = 4; // F32 activations

    // Embedding output
    let embed = max_num_batched_tokens * model_config.hidden_size * dtype_bytes;

    // MLP intermediates (gate + up, both are intermediate_size)
    let mlp = max_num_batched_tokens * model_config.intermediate_size * 2 * dtype_bytes;

    // Attention QKV + output (per layer, but only one layer active at a time)
    let attn = max_num_batched_tokens * model_config.hidden_size * 4 * dtype_bytes;

    // Logits output
    let logits = max_num_batched_tokens * model_config.vocab_size * dtype_bytes;

    // Only one layer's activations are live at a time, plus embeddings + logits
    embed + mlp.max(attn) + logits
}

/// Profile GPU memory and calculate the optimal number of KV cache blocks.
///
/// # Arguments
/// - `model_config`: Model architecture parameters
/// - `device`: The device being used (for memory queries)
/// - `gpu_memory_utilization`: Fraction of total memory to use (0.0–1.0)
/// - `max_num_batched_tokens`: Maximum tokens per batch step
/// - `kv_dtype_bytes`: Bytes per KV cache element (4 for F32, 2 for F16)
pub fn profile_gpu_memory(
    model_config: &ModelConfig,
    device: &candle_core::Device,
    gpu_memory_utilization: f64,
    max_num_batched_tokens: usize,
    kv_dtype_bytes: usize,
) -> GpuMemoryProfile {
    let total_memory = query_device_memory(device);
    let weights_memory = estimate_weights_memory(model_config);
    let activation_memory = estimate_activation_memory(model_config, max_num_batched_tokens);

    let usable_memory = (total_memory as f64 * gpu_memory_utilization) as usize;
    let available_for_kv_cache = usable_memory.saturating_sub(weights_memory + activation_memory);

    let bytes_per_block = kv_cache_block_bytes(model_config, kv_dtype_bytes);
    let num_blocks = if bytes_per_block > 0 {
        available_for_kv_cache / bytes_per_block
    } else {
        0
    };

    // Clamp to a reasonable range
    let num_blocks = num_blocks.clamp(16, 65536);

    GpuMemoryProfile {
        total_memory,
        weights_memory,
        activation_memory,
        available_for_kv_cache,
        num_blocks,
        bytes_per_block,
    }
}

/// Query total device memory for the given device.
///
/// - For CUDA: queries GPU VRAM via cudarc/nvidia-smi
/// - For Metal/CPU: queries system physical RAM
fn query_device_memory(device: &candle_core::Device) -> usize {
    match device {
        #[cfg(feature = "cuda")]
        candle_core::Device::Cuda(_) => query_cuda_memory(),
        _ => query_system_memory(),
    }
}

/// Query total system physical memory (macOS + Linux).
fn query_system_memory() -> usize {
    #[cfg(target_os = "macos")]
    {
        use std::mem;
        let mut size: u64 = 0;
        let mut len = mem::size_of::<u64>();
        let mib = [libc::CTL_HW, libc::HW_MEMSIZE];
        let ret = unsafe {
            libc::sysctl(
                mib.as_ptr() as *mut _,
                2,
                &mut size as *mut u64 as *mut _,
                &mut len as *mut _,
                std::ptr::null_mut(),
                0,
            )
        };
        if ret == 0 {
            size as usize
        } else {
            0
        }
    }

    #[cfg(target_os = "linux")]
    {
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) } as usize;
        page_size * pages
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        0
    }
}

/// Live GPU memory information from the CUDA driver.
#[derive(Debug, Clone)]
pub struct GpuMemoryInfo {
    /// Total GPU VRAM in bytes.
    pub total_bytes: usize,
    /// Free (unallocated) GPU VRAM in bytes.
    pub free_bytes: usize,
    /// Used GPU VRAM in bytes (`total - free`).
    pub used_bytes: usize,
    /// Utilization ratio (0.0–1.0).
    pub utilization: f64,
}

/// Query current GPU memory usage via the CUDA driver API.
///
/// Returns `None` if CUDA is not available or the query fails.
#[cfg(feature = "cuda")]
pub fn query_gpu_memory_info() -> Option<GpuMemoryInfo> {
    use candle_core::cuda_backend::cudarc::driver::sys;

    // cuMemGetInfo requires an active CUDA context. We create a device
    // handle (which sets the context) and then query.
    let _device = candle_core::CudaDevice::new_with_stream(0).ok()?;

    let mut free: usize = 0;
    let mut total: usize = 0;
    let result = unsafe { sys::cuMemGetInfo_v2(&mut free as *mut _, &mut total as *mut _) };
    if result != sys::CUresult::CUDA_SUCCESS {
        return None;
    }

    let used = total.saturating_sub(free);
    let utilization = if total > 0 {
        used as f64 / total as f64
    } else {
        0.0
    };

    Some(GpuMemoryInfo {
        total_bytes: total,
        free_bytes: free,
        used_bytes: used,
        utilization,
    })
}

#[cfg(not(feature = "cuda"))]
pub fn query_gpu_memory_info() -> Option<GpuMemoryInfo> {
    None
}

/// Query CUDA GPU memory (total VRAM in bytes).
#[cfg(feature = "cuda")]
fn query_cuda_memory() -> usize {
    // Use the CUDA driver API directly for accurate total memory.
    if let Some(info) = query_gpu_memory_info() {
        return info.total_bytes;
    }
    // Fallback: 6GB (common for RTX 3060 etc.)
    6 * 1024 * 1024 * 1024
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_model_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 4096,
            intermediate_size: 11008,
            num_heads: 32,
            num_kv_heads: 8,
            num_layers: 32,
            head_size: 128,
            vocab_size: 32000,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_dim: 128,
            max_seq_len: 4096,
        }
    }

    #[test]
    fn test_kv_cache_block_bytes() {
        let config = test_model_config();
        let bytes = kv_cache_block_bytes(&config, 4); // F32
        // 2(K+V) * 8 kv_heads * 128 head_size * 16 block_size * 4 bytes * 32 layers
        let expected = 2 * 8 * 128 * 16 * 4 * 32;
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_kv_cache_block_bytes_f16() {
        let config = test_model_config();
        let bytes = kv_cache_block_bytes(&config, 2); // F16
        let expected = 2 * 8 * 128 * 16 * 2 * 32;
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_estimate_weights_memory() {
        let config = test_model_config();
        let mem = estimate_weights_memory(&config);
        // Llama 7B has ~7B params → ~14GB at 2 bytes each
        // Our estimate should be in the right ballpark (5-20 GB)
        assert!(mem > 5_000_000_000, "weights memory too low: {mem}");
        assert!(mem < 20_000_000_000, "weights memory too high: {mem}");
    }

    #[test]
    fn test_estimate_activation_memory() {
        let config = test_model_config();
        let mem = estimate_activation_memory(&config, 2048);
        // Should be a few hundred MB to a few GB
        assert!(mem > 100_000_000, "activation memory too low: {mem}");
        assert!(mem < 5_000_000_000, "activation memory too high: {mem}");
    }

    #[test]
    fn test_profile_gpu_memory() {
        let config = test_model_config();
        let device = candle_core::Device::Cpu;
        let profile = profile_gpu_memory(&config, &device, 0.90, 2048, 4);

        assert!(profile.num_blocks >= 16, "too few blocks: {}", profile.num_blocks);
        assert!(profile.bytes_per_block > 0);
        assert!(profile.weights_memory > 0);
        assert!(profile.activation_memory > 0);
    }

    #[test]
    fn test_query_system_memory() {
        let mem = query_system_memory();
        // Should be at least 1GB on any modern system running tests
        assert!(mem > 1_000_000_000, "system memory too low: {mem}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_query_gpu_memory_info() {
        if candle_core::Device::new_cuda(0).is_err() {
            eprintln!("SKIP: no CUDA device");
            return;
        }
        let info = query_gpu_memory_info().expect("GPU memory query failed");
        // RTX 3060 has 6GB, should be at least 4GB total
        assert!(info.total_bytes > 4_000_000_000, "GPU total too low: {}", info.total_bytes);
        // Some memory should be free
        assert!(info.free_bytes > 0, "GPU free should be > 0");
        // Used = total - free
        assert_eq!(info.used_bytes, info.total_bytes - info.free_bytes);
        // Utilization should be between 0 and 1
        assert!(info.utilization >= 0.0 && info.utilization <= 1.0,
            "utilization out of range: {}", info.utilization);
    }
}
