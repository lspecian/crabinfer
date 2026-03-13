//! CUDA backend for paged attention operations on NVIDIA GPUs.
//!
//! Uses candle's re-exported `cudarc` (0.19) for GPU management and launches
//! custom CUDA kernels compiled from embedded source at first use.
//! All cudarc types come from `candle_core::cuda_backend::cudarc` to avoid
//! version conflicts with candle's own cudarc dependency.

use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::{DType, Device, Result, Tensor};
use half::f16;

use super::backend::{KernelBackend, PagedAttentionConfig};
use super::BLOCK_SIZE;

// ─── Embedded CUDA kernel source ──────────────────────────────────────────

const PAGED_ATTENTION_CU: &str = include_str!("paged_attention.cu");

/// Module name used for caching the compiled PTX in candle's CudaDevice.
const MODULE_NAME: &str = "crabinfer_paged_attn";

// ─── CudaBackend ──────────────────────────────────────────────────────────

/// NVIDIA CUDA backend implementing `KernelBackend`.
///
/// Uses candle's `CudaDevice` for kernel compilation, loading, and launch.
/// CUDA source is compiled to PTX at construction time via NVRTC.
pub struct CudaBackend {
    device: candle_core::CudaDevice,
    /// Pre-compiled PTX from the paged attention CUDA source.
    compiled_ptx: String,
}

impl CudaBackend {
    /// Create a new CUDA backend on the given device ordinal (usually 0).
    ///
    /// Compiles the embedded CUDA kernels to PTX using NVRTC at construction time.
    pub fn new(ordinal: usize) -> std::result::Result<Self, String> {
        let device = candle_core::CudaDevice::new_with_stream(ordinal)
            .map_err(|e| format!("failed to open CUDA device {ordinal}: {e}"))?;

        // Compile CUDA source to PTX using NVRTC.
        // We need to pass the CUDA include path for headers like cuda_fp16.h.
        let cuda_include = find_cuda_include_dir();
        let include_flag = format!("-I{cuda_include}");
        let opts = candle_core::cuda_backend::cudarc::nvrtc::CompileOptions {
            use_fast_math: Some(true),
            options: vec![include_flag],
            ..Default::default()
        };
        let ptx = candle_core::cuda_backend::cudarc::nvrtc::safe::compile_ptx_with_opts(
            PAGED_ATTENTION_CU,
            opts,
        )
        .map_err(|e| format!("failed to compile CUDA kernels: {e}"))?;

        // Extract compiled PTX text so it can be loaded by candle's module system.
        // `to_src()` converts the NVRTC-compiled image into a PTX source string.
        let compiled_ptx = ptx.to_src();

        Ok(Self {
            device,
            compiled_ptx,
        })
    }

    /// Get a CudaFunc for the named kernel, loading from pre-compiled PTX.
    fn get_func(&self, func_name: &str) -> Result<candle_core::cuda_backend::CudaFunc> {
        self.device
            .get_or_load_custom_func(func_name, MODULE_NAME, &self.compiled_ptx)
    }
}

/// Find the CUDA toolkit include directory for NVRTC compilation.
///
/// Searches: `$CUDA_HOME/include`, `/usr/local/cuda/include`, and
/// versioned paths like `/usr/local/cuda-12.0/include`.
fn find_cuda_include_dir() -> String {
    // 1. CUDA_HOME environment variable
    if let Ok(home) = std::env::var("CUDA_HOME") {
        let p = format!("{home}/include");
        if std::path::Path::new(&p).join("cuda_fp16.h").exists() {
            return p;
        }
    }
    // 2. Common paths
    for candidate in &[
        "/usr/local/cuda/include",
        "/usr/local/cuda-12.0/include",
        "/usr/local/cuda-12/include",
        "/usr/local/cuda-12.2/include",
        "/usr/local/cuda-11.8/include",
        "/opt/cuda/include",
    ] {
        if std::path::Path::new(candidate)
            .join("cuda_fp16.h")
            .exists()
        {
            return candidate.to_string();
        }
    }
    // Fallback — NVRTC may find it via its own search
    "/usr/local/cuda/include".to_string()
}

/// Helper to extract CudaStorage from a tensor's storage guard.
fn as_cuda(
    storage: &candle_core::Storage,
) -> Result<&candle_core::cuda_backend::CudaStorage> {
    match storage {
        candle_core::Storage::Cuda(s) => Ok(s),
        _ => candle_core::bail!("expected CUDA tensor"),
    }
}

/// Launch paged_attention kernel for a specific float type.
///
/// Generic over `T` (f32 or f16) to handle typed CUDA slice access.
fn launch_paged_attention<T: candle_core::cuda_backend::cudarc::driver::DeviceRepr + candle_core::cuda_backend::CudaDType>(
    func: candle_core::cuda_backend::CudaFunc,
    cfg: LaunchConfig,
    output: &Tensor,
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    num_kv_heads: i32,
    scale: f32,
    max_blocks: i32,
    q_stride: i32,
    kv_block_stride: i32,
    kv_head_stride: i32,
    head_size: i32,
    block_size: i32,
) -> Result<()> {
    let (o_st, o_lay) = output.storage_and_layout();
    let (q_st, q_lay) = query.storage_and_layout();
    let (kc_st, kc_lay) = key_cache.storage_and_layout();
    let (vc_st, vc_lay) = value_cache.storage_and_layout();
    let (bt_st, bt_lay) = block_tables.storage_and_layout();
    let (cl_st, cl_lay) = context_lens.storage_and_layout();

    let o_slice = as_cuda(&o_st)?.as_cuda_slice::<T>()?;
    let q_slice = as_cuda(&q_st)?.as_cuda_slice::<T>()?;
    let kc_slice = as_cuda(&kc_st)?.as_cuda_slice::<T>()?;
    let vc_slice = as_cuda(&vc_st)?.as_cuda_slice::<T>()?;
    let bt_slice = as_cuda(&bt_st)?.as_cuda_slice::<i32>()?;
    let cl_slice = as_cuda(&cl_st)?.as_cuda_slice::<i32>()?;

    let o_view = o_slice.slice(o_lay.start_offset()..);
    let q_view = q_slice.slice(q_lay.start_offset()..);
    let kc_view = kc_slice.slice(kc_lay.start_offset()..);
    let vc_view = vc_slice.slice(vc_lay.start_offset()..);
    let bt_view = bt_slice.slice(bt_lay.start_offset()..);
    let cl_view = cl_slice.slice(cl_lay.start_offset()..);

    let mut builder = func.builder();
    builder.arg(&o_view);
    builder.arg(&q_view);
    builder.arg(&kc_view);
    builder.arg(&vc_view);
    builder.arg(&bt_view);
    builder.arg(&cl_view);
    builder.arg(&num_kv_heads);
    builder.arg(&scale);
    builder.arg(&max_blocks);
    builder.arg(&q_stride);
    builder.arg(&kv_block_stride);
    builder.arg(&kv_head_stride);
    builder.arg(&head_size);
    builder.arg(&block_size);

    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| candle_core::Error::Msg(format!("CUDA launch paged_attention: {e}")))?;
    }
    Ok(())
}

/// Launch reshape_and_cache kernel for a specific float type.
fn launch_reshape_and_cache<T: candle_core::cuda_backend::cudarc::driver::DeviceRepr + candle_core::cuda_backend::CudaDType>(
    func: candle_core::cuda_backend::CudaFunc,
    cfg: LaunchConfig,
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
    key_stride: i32,
    value_stride: i32,
    num_heads: i32,
    head_size: i32,
    block_size: i32,
    x: i32,
) -> Result<()> {
    let (k_st, k_lay) = key.storage_and_layout();
    let (v_st, v_lay) = value.storage_and_layout();
    let (kc_st, kc_lay) = key_cache.storage_and_layout();
    let (vc_st, vc_lay) = value_cache.storage_and_layout();
    let (sm_st, sm_lay) = slot_mapping.storage_and_layout();

    let k_slice = as_cuda(&k_st)?.as_cuda_slice::<T>()?;
    let v_slice = as_cuda(&v_st)?.as_cuda_slice::<T>()?;
    let kc_slice = as_cuda(&kc_st)?.as_cuda_slice::<T>()?;
    let vc_slice = as_cuda(&vc_st)?.as_cuda_slice::<T>()?;
    let sm_slice = as_cuda(&sm_st)?.as_cuda_slice::<i32>()?;

    let k_view = k_slice.slice(k_lay.start_offset()..);
    let v_view = v_slice.slice(v_lay.start_offset()..);
    let kc_view = kc_slice.slice(kc_lay.start_offset()..);
    let vc_view = vc_slice.slice(vc_lay.start_offset()..);
    let sm_view = sm_slice.slice(sm_lay.start_offset()..);

    let mut builder = func.builder();
    builder.arg(&k_view);
    builder.arg(&v_view);
    builder.arg(&kc_view);
    builder.arg(&vc_view);
    builder.arg(&sm_view);
    builder.arg(&key_stride);
    builder.arg(&value_stride);
    builder.arg(&num_heads);
    builder.arg(&head_size);
    builder.arg(&block_size);
    builder.arg(&x);

    unsafe {
        builder.launch(cfg).map_err(|e| {
            candle_core::Error::Msg(format!("CUDA launch reshape_and_cache: {e}"))
        })?;
    }
    Ok(())
}

/// Launch copy_blocks kernel for a specific float type.
fn launch_copy_blocks<T: candle_core::cuda_backend::cudarc::driver::DeviceRepr + candle_core::cuda_backend::CudaDType>(
    func: candle_core::cuda_backend::CudaFunc,
    cfg: LaunchConfig,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_mapping: &Tensor,
    numel: i32,
) -> Result<()> {
    let (kc_st, kc_lay) = key_cache.storage_and_layout();
    let (vc_st, vc_lay) = value_cache.storage_and_layout();
    let (bm_st, bm_lay) = block_mapping.storage_and_layout();

    let kc_slice = as_cuda(&kc_st)?.as_cuda_slice::<T>()?;
    let vc_slice = as_cuda(&vc_st)?.as_cuda_slice::<T>()?;
    let bm_slice = as_cuda(&bm_st)?.as_cuda_slice::<i32>()?;

    let kc_view = kc_slice.slice(kc_lay.start_offset()..);
    let vc_view = vc_slice.slice(vc_lay.start_offset()..);
    let bm_view = bm_slice.slice(bm_lay.start_offset()..);

    let mut builder = func.builder();
    builder.arg(&kc_view);
    builder.arg(&vc_view);
    builder.arg(&bm_view);
    builder.arg(&numel);

    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| candle_core::Error::Msg(format!("CUDA launch copy_blocks: {e}")))?;
    }
    Ok(())
}

impl KernelBackend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
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
        let num_seqs = query.dims()[0];
        let num_heads = query.dims()[1];
        let head_size = config.head_size;
        let num_kv_heads = config.num_kv_heads;
        let max_blocks_per_seq = block_tables.dims()[1];

        let func_name = match query.dtype() {
            DType::F32 => "paged_attention_v1_f32",
            DType::F16 => "paged_attention_v1_f16",
            d => candle_core::bail!("paged_attention: unsupported dtype {d:?}"),
        };

        let func = self.get_func(func_name)?;

        let cfg = LaunchConfig {
            grid_dim: (num_heads as u32, num_seqs as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        let arg_num_kv_heads = num_kv_heads as i32;
        let arg_scale = config.scale;
        let arg_max_blocks = max_blocks_per_seq as i32;
        let arg_q_stride = (num_heads * head_size) as i32;
        let arg_kv_block_stride = (num_kv_heads * head_size * BLOCK_SIZE) as i32;
        let arg_kv_head_stride = (head_size * BLOCK_SIZE) as i32;
        let arg_head_size = head_size as i32;
        let arg_block_size = BLOCK_SIZE as i32;

        match query.dtype() {
            DType::F32 => launch_paged_attention::<f32>(
                func, cfg, output, query, key_cache, value_cache, block_tables, context_lens,
                arg_num_kv_heads, arg_scale, arg_max_blocks, arg_q_stride,
                arg_kv_block_stride, arg_kv_head_stride, arg_head_size, arg_block_size,
            ),
            DType::F16 => launch_paged_attention::<f16>(
                func, cfg, output, query, key_cache, value_cache, block_tables, context_lens,
                arg_num_kv_heads, arg_scale, arg_max_blocks, arg_q_stride,
                arg_kv_block_stride, arg_kv_head_stride, arg_head_size, arg_block_size,
            ),
            d => candle_core::bail!("paged_attention: unsupported dtype {d:?}"),
        }
    }

    fn reshape_and_cache(
        &self,
        key: &Tensor,
        value: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        slot_mapping: &Tensor,
    ) -> Result<()> {
        let dims = key.dims();
        let num_tokens = dims[0];
        let num_heads = dims[1];
        let head_size = dims[2];
        let dtype_bytes = key.dtype().size_in_bytes();

        let func_name = match key.dtype() {
            DType::F32 => "reshape_and_cache_f32",
            DType::F16 => "reshape_and_cache_f16",
            d => candle_core::bail!("reshape_and_cache: unsupported dtype {d:?}"),
        };

        let func = self.get_func(func_name)?;

        let threads_per_block = (num_heads * head_size).min(512) as u32;
        let cfg = LaunchConfig {
            grid_dim: (num_tokens as u32, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        let arg_key_stride = (num_heads * head_size) as i32;
        let arg_value_stride = arg_key_stride;
        let arg_num_heads = num_heads as i32;
        let arg_head_size = head_size as i32;
        let arg_block_size = BLOCK_SIZE as i32;
        let arg_x = (16 / dtype_bytes) as i32;

        match key.dtype() {
            DType::F32 => launch_reshape_and_cache::<f32>(
                func, cfg, key, value, key_cache, value_cache, slot_mapping,
                arg_key_stride, arg_value_stride, arg_num_heads, arg_head_size,
                arg_block_size, arg_x,
            ),
            DType::F16 => launch_reshape_and_cache::<f16>(
                func, cfg, key, value, key_cache, value_cache, slot_mapping,
                arg_key_stride, arg_value_stride, arg_num_heads, arg_head_size,
                arg_block_size, arg_x,
            ),
            d => candle_core::bail!("reshape_and_cache: unsupported dtype {d:?}"),
        }
    }

    fn copy_blocks(
        &self,
        key_cache: &Tensor,
        value_cache: &Tensor,
        block_mapping: &Tensor,
        numel_per_block: usize,
    ) -> Result<()> {
        let num_pairs = block_mapping.dims()[0] / 2;
        if num_pairs == 0 {
            return Ok(());
        }

        let func_name = match key_cache.dtype() {
            DType::F32 => "copy_blocks_f32",
            DType::F16 => "copy_blocks_f16",
            d => candle_core::bail!("copy_blocks: unsupported dtype {d:?}"),
        };

        let func = self.get_func(func_name)?;

        let threads_per_block = numel_per_block.min(1024) as u32;
        let cfg = LaunchConfig {
            grid_dim: (num_pairs as u32, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        let arg_numel = numel_per_block as i32;

        match key_cache.dtype() {
            DType::F32 => launch_copy_blocks::<f32>(
                func, cfg, key_cache, value_cache, block_mapping, arg_numel,
            ),
            DType::F16 => launch_copy_blocks::<f16>(
                func, cfg, key_cache, value_cache, block_mapping, arg_numel,
            ),
            d => candle_core::bail!("copy_blocks: unsupported dtype {d:?}"),
        }
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
        let x = 16 / dtype.size_in_bytes();
        let mut key_caches = Vec::with_capacity(num_layers);
        let mut value_caches = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            let key_cache = Tensor::zeros(
                (num_blocks, num_kv_heads, head_size / x, BLOCK_SIZE, x),
                dtype,
                device,
            )?;
            let value_cache = Tensor::zeros(
                (num_blocks, num_kv_heads, head_size, BLOCK_SIZE),
                dtype,
                device,
            )?;
            key_caches.push(key_cache);
            value_caches.push(value_cache);
        }

        Ok((key_caches, value_caches))
    }

    fn marlin_gemm(
        &self,
        input: &Tensor,
        qweight_marlin: &Tensor,
        scales: &Tensor,
        qzeros: &Tensor,
        group_size: usize,
    ) -> Result<Tensor> {
        self.marlin_gemm_impl(input, qweight_marlin, scales, qzeros, group_size)
    }

    fn fused_silu_mul(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        let total_elements = gate.elem_count();
        let output = Tensor::zeros(gate.shape(), gate.dtype(), gate.device())?;

        let func_name = match gate.dtype() {
            DType::F32 => "fused_silu_mul_f32",
            DType::F16 => "fused_silu_mul_f16",
            d => candle_core::bail!("fused_silu_mul: unsupported dtype {d:?}"),
        };

        let func = self.get_func(func_name)?;

        let threads = 256u32;
        let blocks = ((total_elements as u32) + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        {
            let (o_st, _) = output.storage_and_layout();
            let (g_st, _) = gate.storage_and_layout();
            let (u_st, _) = up.storage_and_layout();
            let o_cuda = as_cuda(&o_st)?;
            let g_cuda = as_cuda(&g_st)?;
            let u_cuda = as_cuda(&u_st)?;
            let arg_total = total_elements as i32;

            match gate.dtype() {
                DType::F32 => {
                    let mut builder = func.builder();
                    builder.arg(o_cuda.as_cuda_slice::<f32>()?);
                    builder.arg(g_cuda.as_cuda_slice::<f32>()?);
                    builder.arg(u_cuda.as_cuda_slice::<f32>()?);
                    builder.arg(&arg_total);
                    unsafe { builder.launch(cfg) }
                        .map_err(|e| candle_core::Error::Msg(format!("fused_silu_mul_f32 launch: {e}")))?;
                }
                DType::F16 => {
                    let mut builder = func.builder();
                    builder.arg(o_cuda.as_cuda_slice::<f16>()?);
                    builder.arg(g_cuda.as_cuda_slice::<f16>()?);
                    builder.arg(u_cuda.as_cuda_slice::<f16>()?);
                    builder.arg(&arg_total);
                    unsafe { builder.launch(cfg) }
                        .map_err(|e| candle_core::Error::Msg(format!("fused_silu_mul_f16 launch: {e}")))?;
                }
                _ => unreachable!(),
            }
        }

        Ok(output)
    }

    fn fused_rope(
        &self,
        x: &Tensor,
        positions: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        num_heads: usize,
        head_size: usize,
        rope_dim: usize,
    ) -> Result<Tensor> {
        let total_tokens = x.dims()[0];
        let half_rope = rope_dim / 2;

        // We need contiguous data to write in-place. Clone to get an owned copy.
        let output = x.contiguous()?.copy()?;

        let func_name = match x.dtype() {
            DType::F32 => "fused_rope_f32",
            DType::F16 => "fused_rope_f16",
            d => candle_core::bail!("fused_rope: unsupported dtype {d:?}"),
        };

        let func = self.get_func(func_name)?;

        // Positions should be U32 — the CUDA kernel accepts unsigned int*.
        let positions_u32 = if positions.dtype() != candle_core::DType::U32 {
            positions.to_dtype(candle_core::DType::U32)?
        } else {
            positions.contiguous()?
        };

        let threads_x = half_rope.min(256) as u32;
        let blocks_x = ((half_rope as u32) + threads_x - 1) / threads_x;
        let cfg = LaunchConfig {
            grid_dim: (blocks_x, num_heads as u32, total_tokens as u32),
            block_dim: (threads_x, 1, 1),
            shared_mem_bytes: 0,
        };

        // Get raw CUDA slices
        {
            let (o_st, _) = output.storage_and_layout();
            let (p_st, _) = positions_u32.storage_and_layout();
            let (c_st, _) = cos.storage_and_layout();
            let (s_st, _) = sin.storage_and_layout();

            let o_cuda = as_cuda(&o_st)?;
            let p_cuda = as_cuda(&p_st)?;
            let c_cuda = as_cuda(&c_st)?;
            let s_cuda = as_cuda(&s_st)?;

            let pos_slice = p_cuda.as_cuda_slice::<u32>()?;
            let cos_slice = c_cuda.as_cuda_slice::<f32>()?;
            let sin_slice = s_cuda.as_cuda_slice::<f32>()?;

            let arg_num_heads = num_heads as i32;
            let arg_head_size = head_size as i32;
            let arg_rope_dim = rope_dim as i32;

            match x.dtype() {
                DType::F32 => {
                    let x_slice = o_cuda.as_cuda_slice::<f32>()?;
                    let mut builder = func.builder();
                    builder.arg(x_slice);
                    builder.arg(pos_slice);
                    builder.arg(cos_slice);
                    builder.arg(sin_slice);
                    builder.arg(&arg_num_heads);
                    builder.arg(&arg_head_size);
                    builder.arg(&arg_rope_dim);
                    unsafe { builder.launch(cfg) }
                        .map_err(|e| candle_core::Error::Msg(format!("fused_rope_f32 launch: {e}")))?;
                }
                DType::F16 => {
                    let x_slice = o_cuda.as_cuda_slice::<f16>()?;
                    let mut builder = func.builder();
                    builder.arg(x_slice);
                    builder.arg(pos_slice);
                    builder.arg(cos_slice);
                    builder.arg(sin_slice);
                    builder.arg(&arg_num_heads);
                    builder.arg(&arg_head_size);
                    builder.arg(&arg_rope_dim);
                    unsafe { builder.launch(cfg) }
                        .map_err(|e| candle_core::Error::Msg(format!("fused_rope_f16 launch: {e}")))?;
                }
                _ => unreachable!(),
            }
        }

        Ok(output)
    }

    fn fused_rmsnorm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let dims = x.dims();
        if dims.len() != 2 {
            candle_core::bail!("fused_rmsnorm: expected 2D input, got {:?}", dims);
        }
        let num_rows = dims[0];
        let hidden_size = dims[1];

        let output = Tensor::zeros(x.shape(), x.dtype(), x.device())?;

        let func_name = match x.dtype() {
            DType::F32 => "fused_rmsnorm_f32",
            DType::F16 => "fused_rmsnorm_f16",
            d => candle_core::bail!("fused_rmsnorm: unsupported dtype {d:?}"),
        };

        let func = self.get_func(func_name)?;

        // One block per row, threads cooperate on the reduction.
        // Use min(hidden_size, 1024) threads, rounded to next power of 2 for tree reduction.
        let threads = (hidden_size as u32).min(1024).next_power_of_two();
        let cfg = LaunchConfig {
            grid_dim: (num_rows as u32, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: threads * 4, // float per thread for reduction
        };

        // Weight must be F32 for the kernel
        let weight_f32 = weight.to_dtype(DType::F32)?;

        {
            let (o_st, _) = output.storage_and_layout();
            let (x_st, _) = x.storage_and_layout();
            let (w_st, _) = weight_f32.storage_and_layout();
            let o_cuda = as_cuda(&o_st)?;
            let x_cuda = as_cuda(&x_st)?;
            let w_cuda = as_cuda(&w_st)?;
            let arg_hidden = hidden_size as i32;
            let arg_eps = eps;

            match x.dtype() {
                DType::F32 => {
                    let mut builder = func.builder();
                    builder.arg(o_cuda.as_cuda_slice::<f32>()?);
                    builder.arg(x_cuda.as_cuda_slice::<f32>()?);
                    builder.arg(w_cuda.as_cuda_slice::<f32>()?);
                    builder.arg(&arg_hidden);
                    builder.arg(&arg_eps);
                    unsafe { builder.launch(cfg) }
                        .map_err(|e| candle_core::Error::Msg(format!("fused_rmsnorm_f32 launch: {e}")))?;
                }
                DType::F16 => {
                    let mut builder = func.builder();
                    builder.arg(o_cuda.as_cuda_slice::<f16>()?);
                    builder.arg(x_cuda.as_cuda_slice::<f16>()?);
                    builder.arg(w_cuda.as_cuda_slice::<f32>()?);
                    builder.arg(&arg_hidden);
                    builder.arg(&arg_eps);
                    unsafe { builder.launch(cfg) }
                        .map_err(|e| candle_core::Error::Msg(format!("fused_rmsnorm_f16 launch: {e}")))?;
                }
                _ => unreachable!(),
            }
        }

        Ok(output)
    }

    fn fused_add_rmsnorm(
        &self,
        x: &Tensor,
        residual: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor)> {
        let dims = x.dims();
        if dims.len() != 2 {
            candle_core::bail!("fused_add_rmsnorm: expected 2D input, got {:?}", dims);
        }
        let num_rows = dims[0];
        let hidden_size = dims[1];

        // x_buf will be modified in-place: x_buf = x + residual
        // output gets the normalized result
        let x_buf = x.copy()?;
        let output = Tensor::zeros(x.shape(), x.dtype(), x.device())?;

        let func_name = match x.dtype() {
            DType::F32 => "fused_add_rmsnorm_f32",
            DType::F16 => "fused_add_rmsnorm_f16",
            d => candle_core::bail!("fused_add_rmsnorm: unsupported dtype {d:?}"),
        };

        let func = self.get_func(func_name)?;

        let threads = (hidden_size as u32).min(1024).next_power_of_two();
        let cfg = LaunchConfig {
            grid_dim: (num_rows as u32, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: threads * 4,
        };

        let weight_f32 = weight.to_dtype(DType::F32)?;

        {
            let (o_st, _) = output.storage_and_layout();
            let (xb_st, _) = x_buf.storage_and_layout();
            let (r_st, _) = residual.storage_and_layout();
            let (w_st, _) = weight_f32.storage_and_layout();
            let o_cuda = as_cuda(&o_st)?;
            let xb_cuda = as_cuda(&xb_st)?;
            let r_cuda = as_cuda(&r_st)?;
            let w_cuda = as_cuda(&w_st)?;
            let arg_hidden = hidden_size as i32;
            let arg_eps = eps;

            match x.dtype() {
                DType::F32 => {
                    let mut builder = func.builder();
                    builder.arg(o_cuda.as_cuda_slice::<f32>()?);
                    builder.arg(xb_cuda.as_cuda_slice::<f32>()?);
                    builder.arg(r_cuda.as_cuda_slice::<f32>()?);
                    builder.arg(w_cuda.as_cuda_slice::<f32>()?);
                    builder.arg(&arg_hidden);
                    builder.arg(&arg_eps);
                    unsafe { builder.launch(cfg) }
                        .map_err(|e| candle_core::Error::Msg(format!("fused_add_rmsnorm_f32 launch: {e}")))?;
                }
                DType::F16 => {
                    let mut builder = func.builder();
                    builder.arg(o_cuda.as_cuda_slice::<f16>()?);
                    builder.arg(xb_cuda.as_cuda_slice::<f16>()?);
                    builder.arg(r_cuda.as_cuda_slice::<f16>()?);
                    builder.arg(w_cuda.as_cuda_slice::<f32>()?);
                    builder.arg(&arg_hidden);
                    builder.arg(&arg_eps);
                    unsafe { builder.launch(cfg) }
                        .map_err(|e| candle_core::Error::Msg(format!("fused_add_rmsnorm_f16 launch: {e}")))?;
                }
                _ => unreachable!(),
            }
        }

        Ok((output, x_buf))
    }

    fn fused_layernorm_linear(
        &self,
        x: &Tensor,
        norm_weight: &Tensor,
        eps: f32,
        linear_weight: &Tensor,
    ) -> Result<Tensor> {
        let dims = x.dims();
        if dims.len() != 2 {
            candle_core::bail!("fused_layernorm_linear: expected 2D input, got {:?}", dims);
        }
        let num_rows = dims[0];
        let hidden_size = dims[1];

        let lw_dims = linear_weight.dims();
        if lw_dims.len() != 2 || lw_dims[1] != hidden_size {
            candle_core::bail!(
                "fused_layernorm_linear: linear_weight shape mismatch: {:?} vs hidden_size={}",
                lw_dims,
                hidden_size
            );
        }
        let out_features = lw_dims[0];

        // For large hidden_size or non-F16 weights, fall back to unfused path.
        // The fused kernel targets the common case of moderate hidden_size.
        if hidden_size > 256 * 4 || !matches!(x.dtype(), DType::F32 | DType::F16) {
            let normed = candle_nn::ops::rms_norm(x, norm_weight, eps)?;
            return normed.matmul(&linear_weight.t()?);
        }

        let output = Tensor::zeros(&[num_rows, out_features], x.dtype(), x.device())?;

        let func_name = match x.dtype() {
            DType::F32 => "fused_layernorm_linear_f32",
            DType::F16 => "fused_layernorm_linear_f16",
            d => candle_core::bail!("fused_layernorm_linear: unsupported dtype {d:?}"),
        };

        let func = self.get_func(func_name)?;

        let threads = 256u32;
        let cfg = LaunchConfig {
            grid_dim: (num_rows as u32, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: threads * 4,
        };

        // Norm weight must be F32 for the kernel
        let weight_f32 = norm_weight.to_dtype(DType::F32)?;
        // Linear weight must match x's dtype
        let lw = linear_weight.to_dtype(x.dtype())?;

        {
            let (o_st, _) = output.storage_and_layout();
            let (x_st, _) = x.storage_and_layout();
            let (nw_st, _) = weight_f32.storage_and_layout();
            let (lw_st, _) = lw.storage_and_layout();
            let o_cuda = as_cuda(&o_st)?;
            let x_cuda = as_cuda(&x_st)?;
            let nw_cuda = as_cuda(&nw_st)?;
            let lw_cuda = as_cuda(&lw_st)?;
            let arg_hidden = hidden_size as i32;
            let arg_out = out_features as i32;
            let arg_eps = eps;

            match x.dtype() {
                DType::F32 => {
                    let mut builder = func.builder();
                    builder.arg(o_cuda.as_cuda_slice::<f32>()?);
                    builder.arg(x_cuda.as_cuda_slice::<f32>()?);
                    builder.arg(nw_cuda.as_cuda_slice::<f32>()?);
                    builder.arg(lw_cuda.as_cuda_slice::<f32>()?);
                    builder.arg(&arg_hidden);
                    builder.arg(&arg_out);
                    builder.arg(&arg_eps);
                    unsafe { builder.launch(cfg) }
                        .map_err(|e| candle_core::Error::Msg(format!("fused_layernorm_linear_f32 launch: {e}")))?;
                }
                DType::F16 => {
                    let mut builder = func.builder();
                    builder.arg(o_cuda.as_cuda_slice::<f16>()?);
                    builder.arg(x_cuda.as_cuda_slice::<f16>()?);
                    builder.arg(nw_cuda.as_cuda_slice::<f32>()?);
                    builder.arg(lw_cuda.as_cuda_slice::<f16>()?);
                    builder.arg(&arg_hidden);
                    builder.arg(&arg_out);
                    builder.arg(&arg_eps);
                    unsafe { builder.launch(cfg) }
                        .map_err(|e| candle_core::Error::Msg(format!("fused_layernorm_linear_f16 launch: {e}")))?;
                }
                _ => unreachable!(),
            }
        }

        Ok(output)
    }
}

// ─── Marlin fused dequant+GEMM via CUDA kernel ──────────────────────────────

impl CudaBackend {
    /// Fused INT4 dequant + GEMM using the Marlin-style CUDA kernel.
    ///
    /// Weights must be pre-reformatted into Marlin tile layout `[K/16, N/64, 128]`
    /// via `GptqLinear::reformat_for_marlin()`.
    fn marlin_gemm_impl(
        &self,
        input: &Tensor,
        qweight_marlin: &Tensor,
        scales: &Tensor,
        qzeros: &Tensor,
        group_size: usize,
    ) -> Result<Tensor> {
        let dims = input.dims();
        let m = dims[0]; // batch size
        let k = dims[1]; // in_features

        // Infer N from qweight_marlin shape: [K/16, N/64, 128]
        let qw_dims = qweight_marlin.dims();
        let n = qw_dims[1] * 64; // out_features

        let device = input.device().clone();

        // Allocate output: [M, N] as F16
        let output = Tensor::zeros((m, n), DType::F16, &device)?;

        // Ensure input is F16
        let input_f16 = if input.dtype() != DType::F16 {
            input.to_dtype(DType::F16)?
        } else {
            input.contiguous()?
        };

        // Ensure scales are F16
        let scales_f16 = if scales.dtype() != DType::F16 {
            scales.to_dtype(DType::F16)?
        } else {
            scales.contiguous()?
        };

        let func = self.get_func("marlin_gemm_f16")?;

        // Grid: (N/64, M, 1), Block: (256, 1, 1)
        let cfg = LaunchConfig {
            grid_dim: ((n / 64) as u32, m as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * 4, // float per thread for K-reduction
        };

        {
            let (o_st, _) = output.storage_and_layout();
            let (i_st, _) = input_f16.storage_and_layout();
            let (qw_st, _) = qweight_marlin.storage_and_layout();
            let (sc_st, _) = scales_f16.storage_and_layout();
            let (qz_st, _) = qzeros.storage_and_layout();

            let o_cuda = as_cuda(&o_st)?;
            let i_cuda = as_cuda(&i_st)?;
            let qw_cuda = as_cuda(&qw_st)?;
            let sc_cuda = as_cuda(&sc_st)?;
            let qz_cuda = as_cuda(&qz_st)?;

            let arg_m = m as i32;
            let arg_n = n as i32;
            let arg_k = k as i32;
            let arg_group = group_size as i32;

            let mut builder = func.builder();
            builder.arg(o_cuda.as_cuda_slice::<f16>()?);
            builder.arg(i_cuda.as_cuda_slice::<f16>()?);
            builder.arg(qw_cuda.as_cuda_slice::<u32>()?);
            builder.arg(sc_cuda.as_cuda_slice::<f16>()?);
            builder.arg(qz_cuda.as_cuda_slice::<u32>()?);
            builder.arg(&arg_m);
            builder.arg(&arg_n);
            builder.arg(&arg_k);
            builder.arg(&arg_group);

            unsafe {
                builder
                    .launch(cfg)
                    .map_err(|e| candle_core::Error::Msg(format!("marlin_gemm_f16 launch: {e}")))?;
            }
        }

        Ok(output)
    }
}

// ─── GPTQ dequantization via CUDA kernel ──────────────────────────────────

impl CudaBackend {
    /// Dequantize GPTQ INT4 packed weights to F16/F32 using a CUDA kernel.
    ///
    /// This is much faster than the CPU path because it avoids unpacking
    /// and dequantizing element-by-element on the host.
    ///
    /// # Arguments
    /// - `qweight`: `[in_features / 8, out_features]` as U32
    /// - `scales`: `[num_groups, out_features]` as F16 or F32
    /// - `qzeros`: `[num_groups, out_features / 8]` as U32
    /// - `in_features`: number of input features
    /// - `out_features`: number of output features
    /// - `group_size`: GPTQ group size (typically 128)
    /// - `output_dtype`: DType::F16 or DType::F32
    ///
    /// Returns: `[out_features, in_features]` tensor in the requested dtype.
    pub fn gptq_dequantize(
        &self,
        qweight: &Tensor,
        scales: &Tensor,
        qzeros: &Tensor,
        in_features: usize,
        out_features: usize,
        group_size: usize,
        output_dtype: DType,
    ) -> Result<Tensor> {
        let device = Device::Cuda(self.device.clone());

        // Allocate output tensor: [out_features, in_features]
        let output = Tensor::zeros((out_features, in_features), output_dtype, &device)?;

        // Select kernel variant
        let func_name = match output_dtype {
            DType::F16 => "gptq_dequant_f16",
            DType::F32 => "gptq_dequant_f32",
            d => candle_core::bail!("gptq_dequantize: unsupported output dtype {d:?}"),
        };

        let func = self.get_func(func_name)?;

        // Get raw CUDA slices
        {
            let (o_st, _) = output.storage_and_layout();
            let (qw_st, _) = qweight.storage_and_layout();
            let (sc_st, _) = scales.storage_and_layout();
            let (qz_st, _) = qzeros.storage_and_layout();

            let o_cuda = as_cuda(&o_st)?;
            let qw_cuda = as_cuda(&qw_st)?;
            let sc_cuda = as_cuda(&sc_st)?;
            let qz_cuda = as_cuda(&qz_st)?;

            // Launch config: one thread per (in_feature, out_feature) element
            let block_x = 16u32;
            let block_y = 16u32;
            let grid_x = ((in_features as u32) + block_x - 1) / block_x;
            let grid_y = ((out_features as u32) + block_y - 1) / block_y;
            let cfg = LaunchConfig {
                grid_dim: (grid_x, grid_y, 1),
                block_dim: (block_x, block_y, 1),
                shared_mem_bytes: 0,
            };

            let arg_in_features = in_features as i32;
            let arg_out_features = out_features as i32;
            let arg_group_size = group_size as i32;

            match output_dtype {
                DType::F16 => {
                    let o_slice = o_cuda.as_cuda_slice::<f16>()?;
                    let qw_slice = qw_cuda.as_cuda_slice::<u32>()?;
                    let sc_slice = sc_cuda.as_cuda_slice::<f16>()?;
                    let qz_slice = qz_cuda.as_cuda_slice::<u32>()?;

                    let mut builder = func.builder();
                    builder.arg(o_slice);
                    builder.arg(qw_slice);
                    builder.arg(sc_slice);
                    builder.arg(qz_slice);
                    builder.arg(&arg_in_features);
                    builder.arg(&arg_out_features);
                    builder.arg(&arg_group_size);

                    unsafe { builder.launch(cfg) }
                        .map_err(|e| candle_core::Error::Msg(format!("gptq_dequant_f16 launch: {e}")))?;
                }
                DType::F32 => {
                    let o_slice = o_cuda.as_cuda_slice::<f32>()?;
                    let qw_slice = qw_cuda.as_cuda_slice::<u32>()?;
                    let sc_slice = sc_cuda.as_cuda_slice::<f32>()?;
                    let qz_slice = qz_cuda.as_cuda_slice::<u32>()?;

                    let mut builder = func.builder();
                    builder.arg(o_slice);
                    builder.arg(qw_slice);
                    builder.arg(sc_slice);
                    builder.arg(qz_slice);
                    builder.arg(&arg_in_features);
                    builder.arg(&arg_out_features);
                    builder.arg(&arg_group_size);

                    unsafe { builder.launch(cfg) }
                        .map_err(|e| candle_core::Error::Msg(format!("gptq_dequant_f32 launch: {e}")))?;
                }
                _ => unreachable!(),
            }
        }

        Ok(output)
    }
}

// ─── GPU Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    /// Try to get a CUDA device, skip test if unavailable.
    fn cuda_device() -> Option<Device> {
        Device::new_cuda(0).ok()
    }

    #[test]
    fn test_cuda_backend_creation() {
        if cuda_device().is_none() {
            eprintln!("SKIP: no CUDA device");
            return;
        }
        let backend = CudaBackend::new(0).expect("failed to create CudaBackend");
        assert_eq!(backend.name(), "cuda");
    }

    #[test]
    fn test_cuda_allocate_kv_caches() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => {
                eprintln!("SKIP: no CUDA device");
                return;
            }
        };
        let backend = CudaBackend::new(0).unwrap();
        let (kc, vc) = backend
            .allocate_kv_caches(2, 4, 8, 128, DType::F32, &dev)
            .expect("allocate_kv_caches failed");

        assert_eq!(kc.len(), 2);
        assert_eq!(vc.len(), 2);
        // x = 16/4 = 4 for F32
        assert_eq!(kc[0].dims(), &[4, 8, 32, BLOCK_SIZE, 4]);
        assert_eq!(vc[0].dims(), &[4, 8, 128, BLOCK_SIZE]);
        assert!(kc[0].device().is_cuda());
    }

    #[test]
    fn test_cuda_reshape_and_cache() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => {
                eprintln!("SKIP: no CUDA device");
                return;
            }
        };
        let backend = CudaBackend::new(0).unwrap();
        let num_blocks = 2;
        let num_kv_heads = 2;
        let head_size = 8;
        let dtype = DType::F32;

        let (kc, vc) = backend
            .allocate_kv_caches(1, num_blocks, num_kv_heads, head_size, dtype, &dev)
            .unwrap();

        // Create key/value tensors with known values
        let key = Tensor::ones((1, num_kv_heads, head_size), dtype, &dev).unwrap();
        let value = Tensor::ones((1, num_kv_heads, head_size), dtype, &dev).unwrap();
        let slot_mapping = Tensor::new(&[3i32], &dev).unwrap();

        backend
            .reshape_and_cache(&key, &value, &kc[0], &vc[0], &slot_mapping)
            .expect("reshape_and_cache failed on CUDA");

        // Copy to CPU and verify
        let vc_cpu: Vec<f32> = vc[0].flatten_all().unwrap().to_vec1().unwrap();
        // slot 3 -> block 0, offset 3
        // Value cache layout: [num_blocks, num_kv_heads, head_size, block_size]
        // For head=0, d=0, offset=3: index = 0 + 0 + 0 + 3 = 3
        assert_eq!(vc_cpu[3], 1.0, "value not written to slot 3");
    }

    #[test]
    fn test_cuda_reshape_and_cache_multiple_tokens() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => {
                eprintln!("SKIP: no CUDA device");
                return;
            }
        };
        let backend = CudaBackend::new(0).unwrap();
        let num_blocks = 4;
        let num_kv_heads = 1;
        let head_size = 4;
        let dtype = DType::F32;

        let (kc, vc) = backend
            .allocate_kv_caches(1, num_blocks, num_kv_heads, head_size, dtype, &dev)
            .unwrap();

        // Write 3 tokens into different slots
        let key_data: Vec<f32> = (0..3 * head_size).map(|i| i as f32).collect();
        let key = Tensor::from_vec(key_data.clone(), (3, num_kv_heads, head_size), &dev).unwrap();
        let value = Tensor::from_vec(key_data, (3, num_kv_heads, head_size), &dev).unwrap();
        let slot_mapping = Tensor::new(&[0i32, 1, 16], &dev).unwrap(); // slot 16 = block 1, offset 0

        backend
            .reshape_and_cache(&key, &value, &kc[0], &vc[0], &slot_mapping)
            .unwrap();

        let vc_cpu: Vec<f32> = vc[0].flatten_all().unwrap().to_vec1().unwrap();
        // Token 0 goes to slot 0 (block 0, offset 0)
        // Value for token 0, head 0, d=0: should be 0.0
        assert_eq!(vc_cpu[0], 0.0);
        // Value for token 0, head 0, d=1 at block_size offset: 1.0
        // index = d * block_size + offset = 1 * 16 + 0 = 16
        assert_eq!(vc_cpu[16], 1.0);
    }

    #[test]
    fn test_cuda_copy_blocks() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => {
                eprintln!("SKIP: no CUDA device");
                return;
            }
        };
        let backend = CudaBackend::new(0).unwrap();
        let num_blocks = 4;
        let num_kv_heads = 1;
        let head_size = 4;
        let dtype = DType::F32;

        let (kc, vc) = backend
            .allocate_kv_caches(1, num_blocks, num_kv_heads, head_size, dtype, &dev)
            .unwrap();

        // Write a token into block 0
        let key = Tensor::ones((1, num_kv_heads, head_size), dtype, &dev).unwrap();
        let value = Tensor::ones((1, num_kv_heads, head_size), dtype, &dev).unwrap();
        let slot_mapping = Tensor::new(&[0i32], &dev).unwrap();
        backend
            .reshape_and_cache(&key, &value, &kc[0], &vc[0], &slot_mapping)
            .unwrap();

        // Copy block 0 -> block 2
        let x = 16 / dtype.size_in_bytes();
        let numel_per_block = num_kv_heads * (head_size / x) * BLOCK_SIZE * x;
        let block_mapping = Tensor::new(&[0i32, 2], &dev).unwrap();
        backend
            .copy_blocks(&kc[0], &vc[0], &block_mapping, numel_per_block)
            .expect("copy_blocks failed on CUDA");

        // Verify block 0 and block 2 are identical
        let kc_cpu: Vec<f32> = kc[0].flatten_all().unwrap().to_vec1().unwrap();
        for i in 0..numel_per_block {
            assert_eq!(
                kc_cpu[i],
                kc_cpu[2 * numel_per_block + i],
                "key cache mismatch at offset {i}"
            );
        }
    }

    #[test]
    fn test_cuda_paged_attention_single_seq() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => {
                eprintln!("SKIP: no CUDA device");
                return;
            }
        };
        let backend = CudaBackend::new(0).unwrap();
        let num_blocks = 2;
        let num_kv_heads = 1;
        let num_heads = 1;
        let head_size = 4;
        let dtype = DType::F32;

        let (kc, vc) = backend
            .allocate_kv_caches(1, num_blocks, num_kv_heads, head_size, dtype, &dev)
            .unwrap();

        // Cache 2 tokens:
        //   token 0: K=[1,0,0,0], V=[1,0,0,0]
        //   token 1: K=[0,1,0,0], V=[0,0,1,0]
        let key = Tensor::new(
            &[[[1.0f32, 0.0, 0.0, 0.0]], [[0.0f32, 1.0, 0.0, 0.0]]],
            &dev,
        )
        .unwrap();
        let value = Tensor::new(
            &[[[1.0f32, 0.0, 0.0, 0.0]], [[0.0f32, 0.0, 1.0, 0.0]]],
            &dev,
        )
        .unwrap();
        let slot_mapping = Tensor::new(&[0i32, 1], &dev).unwrap();
        backend
            .reshape_and_cache(&key, &value, &kc[0], &vc[0], &slot_mapping)
            .unwrap();

        // Query aligns with token 1's key [0,1,0,0]
        let query = Tensor::new(&[[[0.0f32, 1.0, 0.0, 0.0]]], &dev).unwrap();
        let output = Tensor::zeros((1, num_heads, head_size), dtype, &dev).unwrap();
        let block_tables = Tensor::new(&[[0i32]], &dev).unwrap();
        let context_lens = Tensor::new(&[2i32], &dev).unwrap();

        let config = PagedAttentionConfig {
            head_size,
            num_kv_heads,
            scale: 1.0 / (head_size as f32).sqrt(),
            max_context_len: 2,
        };

        backend
            .paged_attention(
                &output,
                &query,
                &kc[0],
                &vc[0],
                &block_tables,
                &context_lens,
                &config,
            )
            .expect("paged_attention failed on CUDA");

        let out_cpu: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        // Token 1's value [0,0,1,0] should dominate since query dot token1_key > query dot token0_key
        assert!(
            out_cpu[2] > out_cpu[0],
            "token 1 value should dominate: output={:?}",
            out_cpu
        );
    }

    #[test]
    fn test_cuda_paged_attention_gqa() {
        // Test Grouped-Query Attention: 4 query heads, 1 KV head
        let dev = match cuda_device() {
            Some(d) => d,
            None => {
                eprintln!("SKIP: no CUDA device");
                return;
            }
        };
        let backend = CudaBackend::new(0).unwrap();
        let num_blocks = 2;
        let num_kv_heads = 1;
        let num_heads = 4;
        let head_size = 8;
        let dtype = DType::F32;

        let (kc, vc) = backend
            .allocate_kv_caches(1, num_blocks, num_kv_heads, head_size, dtype, &dev)
            .unwrap();

        // Cache 1 token
        let key_data: Vec<f32> = (0..head_size).map(|i| (i as f32) * 0.1).collect();
        let key = Tensor::from_vec(key_data.clone(), (1, num_kv_heads, head_size), &dev).unwrap();
        let value = Tensor::from_vec(key_data, (1, num_kv_heads, head_size), &dev).unwrap();
        let slot_mapping = Tensor::new(&[0i32], &dev).unwrap();
        backend
            .reshape_and_cache(&key, &value, &kc[0], &vc[0], &slot_mapping)
            .unwrap();

        // All 4 query heads attend to the single KV head
        let q_data: Vec<f32> = (0..num_heads * head_size).map(|i| (i as f32) * 0.01).collect();
        let query = Tensor::from_vec(q_data, (1, num_heads, head_size), &dev).unwrap();
        let output = Tensor::zeros((1, num_heads, head_size), dtype, &dev).unwrap();
        let block_tables = Tensor::new(&[[0i32]], &dev).unwrap();
        let context_lens = Tensor::new(&[1i32], &dev).unwrap();

        let config = PagedAttentionConfig {
            head_size,
            num_kv_heads,
            scale: 1.0 / (head_size as f32).sqrt(),
            max_context_len: 1,
        };

        backend
            .paged_attention(
                &output,
                &query,
                &kc[0],
                &vc[0],
                &block_tables,
                &context_lens,
                &config,
            )
            .expect("GQA paged_attention failed on CUDA");

        let out_cpu: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        // With single cached token, softmax is trivially 1.0 so output = value
        // All heads should produce the same output (same KV head)
        let head0: Vec<f32> = out_cpu[..head_size].to_vec();
        let head1: Vec<f32> = out_cpu[head_size..2 * head_size].to_vec();
        for (a, b) in head0.iter().zip(head1.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "GQA heads should match: head0={:?} head1={:?}",
                head0,
                head1
            );
        }
    }

    #[test]
    fn test_cuda_paged_attention_multi_seq() {
        // Test batched attention with 2 sequences
        let dev = match cuda_device() {
            Some(d) => d,
            None => {
                eprintln!("SKIP: no CUDA device");
                return;
            }
        };
        let backend = CudaBackend::new(0).unwrap();
        let num_blocks = 4;
        let num_kv_heads = 1;
        let num_heads = 1;
        let head_size = 4;
        let dtype = DType::F32;

        let (kc, vc) = backend
            .allocate_kv_caches(1, num_blocks, num_kv_heads, head_size, dtype, &dev)
            .unwrap();

        // Seq 0: 1 token in block 0, K=[1,0,0,0], V=[1,0,0,0]
        // Seq 1: 1 token in block 1, K=[0,0,0,1], V=[0,0,0,1]
        let key = Tensor::new(
            &[[[1.0f32, 0.0, 0.0, 0.0]], [[0.0f32, 0.0, 0.0, 1.0]]],
            &dev,
        )
        .unwrap();
        let value = Tensor::new(
            &[[[1.0f32, 0.0, 0.0, 0.0]], [[0.0f32, 0.0, 0.0, 1.0]]],
            &dev,
        )
        .unwrap();
        let slot_mapping = Tensor::new(&[0i32, 16], &dev).unwrap(); // block 0 slot 0, block 1 slot 0
        backend
            .reshape_and_cache(&key, &value, &kc[0], &vc[0], &slot_mapping)
            .unwrap();

        // Both sequences query with identity-ish vectors
        let query = Tensor::new(
            &[
                [[1.0f32, 0.0, 0.0, 0.0]], // seq 0
                [[0.0f32, 0.0, 0.0, 1.0]], // seq 1
            ],
            &dev,
        )
        .unwrap();
        let output = Tensor::zeros((2, num_heads, head_size), dtype, &dev).unwrap();
        let block_tables = Tensor::new(&[[0i32], [1i32]], &dev).unwrap();
        let context_lens = Tensor::new(&[1i32, 1], &dev).unwrap();

        let config = PagedAttentionConfig {
            head_size,
            num_kv_heads,
            scale: 1.0 / (head_size as f32).sqrt(),
            max_context_len: 1,
        };

        backend
            .paged_attention(
                &output,
                &query,
                &kc[0],
                &vc[0],
                &block_tables,
                &context_lens,
                &config,
            )
            .expect("batched paged_attention failed on CUDA");

        let out_cpu: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        // Seq 0 output should be V0 = [1,0,0,0], Seq 1 output should be V1 = [0,0,0,1]
        assert!(
            out_cpu[0] > 0.9,
            "seq 0 output[0] should be ~1.0: {:?}",
            &out_cpu[..head_size]
        );
        assert!(
            out_cpu[head_size + 3] > 0.9,
            "seq 1 output[3] should be ~1.0: {:?}",
            &out_cpu[head_size..]
        );
    }

    /// Cross-validate CUDA results against CPU backend.
    #[test]
    fn test_cuda_vs_cpu_paged_attention() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => {
                eprintln!("SKIP: no CUDA device");
                return;
            }
        };
        let cuda_backend = CudaBackend::new(0).unwrap();
        let cpu_backend = crate::serving::kernels::CpuBackend::new();

        let num_blocks = 2;
        let num_kv_heads = 2;
        let num_heads = 2;
        let head_size = 8;
        let dtype = DType::F32;

        // Allocate caches on both devices
        let (kc_cuda, vc_cuda) = cuda_backend
            .allocate_kv_caches(1, num_blocks, num_kv_heads, head_size, dtype, &dev)
            .unwrap();
        let (kc_cpu, vc_cpu) = cpu_backend
            .allocate_kv_caches(1, num_blocks, num_kv_heads, head_size, dtype, &Device::Cpu)
            .unwrap();

        // Same random-ish key/value data
        let n = 3; // 3 tokens
        let kv_data: Vec<f32> = (0..(n * num_kv_heads * head_size))
            .map(|i| ((i * 7 + 3) % 17) as f32 * 0.1 - 0.8)
            .collect();

        let key_cuda =
            Tensor::from_vec(kv_data.clone(), (n, num_kv_heads, head_size), &dev).unwrap();
        let val_cuda =
            Tensor::from_vec(kv_data.clone(), (n, num_kv_heads, head_size), &dev).unwrap();
        let key_cpu = Tensor::from_vec(
            kv_data.clone(),
            (n, num_kv_heads, head_size),
            &Device::Cpu,
        )
        .unwrap();
        let val_cpu =
            Tensor::from_vec(kv_data, (n, num_kv_heads, head_size), &Device::Cpu).unwrap();

        let slots = &[0i32, 1, 2];
        let sm_cuda = Tensor::new(slots, &dev).unwrap();
        let sm_cpu = Tensor::new(slots, &Device::Cpu).unwrap();

        cuda_backend
            .reshape_and_cache(&key_cuda, &val_cuda, &kc_cuda[0], &vc_cuda[0], &sm_cuda)
            .unwrap();
        cpu_backend
            .reshape_and_cache(&key_cpu, &val_cpu, &kc_cpu[0], &vc_cpu[0], &sm_cpu)
            .unwrap();

        // Verify caches match
        let kc_cuda_data: Vec<f32> = kc_cuda[0].flatten_all().unwrap().to_vec1().unwrap();
        let kc_cpu_data: Vec<f32> = kc_cpu[0].flatten_all().unwrap().to_vec1().unwrap();
        for (i, (a, b)) in kc_cuda_data.iter().zip(kc_cpu_data.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "key cache mismatch at {i}: cuda={a}, cpu={b}"
            );
        }

        // Now run paged attention on both
        let q_data: Vec<f32> = (0..(num_heads * head_size))
            .map(|i| ((i * 11 + 5) % 13) as f32 * 0.1 - 0.6)
            .collect();

        let q_cuda = Tensor::from_vec(q_data.clone(), (1, num_heads, head_size), &dev).unwrap();
        let q_cpu =
            Tensor::from_vec(q_data, (1, num_heads, head_size), &Device::Cpu).unwrap();

        let out_cuda = Tensor::zeros((1, num_heads, head_size), dtype, &dev).unwrap();
        let out_cpu = Tensor::zeros((1, num_heads, head_size), dtype, &Device::Cpu).unwrap();

        let bt_cuda = Tensor::new(&[[0i32]], &dev).unwrap();
        let bt_cpu = Tensor::new(&[[0i32]], &Device::Cpu).unwrap();
        let cl_cuda = Tensor::new(&[3i32], &dev).unwrap();
        let cl_cpu = Tensor::new(&[3i32], &Device::Cpu).unwrap();

        let config = PagedAttentionConfig {
            head_size,
            num_kv_heads,
            scale: 1.0 / (head_size as f32).sqrt(),
            max_context_len: 3,
        };

        cuda_backend
            .paged_attention(
                &out_cuda,
                &q_cuda,
                &kc_cuda[0],
                &vc_cuda[0],
                &bt_cuda,
                &cl_cuda,
                &config,
            )
            .unwrap();
        cpu_backend
            .paged_attention(
                &out_cpu,
                &q_cpu,
                &kc_cpu[0],
                &vc_cpu[0],
                &bt_cpu,
                &cl_cpu,
                &config,
            )
            .unwrap();

        let cuda_out: Vec<f32> = out_cuda.flatten_all().unwrap().to_vec1().unwrap();
        let cpu_out: Vec<f32> = out_cpu.flatten_all().unwrap().to_vec1().unwrap();

        for (i, (a, b)) in cuda_out.iter().zip(cpu_out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "attention output mismatch at {i}: cuda={a}, cpu={b}"
            );
        }
    }

    // ─── F16 Tests ───────────────────────────────────────────────────────

    /// Helper to convert f32 data to F16 tensor on CUDA device.
    fn f32_to_f16_tensor(data: &[f32], shape: impl Into<candle_core::Shape>, dev: &Device) -> Tensor {
        let f32_tensor = Tensor::from_vec(data.to_vec(), shape, dev).unwrap();
        f32_tensor.to_dtype(DType::F16).unwrap()
    }

    #[test]
    fn test_cuda_f16_allocate_kv_caches() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => {
                eprintln!("SKIP: no CUDA device");
                return;
            }
        };
        let backend = CudaBackend::new(0).unwrap();
        let (kc, vc) = backend
            .allocate_kv_caches(1, 4, 2, 8, DType::F16, &dev)
            .expect("allocate F16 KV caches failed");

        // x = 16/2 = 8 for F16
        assert_eq!(kc[0].dims(), &[4, 2, 1, BLOCK_SIZE, 8]);
        assert_eq!(vc[0].dims(), &[4, 2, 8, BLOCK_SIZE]);
        assert_eq!(kc[0].dtype(), DType::F16);
    }

    #[test]
    fn test_cuda_f16_reshape_and_cache() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => {
                eprintln!("SKIP: no CUDA device");
                return;
            }
        };
        let backend = CudaBackend::new(0).unwrap();
        let num_blocks = 2;
        let num_kv_heads = 1;
        let head_size = 8;
        let dtype = DType::F16;

        let (kc, vc) = backend
            .allocate_kv_caches(1, num_blocks, num_kv_heads, head_size, dtype, &dev)
            .unwrap();

        // Write a token with known values in F16
        let data: Vec<f32> = (0..head_size).map(|i| (i as f32) * 0.5).collect();
        let key = f32_to_f16_tensor(&data, (1, num_kv_heads, head_size), &dev);
        let value = f32_to_f16_tensor(&data, (1, num_kv_heads, head_size), &dev);
        let slot_mapping = Tensor::new(&[0i32], &dev).unwrap();

        backend
            .reshape_and_cache(&key, &value, &kc[0], &vc[0], &slot_mapping)
            .expect("F16 reshape_and_cache failed");

        // Read back as F32 to verify
        let vc_f32 = vc[0].to_dtype(DType::F32).unwrap();
        let vc_cpu: Vec<f32> = vc_f32.flatten_all().unwrap().to_vec1().unwrap();
        // Value at d=0, block_size offset=0 should be 0.0
        assert!((vc_cpu[0] - 0.0).abs() < 0.01, "F16 value d=0: {}", vc_cpu[0]);
        // Value at d=1, offset = d * block_size + 0 = 1 * 16 = 16
        assert!((vc_cpu[16] - 0.5).abs() < 0.01, "F16 value d=1: {}", vc_cpu[16]);
    }

    #[test]
    fn test_cuda_f16_copy_blocks() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => {
                eprintln!("SKIP: no CUDA device");
                return;
            }
        };
        let backend = CudaBackend::new(0).unwrap();
        let num_blocks = 4;
        let num_kv_heads = 1;
        let head_size = 8;
        let dtype = DType::F16;

        let (kc, vc) = backend
            .allocate_kv_caches(1, num_blocks, num_kv_heads, head_size, dtype, &dev)
            .unwrap();

        // Write a token into block 0
        let data: Vec<f32> = (0..head_size).map(|i| (i as f32) + 1.0).collect();
        let key = f32_to_f16_tensor(&data, (1, num_kv_heads, head_size), &dev);
        let value = f32_to_f16_tensor(&data, (1, num_kv_heads, head_size), &dev);
        let slot_mapping = Tensor::new(&[0i32], &dev).unwrap();
        backend
            .reshape_and_cache(&key, &value, &kc[0], &vc[0], &slot_mapping)
            .unwrap();

        // Copy block 0 -> block 2
        let x = 16 / dtype.size_in_bytes();
        let numel_per_block = num_kv_heads * (head_size / x) * BLOCK_SIZE * x;
        let block_mapping = Tensor::new(&[0i32, 2], &dev).unwrap();
        backend
            .copy_blocks(&kc[0], &vc[0], &block_mapping, numel_per_block)
            .expect("F16 copy_blocks failed");

        // Verify blocks match
        let kc_f32 = kc[0].to_dtype(DType::F32).unwrap();
        let kc_cpu: Vec<f32> = kc_f32.flatten_all().unwrap().to_vec1().unwrap();
        for i in 0..numel_per_block {
            assert!(
                (kc_cpu[i] - kc_cpu[2 * numel_per_block + i]).abs() < 0.01,
                "F16 key cache mismatch at offset {i}"
            );
        }
    }

    #[test]
    fn test_cuda_f16_paged_attention() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => {
                eprintln!("SKIP: no CUDA device");
                return;
            }
        };
        let backend = CudaBackend::new(0).unwrap();
        let num_blocks = 2;
        let num_kv_heads = 1;
        let num_heads = 1;
        // F16 requires head_size >= 8 because x = 16/2 = 8 and key cache dim = head_size/x
        let head_size = 8;
        let dtype = DType::F16;

        let (kc, vc) = backend
            .allocate_kv_caches(1, num_blocks, num_kv_heads, head_size, dtype, &dev)
            .unwrap();

        // Cache 1 token: K=[1,0,0,0,0,0,0,0], V=[1,0,0,0,0,0,0,0]
        let mut k_data = vec![0.0f32; head_size];
        k_data[0] = 1.0;
        let key = f32_to_f16_tensor(&k_data, (1, num_kv_heads, head_size), &dev);
        let value = f32_to_f16_tensor(&k_data, (1, num_kv_heads, head_size), &dev);
        let slot_mapping = Tensor::new(&[0i32], &dev).unwrap();
        backend
            .reshape_and_cache(&key, &value, &kc[0], &vc[0], &slot_mapping)
            .unwrap();

        // Query = [1,0,0,0,...] — should match perfectly, output = V = [1,0,0,0,...]
        let query = f32_to_f16_tensor(&k_data, (1, num_heads, head_size), &dev);
        let output = Tensor::zeros((1, num_heads, head_size), dtype, &dev).unwrap();
        let block_tables = Tensor::new(&[[0i32]], &dev).unwrap();
        let context_lens = Tensor::new(&[1i32], &dev).unwrap();

        let config = PagedAttentionConfig {
            head_size,
            num_kv_heads,
            scale: 1.0 / (head_size as f32).sqrt(),
            max_context_len: 1,
        };

        backend
            .paged_attention(
                &output, &query, &kc[0], &vc[0],
                &block_tables, &context_lens, &config,
            )
            .expect("F16 paged_attention failed");

        // Single cached token → softmax = 1.0 → output = value
        let out_f32 = output.to_dtype(DType::F32).unwrap();
        let out_cpu: Vec<f32> = out_f32.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            out_cpu[0] > 0.9,
            "F16 attention output[0] should be ~1.0: {:?}",
            out_cpu
        );
        assert!(
            out_cpu[1].abs() < 0.1,
            "F16 attention output[1] should be ~0.0: {:?}",
            out_cpu
        );
    }

    #[test]
    fn test_cuda_detect_backend() {
        if cuda_device().is_none() {
            eprintln!("SKIP: no CUDA device");
            return;
        }
        let backend = crate::serving::kernels::detect_backend();
        assert_eq!(backend.name(), "cuda");
    }

    #[test]
    fn test_cuda_gptq_dequant_f32() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => { eprintln!("SKIP: no CUDA device"); return; }
        };
        let backend = CudaBackend::new(0).unwrap();

        // Create a small GPTQ-quantized layer on CPU, then move to GPU
        let w = Tensor::randn(0f32, 0.1, (8, 16), &Device::Cpu).unwrap();
        let gl = crate::serving::quantization::GptqLinear::from_float(&w, None, 8).unwrap();

        // Get the CPU dequantized result for reference
        let w_cpu = gl.dequantize(DType::F32).unwrap();

        // Move packed data to GPU
        let qweight_gpu = gl.qweight.to_device(&dev).unwrap();
        let scales_gpu = gl.scales.to_dtype(DType::F32).unwrap().to_device(&dev).unwrap();
        let qzeros_gpu = gl.qzeros.to_device(&dev).unwrap();

        // Run CUDA dequant kernel
        let w_gpu = backend.gptq_dequantize(
            &qweight_gpu,
            &scales_gpu,
            &qzeros_gpu,
            16, // in_features
            8,  // out_features
            8,  // group_size
            DType::F32,
        ).unwrap();

        assert_eq!(w_gpu.dims(), &[8, 16]);

        // Compare GPU result with CPU reference
        let w_gpu_host: Vec<f32> = w_gpu.to_vec2().unwrap().into_iter().flatten().collect();
        let w_cpu_host: Vec<f32> = w_cpu.to_vec2().unwrap().into_iter().flatten().collect();

        let mut max_diff: f32 = 0.0;
        for (g, c) in w_gpu_host.iter().zip(w_cpu_host.iter()) {
            let diff = (g - c).abs();
            if diff > max_diff { max_diff = diff; }
        }

        assert!(
            max_diff < 0.01,
            "CUDA vs CPU GPTQ dequant max_diff = {max_diff} (should be < 0.01)"
        );
    }

    #[test]
    fn test_cuda_gptq_dequant_f16() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => { eprintln!("SKIP: no CUDA device"); return; }
        };
        let backend = CudaBackend::new(0).unwrap();

        let w = Tensor::randn(0f32, 0.1, (16, 32), &Device::Cpu).unwrap();
        let gl = crate::serving::quantization::GptqLinear::from_float(&w, None, 8).unwrap();

        // Move to GPU
        let qweight_gpu = gl.qweight.to_device(&dev).unwrap();
        let scales_gpu = gl.scales.to_dtype(DType::F16).unwrap().to_device(&dev).unwrap();
        let qzeros_gpu = gl.qzeros.to_device(&dev).unwrap();

        let w_gpu = backend.gptq_dequantize(
            &qweight_gpu,
            &scales_gpu,
            &qzeros_gpu,
            32, // in_features
            16, // out_features
            8,  // group_size
            DType::F16,
        ).unwrap();

        assert_eq!(w_gpu.dims(), &[16, 32]);
        assert_eq!(w_gpu.dtype(), DType::F16);

        // Verify not all zeros (basic sanity check)
        let sum: f32 = w_gpu.to_dtype(DType::F32).unwrap()
            .abs().unwrap()
            .sum_all().unwrap()
            .to_scalar().unwrap();
        assert!(sum > 0.01, "GPTQ F16 dequant should produce non-zero output, got sum={sum}");
    }

    // ─── Fused kernel CUDA tests ─────────────────────────────────────────

    #[test]
    fn test_cuda_fused_silu_mul() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => { eprintln!("SKIP: no CUDA device"); return; }
        };
        let backend = CudaBackend::new(0).unwrap();

        // Create test data on GPU
        let gate_data: Vec<f32> = (0..512).map(|i| (i as f32 - 256.0) * 0.01).collect();
        let up_data: Vec<f32> = (0..512).map(|i| ((i * 7 + 3) % 100) as f32 * 0.02 - 1.0).collect();

        let gate = Tensor::from_vec(gate_data.clone(), (4, 128), &dev).unwrap();
        let up = Tensor::from_vec(up_data.clone(), (4, 128), &dev).unwrap();

        let result = backend.fused_silu_mul(&gate, &up).unwrap();
        assert_eq!(result.dims(), &[4, 128]);

        // Compare with CPU reference
        let gate_cpu = Tensor::from_vec(gate_data, (4, 128), &Device::Cpu).unwrap();
        let up_cpu = Tensor::from_vec(up_data, (4, 128), &Device::Cpu).unwrap();
        let silu_gate = candle_nn::ops::silu(&gate_cpu).unwrap();
        let expected = (silu_gate * &up_cpu).unwrap();

        let result_host: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        let expected_host: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();

        let mut max_diff: f32 = 0.0;
        for (r, e) in result_host.iter().zip(expected_host.iter()) {
            let diff = (r - e).abs();
            if diff > max_diff { max_diff = diff; }
        }
        assert!(
            max_diff < 1e-4,
            "CUDA fused_silu_mul vs CPU: max_diff={max_diff}"
        );
    }

    #[test]
    fn test_cuda_fused_silu_mul_f16() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => { eprintln!("SKIP: no CUDA device"); return; }
        };
        let backend = CudaBackend::new(0).unwrap();

        let gate_data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
        let up_data: Vec<f32> = (0..64).map(|i| ((i * 3 + 1) % 20) as f32 * 0.1 - 1.0).collect();

        let gate = Tensor::from_vec(gate_data, (2, 32), &dev).unwrap().to_dtype(DType::F16).unwrap();
        let up = Tensor::from_vec(up_data, (2, 32), &dev).unwrap().to_dtype(DType::F16).unwrap();

        let result = backend.fused_silu_mul(&gate, &up).unwrap();
        assert_eq!(result.dims(), &[2, 32]);
        assert_eq!(result.dtype(), DType::F16);

        // Verify non-zero output
        let sum: f32 = result.to_dtype(DType::F32).unwrap()
            .abs().unwrap().sum_all().unwrap().to_scalar().unwrap();
        assert!(sum > 0.01, "F16 fused_silu_mul should produce non-zero output");
    }

    #[test]
    fn test_cuda_fused_rope() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => { eprintln!("SKIP: no CUDA device"); return; }
        };
        let backend = CudaBackend::new(0).unwrap();

        let total_tokens = 4;
        let num_heads = 2;
        let head_size = 8;
        let rope_dim = 8;

        // Create data
        let x_data: Vec<f32> = (0..total_tokens * num_heads * head_size)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let x = Tensor::from_vec(x_data.clone(), (total_tokens, num_heads, head_size), &dev).unwrap();
        let positions = Tensor::new(&[0u32, 1, 5, 10], &dev).unwrap();

        // Precompute RoPE on CPU, move to GPU
        let (cos, sin) = crate::serving::models::attention::precompute_rope(
            rope_dim, 10000.0, 128, &Device::Cpu,
        ).unwrap();
        let cos_gpu = cos.to_device(&dev).unwrap();
        let sin_gpu = sin.to_device(&dev).unwrap();

        let result = backend.fused_rope(
            &x, &positions, &cos_gpu, &sin_gpu,
            num_heads, head_size, rope_dim,
        ).unwrap();
        assert_eq!(result.dims(), &[total_tokens, num_heads, head_size]);

        // Compare with CPU default implementation
        let cpu_backend = crate::serving::kernels::CpuBackend::new();
        let x_cpu = Tensor::from_vec(x_data, (total_tokens, num_heads, head_size), &Device::Cpu).unwrap();
        let positions_cpu = Tensor::new(&[0u32, 1, 5, 10], &Device::Cpu).unwrap();

        let expected = cpu_backend.fused_rope(
            &x_cpu, &positions_cpu, &cos, &sin,
            num_heads, head_size, rope_dim,
        ).unwrap();

        let result_host: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        let expected_host: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();

        let mut max_diff: f32 = 0.0;
        for (i, (r, e)) in result_host.iter().zip(expected_host.iter()).enumerate() {
            let diff = (r - e).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        assert!(
            max_diff < 1e-4,
            "CUDA fused_rope vs CPU: max_diff={max_diff}"
        );
    }

    #[test]
    fn test_cuda_fused_rope_position_zero_identity() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => { eprintln!("SKIP: no CUDA device"); return; }
        };
        let backend = CudaBackend::new(0).unwrap();

        let num_heads = 2;
        let head_size = 4;
        let rope_dim = 4;

        let x_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x = Tensor::from_vec(x_data.clone(), (1, num_heads, head_size), &dev).unwrap();
        let positions = Tensor::new(&[0u32], &dev).unwrap();

        let (cos, sin) = crate::serving::models::attention::precompute_rope(
            rope_dim, 10000.0, 128, &Device::Cpu,
        ).unwrap();
        let cos_gpu = cos.to_device(&dev).unwrap();
        let sin_gpu = sin.to_device(&dev).unwrap();

        let result = backend.fused_rope(
            &x, &positions, &cos_gpu, &sin_gpu,
            num_heads, head_size, rope_dim,
        ).unwrap();

        let result_host: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        for (i, (r, orig)) in result_host.iter().zip(x_data.iter()).enumerate() {
            assert!(
                (r - orig).abs() < 1e-4,
                "pos-0 CUDA identity failed at {i}: got {r}, expected {orig}"
            );
        }
    }

    #[test]
    fn test_cuda_fused_rmsnorm() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => { eprintln!("SKIP: no CUDA device"); return; }
        };
        let backend = CudaBackend::new(0).unwrap();

        let x = Tensor::randn(0f32, 1.0, (8, 128), &Device::Cpu).unwrap();
        let weight = Tensor::randn(0f32, 1.0, 128, &Device::Cpu).unwrap();
        let eps = 1e-5f32;

        // CPU reference
        let reference = candle_nn::ops::rms_norm(&x, &weight, eps).unwrap();
        let ref_data: Vec<f32> = reference.flatten_all().unwrap().to_vec1().unwrap();

        // CUDA
        let x_gpu = x.to_device(&dev).unwrap();
        let weight_gpu = weight.to_device(&dev).unwrap();
        let result = backend.fused_rmsnorm(&x_gpu, &weight_gpu, eps).unwrap();
        let result_data: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();

        let max_diff = ref_data.iter().zip(result_data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "CUDA fused_rmsnorm vs CPU: max_diff={max_diff}"
        );
    }

    #[test]
    fn test_cuda_fused_rmsnorm_f16() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => { eprintln!("SKIP: no CUDA device"); return; }
        };
        let backend = CudaBackend::new(0).unwrap();

        let x = Tensor::randn(0f32, 0.5, (4, 64), &Device::Cpu).unwrap();
        let x_f16 = x.to_dtype(DType::F16).unwrap().to_device(&dev).unwrap();
        let weight = Tensor::ones(64, DType::F32, &Device::Cpu).unwrap().to_device(&dev).unwrap();
        let eps = 1e-5f32;

        let result = backend.fused_rmsnorm(&x_f16, &weight, eps).unwrap();
        assert_eq!(result.dtype(), DType::F16);
        assert_eq!(result.dims(), &[4, 64]);

        let result_f32: Vec<f32> = result.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let sum: f32 = result_f32.iter().map(|v| v.abs()).sum();
        assert!(sum > 0.1, "F16 fused_rmsnorm should produce non-zero output, got sum={sum}");
    }

    #[test]
    fn test_cuda_fused_add_rmsnorm() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => { eprintln!("SKIP: no CUDA device"); return; }
        };
        let backend = CudaBackend::new(0).unwrap();
        let x = Tensor::randn(0f32, 1.0, (4, 64), &dev).unwrap();
        let residual = Tensor::randn(0f32, 1.0, (4, 64), &dev).unwrap();
        let weight = Tensor::randn(0f32, 1.0, 64, &dev).unwrap();
        let eps = 1e-5f32;

        let (normed, x_out) = backend.fused_add_rmsnorm(&x, &residual, &weight, eps).unwrap();
        assert_eq!(normed.dims(), &[4, 64]);
        assert_eq!(x_out.dims(), &[4, 64]);

        // Cross-validate against CPU reference
        let x_cpu = x.to_device(&Device::Cpu).unwrap();
        let res_cpu = residual.to_device(&Device::Cpu).unwrap();
        let w_cpu = weight.to_device(&Device::Cpu).unwrap();
        let ref_x = (&x_cpu + &res_cpu).unwrap();
        let ref_norm = candle_nn::ops::rms_norm(&ref_x, &w_cpu, eps).unwrap();

        let x_out_cpu: Vec<f32> = x_out.to_device(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let ref_x_data: Vec<f32> = ref_x.flatten_all().unwrap().to_vec1().unwrap();
        let x_max_diff: f32 = x_out_cpu.iter().zip(ref_x_data.iter())
            .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(x_max_diff < 1e-4, "x_out diff={x_max_diff}");

        let normed_cpu: Vec<f32> = normed.to_device(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let ref_norm_data: Vec<f32> = ref_norm.flatten_all().unwrap().to_vec1().unwrap();
        let n_max_diff: f32 = normed_cpu.iter().zip(ref_norm_data.iter())
            .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(n_max_diff < 1e-4, "normed diff={n_max_diff}");
    }

    #[test]
    fn test_cuda_fused_add_rmsnorm_f16() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => { eprintln!("SKIP: no CUDA device"); return; }
        };
        let backend = CudaBackend::new(0).unwrap();
        let x = Tensor::randn(0f32, 1.0, (4, 64), &dev).unwrap().to_dtype(DType::F16).unwrap();
        let residual = Tensor::randn(0f32, 1.0, (4, 64), &dev).unwrap().to_dtype(DType::F16).unwrap();
        let weight = Tensor::randn(0f32, 1.0, 64, &dev).unwrap();
        let eps = 1e-5f32;

        let (normed, x_out) = backend.fused_add_rmsnorm(&x, &residual, &weight, eps).unwrap();
        assert_eq!(normed.dtype(), DType::F16);
        assert_eq!(x_out.dtype(), DType::F16);
        assert_eq!(normed.dims(), &[4, 64]);
        assert_eq!(x_out.dims(), &[4, 64]);

        // Verify non-zero
        let sum: f32 = normed.to_dtype(DType::F32).unwrap().flatten_all().unwrap()
            .to_vec1::<f32>().unwrap().iter().map(|v| v.abs()).sum();
        assert!(sum > 0.1, "F16 fused_add_rmsnorm should produce non-zero output");
    }
}
