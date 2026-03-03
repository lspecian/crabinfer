//! Metal kernel dispatch implementation.
//!
//! Compiles the paged attention, reshape_and_cache, and copy_blocks Metal
//! shaders at first use, caches pipelines, and provides Rust dispatch functions.

use candle_core::{DType, Device, Result, Tensor};
use candle_metal_kernels::metal::ComputeCommandEncoder;
use objc2_metal::MTLSize;
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

use super::{HeadSize, BLOCK_SIZE, NUM_THREADS, PARTITION_SIZE};

// ─── Metal source (embedded at compile time) ─────────────────────────────

const PAGED_ATTENTION_SOURCE: &str = include_str!("paged_attention.metal");
const RESHAPE_AND_CACHE_SOURCE: &str = include_str!("reshape_and_cache.metal");
const COPY_BLOCKS_SOURCE: &str = include_str!("copy_blocks.metal");

// ─── Pipeline cache ──────────────────────────────────────────────────────

/// Thread-safe cache of compiled Metal compute pipelines.
pub struct PagedAttentionKernels {
    pipelines: RwLock<HashMap<String, candle_metal_kernels::metal::ComputePipeline>>,
}

impl PagedAttentionKernels {
    fn new() -> Self {
        Self {
            pipelines: RwLock::new(HashMap::new()),
        }
    }

    /// Get or compile a pipeline for the given kernel name.
    fn get_pipeline(
        &self,
        device: &candle_metal_kernels::metal::Device,
        source: &str,
        kernel_name: &str,
    ) -> Result<candle_metal_kernels::metal::ComputePipeline> {
        // Fast path: check read lock
        {
            let cache = self.pipelines.read().map_err(|e| {
                candle_core::Error::Msg(format!("pipeline cache lock poisoned: {e}"))
            })?;
            if let Some(pipeline) = cache.get(kernel_name) {
                return Ok(pipeline.clone());
            }
        }

        // Slow path: compile and insert
        let library = device.new_library_with_source(source, None).map_err(|e| {
            candle_core::Error::Msg(format!("Metal shader compilation failed: {e}"))
        })?;
        let function = library.get_function(kernel_name, None).map_err(|e| {
            candle_core::Error::Msg(format!(
                "Metal function '{kernel_name}' not found: {e}"
            ))
        })?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| {
                candle_core::Error::Msg(format!(
                    "Metal pipeline creation failed for '{kernel_name}': {e}"
                ))
            })?;

        let mut cache = self.pipelines.write().map_err(|e| {
            candle_core::Error::Msg(format!("pipeline cache lock poisoned: {e}"))
        })?;
        cache.insert(kernel_name.to_string(), pipeline.clone());
        Ok(pipeline)
    }
}

/// Global pipeline cache (initialized once per process).
static KERNELS: OnceLock<PagedAttentionKernels> = OnceLock::new();

fn kernels() -> &'static PagedAttentionKernels {
    KERNELS.get_or_init(PagedAttentionKernels::new)
}

// ─── Helper: extract Metal device and storage ────────────────────────────

fn get_metal_device(tensor: &Tensor) -> Result<candle_core::MetalDevice> {
    match tensor.device() {
        Device::Metal(dev) => Ok(dev.clone()),
        _ => candle_core::bail!("expected Metal tensor"),
    }
}


// ─── Reshape and cache ───────────────────────────────────────────────────

/// Write K/V tokens into the paged KV cache.
///
/// # Arguments
/// - `key`: [num_tokens, num_kv_heads, head_size] — key projections
/// - `value`: [num_tokens, num_kv_heads, head_size] — value projections
/// - `key_cache`: [num_blocks, num_kv_heads, head_size/x, block_size, x] — paged key cache
/// - `value_cache`: [num_blocks, num_kv_heads, head_size, block_size] — paged value cache
/// - `slot_mapping`: [num_tokens] i32 — maps each token to a physical cache slot
pub fn reshape_and_cache(
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    let device = get_metal_device(key)?;
    let dtype = key.dtype();

    let kernel_name = match dtype {
        DType::F32 => "reshape_and_cache_float",
        DType::F16 => "reshape_and_cache_half",
        d => candle_core::bail!("reshape_and_cache: unsupported dtype {d:?}"),
    };

    let dims = key.dims();
    let num_tokens = dims[0];
    let num_heads = dims[1];
    let head_size = dims[2];
    let block_size = BLOCK_SIZE as i32;
    let x = (16 / dtype.size_in_bytes()) as i32;

    let key_stride = (num_heads * head_size) as i32;
    let value_stride = (num_heads * head_size) as i32;

    let pipeline = kernels().get_pipeline(
        device.metal_device(),
        RESHAPE_AND_CACHE_SOURCE,
        kernel_name,
    )?;

    let encoder = device.command_encoder()?;
    encoder.set_compute_pipeline_state(&pipeline);

    // Bind buffers — we need to access raw buffers within storage guard scope
    {
        let (k_store, k_layout) = key.storage_and_layout();
        let (v_store, v_layout) = value.storage_and_layout();
        let (kc_store, kc_layout) = key_cache.storage_and_layout();
        let (vc_store, vc_layout) = value_cache.storage_and_layout();
        let (sm_store, sm_layout) = slot_mapping.storage_and_layout();

        let k_buf = match &*k_store { candle_core::Storage::Metal(s) => s.buffer(), _ => candle_core::bail!("Metal") };
        let v_buf = match &*v_store { candle_core::Storage::Metal(s) => s.buffer(), _ => candle_core::bail!("Metal") };
        let kc_buf = match &*kc_store { candle_core::Storage::Metal(s) => s.buffer(), _ => candle_core::bail!("Metal") };
        let vc_buf = match &*vc_store { candle_core::Storage::Metal(s) => s.buffer(), _ => candle_core::bail!("Metal") };
        let sm_buf = match &*sm_store { candle_core::Storage::Metal(s) => s.buffer(), _ => candle_core::bail!("Metal") };

        let k_off = k_layout.start_offset() * dtype.size_in_bytes();
        let v_off = v_layout.start_offset() * dtype.size_in_bytes();
        let kc_off = kc_layout.start_offset() * dtype.size_in_bytes();
        let vc_off = vc_layout.start_offset() * dtype.size_in_bytes();
        let sm_off = sm_layout.start_offset() * slot_mapping.dtype().size_in_bytes();

        encoder.set_buffer(0, Some(k_buf), k_off);
        encoder.set_buffer(1, Some(v_buf), v_off);
        encoder.set_buffer(2, Some(kc_buf), kc_off);
        encoder.set_buffer(3, Some(vc_buf), vc_off);
        encoder.set_buffer(4, Some(sm_buf), sm_off);

        // Inline scalars
        set_bytes(&encoder, 5, &key_stride);
        set_bytes(&encoder, 6, &value_stride);
        set_bytes(&encoder, 7, &(num_heads as i32));
        set_bytes(&encoder, 8, &(head_size as i32));
        set_bytes(&encoder, 9, &block_size);
        set_bytes(&encoder, 10, &x);
    }

    let threads_per_tg = (num_heads * head_size).min(512);
    encoder.dispatch_thread_groups(
        MTLSize { width: num_tokens, height: 1, depth: 1 },
        MTLSize { width: threads_per_tg, height: 1, depth: 1 },
    );

    Ok(())
}

// ─── Copy blocks ─────────────────────────────────────────────────────────

/// Copy KV cache blocks for prefix sharing.
///
/// # Arguments
/// - `key_cache`: the full key cache tensor (mutated in place)
/// - `value_cache`: the full value cache tensor (mutated in place)
/// - `block_mapping`: flat i32 tensor of [src, dst] pairs
/// - `numel_per_block`: number of elements per block
pub fn copy_blocks(
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_mapping: &Tensor,
    numel_per_block: usize,
) -> Result<()> {
    let device = get_metal_device(key_cache)?;
    let dtype = key_cache.dtype();

    let kernel_name = match dtype {
        DType::F32 => "copy_blocks_float",
        DType::F16 => "copy_blocks_half",
        d => candle_core::bail!("copy_blocks: unsupported dtype {d:?}"),
    };

    let num_pairs = block_mapping.dims()[0] / 2;
    if num_pairs == 0 {
        return Ok(());
    }

    let pipeline = kernels().get_pipeline(
        device.metal_device(),
        COPY_BLOCKS_SOURCE,
        kernel_name,
    )?;

    let encoder = device.command_encoder()?;
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let (kc_store, kc_layout) = key_cache.storage_and_layout();
        let (vc_store, vc_layout) = value_cache.storage_and_layout();
        let (bm_store, bm_layout) = block_mapping.storage_and_layout();

        let kc_buf = match &*kc_store { candle_core::Storage::Metal(s) => s.buffer(), _ => candle_core::bail!("Metal") };
        let vc_buf = match &*vc_store { candle_core::Storage::Metal(s) => s.buffer(), _ => candle_core::bail!("Metal") };
        let bm_buf = match &*bm_store { candle_core::Storage::Metal(s) => s.buffer(), _ => candle_core::bail!("Metal") };

        let kc_off = kc_layout.start_offset() * dtype.size_in_bytes();
        let vc_off = vc_layout.start_offset() * dtype.size_in_bytes();
        let bm_off = bm_layout.start_offset() * block_mapping.dtype().size_in_bytes();

        encoder.set_buffer(0, Some(kc_buf), kc_off);
        encoder.set_buffer(1, Some(vc_buf), vc_off);
        encoder.set_buffer(2, Some(bm_buf), bm_off);
        set_bytes(&encoder, 3, &(numel_per_block as i32));
    }

    let threads_per_tg = numel_per_block.min(1024);
    encoder.dispatch_thread_groups(
        MTLSize { width: num_pairs, height: 1, depth: 1 },
        MTLSize { width: threads_per_tg, height: 1, depth: 1 },
    );

    Ok(())
}

// ─── Paged attention ─────────────────────────────────────────────────────

/// Configuration for a paged attention dispatch.
pub struct PagedAttentionConfig {
    pub head_size: HeadSize,
    pub num_kv_heads: usize,
    pub scale: f32,
    pub max_context_len: usize,
}

/// Run paged attention over batched queries and paged KV cache.
///
/// Automatically selects V1 (single partition) or V2 (multi-partition + reduce)
/// based on context length and batch size.
///
/// # Arguments
/// - `output`: [num_seqs, num_heads, head_size] — pre-allocated output tensor
/// - `query`: [num_seqs, num_heads, head_size] — query projections
/// - `key_cache`: paged key cache
/// - `value_cache`: paged value cache
/// - `block_tables`: [num_seqs, max_blocks_per_seq] i32 — block table
/// - `context_lens`: [num_seqs] i32 — context length per sequence
/// - `config`: attention configuration
pub fn paged_attention(
    output: &Tensor,
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    config: &PagedAttentionConfig,
) -> Result<()> {
    let device = get_metal_device(query)?;
    let dtype = query.dtype();
    let head_size = config.head_size.value();

    let q_dims = query.dims();
    let num_seqs = q_dims[0];
    let num_heads = q_dims[1];

    let max_blocks_per_seq = block_tables.dims()[1];

    // V1 vs V2 decision
    let max_num_partitions = (config.max_context_len + PARTITION_SIZE - 1) / PARTITION_SIZE;
    let use_v1 = max_num_partitions == 1
        || (num_seqs * num_heads > 512 && PARTITION_SIZE % BLOCK_SIZE == 0);

    let dtype_str = match dtype {
        DType::F32 => "float",
        DType::F16 => "half",
        d => candle_core::bail!("paged_attention: unsupported dtype {d:?}"),
    };

    if use_v1 {
        dispatch_paged_attention_v1(
            &device, output, query, key_cache, value_cache,
            block_tables, context_lens, config, dtype_str,
            num_seqs, num_heads, head_size, max_blocks_per_seq,
        )
    } else {
        dispatch_paged_attention_v2(
            &device, output, query, key_cache, value_cache,
            block_tables, context_lens, config, dtype_str,
            num_seqs, num_heads, head_size, max_blocks_per_seq,
            max_num_partitions,
        )
    }
}

fn dispatch_paged_attention_v1(
    device: &candle_core::MetalDevice,
    output: &Tensor,
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    config: &PagedAttentionConfig,
    dtype_str: &str,
    num_seqs: usize,
    num_heads: usize,
    head_size: usize,
    max_blocks_per_seq: usize,
) -> Result<()> {
    let dtype = query.dtype();
    let kernel_name = format!(
        "paged_attention_{dtype_str}_hs{head_size}_bs{BLOCK_SIZE}_ps0"
    );

    let pipeline = kernels().get_pipeline(
        device.metal_device(),
        PAGED_ATTENTION_SOURCE,
        &kernel_name,
    )?;

    let encoder = device.command_encoder()?;
    encoder.set_compute_pipeline_state(&pipeline);

    // Compute strides
    // K cache layout: [num_blocks, num_kv_heads, head_size/x, block_size, x]
    // kv_block_stride = stride[0] = num_kv_heads * (head_size/x) * block_size * x
    //                              = num_kv_heads * head_size * block_size
    // kv_head_stride  = stride[1] = (head_size/x) * block_size * x
    //                              = head_size * block_size
    let q_stride = (num_heads * head_size) as i32;
    let kv_block_stride = (config.num_kv_heads * head_size * BLOCK_SIZE) as i32;
    let kv_head_stride = (head_size * BLOCK_SIZE) as i32;

    // Shared memory: max(logits_size, output_scratch)
    let padded_context = ((config.max_context_len + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    let logits_size = padded_context * 4; // f32 per token
    let num_warps = NUM_THREADS / 32;
    let output_scratch = (num_warps / 2) * head_size * 4;
    let shared_mem_size = logits_size.max(output_scratch);

    {
        let (o_store, o_layout) = output.storage_and_layout();
        let (q_store, q_layout) = query.storage_and_layout();
        let (kc_store, kc_layout) = key_cache.storage_and_layout();
        let (vc_store, vc_layout) = value_cache.storage_and_layout();
        let (bt_store, bt_layout) = block_tables.storage_and_layout();
        let (cl_store, cl_layout) = context_lens.storage_and_layout();

        let o_buf = match &*o_store { candle_core::Storage::Metal(s) => s.buffer(), _ => candle_core::bail!("Metal") };
        let q_buf = match &*q_store { candle_core::Storage::Metal(s) => s.buffer(), _ => candle_core::bail!("Metal") };
        let kc_buf = match &*kc_store { candle_core::Storage::Metal(s) => s.buffer(), _ => candle_core::bail!("Metal") };
        let vc_buf = match &*vc_store { candle_core::Storage::Metal(s) => s.buffer(), _ => candle_core::bail!("Metal") };
        let bt_buf = match &*bt_store { candle_core::Storage::Metal(s) => s.buffer(), _ => candle_core::bail!("Metal") };
        let cl_buf = match &*cl_store { candle_core::Storage::Metal(s) => s.buffer(), _ => candle_core::bail!("Metal") };

        // buffer(0) = output
        encoder.set_buffer(0, Some(o_buf), o_layout.start_offset() * dtype.size_in_bytes());
        // buffer(1) = query
        encoder.set_buffer(1, Some(q_buf), q_layout.start_offset() * dtype.size_in_bytes());
        // buffer(2) = key_cache
        encoder.set_buffer(2, Some(kc_buf), kc_layout.start_offset() * dtype.size_in_bytes());
        // buffer(3) = value_cache
        encoder.set_buffer(3, Some(vc_buf), vc_layout.start_offset() * dtype.size_in_bytes());
        // buffer(4) = block_tables (u32)
        encoder.set_buffer(4, Some(bt_buf), bt_layout.start_offset() * block_tables.dtype().size_in_bytes());
        // buffer(5) = context_lens (u32)
        encoder.set_buffer(5, Some(cl_buf), cl_layout.start_offset() * context_lens.dtype().size_in_bytes());

        // Inline scalars
        set_bytes(&encoder, 6, &(config.num_kv_heads as i32));
        set_bytes(&encoder, 7, &config.scale);
        set_bytes(&encoder, 8, &(max_blocks_per_seq as i32));
        set_bytes(&encoder, 9, &q_stride);
        set_bytes(&encoder, 10, &kv_block_stride);
        set_bytes(&encoder, 11, &kv_head_stride);

        // V1: exp_sums and max_logits are not used but must be bound
        // Bind dummy (same as output, will be unused due to PARTITION_SIZE=0)
        encoder.set_buffer(12, Some(o_buf), 0);
        encoder.set_buffer(13, Some(o_buf), 0);
    }

    // Threadgroup memory
    encoder.set_threadgroup_memory_length(0, shared_mem_size);

    // Dispatch: (num_heads, num_seqs, 1)
    encoder.dispatch_thread_groups(
        MTLSize { width: num_heads, height: num_seqs, depth: 1 },
        MTLSize { width: NUM_THREADS, height: 1, depth: 1 },
    );

    Ok(())
}

fn dispatch_paged_attention_v2(
    device: &candle_core::MetalDevice,
    output: &Tensor,
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    config: &PagedAttentionConfig,
    dtype_str: &str,
    num_seqs: usize,
    num_heads: usize,
    head_size: usize,
    max_blocks_per_seq: usize,
    max_num_partitions: usize,
) -> Result<()> {
    let dtype = query.dtype();

    // Allocate temporary buffers for V2
    let exp_sums = Tensor::zeros(
        (num_seqs, num_heads, max_num_partitions),
        DType::F32,
        query.device(),
    )?;
    let max_logits = Tensor::zeros(
        (num_seqs, num_heads, max_num_partitions),
        DType::F32,
        query.device(),
    )?;
    let tmp_output = Tensor::zeros(
        (num_seqs, num_heads, max_num_partitions, head_size),
        dtype,
        query.device(),
    )?;

    // Pass 1: partitioned attention
    let kernel_name = format!(
        "paged_attention_{dtype_str}_hs{head_size}_bs{BLOCK_SIZE}_ps{PARTITION_SIZE}"
    );

    let pipeline = kernels().get_pipeline(
        device.metal_device(),
        PAGED_ATTENTION_SOURCE,
        &kernel_name,
    )?;

    let encoder = device.command_encoder()?;
    encoder.set_compute_pipeline_state(&pipeline);

    let q_stride = (num_heads * head_size) as i32;
    let kv_block_stride = (config.num_kv_heads * head_size * BLOCK_SIZE) as i32;
    let kv_head_stride = (head_size * BLOCK_SIZE) as i32;

    let padded_context = ((config.max_context_len + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    let partition_context = PARTITION_SIZE.min(padded_context);
    let logits_size = partition_context * 4;
    let num_warps = NUM_THREADS / 32;
    let output_scratch = (num_warps / 2) * head_size * 4;
    let shared_mem_size = logits_size.max(output_scratch);

    {
        let (q_store, q_layout) = query.storage_and_layout();
        let (kc_store, kc_layout) = key_cache.storage_and_layout();
        let (vc_store, vc_layout) = value_cache.storage_and_layout();
        let (bt_store, bt_layout) = block_tables.storage_and_layout();
        let (cl_store, cl_layout) = context_lens.storage_and_layout();
        let (to_store, to_layout) = tmp_output.storage_and_layout();
        let (es_store, es_layout) = exp_sums.storage_and_layout();
        let (ml_store, ml_layout) = max_logits.storage_and_layout();

        let q_buf = match &*q_store { candle_core::Storage::Metal(s) => s.buffer(), _ => candle_core::bail!("Metal") };
        let kc_buf = match &*kc_store { candle_core::Storage::Metal(s) => s.buffer(), _ => candle_core::bail!("Metal") };
        let vc_buf = match &*vc_store { candle_core::Storage::Metal(s) => s.buffer(), _ => candle_core::bail!("Metal") };
        let bt_buf = match &*bt_store { candle_core::Storage::Metal(s) => s.buffer(), _ => candle_core::bail!("Metal") };
        let cl_buf = match &*cl_store { candle_core::Storage::Metal(s) => s.buffer(), _ => candle_core::bail!("Metal") };
        let to_buf = match &*to_store { candle_core::Storage::Metal(s) => s.buffer(), _ => candle_core::bail!("Metal") };
        let es_buf = match &*es_store { candle_core::Storage::Metal(s) => s.buffer(), _ => candle_core::bail!("Metal") };
        let ml_buf = match &*ml_store { candle_core::Storage::Metal(s) => s.buffer(), _ => candle_core::bail!("Metal") };

        // V2 pass 1 writes to tmp_output, with exp_sums and max_logits
        encoder.set_buffer(0, Some(to_buf), to_layout.start_offset() * dtype.size_in_bytes());
        encoder.set_buffer(1, Some(q_buf), q_layout.start_offset() * dtype.size_in_bytes());
        encoder.set_buffer(2, Some(kc_buf), kc_layout.start_offset() * dtype.size_in_bytes());
        encoder.set_buffer(3, Some(vc_buf), vc_layout.start_offset() * dtype.size_in_bytes());
        encoder.set_buffer(4, Some(bt_buf), bt_layout.start_offset() * block_tables.dtype().size_in_bytes());
        encoder.set_buffer(5, Some(cl_buf), cl_layout.start_offset() * context_lens.dtype().size_in_bytes());
        set_bytes(&encoder, 6, &(config.num_kv_heads as i32));
        set_bytes(&encoder, 7, &config.scale);
        set_bytes(&encoder, 8, &(max_blocks_per_seq as i32));
        set_bytes(&encoder, 9, &q_stride);
        set_bytes(&encoder, 10, &kv_block_stride);
        set_bytes(&encoder, 11, &kv_head_stride);
        encoder.set_buffer(12, Some(es_buf), es_layout.start_offset() * exp_sums.dtype().size_in_bytes());
        encoder.set_buffer(13, Some(ml_buf), ml_layout.start_offset() * max_logits.dtype().size_in_bytes());
    }

    encoder.set_threadgroup_memory_length(0, shared_mem_size);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: num_heads,
            height: num_seqs,
            depth: max_num_partitions,
        },
        MTLSize { width: NUM_THREADS, height: 1, depth: 1 },
    );

    // Pass 2: reduce partitions
    let reduce_name = format!(
        "paged_attention_v2_reduce_{dtype_str}_hs{head_size}"
    );
    let reduce_pipeline = kernels().get_pipeline(
        device.metal_device(),
        PAGED_ATTENTION_SOURCE,
        &reduce_name,
    )?;

    let encoder2 = device.command_encoder()?;
    encoder2.set_compute_pipeline_state(&reduce_pipeline);

    {
        let (o_store, o_layout) = output.storage_and_layout();
        let (es_store, _) = exp_sums.storage_and_layout();
        let (ml_store, _) = max_logits.storage_and_layout();
        let (to_store, _) = tmp_output.storage_and_layout();
        let (cl_store, cl_layout) = context_lens.storage_and_layout();

        let o_buf = match &*o_store { candle_core::Storage::Metal(s) => s.buffer(), _ => candle_core::bail!("Metal") };
        let es_buf = match &*es_store { candle_core::Storage::Metal(s) => s.buffer(), _ => candle_core::bail!("Metal") };
        let ml_buf = match &*ml_store { candle_core::Storage::Metal(s) => s.buffer(), _ => candle_core::bail!("Metal") };
        let to_buf = match &*to_store { candle_core::Storage::Metal(s) => s.buffer(), _ => candle_core::bail!("Metal") };
        let cl_buf = match &*cl_store { candle_core::Storage::Metal(s) => s.buffer(), _ => candle_core::bail!("Metal") };

        encoder2.set_buffer(0, Some(o_buf), o_layout.start_offset() * dtype.size_in_bytes());
        encoder2.set_buffer(1, Some(es_buf), 0);
        encoder2.set_buffer(2, Some(ml_buf), 0);
        encoder2.set_buffer(3, Some(to_buf), 0);
        encoder2.set_buffer(4, Some(cl_buf), cl_layout.start_offset() * context_lens.dtype().size_in_bytes());
        set_bytes(&encoder2, 5, &(max_num_partitions as i32));
    }

    let reduce_shared = 2 * max_num_partitions * 4;
    encoder2.set_threadgroup_memory_length(0, reduce_shared);
    encoder2.dispatch_thread_groups(
        MTLSize { width: num_heads, height: num_seqs, depth: 1 },
        MTLSize { width: NUM_THREADS, height: 1, depth: 1 },
    );

    Ok(())
}

// ─── Utility ─────────────────────────────────────────────────────────────

/// Set inline bytes on a compute encoder at the given buffer index.
fn set_bytes<T>(encoder: &ComputeCommandEncoder, index: usize, data: &T) {
    encoder.set_bytes(index, data);
}

// ─── Allocate KV cache tensors ───────────────────────────────────────────

/// Allocate the KV cache tensors for the paged attention engine.
///
/// Returns `(key_caches, value_caches)` — one pair per layer.
///
/// K cache layout: [num_blocks, num_kv_heads, head_size/x, block_size, x]
/// V cache layout: [num_blocks, num_kv_heads, head_size, block_size]
pub fn allocate_kv_caches(
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
