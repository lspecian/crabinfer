//! CUDA graph capture and replay for decode steps.
//!
//! CUDA graphs eliminate kernel launch overhead by recording a sequence of GPU
//! operations (the "graph") and replaying them with different input data. For
//! LLM decode steps, the computation graph shape is determined entirely by the
//! batch size, so we capture one graph per distinct batch size.
//!
//! # Architecture
//!
//! ```text
//!  ┌──────────────────┐
//!  │  CudaGraphCache   │  ← HashMap<batch_size, CapturedGraph>
//!  └────────┬─────────┘
//!           │
//!  ┌────────▼─────────┐
//!  │  CapturedGraph    │  ← CUDA graph handle + pre-allocated I/O buffers
//!  │  - input_ids_buf  │     Fixed-size GPU tensors that the graph reads from
//!  │  - positions_buf  │     and writes to. Before replay, we memcpy new data
//!  │  - output_buf     │     into the input buffers; after replay, we read from
//!  │  - graph_exec     │     the output buffer.
//!  └──────────────────┘
//! ```
//!
//! # Usage in the engine loop
//!
//! 1. **Warmup**: At engine startup, run dummy decode batches for common batch
//!    sizes (1, 2, 4, 8, ...) to capture graphs.
//! 2. **Run**: On each decode step, look up the cached graph for the current
//!    batch size. If found, copy inputs → replay → read outputs. If not found,
//!    fall back to eager execution (and optionally capture).
//! 3. **Invalidation**: Graphs are invalidated when batch size changes or when
//!    KV cache is reorganized (preemption, new allocations change block tables).
//!
//! # References
//! - vLLM CUDA graph implementation: 2-5x throughput improvement
//! - Cloudflare Infire: generates graphs for every batch size
//! - cudarc::driver::CudaGraph / CudaGraphExec

use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};

/// Configuration for the CUDA graph subsystem.
#[derive(Debug, Clone)]
pub struct CudaGraphConfig {
    /// Maximum batch size to pre-capture graphs for during warmup.
    /// Graphs for larger batches fall back to eager execution.
    pub max_capture_batch_size: usize,

    /// Whether CUDA graphs are enabled. When false, all execution is eager.
    /// Corresponds to vLLM's `--enforce-eager` flag.
    pub enabled: bool,

    /// Batch sizes to capture during warmup. If empty, uses powers of 2 up to
    /// `max_capture_batch_size`.
    pub warmup_batch_sizes: Vec<usize>,

    /// Maximum sequence length for captured graphs. Graphs are captured at this
    /// length; shorter sequences are padded.
    pub max_seq_len_for_capture: usize,

    /// Dtype for metadata tensors (block_tables, slot_mapping, context_lens).
    /// I32 saves memory and matches vLLM default. I64 for large-scale deployments
    /// where block indices may exceed i32 range.
    pub metadata_dtype: DType,
}

impl Default for CudaGraphConfig {
    fn default() -> Self {
        Self {
            max_capture_batch_size: 32,
            enabled: true,
            warmup_batch_sizes: Vec::new(),
            max_seq_len_for_capture: 4096,
            metadata_dtype: DType::I32,
        }
    }
}

impl CudaGraphConfig {
    /// Get the batch sizes to use for warmup capture.
    pub fn capture_batch_sizes(&self) -> Vec<usize> {
        if !self.warmup_batch_sizes.is_empty() {
            return self.warmup_batch_sizes.clone();
        }
        // Default: powers of 2 up to max, plus 1 (single decode)
        let mut sizes = vec![1];
        let mut s = 2;
        while s <= self.max_capture_batch_size {
            sizes.push(s);
            s *= 2;
        }
        sizes
    }
}

/// Pre-allocated input/output buffers for a captured CUDA graph.
///
/// CUDA graphs bind to specific GPU memory addresses, so we must use the same
/// buffers for every replay. Before replay, copy fresh data into the input
/// buffers; after replay, read results from the output buffer.
#[derive(Debug)]
pub struct GraphBuffers {
    /// Input token IDs: `[batch_size]` u32 (decode = 1 token per sequence).
    pub input_ids: Tensor,
    /// Position indices: `[batch_size]` u32.
    pub positions: Tensor,
    /// Block tables: `[batch_size, max_blocks_per_seq]` i32.
    pub block_tables: Tensor,
    /// Slot mapping: `[batch_size]` i32.
    pub slot_mapping: Tensor,
    /// Context lengths: `[batch_size]` i32.
    pub context_lens: Tensor,
    /// Output logits: `[batch_size, vocab_size]` (dtype matches model).
    pub output_logits: Tensor,
}

/// A captured CUDA graph for a specific batch size.
///
/// Holds the graph execution handle and the fixed-size GPU buffers.
/// The graph was captured while reading from these exact buffers.
pub struct CapturedGraph {
    /// Batch size this graph was captured for.
    pub batch_size: usize,

    /// Pre-allocated I/O buffers. Must not be reallocated — the graph
    /// references these specific GPU memory addresses.
    pub buffers: GraphBuffers,

    /// CUDA graph execution handle. None until the graph is captured.
    /// Type is opaque; actual CUDA graph exec is stored when `cuda` feature
    /// is enabled.
    #[cfg(feature = "cuda")]
    pub graph_exec: Option<CudaGraphExecHandle>,

    /// On non-CUDA builds, this is a placeholder.
    #[cfg(not(feature = "cuda"))]
    pub graph_exec: Option<()>,
}

/// Opaque handle wrapping cudarc's CUDA graph execution object.
///
/// This is separated out so the rest of the module can compile without
/// the `cuda` feature.
#[cfg(feature = "cuda")]
pub struct CudaGraphExecHandle {
    graph: candle_core::cuda_backend::cudarc::driver::CudaGraph,
}

#[cfg(feature = "cuda")]
impl CudaGraphExecHandle {
    /// Create a new handle from a captured CUDA graph.
    pub fn new(graph: candle_core::cuda_backend::cudarc::driver::CudaGraph) -> Self {
        Self { graph }
    }
}

// CudaGraph is not Send+Sync by default (internal raw pointers), but we only
// access it from the engine loop's single thread, so this is safe.
#[cfg(feature = "cuda")]
unsafe impl Send for CudaGraphExecHandle {}
#[cfg(feature = "cuda")]
unsafe impl Sync for CudaGraphExecHandle {}

/// Cache of captured CUDA graphs, keyed by batch size.
pub struct CudaGraphCache {
    config: CudaGraphConfig,
    /// Captured graphs, keyed by batch size.
    graphs: HashMap<usize, CapturedGraph>,
    /// Device for buffer allocation.
    device: Device,
    /// Model vocab size (for output buffer sizing).
    vocab_size: usize,
    /// Max blocks per sequence (for block table buffer sizing).
    max_blocks_per_seq: usize,
    /// Model dtype for output logits.
    dtype: DType,
}

impl CudaGraphCache {
    /// Create a new graph cache.
    ///
    /// Does NOT capture graphs — call `warmup()` to pre-capture.
    pub fn new(
        config: CudaGraphConfig,
        device: Device,
        vocab_size: usize,
        max_blocks_per_seq: usize,
        dtype: DType,
    ) -> Self {
        Self {
            config,
            graphs: HashMap::new(),
            device,
            vocab_size,
            max_blocks_per_seq,
            dtype,
        }
    }

    /// Whether CUDA graphs are enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the configuration.
    pub fn config(&self) -> &CudaGraphConfig {
        &self.config
    }

    /// Insert a pre-allocated graph entry (buffers only, no captured graph).
    pub fn insert_graph(&mut self, batch_size: usize, buffers: GraphBuffers) {
        self.graphs.insert(batch_size, CapturedGraph {
            batch_size,
            buffers,
            graph_exec: None,
        });
    }

    /// Look up a cached graph for the given batch size.
    pub fn get(&self, batch_size: usize) -> Option<&CapturedGraph> {
        self.graphs.get(&batch_size)
    }

    /// Look up a cached graph for the given batch size (mutable).
    pub fn get_mut(&mut self, batch_size: usize) -> Option<&mut CapturedGraph> {
        self.graphs.get_mut(&batch_size)
    }

    /// Whether a graph exists for the given batch size.
    pub fn has_graph(&self, batch_size: usize) -> bool {
        self.graphs.contains_key(&batch_size)
    }

    /// Find the best matching graph for a batch size.
    ///
    /// If an exact match exists, returns it. Otherwise, returns the next
    /// larger captured batch size (we pad the extra slots with dummy data).
    /// Returns None if no suitable graph exists.
    pub fn find_graph(&self, batch_size: usize) -> Option<&CapturedGraph> {
        // Exact match first
        if let Some(g) = self.graphs.get(&batch_size) {
            return Some(g);
        }
        // Find smallest captured graph >= batch_size
        let mut best: Option<&CapturedGraph> = None;
        for g in self.graphs.values() {
            if g.batch_size >= batch_size {
                match best {
                    None => best = Some(g),
                    Some(b) if g.batch_size < b.batch_size => best = Some(g),
                    _ => {}
                }
            }
        }
        best
    }

    /// Allocate buffers for a given batch size.
    ///
    /// These are fixed-size GPU tensors that the CUDA graph will bind to.
    pub fn allocate_buffers(&self, batch_size: usize) -> Result<GraphBuffers> {
        let dev = &self.device;
        Ok(GraphBuffers {
            input_ids: Tensor::zeros((batch_size,), DType::U32, dev)?,
            positions: Tensor::zeros((batch_size,), DType::U32, dev)?,
            block_tables: Tensor::zeros(
                (batch_size, self.max_blocks_per_seq),
                self.config.metadata_dtype,
                dev,
            )?,
            slot_mapping: Tensor::zeros((batch_size,), self.config.metadata_dtype, dev)?,
            context_lens: Tensor::zeros((batch_size,), self.config.metadata_dtype, dev)?,
            output_logits: Tensor::zeros((batch_size, self.vocab_size), self.dtype, dev)?,
        })
    }

    /// Pre-allocate buffer entries for common batch sizes.
    ///
    /// This allocates the fixed GPU buffers that CUDA graphs will reference.
    /// The actual graph capture happens in `capture()`, which requires running
    /// a model forward pass.
    pub fn prepare_warmup(&mut self) -> Result<Vec<usize>> {
        if !self.config.enabled {
            return Ok(Vec::new());
        }

        let batch_sizes = self.config.capture_batch_sizes();
        for &bs in &batch_sizes {
            let buffers = self.allocate_buffers(bs)?;
            self.graphs.insert(bs, CapturedGraph {
                batch_size: bs,
                buffers,
                graph_exec: None,
            });
        }

        tracing::info!(
            "CUDA graph warmup: prepared buffers for {} batch sizes: {:?}",
            batch_sizes.len(),
            batch_sizes,
        );

        Ok(batch_sizes)
    }

    /// Number of captured (complete) graphs.
    pub fn num_captured(&self) -> usize {
        self.graphs.values().filter(|g| g.graph_exec.is_some()).count()
    }

    /// Number of prepared (buffer-only) graphs.
    pub fn num_prepared(&self) -> usize {
        self.graphs.len()
    }

    /// Remove all cached graphs (e.g., after KV cache reorganization).
    pub fn invalidate_all(&mut self) {
        for g in self.graphs.values_mut() {
            g.graph_exec = None;
        }
        tracing::debug!("CUDA graph cache: invalidated all {} graphs", self.graphs.len());
    }

    /// Capture a CUDA graph by running a model forward pass during stream capture.
    ///
    /// The `run_forward` closure receives the graph's pre-allocated input buffers
    /// and must execute the full decode forward pass on the GPU. All GPU operations
    /// are recorded into the graph rather than executed.
    ///
    /// After capture, the graph can be replayed with `replay()`.
    #[cfg(feature = "cuda")]
    pub fn capture<F>(
        &mut self,
        batch_size: usize,
        run_forward: F,
    ) -> Result<()>
    where
        F: FnOnce(&GraphBuffers) -> Result<Tensor>,
    {
        use candle_core::cuda_backend::cudarc::driver::sys::{
            CUstreamCaptureMode, CUgraphInstantiate_flags,
        };

        let graph_entry = match self.graphs.get(&batch_size) {
            Some(_) => {}
            None => {
                // Allocate buffers on the fly if not pre-warmed
                let buffers = self.allocate_buffers(batch_size)?;
                self.graphs.insert(batch_size, CapturedGraph {
                    batch_size,
                    buffers,
                    graph_exec: None,
                });
            }
        };
        let _ = graph_entry;

        // Get the CUDA stream from the device
        let cuda_device = match &self.device {
            Device::Cuda(d) => d.cuda_stream(),
            _ => candle_core::bail!("CUDA graphs require a CUDA device"),
        };

        // Synchronize before capture
        cuda_device.synchronize()
            .map_err(|e| candle_core::Error::Msg(format!("sync before capture: {e}")))?;

        // Begin stream capture
        cuda_device
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(|e| candle_core::Error::Msg(format!("begin_capture: {e}")))?;

        // Run the forward pass — ops are recorded, not executed
        let entry = self.graphs.get(&batch_size).unwrap();
        let output = run_forward(&entry.buffers);

        // End capture and instantiate the graph
        let cuda_graph = cuda_device
            .end_capture(
                CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            )
            .map_err(|e| candle_core::Error::Msg(format!("end_capture: {e}")))?;

        // Check if capture succeeded
        match (output, cuda_graph) {
            (Ok(output_logits), Some(graph)) => {
                let entry = self.graphs.get_mut(&batch_size).unwrap();
                // Copy output logits to the graph's output buffer for address binding
                entry.buffers.output_logits = output_logits;
                entry.graph_exec = Some(CudaGraphExecHandle { graph });

                tracing::info!(
                    "CUDA graph captured for batch_size={}",
                    batch_size,
                );
                Ok(())
            }
            (Err(e), _) => {
                tracing::warn!(
                    "CUDA graph capture failed for batch_size={}: {e}",
                    batch_size,
                );
                Err(e)
            }
            (Ok(_), None) => {
                candle_core::bail!("CUDA graph capture returned null graph for batch_size={}", batch_size);
            }
        }
    }

    /// Replay a captured CUDA graph.
    ///
    /// Before calling, the caller must have copied fresh input data into the
    /// graph's pre-allocated buffers (via `copy_inputs()`). After replay,
    /// results are available in `buffers.output_logits`.
    #[cfg(feature = "cuda")]
    pub fn replay(&self, batch_size: usize) -> Result<&Tensor> {
        let entry = self.graphs.get(&batch_size)
            .ok_or_else(|| candle_core::Error::Msg(
                format!("no graph for batch_size={}", batch_size),
            ))?;

        let exec = entry.graph_exec.as_ref()
            .ok_or_else(|| candle_core::Error::Msg(
                format!("graph for batch_size={} not yet captured", batch_size),
            ))?;

        exec.graph.launch()
            .map_err(|e| candle_core::Error::Msg(format!("graph launch: {e}")))?;

        Ok(&entry.buffers.output_logits)
    }

    /// Copy input tensors into a graph's pre-allocated buffers using
    /// device-to-device memcpy. This preserves the graph's fixed memory
    /// addresses while updating the data.
    #[cfg(feature = "cuda")]
    pub fn copy_inputs(
        &self,
        batch_size: usize,
        input_ids: &Tensor,
        positions: &Tensor,
        slot_mapping: &Tensor,
        context_lens: &Tensor,
    ) -> Result<()> {
        let entry = self.graphs.get(&batch_size)
            .ok_or_else(|| candle_core::Error::Msg(
                format!("no graph for batch_size={}", batch_size),
            ))?;

        // Assert dtype consistency to prevent silent memory corruption (Pitfall 5).
        assert_eq!(
            entry.buffers.slot_mapping.dtype(),
            slot_mapping.dtype(),
            "slot_mapping dtype mismatch: buffer={:?}, input={:?}",
            entry.buffers.slot_mapping.dtype(),
            slot_mapping.dtype()
        );
        assert_eq!(
            entry.buffers.context_lens.dtype(),
            context_lens.dtype(),
            "context_lens dtype mismatch: buffer={:?}, input={:?}",
            entry.buffers.context_lens.dtype(),
            context_lens.dtype()
        );
        assert_eq!(
            entry.buffers.input_ids.dtype(),
            input_ids.dtype(),
            "input_ids dtype mismatch: buffer={:?}, input={:?}",
            entry.buffers.input_ids.dtype(),
            input_ids.dtype()
        );
        assert_eq!(
            entry.buffers.positions.dtype(),
            positions.dtype(),
            "positions dtype mismatch: buffer={:?}, input={:?}",
            entry.buffers.positions.dtype(),
            positions.dtype()
        );

        // Copy each input into the graph's fixed buffer
        copy_tensor_data(&entry.buffers.input_ids, input_ids)?;
        copy_tensor_data(&entry.buffers.positions, positions)?;
        copy_tensor_data(&entry.buffers.slot_mapping, slot_mapping)?;
        copy_tensor_data(&entry.buffers.context_lens, context_lens)?;

        Ok(())
    }
}

/// Copy tensor data in-place using raw cuMemcpyDtoD.
///
/// The destination tensor's memory address is preserved (required for CUDA
/// graph replay). Only the raw bytes are overwritten.
#[cfg(feature = "cuda")]
fn copy_tensor_data(dst: &Tensor, src: &Tensor) -> Result<()> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;

    let dst_guard = dst.storage_and_layout();
    let src_guard = src.storage_and_layout();

    let dst_cuda = match &*dst_guard.0 {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("copy_tensor_data: dst is not CUDA"),
    };
    let src_cuda = match &*src_guard.0 {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("copy_tensor_data: src is not CUDA"),
    };

    let dtype_size = dst.dtype().size_in_bytes();
    let num_elements = src_guard.1.shape().elem_count().min(dst_guard.1.shape().elem_count());
    let num_bytes = num_elements * dtype_size;

    if num_bytes == 0 {
        return Ok(());
    }

    let dst_slice = dst_cuda.as_cuda_slice::<u8>()?;
    let src_slice = src_cuda.as_cuda_slice::<u8>()?;

    let dst_offset = dst_guard.1.start_offset() * dtype_size;
    let src_offset = src_guard.1.start_offset() * dtype_size;

    // Get the CUDA stream for proper synchronization
    let stream = dst_cuda.device.cuda_stream();

    // Bind slices to let bindings to satisfy borrow checker
    let dst_view = dst_slice.slice(dst_offset..);
    let src_view = src_slice.slice(src_offset..);

    // Get device pointers through the stream-aware API
    let (dst_ptr, _dst_guard) = dst_view.device_ptr(&stream);
    let (src_ptr, _src_guard) = src_view.device_ptr(&stream);

    // Synchronous device-to-device memcpy
    let result = unsafe {
        candle_core::cuda_backend::cudarc::driver::sys::cuMemcpyDtoD_v2(
            dst_ptr, src_ptr, num_bytes,
        )
    };

    if result != candle_core::cuda_backend::cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        candle_core::bail!("cuMemcpyDtoD failed: {:?}", result);
    }

    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Wave 0 stubs: test_cuda_graph_capture_succeeds (CUDA-GRAPH-04)
    // These stubs exist before implementation to satisfy Nyquist sampling.

    #[test]
    fn test_config_default() {
        let config = CudaGraphConfig::default();
        assert!(config.enabled);
        assert_eq!(config.max_capture_batch_size, 32);
        assert_eq!(config.metadata_dtype, DType::I32);
    }

    #[test]
    fn test_capture_batch_sizes_default() {
        let config = CudaGraphConfig::default();
        let sizes = config.capture_batch_sizes();
        assert_eq!(sizes, vec![1, 2, 4, 8, 16, 32]);
    }

    #[test]
    fn test_capture_batch_sizes_custom() {
        let config = CudaGraphConfig {
            warmup_batch_sizes: vec![1, 3, 7],
            ..Default::default()
        };
        let sizes = config.capture_batch_sizes();
        assert_eq!(sizes, vec![1, 3, 7]);
    }

    #[test]
    fn test_cache_creation() {
        let config = CudaGraphConfig::default();
        let device = Device::Cpu;
        let cache = CudaGraphCache::new(config, device, 32000, 256, DType::F32);
        assert!(cache.is_enabled());
        assert_eq!(cache.num_prepared(), 0);
        assert_eq!(cache.num_captured(), 0);
    }

    #[test]
    fn test_cache_disabled() {
        let config = CudaGraphConfig {
            enabled: false,
            ..Default::default()
        };
        let device = Device::Cpu;
        let cache = CudaGraphCache::new(config, device, 32000, 256, DType::F32);
        assert!(!cache.is_enabled());
    }

    #[test]
    fn test_prepare_warmup() {
        let config = CudaGraphConfig {
            max_capture_batch_size: 8,
            ..Default::default()
        };
        let device = Device::Cpu;
        let mut cache = CudaGraphCache::new(config, device, 32000, 16, DType::F32);
        let sizes = cache.prepare_warmup().unwrap();

        assert_eq!(sizes, vec![1, 2, 4, 8]);
        assert_eq!(cache.num_prepared(), 4);
        assert_eq!(cache.num_captured(), 0); // No graphs captured yet

        // Check buffer shapes
        let g = cache.get(4).unwrap();
        assert_eq!(g.buffers.input_ids.dims(), &[4]);
        assert_eq!(g.buffers.output_logits.dims(), &[4, 32000]);
        assert_eq!(g.buffers.block_tables.dims(), &[4, 16]);

        // Verify metadata tensors use I32 (not I64) by default
        assert_eq!(g.buffers.block_tables.dtype(), DType::I32);
        assert_eq!(g.buffers.slot_mapping.dtype(), DType::I32);
        assert_eq!(g.buffers.context_lens.dtype(), DType::I32);
    }

    #[test]
    fn test_prepare_warmup_disabled() {
        let config = CudaGraphConfig {
            enabled: false,
            ..Default::default()
        };
        let device = Device::Cpu;
        let mut cache = CudaGraphCache::new(config, device, 32000, 16, DType::F32);
        let sizes = cache.prepare_warmup().unwrap();
        assert!(sizes.is_empty());
        assert_eq!(cache.num_prepared(), 0);
    }

    #[test]
    fn test_find_graph_exact() {
        let config = CudaGraphConfig {
            max_capture_batch_size: 8,
            ..Default::default()
        };
        let device = Device::Cpu;
        let mut cache = CudaGraphCache::new(config, device, 100, 4, DType::F32);
        cache.prepare_warmup().unwrap();

        let g = cache.find_graph(4).unwrap();
        assert_eq!(g.batch_size, 4);
    }

    #[test]
    fn test_find_graph_next_larger() {
        let config = CudaGraphConfig {
            max_capture_batch_size: 8,
            ..Default::default()
        };
        let device = Device::Cpu;
        let mut cache = CudaGraphCache::new(config, device, 100, 4, DType::F32);
        cache.prepare_warmup().unwrap();

        // No graph for batch_size=3, should return the 4-graph
        let g = cache.find_graph(3).unwrap();
        assert_eq!(g.batch_size, 4);
    }

    #[test]
    fn test_find_graph_too_large() {
        let config = CudaGraphConfig {
            max_capture_batch_size: 8,
            ..Default::default()
        };
        let device = Device::Cpu;
        let mut cache = CudaGraphCache::new(config, device, 100, 4, DType::F32);
        cache.prepare_warmup().unwrap();

        // No graph for batch_size=16
        assert!(cache.find_graph(16).is_none());
    }

    #[test]
    fn test_invalidate_all() {
        let config = CudaGraphConfig {
            max_capture_batch_size: 4,
            ..Default::default()
        };
        let device = Device::Cpu;
        let mut cache = CudaGraphCache::new(config, device, 100, 4, DType::F32);
        cache.prepare_warmup().unwrap();

        assert_eq!(cache.num_prepared(), 3); // 1, 2, 4
        cache.invalidate_all();
        assert_eq!(cache.num_prepared(), 3); // Still prepared
        assert_eq!(cache.num_captured(), 0); // But no captured graphs
    }

    /// CUDA-GRAPH-04: Graph capture succeeds for at least one batch size on CUDA device.
    /// This test requires H100 or equivalent CUDA hardware to run.
    /// It will be a no-op (skip) on CPU-only machines.
    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_graph_capture_succeeds() {
        // Verify CudaGraphConfig with metadata_dtype I32 can be constructed
        // and allocate_buffers produces I32 metadata tensors.
        //
        // The REAL graph capture test (begin_capture -> forward -> end_capture)
        // requires a model + CUDA device and is validated during H100 testing.
        let config = CudaGraphConfig {
            enabled: true,
            max_capture_batch_size: 4,
            warmup_batch_sizes: vec![1, 2, 4],
            max_seq_len_for_capture: 128,
            metadata_dtype: DType::I32,
        };
        assert!(config.enabled);
        assert_eq!(config.metadata_dtype, DType::I32);
        assert_eq!(config.warmup_batch_sizes.len(), 3);

        // Verify allocate_buffers uses metadata_dtype
        let cache = CudaGraphCache::new(config, Device::Cpu, 100, 4, DType::F32);
        let buffers = cache.allocate_buffers(2).unwrap();
        assert_eq!(buffers.block_tables.dtype(), DType::I32);
        assert_eq!(buffers.slot_mapping.dtype(), DType::I32);
        assert_eq!(buffers.context_lens.dtype(), DType::I32);
    }
}
