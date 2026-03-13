//! Arena allocator and tensor buffer pool for zero-allocation inference.
//!
//! - [`TensorArena`]: Bump-pointer arena for CPU-side temporary byte buffers
//!   used when building batch inputs (token IDs, positions, slot mappings).
//! - [`TensorBufferPool`]: Pre-allocates device-side tensors at engine init
//!   and recycles them across forward passes via checkout/checkin.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};

// ─── TensorArena ─────────────────────────────────────────────────────────

/// Bump-pointer arena for per-step scratch memory.
///
/// Allocates contiguous byte slices from a pre-allocated buffer. Reset is O(1)
/// (just resets the offset). Used for gathering token IDs, positions, and slot
/// mappings into contiguous arrays before creating tensors.
pub struct TensorArena {
    buffer: Vec<u8>,
    offset: usize,
}

impl TensorArena {
    /// Create a new arena with the given capacity in bytes.
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            buffer: vec![0u8; capacity_bytes],
            offset: 0,
        }
    }

    /// Allocate `size` bytes from the arena. Returns a mutable slice.
    /// Returns Err if remaining capacity is insufficient.
    pub fn allocate(&mut self, size: usize) -> candle_core::Result<&mut [u8]> {
        if size > self.remaining() {
            return Err(candle_core::Error::Msg(format!(
                "TensorArena OOM: requested {} bytes but only {} remaining",
                size,
                self.remaining()
            )));
        }
        let start = self.offset;
        self.offset += size;
        Ok(&mut self.buffer[start..self.offset])
    }

    /// Reset the arena for the next step. O(1) -- just resets the offset.
    pub fn reset(&mut self) {
        self.offset = 0;
    }

    /// Remaining bytes available.
    pub fn remaining(&self) -> usize {
        self.buffer.len() - self.offset
    }

    /// Current offset (bytes used).
    pub fn used(&self) -> usize {
        self.offset
    }

    /// Convenience: allocate a typed slice of `count` elements.
    ///
    /// The returned slice is aligned to `align_of::<T>()`. The arena bumps
    /// its offset past any alignment padding automatically.
    pub fn allocate_slice<T: Copy>(&mut self, count: usize) -> candle_core::Result<&mut [T]> {
        let align = std::mem::align_of::<T>();
        // Round offset up to alignment boundary
        let aligned_offset = (self.offset + align - 1) & !(align - 1);
        let size = count * std::mem::size_of::<T>();
        let end = aligned_offset + size;
        if end > self.buffer.len() {
            return Err(candle_core::Error::Msg(format!(
                "TensorArena OOM: requested {} bytes (aligned) but only {} remaining",
                size,
                self.buffer.len() - aligned_offset
            )));
        }
        self.offset = end;
        let ptr = self.buffer[aligned_offset..end].as_mut_ptr() as *mut T;
        // Safety: buffer is contiguous, properly aligned after rounding,
        // and lifetime is tied to &mut self.
        Ok(unsafe { std::slice::from_raw_parts_mut(ptr, count) })
    }
}

// ─── TensorBufferPool ────────────────────────────────────────────────────

/// Key for buffer pool lookup: (shape, dtype).
type PoolKey = (Vec<usize>, DType);

/// Pre-allocated pool of reusable tensors keyed by (shape, dtype).
///
/// At engine init, tensors are allocated for known batch-input shapes.
/// During inference, tensors are checked out (popped from the free list)
/// and checked back in after the forward pass completes.
pub struct TensorBufferPool {
    /// Free tensors available for checkout, keyed by (shape, dtype).
    free: HashMap<PoolKey, Vec<Tensor>>,
    device: Device,
}

impl TensorBufferPool {
    /// Create a pool and pre-allocate `count` tensors for each (shape, dtype) pair.
    pub fn new(
        specs: &[(&[usize], DType, usize)],
        device: &Device,
    ) -> candle_core::Result<Self> {
        let mut free: HashMap<PoolKey, Vec<Tensor>> = HashMap::new();
        for &(shape, dtype, count) in specs {
            let key = (shape.to_vec(), dtype);
            let tensors = free.entry(key).or_default();
            for _ in 0..count {
                tensors.push(Tensor::zeros(shape, dtype, device)?);
            }
        }
        Ok(Self {
            free,
            device: device.clone(),
        })
    }

    /// Check out a tensor with the given shape and dtype.
    /// Returns a pre-allocated tensor if available, otherwise allocates a new one.
    pub fn checkout(&mut self, shape: &[usize], dtype: DType) -> candle_core::Result<Tensor> {
        let key = (shape.to_vec(), dtype);
        if let Some(list) = self.free.get_mut(&key) {
            if let Some(tensor) = list.pop() {
                return Ok(tensor);
            }
        }
        // Fallback: allocate new (slow path, should be rare after warmup)
        Tensor::zeros(shape, dtype, &self.device)
    }

    /// Return a tensor to the pool for reuse.
    pub fn checkin(&mut self, tensor: Tensor) {
        let key = (tensor.dims().to_vec(), tensor.dtype());
        self.free.entry(key).or_default().push(tensor);
    }

    /// Number of free tensors for a given (shape, dtype).
    pub fn free_count(&self, shape: &[usize], dtype: DType) -> usize {
        let key = (shape.to_vec(), dtype);
        self.free.get(&key).map(|v| v.len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    // MOPT-01: Arena allocator

    #[test]
    fn test_arena_allocate_returns_subslice() {
        let mut arena = TensorArena::new(1024);
        let slice1 = arena.allocate(256).unwrap();
        assert_eq!(slice1.len(), 256);

        let slice2 = arena.allocate(256).unwrap();
        assert_eq!(slice2.len(), 256);

        // The two slices should not overlap (offset should have advanced)
        assert_eq!(arena.used(), 512);
        assert_eq!(arena.remaining(), 512);
    }

    #[test]
    fn test_arena_reset_reuses_memory() {
        let mut arena = TensorArena::new(1024);
        let _ = arena.allocate(512).unwrap();
        assert_eq!(arena.used(), 512);

        arena.reset();
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.remaining(), 1024);

        // Next allocation starts from the beginning
        let slice = arena.allocate(256).unwrap();
        assert_eq!(slice.len(), 256);
        assert_eq!(arena.used(), 256);
    }

    #[test]
    fn test_arena_oom_returns_error() {
        let mut arena = TensorArena::new(256);
        // First allocation succeeds
        let _ = arena.allocate(200).unwrap();
        // Second allocation exceeds remaining capacity
        let result = arena.allocate(100);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("OOM"));
    }

    // MOPT-02: Buffer pool

    #[test]
    fn test_buffer_pool_checkout_returns_tensor() {
        let device = Device::Cpu;
        let pool = TensorBufferPool::new(
            &[(&[4, 8], DType::F32, 2)],
            &device,
        )
        .unwrap();
        assert_eq!(pool.free_count(&[4, 8], DType::F32), 2);

        let mut pool = pool;
        let tensor = pool.checkout(&[4, 8], DType::F32).unwrap();
        assert_eq!(tensor.dims(), &[4, 8]);
        assert_eq!(tensor.dtype(), DType::F32);
        assert_eq!(pool.free_count(&[4, 8], DType::F32), 1);
    }

    #[test]
    fn test_buffer_pool_checkin_reuses_buffer() {
        let device = Device::Cpu;
        let mut pool = TensorBufferPool::new(
            &[(&[16], DType::U32, 1)],
            &device,
        )
        .unwrap();

        // Check out the only tensor
        let tensor = pool.checkout(&[16], DType::U32).unwrap();
        assert_eq!(pool.free_count(&[16], DType::U32), 0);

        // Check it back in
        pool.checkin(tensor);
        assert_eq!(pool.free_count(&[16], DType::U32), 1);

        // Check out again -- should get the same storage (no new alloc needed)
        let tensor2 = pool.checkout(&[16], DType::U32).unwrap();
        assert_eq!(tensor2.dims(), &[16]);
        assert_eq!(tensor2.dtype(), DType::U32);
    }

    #[test]
    fn test_buffer_pool_shape_mismatch_allocates_new() {
        let device = Device::Cpu;
        let mut pool = TensorBufferPool::new(
            &[(&[4, 8], DType::F32, 1)],
            &device,
        )
        .unwrap();

        // Checking out a shape not in the pool falls back to fresh allocation
        let tensor = pool.checkout(&[3, 5], DType::F32).unwrap();
        assert_eq!(tensor.dims(), &[3, 5]);
        assert_eq!(tensor.dtype(), DType::F32);
    }

    // MOPT-03: Zero alloc in hot path

    #[test]
    fn test_zero_alloc_forward_pass_uses_pool() {
        // Simulates a forward step using only pool buffers (checkout/checkin cycle).
        let device = Device::Cpu;
        let mut pool = TensorBufferPool::new(
            &[
                (&[128], DType::U32, 2),   // input_ids
                (&[128], DType::F32, 2),    // positions
                (&[64, 16], DType::F32, 2), // block_table
            ],
            &device,
        )
        .unwrap();

        // Simulate 3 forward passes reusing the same pool
        for _ in 0..3 {
            let ids = pool.checkout(&[128], DType::U32).unwrap();
            let pos = pool.checkout(&[128], DType::F32).unwrap();
            let bt = pool.checkout(&[64, 16], DType::F32).unwrap();

            // "forward pass" happens here -- tensors are used
            assert_eq!(ids.dims(), &[128]);
            assert_eq!(pos.dims(), &[128]);
            assert_eq!(bt.dims(), &[64, 16]);

            // Return to pool
            pool.checkin(ids);
            pool.checkin(pos);
            pool.checkin(bt);
        }

        // Pool still has all buffers available after recycling
        assert_eq!(pool.free_count(&[128], DType::U32), 2);
        assert_eq!(pool.free_count(&[128], DType::F32), 2);
        assert_eq!(pool.free_count(&[64, 16], DType::F32), 2);
    }

    #[test]
    fn test_arena_allocate_slice_typed() {
        let mut arena = TensorArena::new(4096);
        let u32_slice = arena.allocate_slice::<u32>(64).unwrap();
        assert_eq!(u32_slice.len(), 64);

        // Write some values
        for (i, v) in u32_slice.iter_mut().enumerate() {
            *v = i as u32;
        }
        assert_eq!(u32_slice[0], 0);
        assert_eq!(u32_slice[63], 63);

        // Allocate another typed slice
        let f32_slice = arena.allocate_slice::<f32>(32).unwrap();
        assert_eq!(f32_slice.len(), 32);
    }
}
