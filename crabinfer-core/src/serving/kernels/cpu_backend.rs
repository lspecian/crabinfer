//! CPU fallback backend for paged attention operations.
//!
//! Pure scalar Rust — no GPU required. Enables `cargo test` on Linux CI
//! and any platform without Metal or CUDA. Not performance-optimized.

use candle_core::{DType, Device, Result, Tensor};

use super::backend::{KernelBackend, PagedAttentionConfig};
use super::BLOCK_SIZE;

/// CPU fallback implementing `KernelBackend` with scalar loops.
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }
}

/// Get a mutable f32 slice from a CPU tensor's storage.
///
/// # Safety
/// Caller must ensure exclusive access to the tensor's data.
/// This mirrors GPU backends which mutate buffer memory through shared
/// tensor references via command encoders / kernel launches.
unsafe fn cpu_f32_mut(tensor: &Tensor) -> Result<&mut [f32]> {
    let (mut storage, layout) = tensor.storage_mut_and_layout();
    let offset = layout.start_offset();
    match &mut *storage {
        candle_core::Storage::Cpu(cpu_storage) => match cpu_storage {
            candle_core::CpuStorage::F32(ref mut data) => {
                let slice = &mut data[offset..];
                // Extend lifetime to match tensor — safe because the engine loop
                // is single-threaded and caches outlive all operations.
                Ok(std::mem::transmute::<&mut [f32], &mut [f32]>(slice))
            }
            _ => candle_core::bail!("expected F32 CPU storage"),
        },
        _ => candle_core::bail!("expected CPU tensor"),
    }
}

impl KernelBackend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
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
        let gqa_ratio = num_heads / num_kv_heads;

        let q_data: Vec<f32> = query.flatten_all()?.to_vec1()?;
        let bt_data: Vec<i32> = block_tables.flatten_all()?.to_vec1()?;
        let cl_data: Vec<i32> = context_lens.flatten_all()?.to_vec1()?;
        let max_blocks_per_seq = block_tables.dims()[1];

        let kc_data: Vec<f32> = key_cache.flatten_all()?.to_vec1()?;
        let vc_data: Vec<f32> = value_cache.flatten_all()?.to_vec1()?;

        let vc_block_stride = num_kv_heads * head_size * BLOCK_SIZE;
        let vc_head_stride = head_size * BLOCK_SIZE;

        let dtype_bytes = query.dtype().size_in_bytes();
        let x = 16 / dtype_bytes;
        let kc_block_stride = num_kv_heads * (head_size / x) * BLOCK_SIZE * x;
        let kc_head_stride = (head_size / x) * BLOCK_SIZE * x;

        // SAFETY: Single-threaded engine loop, output is freshly allocated by caller.
        let out_slice = unsafe { cpu_f32_mut(output)? };
        for v in out_slice.iter_mut() {
            *v = 0.0;
        }

        for seq in 0..num_seqs {
            let ctx_len = cl_data[seq] as usize;
            if ctx_len == 0 {
                continue;
            }

            for head in 0..num_heads {
                let kv_head = head / gqa_ratio;
                let q_offset = (seq * num_heads + head) * head_size;
                let q_vec = &q_data[q_offset..q_offset + head_size];

                let mut scores = Vec::with_capacity(ctx_len);
                for tok in 0..ctx_len {
                    let block_idx = tok / BLOCK_SIZE;
                    let block_offset = tok % BLOCK_SIZE;
                    let physical_block = bt_data[seq * max_blocks_per_seq + block_idx] as usize;

                    let mut dot = 0.0f32;
                    for d in 0..head_size {
                        let d_outer = d / x;
                        let d_inner = d % x;
                        let k_idx = physical_block * kc_block_stride
                            + kv_head * kc_head_stride
                            + d_outer * BLOCK_SIZE * x
                            + block_offset * x
                            + d_inner;
                        dot += q_vec[d] * kc_data[k_idx];
                    }
                    scores.push(dot * config.scale);
                }

                // Softmax
                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0f32;
                for s in scores.iter_mut() {
                    *s = (*s - max_score).exp();
                    exp_sum += *s;
                }
                if exp_sum > 0.0 {
                    for s in scores.iter_mut() {
                        *s /= exp_sum;
                    }
                }

                // Weighted sum of values
                let o_offset = (seq * num_heads + head) * head_size;
                for tok in 0..ctx_len {
                    let block_idx = tok / BLOCK_SIZE;
                    let block_offset = tok % BLOCK_SIZE;
                    let physical_block = bt_data[seq * max_blocks_per_seq + block_idx] as usize;

                    let weight = scores[tok];
                    for d in 0..head_size {
                        let v_idx = physical_block * vc_block_stride
                            + kv_head * vc_head_stride
                            + d * BLOCK_SIZE
                            + block_offset;
                        out_slice[o_offset + d] += weight * vc_data[v_idx];
                    }
                }
            }
        }

        Ok(())
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
        let x = 16 / dtype_bytes;

        let slots: Vec<i32> = slot_mapping.flatten_all()?.to_vec1()?;
        let k_data: Vec<f32> = key.flatten_all()?.to_vec1()?;
        let v_data: Vec<f32> = value.flatten_all()?.to_vec1()?;

        // SAFETY: Single-threaded engine loop, caches are only accessed here.
        let kc_data = unsafe { cpu_f32_mut(key_cache)? };
        let vc_data = unsafe { cpu_f32_mut(value_cache)? };

        let kc_head_stride = (head_size / x) * BLOCK_SIZE * x;
        let kc_block_stride = num_heads * kc_head_stride;
        let vc_head_stride = head_size * BLOCK_SIZE;
        let vc_block_stride = num_heads * vc_head_stride;

        for token in 0..num_tokens {
            let slot = slots[token] as usize;
            let block_idx = slot / BLOCK_SIZE;
            let block_offset = slot % BLOCK_SIZE;

            for head in 0..num_heads {
                let src_offset = (token * num_heads + head) * head_size;

                for d in 0..head_size {
                    let d_outer = d / x;
                    let d_inner = d % x;
                    let kc_idx = block_idx * kc_block_stride
                        + head * kc_head_stride
                        + d_outer * BLOCK_SIZE * x
                        + block_offset * x
                        + d_inner;
                    kc_data[kc_idx] = k_data[src_offset + d];
                }

                for d in 0..head_size {
                    let vc_idx = block_idx * vc_block_stride
                        + head * vc_head_stride
                        + d * BLOCK_SIZE
                        + block_offset;
                    vc_data[vc_idx] = v_data[src_offset + d];
                }
            }
        }

        Ok(())
    }

    fn copy_blocks(
        &self,
        key_cache: &Tensor,
        value_cache: &Tensor,
        block_mapping: &Tensor,
        numel_per_block: usize,
    ) -> Result<()> {
        let mapping: Vec<i32> = block_mapping.flatten_all()?.to_vec1()?;
        let num_pairs = mapping.len() / 2;
        if num_pairs == 0 {
            return Ok(());
        }

        // SAFETY: Single-threaded engine loop, caches are only accessed here.
        let kc_data = unsafe { cpu_f32_mut(key_cache)? };
        let vc_data = unsafe { cpu_f32_mut(value_cache)? };

        for i in 0..num_pairs {
            let src = mapping[i * 2] as usize;
            let dst = mapping[i * 2 + 1] as usize;
            let src_start = src * numel_per_block;
            let dst_start = dst * numel_per_block;

            for j in 0..numel_per_block {
                kc_data[dst_start + j] = kc_data[src_start + j];
            }
            for j in 0..numel_per_block {
                vc_data[dst_start + j] = vc_data[src_start + j];
            }
        }

        Ok(())
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
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    fn make_cpu_backend() -> CpuBackend {
        CpuBackend::new()
    }

    #[test]
    fn test_cpu_backend_name() {
        let b = make_cpu_backend();
        assert_eq!(b.name(), "cpu");
    }

    #[test]
    fn test_allocate_kv_caches() {
        let b = make_cpu_backend();
        let (kc, vc) = b
            .allocate_kv_caches(2, 4, 8, 128, DType::F32, &Device::Cpu)
            .unwrap();
        assert_eq!(kc.len(), 2);
        assert_eq!(vc.len(), 2);
        assert_eq!(kc[0].dims(), &[4, 8, 32, BLOCK_SIZE, 4]);
        assert_eq!(vc[0].dims(), &[4, 8, 128, BLOCK_SIZE]);
    }

    #[test]
    fn test_reshape_and_cache_single_token() {
        let b = make_cpu_backend();
        let num_blocks = 2;
        let num_kv_heads = 2;
        let head_size = 8;
        let dtype = DType::F32;
        let dev = &Device::Cpu;

        let (kc, vc) = b
            .allocate_kv_caches(1, num_blocks, num_kv_heads, head_size, dtype, dev)
            .unwrap();

        let key = Tensor::ones((1, num_kv_heads, head_size), dtype, dev).unwrap();
        let value = Tensor::ones((1, num_kv_heads, head_size), dtype, dev).unwrap();
        let slot_mapping = Tensor::new(&[3i32], dev).unwrap();

        b.reshape_and_cache(&key, &value, &kc[0], &vc[0], &slot_mapping)
            .unwrap();

        let vc_data: Vec<f32> = vc[0].flatten_all().unwrap().to_vec1().unwrap();
        let idx = 3; // block=0, head=0, d=0, offset=3
        assert_eq!(vc_data[idx], 1.0);
    }

    #[test]
    fn test_paged_attention_single_seq() {
        let b = make_cpu_backend();
        let num_blocks = 2;
        let num_kv_heads = 1;
        let num_heads = 1;
        let head_size = 4;
        let dtype = DType::F32;
        let dev = &Device::Cpu;

        let (kc, vc) = b
            .allocate_kv_caches(1, num_blocks, num_kv_heads, head_size, dtype, dev)
            .unwrap();

        let key = Tensor::new(&[[[1.0f32, 0.0, 0.0, 0.0]], [[0.0f32, 1.0, 0.0, 0.0]]], dev)
            .unwrap();
        let value = Tensor::new(&[[[1.0f32, 0.0, 0.0, 0.0]], [[0.0f32, 0.0, 1.0, 0.0]]], dev)
            .unwrap();
        let slot_mapping = Tensor::new(&[0i32, 1], dev).unwrap();
        b.reshape_and_cache(&key, &value, &kc[0], &vc[0], &slot_mapping)
            .unwrap();

        let query = Tensor::new(&[[[0.0f32, 1.0, 0.0, 0.0]]], dev).unwrap();
        let output = Tensor::zeros((1, num_heads, head_size), dtype, dev).unwrap();
        let block_tables = Tensor::new(&[[0i32]], dev).unwrap();
        let context_lens = Tensor::new(&[2i32], dev).unwrap();

        let config = PagedAttentionConfig {
            head_size,
            num_kv_heads,
            scale: 1.0 / (head_size as f32).sqrt(),
            max_context_len: 2,
        };

        b.paged_attention(
            &output, &query, &kc[0], &vc[0], &block_tables, &context_lens, &config,
        )
        .unwrap();

        let out_data: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            out_data[2] > out_data[0],
            "second token should dominate: {:?}",
            out_data
        );
    }

    #[test]
    fn test_copy_blocks() {
        let b = make_cpu_backend();
        let num_blocks = 4;
        let num_kv_heads = 1;
        let head_size = 4;
        let dtype = DType::F32;
        let dev = &Device::Cpu;

        let (kc, vc) = b
            .allocate_kv_caches(1, num_blocks, num_kv_heads, head_size, dtype, dev)
            .unwrap();

        let key = Tensor::ones((1, num_kv_heads, head_size), dtype, dev).unwrap();
        let value = Tensor::ones((1, num_kv_heads, head_size), dtype, dev).unwrap();
        let slot_mapping = Tensor::new(&[0i32], dev).unwrap();
        b.reshape_and_cache(&key, &value, &kc[0], &vc[0], &slot_mapping)
            .unwrap();

        let x = 16 / dtype.size_in_bytes();
        let numel_per_block = num_kv_heads * (head_size / x) * BLOCK_SIZE * x;
        let block_mapping = Tensor::new(&[0i32, 2], dev).unwrap();
        b.copy_blocks(&kc[0], &vc[0], &block_mapping, numel_per_block)
            .unwrap();

        let kc_data: Vec<f32> = kc[0].flatten_all().unwrap().to_vec1().unwrap();
        for i in 0..numel_per_block {
            assert_eq!(
                kc_data[i],
                kc_data[2 * numel_per_block + i],
                "key cache mismatch at offset {i}"
            );
        }
    }

    // ─── Fused kernel tests ──────────────────────────────────────────────

    #[test]
    fn test_fused_silu_mul_shape() {
        let b = make_cpu_backend();
        let gate = Tensor::randn(0f32, 1.0, (4, 128), &Device::Cpu).unwrap();
        let up = Tensor::randn(0f32, 1.0, (4, 128), &Device::Cpu).unwrap();
        let result = b.fused_silu_mul(&gate, &up).unwrap();
        assert_eq!(result.dims(), &[4, 128]);
    }

    #[test]
    fn test_fused_silu_mul_values() {
        let b = make_cpu_backend();
        let gate = Tensor::new(&[[1.0f32, -1.0, 0.0, 2.0]], &Device::Cpu).unwrap();
        let up = Tensor::new(&[[1.0f32, 1.0, 1.0, 1.0]], &Device::Cpu).unwrap();
        let result = b.fused_silu_mul(&gate, &up).unwrap();
        let data: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();

        // silu(x) = x * sigmoid(x)
        // silu(1.0) = 1.0 * 0.7311 = 0.7311
        // silu(-1.0) = -1.0 * 0.2689 = -0.2689
        // silu(0.0) = 0.0
        // silu(2.0) = 2.0 * 0.8808 = 1.7616
        assert!((data[0] - 0.7311).abs() < 0.01, "silu(1)={}", data[0]);
        assert!((data[1] - (-0.2689)).abs() < 0.01, "silu(-1)={}", data[1]);
        assert!(data[2].abs() < 0.001, "silu(0)={}", data[2]);
        assert!((data[3] - 1.7616).abs() < 0.01, "silu(2)={}", data[3]);
    }

    #[test]
    fn test_fused_silu_mul_matches_unfused() {
        let b = make_cpu_backend();
        let gate = Tensor::randn(0f32, 1.0, (8, 64), &Device::Cpu).unwrap();
        let up = Tensor::randn(0f32, 1.0, (8, 64), &Device::Cpu).unwrap();

        // Fused path
        let fused = b.fused_silu_mul(&gate, &up).unwrap();

        // Unfused reference
        let silu_gate = candle_nn::ops::silu(&gate).unwrap();
        let unfused = (silu_gate * &up).unwrap();

        let fused_data: Vec<f32> = fused.flatten_all().unwrap().to_vec1().unwrap();
        let unfused_data: Vec<f32> = unfused.flatten_all().unwrap().to_vec1().unwrap();
        for (i, (f, u)) in fused_data.iter().zip(unfused_data.iter()).enumerate() {
            assert!(
                (f - u).abs() < 1e-5,
                "mismatch at {i}: fused={f}, unfused={u}"
            );
        }
    }

    #[test]
    fn test_fused_rope_shape() {
        let b = make_cpu_backend();
        let total_tokens = 4;
        let num_heads = 8;
        let head_size = 64;
        let rope_dim = 64;

        let x = Tensor::randn(0f32, 1.0, (total_tokens, num_heads, head_size), &Device::Cpu).unwrap();
        let positions = Tensor::new(&[0u32, 1, 2, 3], &Device::Cpu).unwrap();

        let (cos, sin) = crate::serving::models::attention::precompute_rope(
            rope_dim, 10000.0, 128, &Device::Cpu,
        ).unwrap();

        let result = b.fused_rope(&x, &positions, &cos, &sin, num_heads, head_size, rope_dim).unwrap();
        assert_eq!(result.dims(), &[total_tokens, num_heads, head_size]);
    }

    #[test]
    fn test_fused_rope_position_zero_is_identity() {
        // At position 0, cos=1 sin=0, so RoPE should be close to identity
        let b = make_cpu_backend();
        let num_heads = 2;
        let head_size = 4;
        let rope_dim = 4;

        let x = Tensor::new(&[[[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]], &Device::Cpu).unwrap();
        let positions = Tensor::new(&[0u32], &Device::Cpu).unwrap();

        let (cos, sin) = crate::serving::models::attention::precompute_rope(
            rope_dim, 10000.0, 128, &Device::Cpu,
        ).unwrap();

        let result = b.fused_rope(&x, &positions, &cos, &sin, num_heads, head_size, rope_dim).unwrap();
        let result_data: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        let x_data: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();

        // At position 0, cos(0)=1, sin(0)=0, so output should equal input
        for (i, (r, orig)) in result_data.iter().zip(x_data.iter()).enumerate() {
            assert!(
                (r - orig).abs() < 1e-4,
                "pos-0 identity failed at {i}: got {r}, expected {orig}"
            );
        }
    }

    // ── Fused RMSNorm tests ────────────────────────────────────────────

    #[test]
    fn test_fused_rmsnorm_shape() {
        let b = make_cpu_backend();
        let x = Tensor::randn(0f32, 1.0, (4, 64), &Device::Cpu).unwrap();
        let weight = Tensor::ones(64, DType::F32, &Device::Cpu).unwrap();
        let result = b.fused_rmsnorm(&x, &weight, 1e-5).unwrap();
        assert_eq!(result.dims(), &[4, 64]);
    }

    #[test]
    fn test_fused_rmsnorm_values() {
        let b = make_cpu_backend();
        // [1, 4] with values [2, 2, 2, 2], weight = [1, 1, 1, 1]
        // RMS = sqrt(mean(4,4,4,4) + eps) = sqrt(4 + eps) ≈ 2
        // output = x / rms * weight ≈ [1, 1, 1, 1]
        let x = Tensor::new(&[[2.0f32, 2.0, 2.0, 2.0]], &Device::Cpu).unwrap();
        let weight = Tensor::ones(4, DType::F32, &Device::Cpu).unwrap();
        let result = b.fused_rmsnorm(&x, &weight, 1e-6).unwrap();
        let data: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        for v in &data {
            assert!((v - 1.0).abs() < 1e-4, "expected ~1.0, got {v}");
        }
    }

    #[test]
    fn test_fused_rmsnorm_matches_candle() {
        let b = make_cpu_backend();
        let x = Tensor::randn(0f32, 1.0, (8, 128), &Device::Cpu).unwrap();
        let weight = Tensor::randn(0f32, 1.0, 128, &Device::Cpu).unwrap();
        let eps = 1e-5f32;

        let fused = b.fused_rmsnorm(&x, &weight, eps).unwrap();
        let reference = candle_nn::ops::rms_norm(&x, &weight, eps).unwrap();

        let diff = (&fused - &reference).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(0).unwrap().max(0).unwrap().to_scalar().unwrap();
        assert!(max_diff < 1e-5, "fused vs candle max_diff={max_diff}");
    }

    #[test]
    fn test_fused_rope_different_positions_differ() {
        let b = make_cpu_backend();
        let num_heads = 2;
        let head_size = 4;
        let rope_dim = 4;

        let x = Tensor::ones((2, num_heads, head_size), DType::F32, &Device::Cpu).unwrap();
        let positions = Tensor::new(&[0u32, 10], &Device::Cpu).unwrap();

        let (cos, sin) = crate::serving::models::attention::precompute_rope(
            rope_dim, 10000.0, 128, &Device::Cpu,
        ).unwrap();

        let result = b.fused_rope(&x, &positions, &cos, &sin, num_heads, head_size, rope_dim).unwrap();
        let data: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();

        // Token at position 0 and position 10 should produce different outputs
        let tok0: Vec<f32> = data[..num_heads * head_size].to_vec();
        let tok1: Vec<f32> = data[num_heads * head_size..].to_vec();
        let differs = tok0.iter().zip(tok1.iter()).any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(differs, "different positions should produce different RoPE outputs");
    }

    #[test]
    fn test_fused_add_rmsnorm_shape() {
        let b = make_cpu_backend();
        let x = Tensor::randn(0f32, 1.0, (4, 64), &Device::Cpu).unwrap();
        let residual = Tensor::randn(0f32, 1.0, (4, 64), &Device::Cpu).unwrap();
        let weight = Tensor::ones(64, DType::F32, &Device::Cpu).unwrap();
        let (normed, x_out) = b.fused_add_rmsnorm(&x, &residual, &weight, 1e-5).unwrap();
        assert_eq!(normed.dims(), &[4, 64]);
        assert_eq!(x_out.dims(), &[4, 64]);
    }

    #[test]
    fn test_fused_add_rmsnorm_values() {
        let b = make_cpu_backend();
        let x = Tensor::new(&[[1.0f32, 1.0, 1.0, 1.0]], &Device::Cpu).unwrap();
        let residual = Tensor::new(&[[1.0f32, 1.0, 1.0, 1.0]], &Device::Cpu).unwrap();
        let weight = Tensor::ones(4, DType::F32, &Device::Cpu).unwrap();
        let (normed, x_out) = b.fused_add_rmsnorm(&x, &residual, &weight, 1e-6).unwrap();
        // x_out = [2,2,2,2]
        let x_data: Vec<f32> = x_out.flatten_all().unwrap().to_vec1().unwrap();
        for v in &x_data {
            assert!((v - 2.0).abs() < 1e-4, "expected ~2.0, got {v}");
        }
        // rmsnorm([2,2,2,2]) = [1,1,1,1] with weight=1
        let n_data: Vec<f32> = normed.flatten_all().unwrap().to_vec1().unwrap();
        for v in &n_data {
            assert!((v - 1.0).abs() < 1e-4, "expected ~1.0, got {v}");
        }
    }

    #[test]
    fn test_fused_add_rmsnorm_matches_unfused() {
        let b = make_cpu_backend();
        let x = Tensor::randn(0f32, 1.0, (8, 128), &Device::Cpu).unwrap();
        let residual = Tensor::randn(0f32, 1.0, (8, 128), &Device::Cpu).unwrap();
        let weight = Tensor::randn(0f32, 1.0, 128, &Device::Cpu).unwrap();
        let eps = 1e-5f32;

        let (fused_norm, fused_x) = b.fused_add_rmsnorm(&x, &residual, &weight, eps).unwrap();

        // Reference: separate add + rmsnorm
        let ref_x = (&x + &residual).unwrap();
        let ref_norm = candle_nn::ops::rms_norm(&ref_x, &weight, eps).unwrap();

        let x_diff: f32 = (&fused_x - &ref_x).unwrap().abs().unwrap()
            .max(0).unwrap().max(0).unwrap().to_scalar().unwrap();
        assert!(x_diff < 1e-5, "x_out diff={x_diff}");

        let n_diff: f32 = (&fused_norm - &ref_norm).unwrap().abs().unwrap()
            .max(0).unwrap().max(0).unwrap().to_scalar().unwrap();
        assert!(n_diff < 1e-5, "normed diff={n_diff}");
    }
}
