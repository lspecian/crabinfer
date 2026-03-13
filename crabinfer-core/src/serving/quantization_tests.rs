//! Tests for Marlin reformat, forward dispatch, and alignment on GptqLinear.
//!
//! These tests validate the Marlin tile layout reformatting, forward dispatch
//! logic, and alignment checks introduced in Plans 01-02 and 01-03.

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use crate::serving::quantization::GptqLinear;
    use crate::serving::kernels::backend::KernelBackend;
    use crate::serving::kernels::cpu_backend::CpuBackend;
    use candle_core::{Device, Tensor, DType, Result};
    use candle_nn::Module as _;

    /// Create a minimal GptqLinear via from_float for testing.
    fn make_gptq_linear(out: usize, in_: usize, group_size: usize) -> GptqLinear {
        let w = Tensor::randn(0f32, 0.1, (out, in_), &Device::Cpu).unwrap();
        GptqLinear::from_float(&w, None, group_size).unwrap()
    }

    /// Verify reformat_for_marlin with aligned dims skips on CPU backend (returns false).
    #[test]
    fn test_marlin_reformat_aligned_cpu_skips() {
        // CPU backend is not "cuda", so reformat_for_marlin always returns false
        let mut gl = make_gptq_linear(64, 128, 16);
        assert!(gl.qweight_marlin.is_none(), "qweight_marlin should start as None");

        let backend = CpuBackend::new();
        let result = gl.reformat_for_marlin(&backend).unwrap();
        assert!(!result, "reformat_for_marlin should return false on CPU backend");
        assert!(gl.qweight_marlin.is_none(), "should not populate qweight_marlin on CPU");
    }

    /// Verify reformat_for_marlin returns false for unaligned dims.
    #[test]
    fn test_marlin_reformat_unaligned_skips() {
        // N=24 is not a multiple of 64 — Marlin should skip.
        let w = Tensor::randn(0f32, 0.1, (24, 128), &Device::Cpu).unwrap();
        let mut gl = GptqLinear::from_float(&w, None, 128).unwrap();

        // Use a mock "cuda" backend so we get past the name check
        let backend = MockCudaBackend;
        let result = gl.reformat_for_marlin(&backend).unwrap();
        assert!(!result, "should return false for unaligned N=24 (not multiple of 64)");
        assert!(gl.qweight_marlin.is_none());
    }

    /// Verify reformat_for_marlin with aligned dims on "cuda" backend populates qweight_marlin.
    #[test]
    fn test_marlin_reformat_aligned_produces_some() {
        let mut gl = make_gptq_linear(64, 128, 16);
        assert!(gl.qweight_marlin.is_none());

        let backend = MockCudaBackend;
        let result = gl.reformat_for_marlin(&backend).unwrap();
        assert!(result, "reformat_for_marlin should return true for aligned dims on cuda");
        assert!(gl.qweight_marlin.is_some(), "qweight_marlin should be populated");

        // Check shape: [K/16, N/64, 128] = [128/16, 64/64, 128] = [8, 1, 128]
        let qwm = gl.qweight_marlin.as_ref().unwrap();
        assert_eq!(qwm.dims(), &[8, 1, 128]);
    }

    /// Verify Marlin-reformatted weights dequantize to same values as original.
    #[test]
    fn test_marlin_reformat_correctness() {
        // Create small GPTQ: K=32, N=64, group_size=16
        let w = Tensor::randn(0f32, 0.5, (64, 32), &Device::Cpu).unwrap();
        let mut gl = GptqLinear::from_float(&w, None, 16).unwrap();

        // Dequantize before reformat (original GPTQ layout)
        let w_before = gl.dequantize(DType::F32).unwrap();

        // Reformat to Marlin tile layout
        let backend = MockCudaBackend;
        let result = gl.reformat_for_marlin(&backend).unwrap();
        assert!(result);

        // Dequantize after reformat (original qweight still intact)
        let w_after = gl.dequantize(DType::F32).unwrap();

        // Values should be identical (dequantize uses the original qweight, not qweight_marlin)
        let diff: f32 = (&w_before - &w_after)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(diff < 1e-6, "dequantized values should be identical, diff={diff}");

        // Also verify that qweight_marlin contains correct values by manually
        // extracting a few INT4 values and comparing to original
        let qwm = gl.qweight_marlin.as_ref().unwrap();
        let qwm_data: Vec<u32> = qwm.flatten_all().unwrap().to_vec1().unwrap();

        let qw_data: Vec<u32> = gl.qweight.flatten_all().unwrap().to_vec1().unwrap();
        let k = 32usize;
        let n = 64usize;

        // Check a few specific positions
        for row in [0, 5, 15] {
            for col in [0, 13, 63] {
                // Original GPTQ: extract from qw_data
                let pack_row = row / 8;
                let bit_pos = row % 8;
                let packed_orig = qw_data[pack_row * n + col];
                let val_orig = (packed_orig >> (bit_pos as u32 * 4)) & 0xF;

                // Marlin tile: row is in tile kt=row/16, r=row%16
                // col is in tile nt=col/64, c=col%64
                let kt = row / 16;
                let nt = col / 64;
                let r = row % 16;
                let c = col % 64;
                let tile_base = (kt * (n / 64) + nt) * 128;
                let linear_idx = r * 64 + c;
                let u32_idx = linear_idx / 8;
                let bit_pos_m = linear_idx % 8;
                let packed_marlin = qwm_data[tile_base + u32_idx];
                let val_marlin = (packed_marlin >> (bit_pos_m as u32 * 4)) & 0xF;

                assert_eq!(
                    val_orig, val_marlin,
                    "INT4 mismatch at row={row}, col={col}: orig={val_orig}, marlin={val_marlin}"
                );
            }
        }
    }

    /// Verify forward() falls back to naive path when qweight_marlin is None.
    #[test]
    fn test_marlin_forward_dispatch_fallback() {
        let w = Tensor::randn(0f32, 0.1, (16, 64), &Device::Cpu).unwrap();
        let gl = GptqLinear::from_float(&w, None, 8).unwrap();

        assert!(gl.qweight_marlin.is_none());
        assert!(gl.backend.is_none());

        let x = Tensor::randn(0f32, 1.0, (4, 64), &Device::Cpu).unwrap();
        let y = gl.forward(&x).unwrap();
        assert_eq!(y.dims(), &[4, 16]);
    }

    /// Verify forward() dispatches to backend.marlin_gemm() when both fields are set.
    #[test]
    fn test_marlin_forward_dispatch_calls_backend() {
        use std::sync::atomic::{AtomicBool, Ordering};

        // Create a mock backend that tracks whether marlin_gemm was called
        static MARLIN_CALLED: AtomicBool = AtomicBool::new(false);

        struct TrackingBackend;
        impl KernelBackend for TrackingBackend {
            fn name(&self) -> &'static str { "cuda" }

            fn marlin_gemm(
                &self,
                input: &Tensor,
                _qweight_marlin: &Tensor,
                _scales: &Tensor,
                _qzeros: &Tensor,
                _group_size: usize,
            ) -> Result<Tensor> {
                MARLIN_CALLED.store(true, Ordering::SeqCst);
                // Return a tensor of the right shape: [M, N]
                // N = qweight_marlin.dims()[1] * 64
                let n = _qweight_marlin.dims()[1] * 64;
                let m = input.dims()[0];
                Tensor::zeros((m, n), DType::F32, &Device::Cpu)
            }

            fn paged_attention(&self, _: &Tensor, _: &Tensor, _: &Tensor, _: &Tensor, _: &Tensor, _: &Tensor, _: &crate::serving::kernels::backend::PagedAttentionConfig) -> Result<()> { Ok(()) }
            fn reshape_and_cache(&self, _: &Tensor, _: &Tensor, _: &Tensor, _: &Tensor, _: &Tensor) -> Result<()> { Ok(()) }
            fn copy_blocks(&self, _: &Tensor, _: &Tensor, _: &Tensor, _: usize) -> Result<()> { Ok(()) }
            fn allocate_kv_caches(&self, _: usize, _: usize, _: usize, _: usize, _: DType, _: &Device) -> Result<(Vec<Tensor>, Vec<Tensor>)> { Ok((vec![], vec![])) }
        }

        MARLIN_CALLED.store(false, Ordering::SeqCst);

        let mut gl = make_gptq_linear(64, 128, 16);

        // Reformat to Marlin layout
        let mock_cuda = MockCudaBackend;
        gl.reformat_for_marlin(&mock_cuda).unwrap();
        assert!(gl.qweight_marlin.is_some());

        // Set tracking backend
        gl.backend = Some(Arc::new(TrackingBackend));

        let x = Tensor::randn(0f32, 1.0, (2, 128), &Device::Cpu).unwrap();
        let y = gl.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 64]);
        assert!(MARLIN_CALLED.load(Ordering::SeqCst), "marlin_gemm should have been called");
    }

    /// Verify backend field is None after construction.
    #[test]
    fn test_gptq_backend_field_starts_none() {
        let gl = make_gptq_linear(16, 32, 8);
        assert!(gl.backend.is_none(), "backend should start as None");
    }

    /// Verify K not multiple of 16 skips Marlin.
    #[test]
    fn test_marlin_reformat_k_unaligned_skips() {
        // K=24 is not a multiple of 16 (24/16 = 1.5)
        // But in_features must be multiple of 8 for GPTQ packing: 24/8=3, ok.
        let w = Tensor::randn(0f32, 0.1, (64, 24), &Device::Cpu).unwrap();
        let mut gl = GptqLinear::from_float(&w, None, 24).unwrap();

        let backend = MockCudaBackend;
        let result = gl.reformat_for_marlin(&backend).unwrap();
        assert!(!result, "should skip: K=24 is not a multiple of 16");
        assert!(gl.qweight_marlin.is_none());
    }

    /// Verify large aligned dims work (N=4096, K=4096, group_size=128).
    #[test]
    fn test_marlin_reformat_large_aligned() {
        // Use smaller values to avoid slow test: N=256, K=256, group_size=128
        let mut gl = make_gptq_linear(256, 256, 128);

        let backend = MockCudaBackend;
        let result = gl.reformat_for_marlin(&backend).unwrap();
        assert!(result, "should succeed for aligned N=256, K=256");
        assert!(gl.qweight_marlin.is_some());

        // Shape: [256/16, 256/64, 128] = [16, 4, 128]
        let qwm = gl.qweight_marlin.as_ref().unwrap();
        assert_eq!(qwm.dims(), &[16, 4, 128]);
    }

    // ─── Mock "cuda" backend for testing ─────────────────────────────────

    /// A minimal mock backend that returns "cuda" as its name but runs on CPU.
    /// This allows testing reformat_for_marlin without actual CUDA hardware.
    struct MockCudaBackend;

    impl KernelBackend for MockCudaBackend {
        fn name(&self) -> &'static str {
            "cuda"
        }

        fn paged_attention(
            &self, _: &Tensor, _: &Tensor, _: &Tensor, _: &Tensor,
            _: &Tensor, _: &Tensor, _: &crate::serving::kernels::backend::PagedAttentionConfig,
        ) -> Result<()> {
            Ok(())
        }

        fn reshape_and_cache(
            &self, _: &Tensor, _: &Tensor, _: &Tensor, _: &Tensor, _: &Tensor,
        ) -> Result<()> {
            Ok(())
        }

        fn copy_blocks(
            &self, _: &Tensor, _: &Tensor, _: &Tensor, _: usize,
        ) -> Result<()> {
            Ok(())
        }

        fn allocate_kv_caches(
            &self, _: usize, _: usize, _: usize, _: usize, _: DType, _: &Device,
        ) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
            Ok((vec![], vec![]))
        }
    }
}
