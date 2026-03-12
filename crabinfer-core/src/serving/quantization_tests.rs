//! Tests for Marlin reformat readiness fields on GptqLinear.
//!
//! These tests validate the structural changes to GptqLinear introduced in Plan 01-02
//! (qweight_marlin and backend fields, reformat_for_marlin stub, Marlin dispatch stub
//! in forward()).

#[cfg(test)]
mod tests {
    use crate::serving::quantization::GptqLinear;
    use crate::serving::kernels::cpu_backend::CpuBackend;
    use candle_core::{Device, Tensor, DType};
    use candle_nn::Module as _;

    /// Create a minimal GptqLinear via from_float for testing.
    fn make_gptq_linear(out: usize, in_: usize, group_size: usize) -> GptqLinear {
        let w = Tensor::randn(0f32, 0.1, (out, in_), &Device::Cpu).unwrap();
        GptqLinear::from_float(&w, None, group_size).unwrap()
    }

    /// Verify qweight_marlin is None after from_parts/from_float (no reformatting yet).
    #[test]
    fn test_marlin_reformat_aligned_produces_some() {
        // After from_float, qweight_marlin is None — reformatting is a no-op (stub)
        // and returns Ok(false) regardless of alignment.
        // Plan 03 will implement the real reformatting.
        let mut gl = make_gptq_linear(64, 128, 16);
        assert!(gl.qweight_marlin.is_none(), "qweight_marlin should start as None");

        let backend = CpuBackend::new();
        let result = gl.reformat_for_marlin(&backend).unwrap();
        // Stub always returns false
        assert!(!result, "stub reformat_for_marlin should return false");
        // qweight_marlin remains None (stub doesn't populate it)
        assert!(gl.qweight_marlin.is_none(), "stub should not populate qweight_marlin");
    }

    /// Verify reformat_for_marlin returns false for unaligned dims (stub behavior).
    #[test]
    fn test_marlin_reformat_unaligned_skips() {
        // N=24 is not a multiple of 16 — Marlin would skip (24/16 = 1.5).
        // INT4 packing requires N divisible by 8: 24/8=3, ok.
        // The stub always returns false regardless of alignment.
        let w = Tensor::randn(0f32, 0.1, (24, 128), &Device::Cpu).unwrap();
        let mut gl = GptqLinear::from_float(&w, None, 128).unwrap();

        let backend = CpuBackend::new();
        let result = gl.reformat_for_marlin(&backend).unwrap();
        assert!(!result, "stub returns false (unaligned for Marlin: 24 not multiple of 16)");
        assert!(gl.qweight_marlin.is_none());
    }

    /// Verify forward() on non-CUDA device falls back to naive dequant path.
    #[test]
    fn test_marlin_forward_dispatch_fallback() {
        // Even with qweight_marlin as None (or some placeholder), forward should work
        // via the naive dequant path.
        let w = Tensor::randn(0f32, 0.1, (16, 64), &Device::Cpu).unwrap();
        let gl = GptqLinear::from_float(&w, None, 8).unwrap();

        // qweight_marlin is None — falls through to naive path
        assert!(gl.qweight_marlin.is_none());

        let x = Tensor::randn(0f32, 1.0, (4, 64), &Device::Cpu).unwrap();
        let y = gl.forward(&x).unwrap();
        assert_eq!(y.dims(), &[4, 16]);
    }

    /// Verify backend field is None after construction.
    #[test]
    fn test_gptq_backend_field_starts_none() {
        let gl = make_gptq_linear(16, 32, 8);
        assert!(gl.backend.is_none(), "backend should start as None");
    }
}
