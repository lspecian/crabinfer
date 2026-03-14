//! Wave 0 failing test stubs for CUDA-dependent requirements.
//!
//! These tests compile everywhere but fail at runtime on CUDA hardware
//! because the underlying kernels/dispatch don't exist yet.
//!
//! Requirements covered:
//!   KERNEL-I32-01 - I32 zeros on CUDA
//!   KERNEL-I32-02 - I32 to_dtype(F32) cast on CUDA
//!   KERNEL-I16-01 - I16 zeros on CUDA
//!   BINARY-I32-01 - I32 binary ops (add/mul) on CUDA
//!   LOADER-01     - Safetensors I32 direct GPU load

#[cfg(feature = "cuda")]
mod cuda_tests {
    use candle_core::{DType, Device, Tensor};

    /// KERNEL-I32-01: Tensor::zeros(DType::I32, &cuda_device) creates I32 tensor on GPU without panic.
    #[test]
    fn test_i32_zeros_on_cuda() {
        let device = Device::cuda_if_available(0).expect("CUDA device required");
        if !device.is_cuda() {
            return;
        } // skip on CPU-only
        let t = Tensor::zeros((4, 4), DType::I32, &device)
            .expect("I32 zeros on CUDA should work after kernel patches");
        assert_eq!(t.dtype(), DType::I32);
        assert!(t.device().is_cuda());
    }

    /// KERNEL-I32-02: I32 tensors support to_dtype(DType::F32) on CUDA without error.
    #[test]
    fn test_i32_to_dtype_f32_on_cuda() {
        let device = Device::cuda_if_available(0).expect("CUDA device required");
        if !device.is_cuda() {
            return;
        }
        let t = Tensor::zeros((4,), DType::I32, &device).expect("I32 zeros on CUDA");
        let f = t
            .to_dtype(DType::F32)
            .expect("I32 to F32 cast should work after kernel patches");
        assert_eq!(f.dtype(), DType::F32);
    }

    /// KERNEL-I16-01: Tensor::zeros(DType::I16, &cuda_device) creates I16 tensor on GPU without panic.
    #[test]
    fn test_i16_zeros_on_cuda() {
        let device = Device::cuda_if_available(0).expect("CUDA device required");
        if !device.is_cuda() {
            return;
        }
        let t = Tensor::zeros((4, 4), DType::I16, &device)
            .expect("I16 zeros on CUDA should work after kernel patches");
        assert_eq!(t.dtype(), DType::I16);
    }

    /// BINARY-I32-01: I32 tensors support add/mul on CUDA.
    #[test]
    fn test_binary_i32_ops_on_cuda() {
        let device = Device::cuda_if_available(0).expect("CUDA device required");
        if !device.is_cuda() {
            return;
        }
        let a = Tensor::zeros((4,), DType::I32, &device).expect("I32 zeros");
        let b = Tensor::ones((4,), DType::I32, &device).expect("I32 ones");
        let sum = (&a + &b).expect("I32 add on CUDA should work after kernel patches");
        assert_eq!(sum.dtype(), DType::I32);
        let product = (&a * &b).expect("I32 mul on CUDA");
        assert_eq!(product.dtype(), DType::I32);
    }

    /// LOADER-01: Safetensors I32 tensors load directly to GPU without CPU roundtrip.
    #[test]
    fn test_safetensors_i32_direct_gpu_load() {
        use std::collections::HashMap;

        let device = Device::cuda_if_available(0).expect("CUDA device required");
        if !device.is_cuda() {
            return;
        }

        // Create a small I32 tensor, save as safetensors, reload directly to GPU
        let data: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let cpu_tensor = Tensor::new(data.as_slice(), &Device::Cpu).expect("CPU tensor");

        let tmp_dir = std::env::temp_dir().join("crabinfer_test_safetensors");
        std::fs::create_dir_all(&tmp_dir).ok();
        let tmp_path = tmp_dir.join("test_i32.safetensors");

        let mut tensors_map = HashMap::new();
        tensors_map.insert("test_tensor".to_string(), cpu_tensor);
        candle_core::safetensors::save(&tensors_map, &tmp_path).expect("save safetensors");

        // Load directly to CUDA device -- this should work after kernel patches
        let tensors = candle_core::safetensors::load(&tmp_path, &device)
            .expect("load safetensors directly to CUDA with I32 dtype");
        let gpu_tensor = tensors.get("test_tensor").expect("tensor not found");
        assert!(gpu_tensor.device().is_cuda());
        assert_eq!(gpu_tensor.dtype(), DType::I32);

        // Cleanup
        std::fs::remove_file(&tmp_path).ok();
        std::fs::remove_dir(&tmp_dir).ok();
    }
}
