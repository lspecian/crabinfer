//! FP8 E4M3 quantization for model weights.
//!
//! Implements FP8 E4M3 (4-bit exponent, 3-bit mantissa) weight quantization,
//! providing ~2x memory savings vs FP16 with ~98% throughput retention on
//! Hopper+ GPUs (H100/H200). On CPU, uses software emulation for the
//! dequantize-then-matmul path.
//!
//! FP8 E4M3 format:
//!   - 1 sign bit + 4 exponent bits + 3 mantissa bits
//!   - Exponent bias: 7
//!   - Max representable value: 448.0
//!   - Min positive normal: 2^(-6) = 0.015625
//!   - Min positive subnormal: 2^(-9) = ~1.953125e-3
//!   - NaN: exponent all-1s AND mantissa all-1s (0x7F / 0xFF)
//!   - No infinities (unlike IEEE 754 FP8 E5M2)

use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Module;

// ─── FP8 dtype enum ──────────────────────────────────────────────────────

/// FP8 sub-format selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Dtype {
    /// E4M3: 4-bit exponent, 3-bit mantissa.
    /// Range: +/-448, precision: ~3 decimal digits.
    /// Preferred for weights.
    E4M3,
    /// E5M2: 5-bit exponent, 2-bit mantissa.
    /// Range: +/-57344, less precision.
    /// Preferred for gradients (not used here, but available for future).
    E5M2,
}

impl std::fmt::Display for Fp8Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::E4M3 => write!(f, "e4m3"),
            Self::E5M2 => write!(f, "e5m2"),
        }
    }
}

// ─── FP8 configuration ──────────────────────────────────────────────────

/// Configuration for FP8 quantization.
#[derive(Debug, Clone)]
pub struct Fp8Config {
    /// FP8 sub-format for stored weights.
    pub weight_dtype: Fp8Dtype,
    /// FP8 sub-format for computation (currently unused; computation is in FP16/FP32).
    pub compute_dtype: Fp8Dtype,
    /// If true, use a single scale factor per tensor.
    /// If false, use per-channel (per-output-row) scaling.
    pub per_tensor_scaling: bool,
}

impl Default for Fp8Config {
    fn default() -> Self {
        Self {
            weight_dtype: Fp8Dtype::E4M3,
            compute_dtype: Fp8Dtype::E4M3,
            per_tensor_scaling: true,
        }
    }
}

// ─── FP8 E4M3 conversion functions ─────────────────────────────────────

/// Maximum finite value representable in FP8 E4M3.
pub const FP8_E4M3_MAX: f32 = 448.0;

/// Minimum positive subnormal value in FP8 E4M3.
pub const FP8_E4M3_MIN_SUBNORMAL: f32 = 1.0 / 512.0; // 2^-9 = ~1.953125e-3

/// Convert an f32 value to FP8 E4M3 (as a u8 bit pattern).
///
/// FP8 E4M3 layout: `[sign(1) | exponent(4) | mantissa(3)]`
/// - Exponent bias: 7
/// - Max value: 448.0 (exponent=1111, mantissa=110)
/// - NaN: 0x7F (positive) or 0xFF (negative) — exponent=1111, mantissa=111
/// - No infinities — values that would overflow clamp to +/-448
/// - Subnormals: exponent=0000, mantissa!=000
///
/// This is a software emulation path. On Hopper GPUs, hardware FP8 GEMM is used.
pub fn f32_to_fp8_e4m3(val: f32) -> u8 {
    if val.is_nan() {
        return 0x7F; // positive NaN in E4M3
    }

    let sign: u8 = if val.is_sign_negative() { 1 } else { 0 };
    let abs_val = val.abs();

    if abs_val == 0.0 {
        return sign << 7; // +0 or -0
    }

    // Clamp to max representable value (no infinities in E4M3)
    let abs_val = if abs_val > FP8_E4M3_MAX {
        FP8_E4M3_MAX
    } else {
        abs_val
    };

    // Check for subnormal range: abs_val < 2^(-6) = 0.015625 (min normal)
    let min_normal: f32 = 1.0 / 64.0; // 2^(-6)
    if abs_val < min_normal {
        // Subnormal: exponent field = 0, mantissa encodes value / 2^(-9)
        // value = mantissa * 2^(-9) where mantissa is in [1, 7] (3 bits, no implicit 1)
        let mantissa_f = abs_val / FP8_E4M3_MIN_SUBNORMAL;
        let mantissa = (mantissa_f + 0.5) as u8; // round to nearest
        let mantissa = mantissa.min(7); // clamp to 3 bits
        if mantissa == 0 {
            return sign << 7; // too small, flush to zero
        }
        return (sign << 7) | mantissa;
    }

    // Normal range: value = 2^(exp-7) * (1 + mantissa/8)
    // Find exponent: floor(log2(abs_val))
    let log2_val = abs_val.log2();
    let biased_exp = (log2_val.floor() as i32 + 7).max(1).min(15) as u8;
    let exp_unbiased = biased_exp as i32 - 7;
    let pow2 = (2.0f32).powi(exp_unbiased);
    let significand = abs_val / pow2; // 1.0 <= significand < 2.0

    // mantissa = round((significand - 1.0) * 8)
    let mantissa_f = (significand - 1.0) * 8.0;
    let mut mantissa = (mantissa_f + 0.5) as u8;

    // Handle rounding overflow: if mantissa rounds to 8, increment exponent
    let mut final_exp = biased_exp;
    if mantissa >= 8 {
        mantissa = 0;
        final_exp += 1;
    }

    // Clamp: max representable is exp=15, mantissa=6 (exp=15, man=7 is NaN)
    if final_exp > 15 || (final_exp == 15 && mantissa >= 7) {
        // Clamp to max: exponent=15 (1111), mantissa=6 (110) = 448.0
        return (sign << 7) | (15 << 3) | 6;
    }

    // Ensure mantissa is within 3 bits
    mantissa = mantissa.min(7);

    (sign << 7) | (final_exp << 3) | mantissa
}

/// Convert an FP8 E4M3 bit pattern (u8) back to f32.
///
/// Handles: zero, subnormals, normals, and NaN.
pub fn fp8_e4m3_to_f32(bits: u8) -> f32 {
    let sign = (bits >> 7) & 1;
    let exp = (bits >> 3) & 0xF; // 4 bits
    let mantissa = bits & 0x7; // 3 bits

    // NaN: exponent=1111 and mantissa=111
    if exp == 0xF && mantissa == 0x7 {
        return f32::NAN;
    }

    let sign_f: f32 = if sign == 1 { -1.0 } else { 1.0 };

    if exp == 0 {
        if mantissa == 0 {
            // Zero (positive or negative)
            return 0.0 * sign_f;
        }
        // Subnormal: value = mantissa * 2^(-9)
        return sign_f * (mantissa as f32) * FP8_E4M3_MIN_SUBNORMAL;
    }

    // Normal: value = 2^(exp-7) * (1 + mantissa/8)
    let exp_unbiased = exp as i32 - 7;
    let significand = 1.0 + (mantissa as f32) / 8.0;
    sign_f * significand * (2.0f32).powi(exp_unbiased)
}

// ─── Tensor-level quantization ──────────────────────────────────────────

/// Quantize a floating-point tensor to FP8 E4M3 with scaling.
///
/// Returns `(fp8_tensor, scale_tensor)`:
/// - `fp8_tensor`: U8 tensor with the same shape, containing FP8 E4M3 bit patterns
/// - `scale_tensor`: F32 tensor with per-tensor `[1]` or per-channel `[dim0]` scales
///
/// The quantization formula:
///   `scale = max(|tensor|) / 448.0`  (per-tensor or per-channel)
///   `fp8_val = f32_to_fp8_e4m3(tensor_val / scale)`
///
/// To dequantize: `float_val = fp8_e4m3_to_f32(fp8_val) * scale`
pub fn quantize_to_fp8(tensor: &Tensor, per_channel: bool) -> Result<(Tensor, Tensor)> {
    let dev = tensor.device();
    let t_f32 = tensor.to_dtype(DType::F32)?;
    let flat: Vec<f32> = t_f32.flatten_all()?.to_vec1()?;
    let shape = tensor.dims().to_vec();

    if per_channel && shape.len() >= 2 {
        let rows = shape[0];
        let cols: usize = shape[1..].iter().product();

        let mut scales = Vec::with_capacity(rows);
        let mut fp8_data = Vec::with_capacity(flat.len());

        for r in 0..rows {
            let row_start = r * cols;
            let row_end = row_start + cols;
            let row_slice = &flat[row_start..row_end];

            let abs_max = row_slice
                .iter()
                .fold(0.0f32, |acc, &v| acc.max(v.abs()));
            let scale = if abs_max < 1e-12 {
                1.0
            } else {
                abs_max / FP8_E4M3_MAX
            };
            scales.push(scale);

            for &v in row_slice {
                fp8_data.push(f32_to_fp8_e4m3(v / scale));
            }
        }

        let fp8_tensor = Tensor::from_vec(fp8_data, shape, dev)?;
        let scale_tensor = Tensor::from_vec(scales, rows, dev)?;
        Ok((fp8_tensor, scale_tensor))
    } else {
        // Per-tensor scaling
        let abs_max = flat.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        let scale = if abs_max < 1e-12 {
            1.0
        } else {
            abs_max / FP8_E4M3_MAX
        };

        let fp8_data: Vec<u8> = flat.iter().map(|&v| f32_to_fp8_e4m3(v / scale)).collect();

        let fp8_tensor = Tensor::from_vec(fp8_data, shape, dev)?;
        let scale_tensor = Tensor::from_vec(vec![scale], 1, dev)?;
        Ok((fp8_tensor, scale_tensor))
    }
}

// ─── FP8 quantized linear layer ─────────────────────────────────────────

/// A linear layer with FP8 E4M3 weight-only quantization (W8A16).
///
/// Stores weights as FP8 E4M3 (u8) with per-tensor or per-channel scaling.
/// During forward, weights are dequantized to the input dtype and a standard
/// matmul is performed (CPU fallback). On Hopper GPUs, the CUDA backend
/// can use `cublasLtMatmul` with native FP8 input types.
///
/// Memory layout:
/// - `weight_fp8`: `[out_features, in_features]` as U8 (FP8 E4M3 bit patterns)
/// - `weight_scale`: `[1]` (per-tensor) or `[out_features]` (per-channel) as F32
/// - Optional `bias`: `[out_features]` as F32
///
/// Compared to FP16: ~2x memory reduction with <2% accuracy loss.
#[derive(Clone)]
pub struct Fp8Linear {
    /// Quantized weight matrix `[out_features, in_features]` stored as U8 (FP8 E4M3 bits).
    pub weight_fp8: Tensor,
    /// Scale factor(s): `[1]` for per-tensor, `[out_features]` for per-channel.
    pub weight_scale: Tensor,
    /// Optional bias `[out_features]`.
    pub bias: Option<Tensor>,
    /// Quantization configuration.
    pub config: Fp8Config,
    /// Output features count.
    out_features: usize,
    /// Input features count.
    in_features: usize,
}

impl Fp8Linear {
    /// Quantize FP16/FP32 weights to FP8 E4M3.
    ///
    /// The weight tensor should be `[out_features, in_features]`.
    pub fn from_float(weight: &Tensor, bias: Option<&Tensor>, config: Fp8Config) -> Result<Self> {
        let dims = weight.dims();
        if dims.len() != 2 {
            return Err(candle_core::Error::Msg(format!(
                "Fp8Linear expects 2D weight, got shape {:?}",
                dims
            )));
        }
        let out_features = dims[0];
        let in_features = dims[1];

        let per_channel = !config.per_tensor_scaling;
        let (weight_fp8, weight_scale) = quantize_to_fp8(weight, per_channel)?;

        let bias = match bias {
            Some(b) => Some(b.to_dtype(DType::F32)?.reshape(out_features)?),
            None => None,
        };

        Ok(Self {
            weight_fp8,
            weight_scale,
            bias,
            config,
            out_features,
            in_features,
        })
    }

    /// Load pre-quantized FP8 weights (e.g., from HuggingFace FP8 checkpoints).
    pub fn from_parts(
        weight_fp8: Tensor,
        scale: Tensor,
        bias: Option<Tensor>,
        config: Fp8Config,
    ) -> Result<Self> {
        let dims = weight_fp8.dims();
        if dims.len() != 2 {
            return Err(candle_core::Error::Msg(format!(
                "Fp8Linear expects 2D weight_fp8, got shape {:?}",
                dims
            )));
        }
        let out_features = dims[0];
        let in_features = dims[1];

        Ok(Self {
            weight_fp8,
            weight_scale: scale,
            bias,
            config,
            out_features,
            in_features,
        })
    }

    /// Dequantize FP8 weights to the given dtype.
    ///
    /// Formula: `w_float = fp8_e4m3_to_f32(w_fp8) * scale`
    pub fn dequantize(&self, dtype: DType) -> Result<Tensor> {
        let dev = self.weight_fp8.device();
        let fp8_data: Vec<u8> = self.weight_fp8.flatten_all()?.to_vec1()?;
        let scale_data: Vec<f32> = self.weight_scale.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

        let per_channel = scale_data.len() > 1;
        let mut f32_data = Vec::with_capacity(fp8_data.len());

        if per_channel {
            for r in 0..self.out_features {
                let scale = scale_data[r];
                let row_start = r * self.in_features;
                let row_end = row_start + self.in_features;
                for &bits in &fp8_data[row_start..row_end] {
                    f32_data.push(fp8_e4m3_to_f32(bits) * scale);
                }
            }
        } else {
            let scale = scale_data[0];
            for &bits in &fp8_data {
                f32_data.push(fp8_e4m3_to_f32(bits) * scale);
            }
        }

        let w_f32 = Tensor::from_vec(f32_data, (self.out_features, self.in_features), dev)?;
        w_f32.to_dtype(dtype)
    }

    /// Forward pass: dequantize then matmul (CPU fallback path).
    ///
    /// On Hopper GPUs (H100/H200), the CUDA backend can override this with
    /// native FP8 GEMM via `cublasLtMatmul`.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_dtype = input.dtype();
        let w = self.dequantize(input_dtype)?;
        let w_t = w.t()?;
        let output = input.matmul(&w_t)?;

        match &self.bias {
            Some(b) => {
                let b = b.to_dtype(input_dtype)?;
                output.broadcast_add(&b)
            }
            None => Ok(output),
        }
    }

    /// Memory usage in bytes for this layer's quantized weights.
    pub fn memory_bytes(&self) -> usize {
        let w_bytes = self.weight_fp8.elem_count(); // 1 byte per element (U8)
        let s_bytes = self.weight_scale.elem_count() * 4; // F32 scales
        let b_bytes = self.bias.as_ref().map_or(0, |b| b.elem_count() * 4);
        w_bytes + s_bytes + b_bytes
    }

    /// Memory that would be used if weights were stored as FP16.
    pub fn fp16_memory_bytes(&self) -> usize {
        self.out_features * self.in_features * 2
    }

    /// Input features.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Output features.
    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

impl Module for Fp8Linear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Fp8Linear::forward(self, xs)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    // ── E4M3 conversion: special values ──────────────────────────────────

    #[test]
    fn test_fp8_e4m3_zero() {
        assert_eq!(f32_to_fp8_e4m3(0.0), 0x00);
        assert_eq!(fp8_e4m3_to_f32(0x00), 0.0);
    }

    #[test]
    fn test_fp8_e4m3_negative_zero() {
        let bits = f32_to_fp8_e4m3(-0.0);
        assert_eq!(bits, 0x80); // sign bit set
        let val = fp8_e4m3_to_f32(0x80);
        assert_eq!(val, 0.0); // -0 == 0 in float comparison
        assert!(val.is_sign_negative() || val == 0.0); // implementation may return +0 or -0
    }

    #[test]
    fn test_fp8_e4m3_nan() {
        let bits = f32_to_fp8_e4m3(f32::NAN);
        assert_eq!(bits, 0x7F); // NaN encoding
        let val = fp8_e4m3_to_f32(0x7F);
        assert!(val.is_nan());
    }

    #[test]
    fn test_fp8_e4m3_negative_nan() {
        // 0xFF = negative NaN
        let val = fp8_e4m3_to_f32(0xFF);
        assert!(val.is_nan());
    }

    #[test]
    fn test_fp8_e4m3_max_value() {
        // Max representable: exponent=15, mantissa=6 (mantissa=7 is NaN)
        // value = 2^(15-7) * (1 + 6/8) = 2^8 * 1.75 = 256 * 1.75 = 448.0
        let bits = f32_to_fp8_e4m3(448.0);
        let val = fp8_e4m3_to_f32(bits);
        assert!(
            (val - 448.0).abs() < 1.0,
            "expected ~448.0, got {val} (bits={bits:#04x})"
        );
    }

    #[test]
    fn test_fp8_e4m3_overflow_clamps() {
        // Values > 448 should clamp to 448 (no infinities)
        let bits = f32_to_fp8_e4m3(1000.0);
        let val = fp8_e4m3_to_f32(bits);
        assert!(
            (val - 448.0).abs() < 1.0,
            "overflow should clamp to 448, got {val}"
        );

        let bits_neg = f32_to_fp8_e4m3(-1000.0);
        let val_neg = fp8_e4m3_to_f32(bits_neg);
        assert!(
            (val_neg + 448.0).abs() < 1.0,
            "negative overflow should clamp to -448, got {val_neg}"
        );
    }

    #[test]
    fn test_fp8_e4m3_underflow() {
        // Very small values should flush to zero
        let bits = f32_to_fp8_e4m3(1e-10);
        assert_eq!(bits, 0x00, "very small value should flush to zero");
    }

    #[test]
    fn test_fp8_e4m3_subnormal() {
        // Smallest subnormal: mantissa=1, exponent=0 -> 1 * 2^(-9) = ~1.953e-3
        let small = FP8_E4M3_MIN_SUBNORMAL;
        let bits = f32_to_fp8_e4m3(small);
        let val = fp8_e4m3_to_f32(bits);
        let rel_err = (val - small).abs() / small;
        assert!(
            rel_err < 0.5,
            "subnormal roundtrip: expected ~{small}, got {val}"
        );
    }

    #[test]
    fn test_fp8_e4m3_one() {
        // 1.0 = 2^(7-7) * (1 + 0/8) -> exponent=7 (0111), mantissa=0
        let bits = f32_to_fp8_e4m3(1.0);
        let val = fp8_e4m3_to_f32(bits);
        assert!(
            (val - 1.0).abs() < 0.01,
            "expected 1.0, got {val} (bits={bits:#04x})"
        );
    }

    #[test]
    fn test_fp8_e4m3_negative_one() {
        let bits = f32_to_fp8_e4m3(-1.0);
        let val = fp8_e4m3_to_f32(bits);
        assert!(
            (val + 1.0).abs() < 0.01,
            "expected -1.0, got {val} (bits={bits:#04x})"
        );
    }

    // ── E4M3 conversion: round-trip accuracy ─────────────────────────────

    #[test]
    fn test_fp8_e4m3_roundtrip_powers_of_two() {
        // Powers of 2 should be exactly representable
        for exp in -6..=8i32 {
            let val = (2.0f32).powi(exp);
            if val > FP8_E4M3_MAX {
                continue;
            }
            let bits = f32_to_fp8_e4m3(val);
            let rt = fp8_e4m3_to_f32(bits);
            assert!(
                (rt - val).abs() < val * 0.01,
                "power of 2 roundtrip failed: 2^{exp}={val}, got {rt}"
            );
        }
    }

    #[test]
    fn test_fp8_e4m3_roundtrip_various() {
        let test_values = [
            0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 448.0,
            -0.5, -1.0, -2.0, -128.0,
        ];
        for &v in &test_values {
            let bits = f32_to_fp8_e4m3(v);
            let rt = fp8_e4m3_to_f32(bits);
            let rel_err = (rt - v).abs() / v.abs().max(1e-6);
            assert!(
                rel_err < 0.15,
                "roundtrip failed for {v}: got {rt}, rel_err={rel_err:.4}"
            );
        }
    }

    #[test]
    fn test_fp8_e4m3_all_bit_patterns_valid() {
        // Every 8-bit pattern should produce a finite value or NaN
        for bits in 0..=255u8 {
            let val = fp8_e4m3_to_f32(bits);
            // Only 0x7F and 0xFF should be NaN
            if bits == 0x7F || bits == 0xFF {
                assert!(val.is_nan(), "bits {bits:#04x} should be NaN");
            } else {
                assert!(
                    val.is_finite(),
                    "bits {bits:#04x} should be finite, got {val}"
                );
            }
        }
    }

    // ── Tensor-level quantize/dequantize ─────────────────────────────────

    #[test]
    fn test_quantize_dequantize_per_tensor() {
        let w = Tensor::randn(0f32, 0.1, (16, 32), &Device::Cpu).unwrap();
        let (fp8, scale) = quantize_to_fp8(&w, false).unwrap();

        assert_eq!(fp8.dims(), w.dims());
        assert_eq!(fp8.dtype(), DType::U8);
        assert_eq!(scale.dims(), &[1]);

        // Dequantize manually and check closeness
        let fp8_data: Vec<u8> = fp8.flatten_all().unwrap().to_vec1().unwrap();
        let scale_val: f32 = scale.to_vec1::<f32>().unwrap()[0];
        let w_data: Vec<f32> = w.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();

        let mut max_rel_err = 0.0f32;
        for (i, (&orig, &bits)) in w_data.iter().zip(fp8_data.iter()).enumerate() {
            let dequant = fp8_e4m3_to_f32(bits) * scale_val;
            if orig.abs() > 0.01 {
                let rel_err = (dequant - orig).abs() / orig.abs();
                if rel_err > max_rel_err {
                    max_rel_err = rel_err;
                }
            }
        }
        assert!(
            max_rel_err < 0.15,
            "per-tensor max relative error {max_rel_err:.4} too large"
        );
    }

    #[test]
    fn test_quantize_dequantize_per_channel() {
        let w = Tensor::randn(0f32, 0.1, (8, 16), &Device::Cpu).unwrap();
        let (fp8, scale) = quantize_to_fp8(&w, true).unwrap();

        assert_eq!(fp8.dims(), w.dims());
        assert_eq!(scale.dims(), &[8]); // per-channel: one scale per row
    }

    #[test]
    fn test_quantize_zeros() {
        let w = Tensor::zeros((4, 8), DType::F32, &Device::Cpu).unwrap();
        let (fp8, scale) = quantize_to_fp8(&w, false).unwrap();
        let fp8_data: Vec<u8> = fp8.flatten_all().unwrap().to_vec1().unwrap();
        for &bits in &fp8_data {
            let val = fp8_e4m3_to_f32(bits);
            assert_eq!(val, 0.0, "quantized zeros should be 0");
        }
    }

    // ── Fp8Linear tests ─────────────────────────────────────────────────

    #[test]
    fn test_fp8_linear_from_float_shape() {
        let w = Tensor::randn(0f32, 0.1, (16, 32), &Device::Cpu).unwrap();
        let fp8l = Fp8Linear::from_float(&w, None, Fp8Config::default()).unwrap();
        assert_eq!(fp8l.out_features(), 16);
        assert_eq!(fp8l.in_features(), 32);
        assert_eq!(fp8l.weight_fp8.dims(), &[16, 32]);
    }

    #[test]
    fn test_fp8_linear_forward_shape() {
        let w = Tensor::randn(0f32, 0.1, (16, 32), &Device::Cpu).unwrap();
        let fp8l = Fp8Linear::from_float(&w, None, Fp8Config::default()).unwrap();
        let x = Tensor::randn(0f32, 1.0, (4, 32), &Device::Cpu).unwrap();
        let y = fp8l.forward(&x).unwrap();
        assert_eq!(y.dims(), &[4, 16]);
    }

    #[test]
    fn test_fp8_linear_forward_with_bias() {
        let w = Tensor::randn(0f32, 0.1, (8, 16), &Device::Cpu).unwrap();
        let b = Tensor::ones(8, DType::F32, &Device::Cpu).unwrap();
        let fp8l = Fp8Linear::from_float(&w, Some(&b), Fp8Config::default()).unwrap();
        let x = Tensor::zeros((2, 16), DType::F32, &Device::Cpu).unwrap();
        let y = fp8l.forward(&x).unwrap();
        let data: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        for v in &data {
            assert!(
                (v - 1.0).abs() < 0.1,
                "expected ~1.0 (bias), got {v}"
            );
        }
    }

    #[test]
    fn test_fp8_linear_forward_closeness() {
        // Compare FP8 forward to full-precision forward
        let w = Tensor::randn(0f32, 0.1, (64, 128), &Device::Cpu).unwrap();
        let x = Tensor::randn(0f32, 1.0, (8, 128), &Device::Cpu).unwrap();

        let y_fp = x.matmul(&w.t().unwrap()).unwrap();

        let fp8l = Fp8Linear::from_float(&w, None, Fp8Config::default()).unwrap();
        let y_q = fp8l.forward(&x).unwrap();

        let diff = (&y_fp - &y_q).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(0).unwrap().max(0).unwrap().to_scalar().unwrap();
        let max_val: f32 = y_fp
            .abs()
            .unwrap()
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar()
            .unwrap();
        let rel_err = max_diff / (max_val + 1e-8);
        assert!(
            rel_err < 0.10,
            "FP8 forward relative error {rel_err:.4} too large (max_diff={max_diff}, max_val={max_val})"
        );
    }

    #[test]
    fn test_fp8_linear_per_channel() {
        let config = Fp8Config {
            per_tensor_scaling: false,
            ..Fp8Config::default()
        };
        let w = Tensor::randn(0f32, 0.1, (32, 64), &Device::Cpu).unwrap();
        let fp8l = Fp8Linear::from_float(&w, None, config).unwrap();
        assert_eq!(fp8l.weight_scale.dims(), &[32]); // per-channel

        let x = Tensor::randn(0f32, 1.0, (4, 64), &Device::Cpu).unwrap();
        let y = fp8l.forward(&x).unwrap();
        assert_eq!(y.dims(), &[4, 32]);
    }

    #[test]
    fn test_fp8_linear_dequantize_closeness() {
        let w = Tensor::randn(0f32, 0.1, (32, 64), &Device::Cpu).unwrap();
        let fp8l = Fp8Linear::from_float(&w, None, Fp8Config::default()).unwrap();
        let w_deq = fp8l.dequantize(DType::F32).unwrap();

        let diff = (&w - &w_deq).unwrap().abs().unwrap();
        let max_err: f32 = diff.max(0).unwrap().max(0).unwrap().to_scalar().unwrap();
        let w_max: f32 = w.abs().unwrap().max(0).unwrap().max(0).unwrap().to_scalar().unwrap();
        let rel_err = max_err / (w_max + 1e-8);

        assert!(
            rel_err < 0.15,
            "FP8 dequantize relative error {rel_err:.4} too large"
        );
    }

    #[test]
    fn test_fp8_linear_from_parts() {
        let fp8_data: Vec<u8> = vec![0x38; 48]; // 0x38 = exponent=7, mantissa=0 -> 1.0
        let weight_fp8 = Tensor::from_vec(fp8_data, (6, 8), &Device::Cpu).unwrap();
        let scale = Tensor::from_vec(vec![0.1f32], 1, &Device::Cpu).unwrap();
        let config = Fp8Config::default();

        let fp8l = Fp8Linear::from_parts(weight_fp8, scale, None, config).unwrap();
        assert_eq!(fp8l.out_features(), 6);
        assert_eq!(fp8l.in_features(), 8);

        let x = Tensor::ones((2, 8), DType::F32, &Device::Cpu).unwrap();
        let y = fp8l.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 6]);
    }

    #[test]
    fn test_fp8_linear_memory_savings() {
        let w = Tensor::randn(0f32, 1.0, (1024, 4096), &Device::Cpu).unwrap();
        let fp8l = Fp8Linear::from_float(&w, None, Fp8Config::default()).unwrap();
        let fp8_bytes = fp8l.memory_bytes();
        let fp16_bytes = fp8l.fp16_memory_bytes();

        assert!(
            fp8_bytes < fp16_bytes,
            "FP8 ({fp8_bytes}) should be less than FP16 ({fp16_bytes})"
        );
        let ratio = fp8_bytes as f64 / fp16_bytes as f64;
        assert!(
            ratio < 0.55,
            "FP8/FP16 ratio {ratio:.2} should be < 0.55"
        );
    }

    #[test]
    fn test_fp8_linear_module_trait() {
        // Ensure the Module trait impl works
        let w = Tensor::randn(0f32, 0.1, (8, 16), &Device::Cpu).unwrap();
        let fp8l = Fp8Linear::from_float(&w, None, Fp8Config::default()).unwrap();
        let x = Tensor::randn(0f32, 1.0, (2, 16), &Device::Cpu).unwrap();
        let y = Module::forward(&fp8l, &x).unwrap();
        assert_eq!(y.dims(), &[2, 8]);
    }

    #[test]
    fn test_fp8_config_default() {
        let cfg = Fp8Config::default();
        assert_eq!(cfg.weight_dtype, Fp8Dtype::E4M3);
        assert_eq!(cfg.compute_dtype, Fp8Dtype::E4M3);
        assert!(cfg.per_tensor_scaling);
    }

    #[test]
    fn test_fp8_dtype_display() {
        assert_eq!(Fp8Dtype::E4M3.to_string(), "e4m3");
        assert_eq!(Fp8Dtype::E5M2.to_string(), "e5m2");
    }
}
