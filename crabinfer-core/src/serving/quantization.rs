//! Weight quantization for serving.
//!
//! Implements INT8 weight-only quantization (W8A16): weights are stored as
//! `i8` with per-channel scale factors, and dequantized to the activation
//! dtype (F16/F32) on-the-fly during matrix multiplication.
//!
//! This is useful when you want to halve memory usage compared to FP16 without
//! needing pre-quantized checkpoints (GPTQ/AWQ). The quantization happens at
//! model load time with no calibration data required.

use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::Module;

// ─── Quantization config ─────────────────────────────────────────────────

/// Quantization method for model weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMethod {
    /// No additional quantization — use weights as-is from GGUF.
    None,
    /// INT8 weight-only (W8A16): 8-bit signed weights, FP16/F32 activations.
    /// Per-channel symmetric quantization: `w_float = w_int8 * scale`.
    Int8WeightOnly,
    /// GPTQ 4-bit (W4A16): 4-bit weights packed into i32, FP16 activations.
    /// Group-wise asymmetric quantization with zero points.
    /// `w_float = (w_int4 - zeros) * scales`
    Gptq,
    /// AWQ 4-bit (W4A16): Activation-Aware Weight Quantization.
    /// Same INT4 packed format as GPTQ but uses activation-aware calibration
    /// for better accuracy. Dequantization: `w_float = (w_int4 - zeros) * scales`
    Awq,
}

impl Default for QuantizationMethod {
    fn default() -> Self {
        Self::None
    }
}

impl std::fmt::Display for QuantizationMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Int8WeightOnly => write!(f, "int8"),
            Self::Gptq => write!(f, "gptq"),
            Self::Awq => write!(f, "awq"),
        }
    }
}

impl std::str::FromStr for QuantizationMethod {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "none" | "" => Ok(Self::None),
            "int8" | "int8-wo" | "w8a16" => Ok(Self::Int8WeightOnly),
            "gptq" | "w4a16" => Ok(Self::Gptq),
            "awq" => Ok(Self::Awq),
            other => Err(format!("unknown quantization method: {other}")),
        }
    }
}

// ─── KV Cache dtype ──────────────────────────────────────────────────────

/// Data type for the KV cache.
///
/// Lower precision KV caches trade minimal accuracy for significant memory
/// savings, allowing longer context lengths or more concurrent sequences.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KVCacheDType {
    /// Use the model's native compute dtype (currently F32).
    Auto,
    /// FP16 — 2x memory savings vs F32. Good for most models.
    F16,
    /// BF16 — 2x memory savings with better dynamic range than F16.
    /// Preferred for large models that produce extreme KV values.
    BF16,
}

impl Default for KVCacheDType {
    fn default() -> Self {
        Self::Auto
    }
}

impl KVCacheDType {
    /// Resolve to a concrete `DType`. `Auto` maps to the provided default.
    pub fn resolve(&self, default: DType) -> DType {
        match self {
            Self::Auto => default,
            Self::F16 => DType::F16,
            Self::BF16 => DType::BF16,
        }
    }
}

impl std::fmt::Display for KVCacheDType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::F16 => write!(f, "fp16"),
            Self::BF16 => write!(f, "bf16"),
        }
    }
}

impl std::str::FromStr for KVCacheDType {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" | "" => Ok(Self::Auto),
            "fp16" | "f16" | "half" => Ok(Self::F16),
            "bf16" | "bfloat16" => Ok(Self::BF16),
            other => Err(format!("unknown KV cache dtype: {other} (valid: auto, fp16, bf16)")),
        }
    }
}

// ─── INT8 quantized linear layer ─────────────────────────────────────────

/// A linear layer with INT8 weight-only quantization.
///
/// Stores weights as `i8` with per-output-channel scale factors.
/// During forward, weights are dequantized to the input dtype and
/// a standard matmul is performed.
///
/// Memory layout:
/// - `weights_int8`: `[out_features, in_features]` as `i8` (stored as U8 tensor)
/// - `scales`: `[out_features]` as `F32`
/// - Optional `bias`: `[out_features]` as `F32`
///
/// Compared to FP16:
/// - 2x memory reduction for weights
/// - Slight overhead from dequantization (typically <5% for large matrices)
pub struct QuantizedLinear {
    /// Quantized weight matrix `[out_features, in_features]` stored as U8
    /// (representing signed i8 values via offset: stored = i8_val + 128).
    weights_u8: Tensor,
    /// Per-output-channel scale factors `[out_features]`.
    scales: Tensor,
    /// Optional bias `[out_features]`.
    bias: Option<Tensor>,
    /// Output features count (for reshaping).
    out_features: usize,
}

impl QuantizedLinear {
    /// Quantize a floating-point weight tensor to INT8.
    ///
    /// Uses symmetric per-channel quantization:
    /// - For each output channel, find `max_abs = max(|w|)`
    /// - `scale = max_abs / 127.0`
    /// - `w_int8 = round(w / scale)`, clamped to `[-128, 127]`
    ///
    /// The weight tensor should be `[out_features, in_features]`.
    pub fn from_float(weight: &Tensor, bias: Option<&Tensor>) -> Result<Self> {
        let dims = weight.dims();
        if dims.len() != 2 {
            return Err(candle_core::Error::Msg(format!(
                "QuantizedLinear expects 2D weight, got shape {:?}",
                dims
            )));
        }
        let out_features = dims[0];

        // Compute per-channel absolute maximum: [out_features]
        let w_f32 = weight.to_dtype(DType::F32)?;
        let abs_max = w_f32.abs()?.max(1)?; // [out_features]

        // Scale = abs_max / 127.0 (avoid division by zero)
        let epsilon = Tensor::new(&[1e-10f32], weight.device())?
            .broadcast_as(abs_max.shape())?;
        let scales = (abs_max.maximum(&epsilon)? / 127.0)?;

        // Quantize: w_int8 = round(w / scale), clamped to [-128, 127]
        let scales_col = scales.reshape((out_features, 1))?;
        let w_scaled = w_f32.broadcast_div(&scales_col)?;
        let w_rounded = w_scaled.round()?;
        let w_clamped = w_rounded.clamp(-128.0f32, 127.0f32)?;

        // Store as U8 with offset: stored = int8_val + 128
        // This avoids needing a signed integer dtype in candle.
        let w_offset = (w_clamped + 128.0f64)?;
        let weights_u8 = w_offset.to_dtype(DType::U8)?;

        let bias = match bias {
            Some(b) => Some(b.to_dtype(DType::F32)?.reshape(out_features)?),
            None => None,
        };

        Ok(Self {
            weights_u8,
            scales: scales.reshape(out_features)?,
            bias,
            out_features,
        })
    }

    /// Create from pre-quantized INT8 data (for loading saved quantized models).
    pub fn from_parts(
        weights_u8: Tensor,
        scales: Tensor,
        bias: Option<Tensor>,
    ) -> Result<Self> {
        let out_features = weights_u8.dim(0)?;
        Ok(Self {
            weights_u8,
            scales,
            bias,
            out_features,
        })
    }

    /// Dequantize weights to the given dtype.
    ///
    /// `w_float = (w_u8 - 128) * scale`
    fn dequantize(&self, dtype: DType) -> Result<Tensor> {
        // Convert U8 to F32, subtract offset to get signed values
        let w_f32 = (self.weights_u8.to_dtype(DType::F32)? - 128.0f64)?;
        // Multiply by per-channel scale
        let scales_col = self.scales.reshape((self.out_features, 1))?;
        let w_dequant = w_f32.broadcast_mul(&scales_col)?;
        w_dequant.to_dtype(dtype)
    }

    /// Memory usage in bytes for this layer's quantized weights.
    pub fn memory_bytes(&self) -> usize {
        let w_bytes = self.weights_u8.elem_count(); // 1 byte per element
        let s_bytes = self.scales.elem_count() * 4; // f32 scales
        let b_bytes = self.bias.as_ref().map_or(0, |b| b.elem_count() * 4);
        w_bytes + s_bytes + b_bytes
    }

    /// Memory that would be used if weights were stored as FP16.
    pub fn fp16_memory_bytes(&self) -> usize {
        self.weights_u8.elem_count() * 2
    }
}

impl Module for QuantizedLinear {
    /// Forward pass: dequantize weights and perform matmul.
    ///
    /// Input `xs` shape: `[..., in_features]`
    /// Output shape: `[..., out_features]`
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let input_dtype = xs.dtype();
        let w = self.dequantize(input_dtype)?;

        // xs: [..., in_features], w: [out_features, in_features]
        // matmul needs w transposed: [..., in_features] @ [in_features, out_features]
        let w_t = w.t()?;
        let output = xs.matmul(&w_t)?;

        match &self.bias {
            Some(b) => {
                let b = b.to_dtype(input_dtype)?;
                output.broadcast_add(&b)
            }
            None => Ok(output),
        }
    }
}

// ─── GPTQ 4-bit quantized linear layer ────────────────────────────────────

/// GPTQ 4-bit group-wise quantized linear layer (W4A16).
///
/// Stores weights as 4-bit integers packed into `u32` values (8 weights per u32),
/// with per-group scale factors and zero points for asymmetric dequantization.
///
/// Memory layout:
/// - `qweight`: `[in_features / 8, out_features]` as `U32` — packed 4-bit weights
/// - `scales`: `[in_features / group_size, out_features]` as `F16/F32` — per-group scales
/// - `qzeros`: `[in_features / group_size, out_features / 8]` as `U32` — packed 4-bit zeros
/// - Optional `bias`: `[out_features]`
///
/// Dequantization formula: `w_float = (w_int4 - zero) * scale`
///
/// Compared to FP16:
/// - 4x memory reduction for weights (4-bit vs 16-bit)
/// - Slight overhead from dequantization
/// - Group size 128 is typical (tradeoff: smaller groups = better accuracy, more scales)
pub struct GptqLinear {
    /// Packed 4-bit weights `[in_features / 8, out_features]` as U32.
    /// Each u32 stores 8 INT4 values in the low bits: bits[3:0] = w0, bits[7:4] = w1, etc.
    pub qweight: Tensor,
    /// Per-group scale factors `[num_groups, out_features]`.
    pub scales: Tensor,
    /// Packed 4-bit zero points `[num_groups, out_features / 8]` as U32.
    pub qzeros: Tensor,
    /// Optional bias `[out_features]`.
    pub bias: Option<Tensor>,
    /// Group size (number of input features sharing the same scale/zero).
    group_size: usize,
    /// Number of input features.
    in_features: usize,
    /// Number of output features.
    out_features: usize,
    /// Weights reformatted into Marlin tile layout for fast CUDA GEMM.
    /// `None` until `reformat_for_marlin()` is called (Plan 03).
    pub(crate) qweight_marlin: Option<Tensor>,
    /// Kernel backend for dispatching fused ops (set after model load in Plan 03).
    /// `None` when running on CPU or Metal.
    pub(crate) backend: Option<Arc<dyn super::kernels::backend::KernelBackend>>,
}

impl GptqLinear {
    /// Create a GPTQ linear layer from pre-quantized components.
    ///
    /// This is used when loading GPTQ model checkpoints (e.g., from HuggingFace).
    pub fn from_parts(
        qweight: Tensor,
        scales: Tensor,
        qzeros: Tensor,
        bias: Option<Tensor>,
        group_size: usize,
    ) -> Result<Self> {
        let qw_dims = qweight.dims();
        if qw_dims.len() != 2 {
            return Err(candle_core::Error::Msg(format!(
                "GptqLinear: qweight must be 2D, got {:?}",
                qw_dims
            )));
        }
        let in_features = qw_dims[0] * 8; // 8 INT4 values packed per u32
        let out_features = qw_dims[1];

        Ok(Self {
            qweight,
            scales,
            qzeros,
            bias,
            group_size,
            in_features,
            out_features,
            qweight_marlin: None,
            backend: None,
        })
    }

    /// Quantize a floating-point weight tensor to GPTQ 4-bit format.
    ///
    /// This is a simplified quantization (no calibration data / Hessian-based
    /// optimization like the original GPTQ algorithm). It uses simple
    /// min/max group-wise quantization as a baseline.
    pub fn from_float(weight: &Tensor, bias: Option<&Tensor>, group_size: usize) -> Result<Self> {
        let dims = weight.dims();
        if dims.len() != 2 {
            return Err(candle_core::Error::Msg(format!(
                "GptqLinear expects 2D weight [out, in], got {:?}",
                dims
            )));
        }
        let out_features = dims[0];
        let in_features = dims[1];

        if in_features % group_size != 0 {
            return Err(candle_core::Error::Msg(format!(
                "in_features ({in_features}) must be divisible by group_size ({group_size})"
            )));
        }
        if in_features % 8 != 0 {
            return Err(candle_core::Error::Msg(format!(
                "in_features ({in_features}) must be divisible by 8 for INT4 packing"
            )));
        }

        let dev = weight.device();
        let w_f32 = weight.to_dtype(DType::F32)?;
        let num_groups = in_features / group_size;

        // Transpose to [in_features, out_features] for column-major GPTQ packing
        let w_t = w_f32.t()?;

        // Process group by group
        let mut all_scales = Vec::new();
        let mut all_zeros = Vec::new();
        let mut all_qweights = Vec::new();

        for g in 0..num_groups {
            let start = g * group_size;
            let end = start + group_size;
            // Slice [start:end, :] from [in_features, out_features]
            let group_w = w_t.narrow(0, start, group_size)?;

            // Find min/max per output channel across this group
            let gmin = group_w.min(0)?; // [out_features]
            let gmax = group_w.max(0)?; // [out_features]

            // Asymmetric quantization to [0, 15]:
            // scale = (max - min) / 15
            // zero = round(-min / scale)
            // w_int4 = round(w / scale) + zero, clamped to [0, 15]
            let range = (&gmax - &gmin)?;
            let epsilon = Tensor::new(&[1e-10f32], dev)?.broadcast_as(range.shape())?;
            let scale = (range.maximum(&epsilon)? / 15.0)?;

            // zero = round(-min / scale), clamped to [0, 15]
            let neg_min = gmin.neg()?;
            let zero_f = (neg_min.broadcast_div(&scale))?;
            let zero_clamped = zero_f.round()?.clamp(0.0f32, 15.0f32)?;

            all_scales.push(scale);
            all_zeros.push(zero_clamped);

            // Quantize: w_int4 = round(w / scale) + zero, clamped to [0, 15]
            let scale_col = all_scales.last().unwrap().unsqueeze(0)?; // [1, out_features]
            let zero_col = all_zeros.last().unwrap().unsqueeze(0)?;
            let w_scaled = group_w.broadcast_div(&scale_col)?;
            let w_int4 = w_scaled
                .broadcast_add(&zero_col)?
                .round()?
                .clamp(0.0f32, 15.0f32)?;
            all_qweights.push(w_int4);
        }

        // Stack scales: [num_groups, out_features]
        let scales = Tensor::stack(&all_scales, 0)?;

        // Pack zeros: each u32 holds 8 INT4 zero values
        // zeros shape before packing: [num_groups, out_features]
        // After packing: [num_groups, out_features / 8]
        let zeros_stacked = Tensor::stack(&all_zeros, 0)?; // [num_groups, out_features]
        let qzeros = pack_int4_tensor(&zeros_stacked, 1)?; // pack along dim 1

        // Stack and pack qweights:
        // qweights shape before packing: [in_features, out_features]
        // After packing: [in_features / 8, out_features]
        let qweights_stacked = Tensor::cat(&all_qweights, 0)?; // [in_features, out_features]
        let qweight = pack_int4_tensor(&qweights_stacked, 0)?; // pack along dim 0

        let bias = bias.map(|b| b.to_dtype(DType::F32).unwrap());

        Ok(Self {
            qweight,
            scales,
            qzeros,
            bias,
            group_size,
            in_features,
            out_features,
            qweight_marlin: None,
            backend: None,
        })
    }

    /// Reformat weights into Marlin tile layout (call once at load time on CUDA).
    ///
    /// Returns `Ok(true)` if reformatted, `Ok(false)` if skipped (non-CUDA or unaligned
    /// dimensions).
    ///
    /// Alignment requirement: both `out_features` (N) and `in_features` (K) must be
    /// multiples of 16 for the Marlin tile layout.
    ///
    /// NOTE: Full implementation in Plan 03. This stub always returns `Ok(false)`.
    pub fn reformat_for_marlin(
        &mut self,
        _backend: &dyn super::kernels::backend::KernelBackend,
    ) -> Result<bool> {
        // Stub — implemented in Plan 03
        Ok(false)
    }

    /// Dequantize weights to the given dtype.
    ///
    /// Returns `[out_features, in_features]` (standard linear layer layout).
    pub fn dequantize(&self, dtype: DType) -> Result<Tensor> {
        let dev = self.qweight.device();

        // Unpack qweight: [in_features / 8, out_features] → [in_features, out_features]
        let w_int4 = unpack_int4_tensor(&self.qweight, 0, self.in_features)?;

        // Unpack qzeros: [num_groups, out_features / 8] → [num_groups, out_features]
        let zeros = unpack_int4_tensor(&self.qzeros, 1, self.out_features)?;
        let zeros_f32 = zeros.to_dtype(DType::F32)?;

        let w_f32 = w_int4.to_dtype(DType::F32)?;
        let scales_f32 = self.scales.to_dtype(DType::F32)?;

        let num_groups = self.in_features / self.group_size;

        // Dequantize group by group, then concat
        let mut dequant_groups = Vec::with_capacity(num_groups);
        for g in 0..num_groups {
            let start = g * self.group_size;
            // w_group: [group_size, out_features]
            let w_group = w_f32.narrow(0, start, self.group_size)?;

            // scale, zero: [out_features] → [1, out_features]
            let scale = scales_f32.get(g)?.unsqueeze(0)?;
            let zero = zeros_f32.get(g)?.unsqueeze(0)?;

            // dequant = (w_int4 - zero) * scale
            let dequant = (w_group.broadcast_sub(&zero))?.broadcast_mul(&scale)?;
            dequant_groups.push(dequant);
        }

        // Concat: [in_features, out_features], then transpose to [out_features, in_features]
        let w_full = Tensor::cat(&dequant_groups, 0)?;
        w_full.t()?.to_dtype(dtype)
    }

    /// Memory usage in bytes for this layer's quantized weights.
    pub fn memory_bytes(&self) -> usize {
        let qw_bytes = self.qweight.elem_count() * 4; // u32
        let s_bytes = self.scales.elem_count() * 2; // f16 typical
        let qz_bytes = self.qzeros.elem_count() * 4; // u32
        let b_bytes = self.bias.as_ref().map_or(0, |b| b.elem_count() * 4);
        qw_bytes + s_bytes + qz_bytes + b_bytes
    }

    /// Memory that would be used if weights were stored as FP16.
    pub fn fp16_memory_bytes(&self) -> usize {
        self.in_features * self.out_features * 2
    }

    /// Input features.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Output features.
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Group size.
    pub fn group_size(&self) -> usize {
        self.group_size
    }
}

impl Module for GptqLinear {
    /// Forward pass: dequantize INT4 weights and perform matmul.
    ///
    /// Dispatch order:
    /// 1. Marlin fast path (Plan 03: populated qweight_marlin + backend) — currently stub
    /// 2. Naive dequant + matmul path (always active)
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Marlin fast path (Plan 03 will populate qweight_marlin and backend)
        if let Some(ref _qw_marlin) = self.qweight_marlin {
            // TODO: dispatch to self.backend.marlin_gemm() in Plan 03
            // For now, fall through to naive path
        }

        // Naive dequant + matmul path
        let input_dtype = xs.dtype();
        let w = self.dequantize(input_dtype)?;

        // xs: [..., in_features], w: [out_features, in_features]
        let w_t = w.t()?;
        let output = xs.matmul(&w_t)?;

        match &self.bias {
            Some(b) => {
                let b = b.to_dtype(input_dtype)?;
                output.broadcast_add(&b)
            }
            None => Ok(output),
        }
    }
}

// ─── AWQ 4-bit quantized linear layer ──────────────────────────────────────

/// AWQ (Activation-Aware Weight Quantization) 4-bit linear layer (W4A16).
///
/// AWQ uses the same INT4 packed weight format as GPTQ — the difference is in
/// the calibration algorithm (activation-aware scaling) which happens offline.
/// At inference time, the dequantization is identical:
///
///   `w_float = (w_int4 - zero) * scale`
///
/// Memory layout matches GPTQ:
/// - `qweight`: `[in_features / 8, out_features]` as `U32` — packed 4-bit weights
/// - `scales`: `[in_features / group_size, out_features]` as `F16/F32`
/// - `qzeros`: `[in_features / group_size, out_features / 8]` as `U32` — packed 4-bit zeros
/// - Optional `bias`: `[out_features]`
pub struct AwqLinear {
    /// The underlying GPTQ-format layer. AWQ reuses the same packed format.
    inner: GptqLinear,
}

impl AwqLinear {
    /// Create an AWQ linear layer from pre-quantized components.
    ///
    /// This is used when loading AWQ model checkpoints (e.g., from HuggingFace).
    /// The packed format is identical to GPTQ.
    pub fn from_parts(
        qweight: Tensor,
        scales: Tensor,
        qzeros: Tensor,
        bias: Option<Tensor>,
        group_size: usize,
    ) -> Result<Self> {
        let inner = GptqLinear::from_parts(qweight, scales, qzeros, bias, group_size)?;
        Ok(Self { inner })
    }

    /// Quantize a floating-point weight tensor to AWQ 4-bit format.
    ///
    /// This is a simplified quantization without activation-aware calibration.
    /// For production use, pre-quantized AWQ checkpoints from AutoAWQ should be
    /// loaded via `from_parts` instead. This method uses the same simple min/max
    /// quantization as GPTQ as a baseline.
    pub fn from_float(weight: &Tensor, bias: Option<&Tensor>, group_size: usize) -> Result<Self> {
        let inner = GptqLinear::from_float(weight, bias, group_size)?;
        Ok(Self { inner })
    }

    /// Dequantize weights to the given dtype.
    pub fn dequantize(&self, dtype: DType) -> Result<Tensor> {
        self.inner.dequantize(dtype)
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.inner.memory_bytes()
    }

    /// Memory that would be used if weights were FP16.
    pub fn fp16_memory_bytes(&self) -> usize {
        self.inner.fp16_memory_bytes()
    }

    /// Input features.
    pub fn in_features(&self) -> usize {
        self.inner.in_features()
    }

    /// Output features.
    pub fn out_features(&self) -> usize {
        self.inner.out_features()
    }

    /// Group size.
    pub fn group_size(&self) -> usize {
        self.inner.group_size()
    }
}

impl Module for AwqLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.inner.forward(xs)
    }
}

// ─── INT4 packing / unpacking helpers ─────────────────────────────────────

/// Pack a tensor of INT4 values (stored as F32 in [0, 15]) into u32.
///
/// Packs 8 consecutive values along `pack_dim` into one u32.
/// Input shape along `pack_dim` must be divisible by 8.
fn pack_int4_tensor(values: &Tensor, pack_dim: usize) -> Result<Tensor> {
    let dev = values.device();
    let shape = values.dims().to_vec();
    let n = shape[pack_dim];

    if n % 8 != 0 {
        return Err(candle_core::Error::Msg(format!(
            "pack_int4: dimension {pack_dim} size {n} must be divisible by 8"
        )));
    }

    let values_f32 = values.to_dtype(DType::F32)?;
    let flat: Vec<f32> = values_f32.flatten_all()?.to_vec1()?;

    // Compute strides for the packing dimension
    let outer: usize = shape[..pack_dim].iter().product();
    let inner: usize = shape[pack_dim + 1..].iter().product();
    let packed_n = n / 8;

    let mut packed = vec![0u32; outer * packed_n * inner];

    for o in 0..outer {
        for p in 0..packed_n {
            for i in 0..inner {
                let mut val = 0u32;
                for bit in 0..8u32 {
                    let src_idx = o * n * inner + (p * 8 + bit as usize) * inner + i;
                    let nibble = (flat[src_idx] as u32) & 0xF;
                    val |= nibble << (bit * 4);
                }
                let dst_idx = o * packed_n * inner + p * inner + i;
                packed[dst_idx] = val;
            }
        }
    }

    let mut packed_shape = shape;
    packed_shape[pack_dim] = packed_n;

    // Store as U32 tensor
    Tensor::from_vec(packed, packed_shape, dev)
}

/// Unpack a u32 tensor of packed INT4 values back to individual values.
///
/// Each u32 produces 8 INT4 values along `pack_dim`.
/// `target_size` is the unpacked size along that dimension.
fn unpack_int4_tensor(packed: &Tensor, pack_dim: usize, target_size: usize) -> Result<Tensor> {
    let dev = packed.device();
    let shape = packed.dims().to_vec();
    let packed_n = shape[pack_dim];

    if target_size != packed_n * 8 {
        return Err(candle_core::Error::Msg(format!(
            "unpack_int4: target_size {target_size} != packed_n {packed_n} * 8"
        )));
    }

    let outer: usize = shape[..pack_dim].iter().product();
    let inner: usize = shape[pack_dim + 1..].iter().product();

    // Read packed u32 values
    let flat_u32: Vec<u32> = packed.flatten_all()?.to_vec1()?;

    let mut unpacked = vec![0u8; outer * target_size * inner];

    for o in 0..outer {
        for p in 0..packed_n {
            for i in 0..inner {
                let src_idx = o * packed_n * inner + p * inner + i;
                let val = flat_u32[src_idx];
                for bit in 0..8usize {
                    let nibble = ((val >> (bit as u32 * 4)) & 0xF) as u8;
                    let dst_idx = o * target_size * inner + (p * 8 + bit) * inner + i;
                    unpacked[dst_idx] = nibble;
                }
            }
        }
    }

    let mut unpacked_shape = shape;
    unpacked_shape[pack_dim] = target_size;

    Tensor::from_vec(unpacked, unpacked_shape, dev)
}

// ─── Wrapper enum for optional quantization ──────────────────────────────

/// A linear layer that may or may not be quantized.
///
/// This is the main interface used by model code — it wraps either a
/// standard `QMatMul` (from GGUF) or an `Int8` quantized linear layer.
pub enum MaybeQuantizedLinear {
    /// Standard candle quantized matmul (GGUF native quantization).
    QMatMul(candle_core::quantized::QMatMul),
    /// INT8 weight-only quantization (W8A16).
    Int8(QuantizedLinear),
    /// GPTQ 4-bit quantization (W4A16).
    Gptq(GptqLinear),
    /// AWQ 4-bit quantization (W4A16).
    Awq(AwqLinear),
}

impl Module for MaybeQuantizedLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::QMatMul(qmm) => qmm.forward(xs),
            Self::Int8(ql) => ql.forward(xs),
            Self::Gptq(gl) => gl.forward(xs),
            Self::Awq(al) => al.forward(xs),
        }
    }
}

/// Helper: dequantize a QMatMul to F32.
fn dequant_qmatmul(
    qmm: &candle_core::quantized::QMatMul,
    device: &Device,
) -> Result<Tensor> {
    match qmm {
        candle_core::quantized::QMatMul::QTensor(qt) => qt.dequantize(device),
        candle_core::quantized::QMatMul::Tensor(t) => t.to_dtype(DType::F32),
        candle_core::quantized::QMatMul::TensorF16(t) => t.to_dtype(DType::F32),
    }
}

impl MaybeQuantizedLinear {
    /// Create from a candle QMatMul, optionally quantizing to INT8 or GPTQ.
    pub fn from_qmatmul(
        qmm: candle_core::quantized::QMatMul,
        method: QuantizationMethod,
        device: &Device,
    ) -> Result<Self> {
        match method {
            QuantizationMethod::None => Ok(Self::QMatMul(qmm)),
            QuantizationMethod::Int8WeightOnly => {
                let w_f32 = dequant_qmatmul(&qmm, device)?;
                let ql = QuantizedLinear::from_float(&w_f32, None)?;
                Ok(Self::Int8(ql))
            }
            QuantizationMethod::Gptq => {
                // Dequantize to F32, then re-quantize to GPTQ 4-bit
                let w_f32 = dequant_qmatmul(&qmm, device)?;
                let gl = GptqLinear::from_float(&w_f32, None, 128)?;
                Ok(Self::Gptq(gl))
            }
            QuantizationMethod::Awq => {
                // Dequantize to F32, then re-quantize to AWQ 4-bit
                let w_f32 = dequant_qmatmul(&qmm, device)?;
                let al = AwqLinear::from_float(&w_f32, None, 128)?;
                Ok(Self::Awq(al))
            }
        }
    }
}

// ─── Utility ─────────────────────────────────────────────────────────────

/// Compute total INT8 model memory for logging.
pub fn estimate_int8_memory(num_params: usize) -> usize {
    // 1 byte per weight + 4 bytes per output channel (amortized ~1.01 bytes/param)
    num_params
}

/// Compute FP16 model memory for comparison.
pub fn estimate_fp16_memory(num_params: usize) -> usize {
    num_params * 2
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_quantize_dequantize_identity_small() {
        // Small weight matrix — quantize and dequantize should be close to original
        let w = Tensor::new(
            &[[1.0f32, -0.5, 0.25], [0.75, -1.0, 0.0]],
            &Device::Cpu,
        )
        .unwrap();
        let ql = QuantizedLinear::from_float(&w, None).unwrap();
        let w_deq = ql.dequantize(DType::F32).unwrap();
        let diff = (&w - &w_deq).unwrap().abs().unwrap();
        let max_err: f32 = diff
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar()
            .unwrap();
        // Per-channel INT8 with 127 levels: max error ≈ max_abs/127
        assert!(
            max_err < 0.02,
            "max quantization error {max_err} too large"
        );
    }

    #[test]
    fn test_quantize_zeros() {
        let w = Tensor::zeros((4, 8), DType::F32, &Device::Cpu).unwrap();
        let ql = QuantizedLinear::from_float(&w, None).unwrap();
        let w_deq = ql.dequantize(DType::F32).unwrap();
        let max_val: f32 = w_deq
            .abs()
            .unwrap()
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(max_val < 1e-6, "dequantized zeros should be ~0, got {max_val}");
    }

    #[test]
    fn test_quantized_linear_forward_shape() {
        let w = Tensor::randn(0f32, 1.0, (16, 32), &Device::Cpu).unwrap();
        let ql = QuantizedLinear::from_float(&w, None).unwrap();
        let x = Tensor::randn(0f32, 1.0, (4, 32), &Device::Cpu).unwrap();
        let y = ql.forward(&x).unwrap();
        assert_eq!(y.dims(), &[4, 16]);
    }

    #[test]
    fn test_quantized_linear_forward_with_bias() {
        let w = Tensor::randn(0f32, 1.0, (8, 16), &Device::Cpu).unwrap();
        let b = Tensor::ones(8, DType::F32, &Device::Cpu).unwrap();
        let ql = QuantizedLinear::from_float(&w, Some(&b)).unwrap();
        let x = Tensor::zeros((2, 16), DType::F32, &Device::Cpu).unwrap();
        let y = ql.forward(&x).unwrap();
        // With zero input, output should be bias
        let data: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        for v in &data {
            assert!(
                (v - 1.0).abs() < 1e-4,
                "expected ~1.0 (bias), got {v}"
            );
        }
    }

    #[test]
    fn test_quantized_linear_closeness() {
        // Compare quantized forward to full-precision forward
        let w = Tensor::randn(0f32, 0.1, (64, 128), &Device::Cpu).unwrap();
        let x = Tensor::randn(0f32, 1.0, (8, 128), &Device::Cpu).unwrap();

        // Full-precision result
        let y_fp = x.matmul(&w.t().unwrap()).unwrap();

        // Quantized result
        let ql = QuantizedLinear::from_float(&w, None).unwrap();
        let y_q = ql.forward(&x).unwrap();

        // Relative error should be small
        let diff = (&y_fp - &y_q).unwrap().abs().unwrap();
        let y_abs = y_fp.abs().unwrap();
        let max_diff: f32 = diff
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar()
            .unwrap();
        let max_val: f32 = y_abs
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar()
            .unwrap();
        let rel_err = max_diff / (max_val + 1e-8);
        assert!(
            rel_err < 0.05,
            "relative error {rel_err:.4} too large (max_diff={max_diff}, max_val={max_val})"
        );
    }

    #[test]
    fn test_memory_savings() {
        let w = Tensor::randn(0f32, 1.0, (1024, 4096), &Device::Cpu).unwrap();
        let ql = QuantizedLinear::from_float(&w, None).unwrap();
        let int8_bytes = ql.memory_bytes();
        let fp16_bytes = ql.fp16_memory_bytes();
        // INT8 should be roughly half of FP16
        assert!(
            int8_bytes < fp16_bytes,
            "INT8 ({int8_bytes}) should be less than FP16 ({fp16_bytes})"
        );
        let ratio = int8_bytes as f64 / fp16_bytes as f64;
        assert!(
            ratio < 0.55,
            "INT8/FP16 ratio {ratio:.2} should be < 0.55"
        );
    }

    #[test]
    fn test_quantization_method_parse() {
        assert_eq!(
            "int8".parse::<QuantizationMethod>().unwrap(),
            QuantizationMethod::Int8WeightOnly
        );
        assert_eq!(
            "w8a16".parse::<QuantizationMethod>().unwrap(),
            QuantizationMethod::Int8WeightOnly
        );
        assert_eq!(
            "none".parse::<QuantizationMethod>().unwrap(),
            QuantizationMethod::None
        );
        assert_eq!(
            "gptq".parse::<QuantizationMethod>().unwrap(),
            QuantizationMethod::Gptq
        );
        assert_eq!(
            "w4a16".parse::<QuantizationMethod>().unwrap(),
            QuantizationMethod::Gptq
        );
        assert!("fp8".parse::<QuantizationMethod>().is_err());
    }

    #[test]
    fn test_quantization_method_display() {
        assert_eq!(QuantizationMethod::None.to_string(), "none");
        assert_eq!(QuantizationMethod::Int8WeightOnly.to_string(), "int8");
        assert_eq!(QuantizationMethod::Gptq.to_string(), "gptq");
    }

    #[test]
    fn test_maybe_quantized_linear_qmatmul() {
        // Verify MaybeQuantizedLinear::QMatMul variant works
        let w = Tensor::randn(0f32, 1.0, (8, 16), &Device::Cpu).unwrap();
        let qmm = candle_core::quantized::QMatMul::Tensor(w);
        let mql =
            MaybeQuantizedLinear::from_qmatmul(qmm, QuantizationMethod::None, &Device::Cpu)
                .unwrap();
        let x = Tensor::randn(0f32, 1.0, (2, 16), &Device::Cpu).unwrap();
        let y = mql.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 8]);
    }

    #[test]
    fn test_maybe_quantized_linear_int8() {
        // Verify MaybeQuantizedLinear::Int8 variant works via from_qmatmul
        let w = Tensor::randn(0f32, 1.0, (8, 16), &Device::Cpu).unwrap();
        let qmm = candle_core::quantized::QMatMul::Tensor(w);
        let mql = MaybeQuantizedLinear::from_qmatmul(
            qmm,
            QuantizationMethod::Int8WeightOnly,
            &Device::Cpu,
        )
        .unwrap();
        let x = Tensor::randn(0f32, 1.0, (2, 16), &Device::Cpu).unwrap();
        let y = mql.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 8]);
    }

    #[test]
    fn test_from_parts() {
        let weights_u8 =
            Tensor::new(&[[128u8, 129, 127], [128, 128, 128]], &Device::Cpu).unwrap();
        let scales = Tensor::new(&[0.01f32, 0.02], &Device::Cpu).unwrap();
        let ql = QuantizedLinear::from_parts(weights_u8, scales, None).unwrap();
        let x = Tensor::ones((1, 3), DType::F32, &Device::Cpu).unwrap();
        let y = ql.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 2]);
    }

    // ── GPTQ tests ──────────────────────────────────────────────────────

    #[test]
    fn test_int4_pack_unpack_roundtrip() {
        // Create a [2, 8] tensor of INT4 values [0..15]
        let data: Vec<f32> = (0..16).map(|i| (i % 16) as f32).collect();
        let t = Tensor::from_vec(data.clone(), (2, 8), &Device::Cpu).unwrap();

        // Pack along dim 1: [2, 8] → [2, 1]
        let packed = pack_int4_tensor(&t, 1).unwrap();
        assert_eq!(packed.dims(), &[2, 1]);

        // Unpack: [2, 1] → [2, 8]
        let unpacked = unpack_int4_tensor(&packed, 1, 8).unwrap();
        assert_eq!(unpacked.dims(), &[2, 8]);

        // Verify roundtrip
        let unpacked_data: Vec<u8> = unpacked.flatten_all().unwrap().to_vec1().unwrap();
        for (i, (&orig, &rt)) in data.iter().zip(unpacked_data.iter()).enumerate() {
            assert_eq!(
                orig as u8, rt,
                "mismatch at index {i}: expected {orig}, got {rt}"
            );
        }
    }

    #[test]
    fn test_int4_pack_unpack_dim0() {
        // Pack along dim 0: [8, 4] → [1, 4]
        let data: Vec<f32> = (0..32).map(|i| (i % 16) as f32).collect();
        let t = Tensor::from_vec(data.clone(), (8, 4), &Device::Cpu).unwrap();

        let packed = pack_int4_tensor(&t, 0).unwrap();
        assert_eq!(packed.dims(), &[1, 4]);

        let unpacked = unpack_int4_tensor(&packed, 0, 8).unwrap();
        assert_eq!(unpacked.dims(), &[8, 4]);

        let unpacked_data: Vec<u8> = unpacked.flatten_all().unwrap().to_vec1().unwrap();
        for (i, (&orig, &rt)) in data.iter().zip(unpacked_data.iter()).enumerate() {
            assert_eq!(
                orig as u8, rt,
                "mismatch at index {i}: expected {orig}, got {rt}"
            );
        }
    }

    #[test]
    fn test_gptq_from_float_shape() {
        // [out=16, in=32], group_size=8 → 4 groups
        let w = Tensor::randn(0f32, 0.1, (16, 32), &Device::Cpu).unwrap();
        let gl = GptqLinear::from_float(&w, None, 8).unwrap();

        assert_eq!(gl.in_features(), 32);
        assert_eq!(gl.out_features(), 16);
        assert_eq!(gl.group_size(), 8);

        // qweight: [32/8, 16] = [4, 16]
        assert_eq!(gl.qweight.dims(), &[4, 16]);
        // scales: [4, 16] (4 groups)
        assert_eq!(gl.scales.dims(), &[4, 16]);
        // qzeros: [4, 16/8] = [4, 2]
        assert_eq!(gl.qzeros.dims(), &[4, 2]);
    }

    #[test]
    fn test_gptq_forward_shape() {
        let w = Tensor::randn(0f32, 0.1, (16, 64), &Device::Cpu).unwrap();
        let gl = GptqLinear::from_float(&w, None, 8).unwrap();
        let x = Tensor::randn(0f32, 1.0, (4, 64), &Device::Cpu).unwrap();
        let y = gl.forward(&x).unwrap();
        assert_eq!(y.dims(), &[4, 16]);
    }

    #[test]
    fn test_gptq_dequantize_closeness() {
        // Quantize → dequantize should be close to original
        let w = Tensor::randn(0f32, 0.1, (32, 128), &Device::Cpu).unwrap();
        let gl = GptqLinear::from_float(&w, None, 128).unwrap();
        let w_deq = gl.dequantize(DType::F32).unwrap();

        let diff = (&w - &w_deq).unwrap().abs().unwrap();
        let max_err: f32 = diff.max(0).unwrap().max(0).unwrap().to_scalar().unwrap();
        let w_max: f32 = w.abs().unwrap().max(0).unwrap().max(0).unwrap().to_scalar().unwrap();
        let rel_err = max_err / (w_max + 1e-8);

        // 4-bit quantization with 16 levels: expect ~6% max relative error
        assert!(
            rel_err < 0.15,
            "GPTQ relative error {rel_err:.4} too large (max_err={max_err:.6}, w_max={w_max:.4})"
        );
    }

    #[test]
    fn test_gptq_forward_closeness() {
        // GPTQ forward should be close to full-precision forward
        let w = Tensor::randn(0f32, 0.1, (32, 128), &Device::Cpu).unwrap();
        let x = Tensor::randn(0f32, 1.0, (8, 128), &Device::Cpu).unwrap();

        let y_fp = x.matmul(&w.t().unwrap()).unwrap();

        let gl = GptqLinear::from_float(&w, None, 128).unwrap();
        let y_q = gl.forward(&x).unwrap();

        let diff = (&y_fp - &y_q).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(0).unwrap().max(0).unwrap().to_scalar().unwrap();
        let max_val: f32 = y_fp.abs().unwrap().max(0).unwrap().max(0).unwrap().to_scalar().unwrap();
        let rel_err = max_diff / (max_val + 1e-8);

        assert!(
            rel_err < 0.20,
            "GPTQ forward relative error {rel_err:.4} too large"
        );
    }

    #[test]
    fn test_gptq_memory_savings() {
        // GPTQ should use ~4x less memory than FP16
        let w = Tensor::randn(0f32, 1.0, (1024, 4096), &Device::Cpu).unwrap();
        let gl = GptqLinear::from_float(&w, None, 128).unwrap();

        let gptq_bytes = gl.memory_bytes();
        let fp16_bytes = gl.fp16_memory_bytes();

        assert!(
            gptq_bytes < fp16_bytes,
            "GPTQ ({gptq_bytes}) should be less than FP16 ({fp16_bytes})"
        );
        let ratio = gptq_bytes as f64 / fp16_bytes as f64;
        // With scales and zeros overhead, ratio should be ~0.3
        assert!(
            ratio < 0.40,
            "GPTQ/FP16 ratio {ratio:.2} should be < 0.40"
        );
    }

    #[test]
    fn test_gptq_with_bias() {
        let w = Tensor::randn(0f32, 0.1, (8, 16), &Device::Cpu).unwrap();
        let b = Tensor::ones(8, DType::F32, &Device::Cpu).unwrap();
        let gl = GptqLinear::from_float(&w, Some(&b), 8).unwrap();

        let x = Tensor::zeros((2, 16), DType::F32, &Device::Cpu).unwrap();
        let y = gl.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 8]);

        // With zero input, output should be ~bias (plus quantization noise in dequant * 0)
        let data: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        for v in &data {
            assert!(
                (v - 1.0).abs() < 0.1,
                "expected ~1.0 (bias), got {v}"
            );
        }
    }

    #[test]
    fn test_gptq_from_parts() {
        // Manual construction with known packed data
        // 8 INT4 values per u32: values [0,1,2,...,7] packed = 0x76543210
        let packed_val: u32 = 0x76543210;
        let qweight = Tensor::from_vec(vec![packed_val; 8], (1, 8), &Device::Cpu).unwrap();
        let scales = Tensor::from_vec(vec![0.1f32; 8], (1, 8), &Device::Cpu).unwrap();
        // zeros packed: 8 zero values of 0 → all packed to 0
        let qzeros = Tensor::from_vec(vec![0u32; 1], (1, 1), &Device::Cpu).unwrap();

        let gl = GptqLinear::from_parts(qweight, scales, qzeros, None, 8).unwrap();
        assert_eq!(gl.in_features(), 8);
        assert_eq!(gl.out_features(), 8);

        let x = Tensor::ones((1, 8), DType::F32, &Device::Cpu).unwrap();
        let y = gl.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 8]);
    }

    #[test]
    fn test_maybe_quantized_linear_gptq() {
        // in_features must be divisible by 128 (group_size) and 8 (INT4 packing)
        let w = Tensor::randn(0f32, 1.0, (16, 128), &Device::Cpu).unwrap();
        let qmm = candle_core::quantized::QMatMul::Tensor(w);
        let mql = MaybeQuantizedLinear::from_qmatmul(
            qmm,
            QuantizationMethod::Gptq,
            &Device::Cpu,
        )
        .unwrap();
        let x = Tensor::randn(0f32, 1.0, (2, 128), &Device::Cpu).unwrap();
        let y = mql.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 16]);
    }

    // ── KV cache dtype tests ────────────────────────────────────────────

    #[test]
    fn test_kv_cache_dtype_parse() {
        assert_eq!("auto".parse::<KVCacheDType>().unwrap(), KVCacheDType::Auto);
        assert_eq!("fp16".parse::<KVCacheDType>().unwrap(), KVCacheDType::F16);
        assert_eq!("f16".parse::<KVCacheDType>().unwrap(), KVCacheDType::F16);
        assert_eq!("half".parse::<KVCacheDType>().unwrap(), KVCacheDType::F16);
        assert_eq!("bf16".parse::<KVCacheDType>().unwrap(), KVCacheDType::BF16);
        assert_eq!("bfloat16".parse::<KVCacheDType>().unwrap(), KVCacheDType::BF16);
        assert!("fp8".parse::<KVCacheDType>().is_err());
    }

    #[test]
    fn test_kv_cache_dtype_resolve() {
        assert_eq!(KVCacheDType::Auto.resolve(DType::F32), DType::F32);
        assert_eq!(KVCacheDType::Auto.resolve(DType::F16), DType::F16);
        assert_eq!(KVCacheDType::F16.resolve(DType::F32), DType::F16);
        assert_eq!(KVCacheDType::BF16.resolve(DType::F32), DType::BF16);
    }

    #[test]
    fn test_kv_cache_dtype_display() {
        assert_eq!(KVCacheDType::Auto.to_string(), "auto");
        assert_eq!(KVCacheDType::F16.to_string(), "fp16");
        assert_eq!(KVCacheDType::BF16.to_string(), "bf16");
    }

    // ── AWQ tests ───────────────────────────────────────────────────────

    #[test]
    fn test_awq_from_float_shape() {
        let w = Tensor::randn(0f32, 0.1, (16, 32), &Device::Cpu).unwrap();
        let al = AwqLinear::from_float(&w, None, 8).unwrap();
        assert_eq!(al.in_features(), 32);
        assert_eq!(al.out_features(), 16);
        assert_eq!(al.group_size(), 8);
    }

    #[test]
    fn test_awq_forward_shape() {
        let w = Tensor::randn(0f32, 0.1, (16, 64), &Device::Cpu).unwrap();
        let al = AwqLinear::from_float(&w, None, 8).unwrap();
        let x = Tensor::randn(0f32, 1.0, (4, 64), &Device::Cpu).unwrap();
        let y = al.forward(&x).unwrap();
        assert_eq!(y.dims(), &[4, 16]);
    }

    #[test]
    fn test_awq_dequantize_closeness() {
        let w = Tensor::randn(0f32, 0.1, (32, 128), &Device::Cpu).unwrap();
        let al = AwqLinear::from_float(&w, None, 128).unwrap();
        let w_deq = al.dequantize(DType::F32).unwrap();

        let diff = (&w - &w_deq).unwrap().abs().unwrap();
        let max_err: f32 = diff.max(0).unwrap().max(0).unwrap().to_scalar().unwrap();
        let w_max: f32 = w.abs().unwrap().max(0).unwrap().max(0).unwrap().to_scalar().unwrap();
        let rel_err = max_err / (w_max + 1e-8);

        assert!(
            rel_err < 0.15,
            "AWQ relative error {rel_err:.4} too large (max_err={max_err:.6}, w_max={w_max:.4})"
        );
    }

    #[test]
    fn test_awq_forward_closeness() {
        let w = Tensor::randn(0f32, 0.1, (32, 128), &Device::Cpu).unwrap();
        let x = Tensor::randn(0f32, 1.0, (8, 128), &Device::Cpu).unwrap();
        let y_fp = x.matmul(&w.t().unwrap()).unwrap();

        let al = AwqLinear::from_float(&w, None, 128).unwrap();
        let y_q = al.forward(&x).unwrap();

        let diff = (&y_fp - &y_q).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(0).unwrap().max(0).unwrap().to_scalar().unwrap();
        let max_val: f32 = y_fp.abs().unwrap().max(0).unwrap().max(0).unwrap().to_scalar().unwrap();
        let rel_err = max_diff / (max_val + 1e-8);

        assert!(
            rel_err < 0.20,
            "AWQ forward relative error {rel_err:.4} too large"
        );
    }

    #[test]
    fn test_awq_memory_savings() {
        let w = Tensor::randn(0f32, 1.0, (1024, 4096), &Device::Cpu).unwrap();
        let al = AwqLinear::from_float(&w, None, 128).unwrap();

        let awq_bytes = al.memory_bytes();
        let fp16_bytes = al.fp16_memory_bytes();

        assert!(awq_bytes < fp16_bytes);
        let ratio = awq_bytes as f64 / fp16_bytes as f64;
        assert!(
            ratio < 0.40,
            "AWQ/FP16 ratio {ratio:.2} should be < 0.40"
        );
    }

    #[test]
    fn test_awq_from_parts() {
        let packed_val: u32 = 0x76543210;
        let qweight = Tensor::from_vec(vec![packed_val; 8], (1, 8), &Device::Cpu).unwrap();
        let scales = Tensor::from_vec(vec![0.1f32; 8], (1, 8), &Device::Cpu).unwrap();
        let qzeros = Tensor::from_vec(vec![0u32; 1], (1, 1), &Device::Cpu).unwrap();

        let al = AwqLinear::from_parts(qweight, scales, qzeros, None, 8).unwrap();
        assert_eq!(al.in_features(), 8);
        assert_eq!(al.out_features(), 8);

        let x = Tensor::ones((1, 8), DType::F32, &Device::Cpu).unwrap();
        let y = al.forward(&x).unwrap();
        assert_eq!(y.dims(), &[1, 8]);
    }

    #[test]
    fn test_awq_with_bias() {
        let w = Tensor::randn(0f32, 0.1, (8, 16), &Device::Cpu).unwrap();
        let b = Tensor::ones(8, DType::F32, &Device::Cpu).unwrap();
        let al = AwqLinear::from_float(&w, Some(&b), 8).unwrap();

        let x = Tensor::zeros((2, 16), DType::F32, &Device::Cpu).unwrap();
        let y = al.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 8]);

        let data: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        for v in &data {
            assert!(
                (v - 1.0).abs() < 0.1,
                "expected ~1.0 (bias), got {v}"
            );
        }
    }

    #[test]
    fn test_quantization_method_awq_parse() {
        assert_eq!(
            "awq".parse::<QuantizationMethod>().unwrap(),
            QuantizationMethod::Awq
        );
    }

    #[test]
    fn test_quantization_method_awq_display() {
        assert_eq!(QuantizationMethod::Awq.to_string(), "awq");
    }

    #[test]
    fn test_maybe_quantized_linear_awq() {
        let w = Tensor::randn(0f32, 1.0, (16, 128), &Device::Cpu).unwrap();
        let qmm = candle_core::quantized::QMatMul::Tensor(w);
        let mql = MaybeQuantizedLinear::from_qmatmul(
            qmm,
            QuantizationMethod::Awq,
            &Device::Cpu,
        )
        .unwrap();
        let x = Tensor::randn(0f32, 1.0, (2, 128), &Device::Cpu).unwrap();
        let y = mql.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, 16]);
    }

    #[test]
    fn test_kv_cache_memory_savings() {
        use crate::serving::kv_cache::KVCacheConfig;

        let f32_config = KVCacheConfig {
            block_size: 16,
            num_blocks: 100,
            num_kv_heads: 8,
            head_size: 128,
            num_layers: 32,
            enable_prefix_cache: false,
            dtype_bytes: 4, // F32
        };
        let f16_config = KVCacheConfig {
            dtype_bytes: 2, // F16
            ..f32_config.clone()
        };

        let f32_bytes = f32_config.total_memory_bytes();
        let f16_bytes = f16_config.total_memory_bytes();
        assert_eq!(f16_bytes * 2, f32_bytes, "F16 should be exactly half of F32");
    }
}
