//! Vision encoder (ViT) for multimodal model support.
//!
//! Implements a standard Vision Transformer (ViT) that encodes images into
//! feature vectors compatible with the language model's embedding space.
//! Based on the CLIP vision encoder architecture used by LLaVA, Qwen-VL,
//! and similar vision-language models.
//!
//! The pipeline:
//! 1. Preprocess image (resize, normalize, convert to tensor)
//! 2. Patch embedding (conv2d → flatten → project)
//! 3. Prepend CLS token + add position embeddings
//! 4. N transformer layers (LayerNorm → self-attention → residual → LayerNorm → MLP → residual)
//! 5. Project to language model hidden size if different
//! 6. Insert features at `<image>` token positions in the text sequence

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Linear, Module, VarBuilder};
use serde::Deserialize;

// ---- Configuration --------------------------------------------------------

/// Vision encoder configuration, typically parsed from a model's `config.json`
/// (preprocessor_config or vision_config section).
#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    /// Input image resolution (square). Default: 224.
    #[serde(default = "default_image_size")]
    pub image_size: usize,
    /// Patch size for patch embedding convolution. Default: 14.
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    /// Hidden dimension of the vision transformer. Default: 1024.
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    /// Number of transformer layers. Default: 24.
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    /// Number of attention heads. Default: 16.
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    /// Intermediate (MLP) dimension. Default: 4096.
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    /// Layer norm epsilon. Default: 1e-6.
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    /// Hidden size of the language model (for projection). Default: 4096.
    #[serde(default = "default_projection_dim")]
    pub projection_dim: usize,
}

fn default_image_size() -> usize {
    224
}
fn default_patch_size() -> usize {
    14
}
fn default_hidden_size() -> usize {
    1024
}
fn default_num_hidden_layers() -> usize {
    24
}
fn default_num_attention_heads() -> usize {
    16
}
fn default_intermediate_size() -> usize {
    4096
}
fn default_layer_norm_eps() -> f64 {
    1e-6
}
fn default_projection_dim() -> usize {
    4096
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            image_size: default_image_size(),
            patch_size: default_patch_size(),
            hidden_size: default_hidden_size(),
            num_hidden_layers: default_num_hidden_layers(),
            num_attention_heads: default_num_attention_heads(),
            intermediate_size: default_intermediate_size(),
            layer_norm_eps: default_layer_norm_eps(),
            projection_dim: default_projection_dim(),
        }
    }
}

impl VisionConfig {
    /// Number of patches along one dimension.
    pub fn num_patches_per_side(&self) -> usize {
        self.image_size / self.patch_size
    }

    /// Total number of patches (excluding CLS token).
    pub fn num_patches(&self) -> usize {
        let n = self.num_patches_per_side();
        n * n
    }

    /// Total sequence length for the vision transformer (patches + CLS).
    pub fn seq_len(&self) -> usize {
        self.num_patches() + 1 // +1 for CLS token
    }

    /// Per-head dimension.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

// ---- Image preprocessing --------------------------------------------------

/// CLIP default normalization constants.
pub const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
pub const CLIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

/// Preprocesses raw image bytes into a normalized tensor suitable for the
/// vision transformer.
///
/// Steps:
/// 1. Decode image bytes (PNG, JPEG, etc.)
/// 2. Resize to `image_size x image_size` using bilinear interpolation
/// 3. Convert to float tensor `[C, H, W]` in range [0, 1]
/// 4. Normalize with CLIP mean/std
///
/// Returns tensor of shape `[1, 3, image_size, image_size]` (batched).
pub struct ImagePreprocessor {
    image_size: usize,
    mean: [f32; 3],
    std: [f32; 3],
}

impl ImagePreprocessor {
    /// Create a preprocessor with CLIP defaults.
    pub fn new(image_size: usize) -> Self {
        Self {
            image_size,
            mean: CLIP_MEAN,
            std: CLIP_STD,
        }
    }

    /// Create a preprocessor with custom normalization constants.
    pub fn with_normalization(image_size: usize, mean: [f32; 3], std: [f32; 3]) -> Self {
        Self {
            image_size,
            mean,
            std,
        }
    }

    /// Preprocess raw pixel data already in `[H, W, 3]` u8 format.
    ///
    /// `pixels` should be RGB u8 values, row-major, `height x width x 3`.
    /// Returns `[1, 3, image_size, image_size]` f32 tensor.
    pub fn preprocess_rgb(
        &self,
        pixels: &[u8],
        width: u32,
        height: u32,
        device: &Device,
    ) -> Result<Tensor> {
        // 1. Create tensor from raw pixels: [H, W, 3]
        let img_tensor = Tensor::from_vec(
            pixels.to_vec(),
            (height as usize, width as usize, 3),
            device,
        )?
        .to_dtype(DType::F32)?;

        // 2. Normalize to [0, 1]
        let img_tensor = (img_tensor / 255.0)?;

        // 3. Transpose to [3, H, W] (channels first)
        let img_tensor = img_tensor.permute((2, 0, 1))?;

        // 4. Resize using interpolation if needed
        let img_tensor = if width as usize != self.image_size || height as usize != self.image_size
        {
            img_tensor
                .unsqueeze(0)?
                .interpolate2d(self.image_size, self.image_size)?
                .squeeze(0)?
        } else {
            img_tensor
        };

        // 5. Normalize with mean/std per channel
        let mean = Tensor::new(&self.mean, device)?.reshape((3, 1, 1))?;
        let std = Tensor::new(&self.std, device)?.reshape((3, 1, 1))?;
        let img_tensor = img_tensor.broadcast_sub(&mean)?.broadcast_div(&std)?;

        // 6. Add batch dimension: [1, 3, H, W]
        img_tensor.unsqueeze(0)
    }

    /// Decode base64-encoded image data and preprocess it.
    ///
    /// Handles `data:image/...;base64,...` URIs as well as raw base64 strings.
    /// Returns `[1, 3, image_size, image_size]` f32 tensor.
    pub fn preprocess_base64(&self, data_url: &str, device: &Device) -> Result<Tensor> {
        // Strip the data URI prefix if present
        let base64_data = if let Some(idx) = data_url.find(";base64,") {
            &data_url[idx + 8..]
        } else {
            data_url
        };

        // Decode base64
        use base64::Engine;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(base64_data)
            .map_err(|e| candle_core::Error::Msg(format!("Base64 decode error: {e}")))?;

        // Decode image using the `image` crate
        let img = image::load_from_memory(&bytes)
            .map_err(|e| candle_core::Error::Msg(format!("Image decode error: {e}")))?;

        let img = img.resize_exact(
            self.image_size as u32,
            self.image_size as u32,
            image::imageops::FilterType::Triangle, // bilinear
        );
        let rgb = img.to_rgb8();
        let (w, h) = rgb.dimensions();

        self.preprocess_rgb(rgb.as_raw(), w, h, device)
    }
}

// ---- Layer Norm -----------------------------------------------------------

/// Standard Layer Normalization (not RMS — vision transformers use full LN).
#[derive(Clone)]
struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        Self { weight, bias, eps }
    }

    fn load(vb: &VarBuilder, size: usize, eps: f64) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        let bias = vb.get(size, "bias")?;
        Ok(Self::new(weight, bias, eps))
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::layer_norm(x, &self.weight, &self.bias, self.eps as f32)
    }
}

// ---- Patch Embedding ------------------------------------------------------

/// Converts an image into a sequence of patch embeddings using a strided
/// convolution (kernel_size = stride = patch_size).
#[derive(Clone)]
struct PatchEmbedding {
    proj: Conv2d,
    num_patches: usize,
    hidden_size: usize,
}

impl PatchEmbedding {
    fn new(config: &VisionConfig, vb: &VarBuilder) -> Result<Self> {
        let conv_config = Conv2dConfig {
            stride: config.patch_size,
            ..Default::default()
        };
        let proj = candle_nn::conv2d(
            3, // RGB input channels
            config.hidden_size,
            config.patch_size,
            conv_config,
            vb.pp("patch_embedding"),
        )?;
        Ok(Self {
            proj,
            num_patches: config.num_patches(),
            hidden_size: config.hidden_size,
        })
    }

    /// Input: `[batch, 3, image_size, image_size]`
    /// Output: `[batch, num_patches, hidden_size]`
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let batch = x.dim(0)?;
        // Conv2d: [B, 3, H, W] -> [B, hidden, H/patch, W/patch]
        let x = self.proj.forward(x)?;
        // Flatten spatial dims: [B, hidden, num_patches]
        let x = x.reshape((batch, self.hidden_size, self.num_patches))?;
        // Transpose: [B, num_patches, hidden]
        x.transpose(1, 2)
    }
}

// ---- Vision Attention -----------------------------------------------------

/// Multi-head self-attention for vision transformer layers.
#[derive(Clone)]
struct VisionAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl VisionAttention {
    fn new(config: &VisionConfig, vb: &VarBuilder) -> Result<Self> {
        let h = config.hidden_size;
        let q_proj = candle_nn::linear(h, h, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(h, h, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(h, h, vb.pp("v_proj"))?;
        let out_proj = candle_nn::linear(h, h, vb.pp("out_proj"))?;
        let head_dim = config.head_dim();
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads: config.num_attention_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    /// Input/Output: `[batch, seq_len, hidden_size]`
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [batch, seq_len, num_heads, head_dim] then transpose to [batch, num_heads, seq_len, head_dim]
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Scaled dot-product attention
        let attn_weights = (q.matmul(&k.t()?)? * self.scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden]
        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq_len, self.num_heads * self.head_dim))?;

        self.out_proj.forward(&attn_output)
    }
}

// ---- Vision MLP -----------------------------------------------------------

/// GELU-activated MLP used in vision transformer layers.
#[derive(Clone)]
struct VisionMlp {
    fc1: Linear,
    fc2: Linear,
}

impl VisionMlp {
    fn new(config: &VisionConfig, vb: &VarBuilder) -> Result<Self> {
        let fc1 = candle_nn::linear(config.hidden_size, config.intermediate_size, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(config.intermediate_size, config.hidden_size, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = x.gelu_erf()?;
        self.fc2.forward(&x)
    }
}

// ---- Vision Transformer Layer ---------------------------------------------

/// A single vision transformer layer:
/// LayerNorm -> Self-Attention -> Residual -> LayerNorm -> MLP -> Residual
#[derive(Clone)]
struct VisionTransformerLayer {
    ln1: LayerNorm,
    attn: VisionAttention,
    ln2: LayerNorm,
    mlp: VisionMlp,
}

impl VisionTransformerLayer {
    fn new(config: &VisionConfig, vb: &VarBuilder) -> Result<Self> {
        let ln1 = LayerNorm::load(&vb.pp("layer_norm1"), config.hidden_size, config.layer_norm_eps)?;
        let attn = VisionAttention::new(config, &vb.pp("self_attn"))?;
        let ln2 = LayerNorm::load(&vb.pp("layer_norm2"), config.hidden_size, config.layer_norm_eps)?;
        let mlp = VisionMlp::new(config, &vb.pp("mlp"))?;
        Ok(Self { ln1, attn, ln2, mlp })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Pre-norm attention + residual
        let residual = x;
        let x = self.ln1.forward(x)?;
        let x = self.attn.forward(&x)?;
        let x = (x + residual)?;

        // Pre-norm MLP + residual
        let residual = &x;
        let h = self.ln2.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        h + residual
    }
}

// ---- Vision Transformer ---------------------------------------------------

/// Complete Vision Transformer (ViT) encoder.
///
/// Architecture: patch embedding -> CLS + position embeddings -> N layers -> LN
///
/// Produces a sequence of image feature vectors `[batch, num_patches + 1, hidden_size]`.
#[derive(Clone)]
pub struct VisionTransformer {
    patch_embed: PatchEmbedding,
    cls_token: Tensor,
    position_embedding: Tensor,
    pre_layrnorm: LayerNorm,
    layers: Vec<VisionTransformerLayer>,
    post_layernorm: LayerNorm,
    config: VisionConfig,
}

impl VisionTransformer {
    /// Build a ViT from a VarBuilder (e.g., loaded from safetensors).
    pub fn new(config: &VisionConfig, vb: &VarBuilder) -> Result<Self> {
        let patch_embed = PatchEmbedding::new(config, vb)?;
        let cls_token = vb.get((1, 1, config.hidden_size), "class_embedding")?;
        let seq_len = config.seq_len();
        let position_embedding =
            vb.get((1, seq_len, config.hidden_size), "position_embedding.weight")?;
        let pre_layrnorm =
            LayerNorm::load(&vb.pp("pre_layrnorm"), config.hidden_size, config.layer_norm_eps)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(VisionTransformerLayer::new(
                config,
                &vb.pp(format!("encoder.layers.{i}")),
            )?);
        }

        let post_layernorm =
            LayerNorm::load(&vb.pp("post_layernorm"), config.hidden_size, config.layer_norm_eps)?;

        Ok(Self {
            patch_embed,
            cls_token,
            position_embedding,
            pre_layrnorm,
            layers,
            post_layernorm,
            config: config.clone(),
        })
    }

    /// Forward pass: `[batch, 3, image_size, image_size]` -> `[batch, seq_len, hidden_size]`
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let batch = pixel_values.dim(0)?;

        // Patch embedding: [B, 3, H, W] -> [B, num_patches, hidden]
        let patch_embeds = self.patch_embed.forward(pixel_values)?;

        // Expand CLS token to batch: [1, 1, hidden] -> [B, 1, hidden]
        let cls_tokens = self.cls_token.broadcast_as((batch, 1, self.config.hidden_size))?;

        // Prepend CLS: [B, 1 + num_patches, hidden]
        let embeddings = Tensor::cat(&[&cls_tokens, &patch_embeds], 1)?;

        // Add position embeddings
        let embeddings = (embeddings + &self.position_embedding)?;

        // Pre-layer normalization
        let mut hidden_states = self.pre_layrnorm.forward(&embeddings)?;

        // Transformer layers
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }

        // Post-layer normalization
        self.post_layernorm.forward(&hidden_states)
    }

    /// Get the configuration.
    pub fn config(&self) -> &VisionConfig {
        &self.config
    }
}

// ---- Multimodal projection ------------------------------------------------

/// Projects vision encoder output to the language model's hidden size.
///
/// Simple two-layer MLP with GELU activation, as used in LLaVA-1.5.
#[derive(Clone)]
pub struct MultimodalProjector {
    linear1: Linear,
    linear2: Linear,
}

impl MultimodalProjector {
    /// Create a projector from vision hidden size to language model hidden size.
    pub fn new(
        vision_hidden_size: usize,
        text_hidden_size: usize,
        vb: &VarBuilder,
    ) -> Result<Self> {
        let linear1 =
            candle_nn::linear(vision_hidden_size, text_hidden_size, vb.pp("linear_1"))?;
        let linear2 =
            candle_nn::linear(text_hidden_size, text_hidden_size, vb.pp("linear_2"))?;
        Ok(Self { linear1, linear2 })
    }

    /// Project: `[batch, seq_len, vision_hidden]` -> `[batch, seq_len, text_hidden]`
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = x.gelu_erf()?;
        self.linear2.forward(&x)
    }
}

// ---- Multimodal input merging ---------------------------------------------

/// Represents combined text + image input for a multimodal model.
pub struct MultimodalInput {
    /// Merged embeddings `[total_tokens, hidden_size]` where image features
    /// replace `<image>` placeholder tokens.
    pub embeddings: Tensor,
    /// Number of image tokens that were inserted (for tracking).
    pub num_image_tokens: usize,
}

impl MultimodalInput {
    /// Merge text token embeddings with projected image features.
    ///
    /// Scans `input_ids` for occurrences of `image_token_id` and replaces
    /// those positions with the corresponding image feature vectors.
    ///
    /// # Arguments
    /// - `input_ids`: `[seq_len]` token IDs
    /// - `text_embeddings`: `[seq_len, hidden_size]` from the LM's embedding table
    /// - `image_features`: `[num_images, num_patches, hidden_size]` projected vision features
    /// - `image_token_id`: token ID used as `<image>` placeholder
    pub fn merge(
        input_ids: &Tensor,
        text_embeddings: &Tensor,
        image_features: &[Tensor],
        image_token_id: u32,
        _device: &Device,
    ) -> Result<Self> {
        let ids: Vec<u32> = input_ids.to_vec1()?;
        let seq_len = ids.len();
        let _hidden_size = text_embeddings.dim(1)?;

        // Find image token positions
        let image_positions: Vec<usize> = ids
            .iter()
            .enumerate()
            .filter(|(_, &id)| id == image_token_id)
            .map(|(i, _)| i)
            .collect();

        if image_positions.is_empty() || image_features.is_empty() {
            return Ok(Self {
                embeddings: text_embeddings.clone(),
                num_image_tokens: 0,
            });
        }

        // Build the merged sequence by replacing image tokens with feature vectors.
        // Each image's features (num_patches vectors) replace one <image> token,
        // expanding the sequence length.
        let mut segments: Vec<Tensor> = Vec::new();
        let mut prev_pos = 0;
        let mut img_idx = 0;
        let mut total_image_tokens = 0;

        for &pos in &image_positions {
            if img_idx >= image_features.len() {
                break;
            }

            // Text segment before this image token
            if pos > prev_pos {
                segments.push(text_embeddings.narrow(0, prev_pos, pos - prev_pos)?);
            }

            // Image features for this position: [num_patches, hidden]
            let features = &image_features[img_idx];
            let num_patches = features.dim(0)?;
            // If features are [1, num_patches, hidden], squeeze the batch dim
            let features = if features.dims().len() == 3 {
                features.squeeze(0)?
            } else {
                features.clone()
            };
            segments.push(features);
            total_image_tokens += num_patches;
            img_idx += 1;
            prev_pos = pos + 1; // skip the <image> token
        }

        // Remaining text after last image token
        if prev_pos < seq_len {
            segments.push(text_embeddings.narrow(0, prev_pos, seq_len - prev_pos)?);
        }

        let merged = if segments.len() == 1 {
            segments.into_iter().next().unwrap()
        } else {
            let refs: Vec<&Tensor> = segments.iter().collect();
            Tensor::cat(&refs, 0)?
        };

        Ok(Self {
            embeddings: merged,
            num_image_tokens: total_image_tokens,
        })
    }
}

// ---- Tests ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_vision_config_defaults() {
        let config = VisionConfig::default();
        assert_eq!(config.image_size, 224);
        assert_eq!(config.patch_size, 14);
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_hidden_layers, 24);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.intermediate_size, 4096);
    }

    #[test]
    fn test_vision_config_num_patches() {
        let config = VisionConfig {
            image_size: 224,
            patch_size: 14,
            ..Default::default()
        };
        assert_eq!(config.num_patches_per_side(), 16);
        assert_eq!(config.num_patches(), 256);
        assert_eq!(config.seq_len(), 257); // 256 patches + CLS
    }

    #[test]
    fn test_vision_config_head_dim() {
        let config = VisionConfig {
            hidden_size: 1024,
            num_attention_heads: 16,
            ..Default::default()
        };
        assert_eq!(config.head_dim(), 64);
    }

    #[test]
    fn test_vision_config_parse_json() {
        let json = r#"{
            "image_size": 336,
            "patch_size": 14,
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "projection_dim": 4096
        }"#;
        let config: VisionConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.image_size, 336);
        assert_eq!(config.num_patches_per_side(), 24);
        assert_eq!(config.num_patches(), 576);
    }

    #[test]
    fn test_vision_config_parse_partial_json() {
        // Missing fields should use defaults
        let json = r#"{"image_size": 384}"#;
        let config: VisionConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.image_size, 384);
        assert_eq!(config.patch_size, 14); // default
        assert_eq!(config.hidden_size, 1024); // default
    }

    #[test]
    fn test_image_preprocessor_rgb() {
        let preprocessor = ImagePreprocessor::new(224);
        // Create a small 4x4 white image
        let pixels = vec![255u8; 4 * 4 * 3];
        let result = preprocessor.preprocess_rgb(&pixels, 4, 4, &Device::Cpu).unwrap();
        assert_eq!(result.dims(), &[1, 3, 224, 224]);
        assert_eq!(result.dtype(), DType::F32);
    }

    #[test]
    fn test_image_preprocessor_normalization() {
        let preprocessor = ImagePreprocessor::new(4);
        // Create a 4x4 image with known pixel value (128 -> 0.50196)
        let pixels = vec![128u8; 4 * 4 * 3];
        let result = preprocessor.preprocess_rgb(&pixels, 4, 4, &Device::Cpu).unwrap();

        // Channel 0 (R): (128/255 - 0.48145466) / 0.26862954
        let expected_r = (128.0 / 255.0 - CLIP_MEAN[0]) / CLIP_STD[0];
        let pixel_val: f32 = result
            .squeeze(0)
            .unwrap()
            .narrow(0, 0, 1) // channel 0
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert!(
            (pixel_val - expected_r).abs() < 1e-4,
            "expected ~{expected_r}, got {pixel_val}"
        );
    }

    #[test]
    fn test_image_preprocessor_already_correct_size() {
        let preprocessor = ImagePreprocessor::new(4);
        let pixels = vec![100u8; 4 * 4 * 3];
        let result = preprocessor.preprocess_rgb(&pixels, 4, 4, &Device::Cpu).unwrap();
        assert_eq!(result.dims(), &[1, 3, 4, 4]);
    }

    #[test]
    fn test_multimodal_input_no_images() {
        let device = &Device::Cpu;
        let seq_len = 10;
        let hidden = 16;
        let input_ids = Tensor::zeros(seq_len, DType::U32, device).unwrap();
        let text_embeddings = Tensor::randn(0f32, 1.0, (seq_len, hidden), device).unwrap();

        let result = MultimodalInput::merge(
            &input_ids,
            &text_embeddings,
            &[],
            99999, // image token ID (not present)
            device,
        )
        .unwrap();

        assert_eq!(result.embeddings.dims(), &[seq_len, hidden]);
        assert_eq!(result.num_image_tokens, 0);
    }

    #[test]
    fn test_multimodal_input_merge_single_image() {
        let device = &Device::Cpu;
        let hidden = 16;
        let num_patches = 4;
        let image_token_id = 32000u32;

        // Input: [hello, <image>, world] = 3 tokens
        let input_ids = Tensor::new(&[1u32, image_token_id, 2u32], device).unwrap();
        let text_embeddings = Tensor::randn(0f32, 1.0, (3, hidden), device).unwrap();
        let image_features =
            Tensor::randn(0f32, 1.0, (num_patches, hidden), device).unwrap();

        let result = MultimodalInput::merge(
            &input_ids,
            &text_embeddings,
            &[image_features],
            image_token_id,
            device,
        )
        .unwrap();

        // Expected: [hello, img0, img1, img2, img3, world] = 2 + 4 = 6
        assert_eq!(result.embeddings.dims(), &[2 + num_patches, hidden]);
        assert_eq!(result.num_image_tokens, num_patches);
    }

    #[test]
    fn test_multimodal_input_merge_multiple_images() {
        let device = &Device::Cpu;
        let hidden = 16;
        let num_patches = 3;
        let image_token_id = 32000u32;

        // Input: [token, <image>, token, <image>, token] = 5 tokens
        let input_ids =
            Tensor::new(&[1u32, image_token_id, 2u32, image_token_id, 3u32], device).unwrap();
        let text_embeddings = Tensor::randn(0f32, 1.0, (5, hidden), device).unwrap();
        let img1 = Tensor::randn(0f32, 1.0, (num_patches, hidden), device).unwrap();
        let img2 = Tensor::randn(0f32, 1.0, (num_patches, hidden), device).unwrap();

        let result = MultimodalInput::merge(
            &input_ids,
            &text_embeddings,
            &[img1, img2],
            image_token_id,
            device,
        )
        .unwrap();

        // Expected: 3 text tokens + 2 * 3 image tokens = 9
        assert_eq!(result.embeddings.dims(), &[3 + 2 * num_patches, hidden]);
        assert_eq!(result.num_image_tokens, 2 * num_patches);
    }

    #[test]
    fn test_multimodal_input_image_at_start() {
        let device = &Device::Cpu;
        let hidden = 8;
        let num_patches = 2;
        let image_token_id = 100u32;

        // Input: [<image>, token1, token2]
        let input_ids = Tensor::new(&[image_token_id, 1u32, 2u32], device).unwrap();
        let text_embeddings = Tensor::randn(0f32, 1.0, (3, hidden), device).unwrap();
        let img = Tensor::randn(0f32, 1.0, (num_patches, hidden), device).unwrap();

        let result = MultimodalInput::merge(
            &input_ids,
            &text_embeddings,
            &[img],
            image_token_id,
            device,
        )
        .unwrap();

        // Expected: 2 text tokens + 2 image tokens = 4
        assert_eq!(result.embeddings.dims(), &[2 + num_patches, hidden]);
    }

    #[test]
    fn test_multimodal_input_image_at_end() {
        let device = &Device::Cpu;
        let hidden = 8;
        let num_patches = 2;
        let image_token_id = 100u32;

        // Input: [token1, token2, <image>]
        let input_ids = Tensor::new(&[1u32, 2u32, image_token_id], device).unwrap();
        let text_embeddings = Tensor::randn(0f32, 1.0, (3, hidden), device).unwrap();
        let img = Tensor::randn(0f32, 1.0, (num_patches, hidden), device).unwrap();

        let result = MultimodalInput::merge(
            &input_ids,
            &text_embeddings,
            &[img],
            image_token_id,
            device,
        )
        .unwrap();

        // Expected: 2 text tokens + 2 image tokens = 4
        assert_eq!(result.embeddings.dims(), &[2 + num_patches, hidden]);
    }

    #[test]
    fn test_base64_data_url_stripping() {
        // Test the data URL parsing logic
        let url = "data:image/png;base64,iVBORw0KGgo=";
        let base64_data = if let Some(idx) = url.find(";base64,") {
            &url[idx + 8..]
        } else {
            url
        };
        assert_eq!(base64_data, "iVBORw0KGgo=");

        // Test raw base64 (no data URL prefix)
        let raw = "iVBORw0KGgo=";
        let base64_data2 = if let Some(idx) = raw.find(";base64,") {
            &raw[idx + 8..]
        } else {
            raw
        };
        assert_eq!(base64_data2, "iVBORw0KGgo=");
    }
}
