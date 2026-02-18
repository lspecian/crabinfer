/// Memory pressure management for iOS
/// 
/// iOS aggressively kills apps that use too much memory (jetsam).
/// This module monitors memory usage and provides graceful degradation
/// before iOS kills the app.
///
/// Key iOS memory facts:
/// - os_proc_available_memory() returns jetsam-aware available memory
/// - Memory warnings come in stages: normal → warning → critical → terminal
/// - On iPhone 14 Pro (6GB), apps typically get killed at ~3-4GB usage
/// - Model weights loaded via mmap don't count toward dirty memory
///   (they can be evicted by the OS), but KV cache and activations do

use crate::MemoryPressure;

/// Thresholds for memory pressure levels (as percentage of available memory used)
const WARNING_THRESHOLD: f64 = 0.65;
const CRITICAL_THRESHOLD: f64 = 0.80;
const TERMINAL_THRESHOLD: f64 = 0.90;

pub struct MemoryPressureManager {
    /// Memory limit set by the user or auto-detected
    memory_limit_bytes: u64,
    /// Peak memory usage observed
    peak_usage_bytes: u64,
    /// Current memory pressure level
    current_pressure: MemoryPressure,
    /// Callbacks registered for pressure changes
    on_pressure_change: Option<Box<dyn Fn(MemoryPressure) + Send + Sync>>,
}

impl MemoryPressureManager {
    pub fn new(memory_limit_bytes: u64) -> Self {
        Self {
            memory_limit_bytes,
            peak_usage_bytes: 0,
            current_pressure: MemoryPressure::Normal,
            on_pressure_change: None,
        }
    }

    /// Auto-configure based on device capabilities
    pub fn auto_configure() -> Self {
        let available = crate::device::detect_device().available_memory_bytes;
        // Use 70% of available memory as our limit
        // This leaves headroom for iOS and other app activities
        let limit = (available as f64 * 0.70) as u64;
        Self::new(limit)
    }

    /// Check current memory pressure level.
    ///
    /// On iOS/macOS, uses `os_proc_available_memory()` for a real-time,
    /// jetsam-aware check instead of comparing RSS against a stale static limit.
    /// This is more accurate because:
    /// - iOS reclaims memory from background apps (available increases dynamically)
    /// - The static limit (set at init before model load) doesn't reflect post-load state
    /// - `os_proc_available_memory()` is the kernel's own jetsam headroom estimate
    pub fn check_pressure(&mut self) -> MemoryPressure {
        let usage = self.current_memory_usage();
        self.peak_usage_bytes = self.peak_usage_bytes.max(usage);

        let new_pressure = self.determine_pressure(usage);

        if std::mem::discriminant(&new_pressure) != std::mem::discriminant(&self.current_pressure) {
            self.current_pressure = new_pressure.clone();
            if let Some(ref callback) = self.on_pressure_change {
                callback(new_pressure.clone());
            }
        }

        self.current_pressure.clone()
    }

    /// Determine memory pressure level.
    fn determine_pressure(&self, usage: u64) -> MemoryPressure {
        // On iOS/macOS: query os_proc_available_memory() for real-time check.
        // This tells us exactly how much memory we can still allocate before
        // iOS kills the process (jetsam).
        #[cfg(any(target_os = "ios", target_os = "macos"))]
        {
            let remaining = Self::os_available_memory();
            if remaining > 0 {
                // Absolute thresholds for remaining memory before jetsam.
                // Each generated token adds ~100-500 KB to KV cache, so even
                // 50 MB of headroom supports hundreds of tokens safely.
                const TERMINAL_MB: u64 = 50;
                const CRITICAL_MB: u64 = 150;
                const WARNING_MB: u64 = 300;
                let mb = 1024 * 1024;

                return if remaining < TERMINAL_MB * mb {
                    MemoryPressure::Terminal
                } else if remaining < CRITICAL_MB * mb {
                    MemoryPressure::Critical
                } else if remaining < WARNING_MB * mb {
                    MemoryPressure::Warning
                } else {
                    MemoryPressure::Normal
                };
            }
        }

        // Fallback (Linux, or if os_proc_available_memory returned 0):
        // compare process RSS against the static memory limit.
        let usage_ratio = if self.memory_limit_bytes > 0 {
            usage as f64 / self.memory_limit_bytes as f64
        } else {
            0.0
        };

        if usage_ratio >= TERMINAL_THRESHOLD {
            MemoryPressure::Terminal
        } else if usage_ratio >= CRITICAL_THRESHOLD {
            MemoryPressure::Critical
        } else if usage_ratio >= WARNING_THRESHOLD {
            MemoryPressure::Warning
        } else {
            MemoryPressure::Normal
        }
    }

    /// Query the OS for real-time available memory (jetsam-aware on iOS).
    #[cfg(any(target_os = "ios", target_os = "macos"))]
    fn os_available_memory() -> u64 {
        extern "C" {
            fn os_proc_available_memory() -> u64;
        }
        unsafe { os_proc_available_memory() }
    }

    /// Get current process memory usage
    fn current_memory_usage(&self) -> u64 {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            // Use task_info to get resident memory size
            // This is the actual physical memory our process is using
            use std::mem;

            #[repr(C)]
            struct TaskBasicInfo {
                virtual_size: u64,
                resident_size: u64,
                resident_size_max: u64,
                user_time: [u32; 2],
                system_time: [u32; 2],
                policy: i32,
                suspend_count: i32,
            }

            extern "C" {
                fn mach_task_self() -> u32;
                fn task_info(
                    target_task: u32,
                    flavor: u32,
                    task_info: *mut TaskBasicInfo,
                    task_info_count: *mut u32,
                ) -> i32;
            }

            const MACH_TASK_BASIC_INFO: u32 = 20;

            unsafe {
                let mut info: TaskBasicInfo = mem::zeroed();
                let mut count = (mem::size_of::<TaskBasicInfo>() / mem::size_of::<u32>()) as u32;
                let result = task_info(
                    mach_task_self(),
                    MACH_TASK_BASIC_INFO,
                    &mut info as *mut _,
                    &mut count,
                );
                if result == 0 {
                    info.resident_size
                } else {
                    0
                }
            }
        }

        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        {
            // Linux: read from /proc/self/statm
            if let Ok(content) = std::fs::read_to_string("/proc/self/statm") {
                let parts: Vec<&str> = content.split_whitespace().collect();
                if let Some(resident_pages) = parts.get(1) {
                    if let Ok(pages) = resident_pages.parse::<u64>() {
                        return pages * 4096; // page size
                    }
                }
            }
            0
        }
    }

    /// Estimate memory needed for a model.
    ///
    /// Calibrated against real benchmarks on M4 Max (Feb 2025):
    ///   Phi-3 3.82B Q4_K ctx 4096 → actual 3,325 MB, estimate 3,296 MB
    ///   Qwen2 7.6B  Q3_K ctx 4096 → actual 5,832 MB, estimate 5,147 MB
    ///
    /// The 1.35x safety multiplier ensures we reject models that would cause
    /// iOS jetsam kills. Slightly underestimates large-vocab models (150K+
    /// vocab uses F16 embeddings that inflate beyond the quant-based estimate).
    pub fn estimate_model_memory(
        param_count_billions: f32,
        quantization: &str,
        context_length: u32,
    ) -> u64 {
        // Weight memory: depends on quantization (bits per parameter)
        let bits_per_param: f32 = match quantization {
            "Q2_K" => 2.5,
            "Q3_K" | "Q3_K_S" | "Q3_K_M" | "Q3_K_L" => 3.5,
            "Q4_0" | "Q4_1" | "Q4_K" | "Q4_K_S" | "Q4_K_M" => 4.5,
            "Q5_0" | "Q5_1" | "Q5_K" | "Q5_K_S" | "Q5_K_M" => 5.5,
            "Q6_K" => 6.5,
            "Q8_0" | "Q8_K" => 8.5,
            "F16" => 16.0,
            "F32" => 32.0,
            _ => 4.5, // Default to Q4 estimate
        };

        let weight_bytes = (param_count_billions as f64 * 1e9 * bits_per_param as f64 / 8.0) as u64;

        // KV cache + dequantization buffers: ~5 MB per 1K context per billion params.
        // Calibrated from Candle's quantized inference on Apple Silicon — includes
        // key/value tensors in F16 plus intermediate attention score buffers.
        let kv_cache_bytes = (context_length as f64 / 1024.0)
            * 5.0
            * param_count_billions as f64
            * 1024.0 * 1024.0;

        // Activation memory: ~10% of weight memory for FFN intermediates,
        // layer norm buffers, and Metal command buffer overhead.
        let activation_bytes = weight_bytes as f64 * 0.10;

        // 1.35x safety multiplier — accounts for tokenizer RSS, Metal shader
        // compilation, GPU buffer allocation, and process overhead. Calibrated
        // against real M4 Max measurements so the gate errs on the side of
        // rejecting models that are too close to the jetsam limit.
        let total = (weight_bytes as f64 + kv_cache_bytes + activation_bytes) * 1.35;

        total as u64
    }

    /// Check if a model will fit in available memory
    pub fn can_fit_model(
        &self,
        param_count_billions: f32,
        quantization: &str,
        context_length: u32,
    ) -> bool {
        let needed = Self::estimate_model_memory(param_count_billions, quantization, context_length);
        needed < self.memory_limit_bytes
    }

    /// Suggest actions to reduce memory pressure
    pub fn suggest_reduction(&self) -> MemoryReductionStrategy {
        match self.current_pressure {
            MemoryPressure::Normal => MemoryReductionStrategy::None,
            MemoryPressure::Warning => MemoryReductionStrategy::CompactContext,
            MemoryPressure::Critical => MemoryReductionStrategy::FallbackToCpu,
            MemoryPressure::Terminal => MemoryReductionStrategy::UnloadModel,
        }
    }

    pub fn peak_usage(&self) -> u64 {
        self.peak_usage_bytes
    }

    pub fn memory_limit(&self) -> u64 {
        self.memory_limit_bytes
    }
}

/// Strategies for reducing memory pressure
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryReductionStrategy {
    /// No action needed
    None,
    /// Compact the KV cache / reduce context window
    CompactContext,
    /// Fall back from Metal GPU to CPU (frees GPU memory)
    FallbackToCpu,
    /// Unload the model entirely
    UnloadModel,
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MB: f64 = 1024.0 * 1024.0;

    #[test]
    fn test_memory_estimation_7b_q4() {
        // 7B Q4_K_M model with 2048 context
        let estimate = MemoryPressureManager::estimate_model_memory(7.0, "Q4_K_M", 2048);
        let gb = estimate as f64 / GB;
        assert!(gb > 4.0 && gb < 8.0, "7B Q4 estimate was {:.2}GB", gb);
    }

    #[test]
    fn test_memory_estimation_3b_q4() {
        // 3B Q4_K_M model with 2048 context
        let estimate = MemoryPressureManager::estimate_model_memory(3.0, "Q4_K_M", 2048);
        let gb = estimate as f64 / GB;
        assert!(gb > 1.5 && gb < 4.0, "3B Q4 estimate was {:.2}GB", gb);
    }

    /// Calibrated against real M4 Max benchmark: Phi-3 3.82B Q4_K at ctx 4096
    /// used 3,325 MB peak resident memory on Metal.
    #[test]
    fn test_calibration_phi3_3b() {
        let estimate = MemoryPressureManager::estimate_model_memory(3.82, "Q4_K", 4096);
        let mb = estimate as f64 / MB;
        // Should be close to actual 3,325 MB (within ±15%)
        assert!(
            mb > 2800.0 && mb < 3900.0,
            "Phi-3 3.82B Q4_K estimate {:.0} MB, expected ~3325 MB", mb
        );
    }

    /// Calibrated against real M4 Max benchmark: Qwen2 7.6B Q3_K at ctx 4096
    /// used 5,832 MB peak resident memory on Metal.
    #[test]
    fn test_calibration_qwen2_7b() {
        let estimate = MemoryPressureManager::estimate_model_memory(7.6, "Q3_K", 4096);
        let mb = estimate as f64 / MB;
        // Should be in the right ballpark (within ±20% — Qwen2's 152K vocab
        // inflates actual RSS beyond what the quant formula predicts)
        assert!(
            mb > 4600.0 && mb < 7000.0,
            "Qwen2 7.6B Q3_K estimate {:.0} MB, expected ~5832 MB", mb
        );
    }

    /// iPhone 14 Pro Max safety gate: 6GB total, ~3.5GB usable for the app.
    /// Phi-3 3.82B Q4_K MUST fit. Qwen2 7.6B Q3_K MUST NOT fit.
    #[test]
    fn test_iphone14_pro_max_gate() {
        // iPhone 14 Pro Max: 6GB RAM, ~3.5GB usable after iOS overhead
        let iphone_limit: u64 = (3.5 * GB) as u64;
        let manager = MemoryPressureManager::new(iphone_limit);

        // Phi-3 3.82B Q4_K at ctx 4096 → should FIT (~3.3 GB actual)
        assert!(
            manager.can_fit_model(3.82, "Q4_K", 4096),
            "Phi-3 3.82B Q4_K should fit on iPhone 14 Pro Max (3.5GB limit)"
        );

        // Qwen2 7.6B Q3_K at ctx 4096 → should NOT fit (~5.7 GB actual)
        assert!(
            !manager.can_fit_model(7.6, "Q3_K", 4096),
            "Qwen2 7.6B Q3_K should NOT fit on iPhone 14 Pro Max (3.5GB limit)"
        );
    }

    #[test]
    fn test_pressure_levels() {
        let mut manager = MemoryPressureManager::new(4 * 1024 * 1024 * 1024); // 4GB limit
        let pressure = manager.check_pressure();
        // On dev machine, should be Normal unless we're eating tons of memory
        assert!(matches!(pressure, MemoryPressure::Normal | MemoryPressure::Warning));
    }
}
