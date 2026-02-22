/// Device capability detection for iOS/macOS
/// Determines optimal model configuration based on hardware

use crate::DeviceInfo;
use crate::memory::MemoryPressureManager;

/// Detect the current device's capabilities
pub fn detect_device() -> DeviceInfo {
    let total_memory = get_total_memory();
    let available_memory = get_available_memory();
    let has_metal = check_metal_support();

    let (recommended_quant, max_model_size_b) = recommend_config(total_memory, has_metal);
    let chip = get_chip_info();

    DeviceInfo {
        device_model: get_device_model(),
        total_memory_bytes: total_memory,
        available_memory_bytes: available_memory,
        has_metal_gpu: has_metal,
        has_neural_engine: check_neural_engine(),
        recommended_quant,
        max_model_size_b,
        max_model_file_size_bytes: max_safe_model_file_size(),
        chip_name: chip.chip_name,
        chip_variant: chip.chip_variant,
        recommended_context_length: recommended_context_length(total_memory),
    }
}

/// Returns the maximum safe model file size (in bytes) for this device.
///
/// iOS device-tier table (constrained by ~2 GB jetsam limit):
///   4 GB RAM (iPhone 12/13)         -> 200 MB max file
///   6 GB RAM (iPhone 14/15)         -> 400 MB max file
///   8 GB RAM (iPhone 15 Pro/16 Pro) -> 600 MB max file
///  16 GB RAM (iPad Pro M4)          -> 2 GB max file
///
/// macOS tiers (no jetsam, ~60% of RAM is safe for model loading):
///  16 GB RAM (MacBook Air M1/M2)    -> 8 GB max file
///  32 GB RAM (MacBook Pro M3 Pro)   -> 16 GB max file
///  64 GB RAM (Mac Studio M2 Ultra)  -> 32 GB max file
/// 128+ GB RAM (Mac Pro, Mac Studio) -> 64 GB max file
///
/// iOS limits account for the ~2.25x Metal expansion ratio
/// (weights + KV cache + computation buffers) and the jetsam kill threshold.
/// macOS limits are more generous since there's no jetsam and swap is available.
pub fn max_safe_model_file_size() -> u64 {
    let total_mb = get_total_memory() / (1024 * 1024);

    #[cfg(target_os = "ios")]
    {
        match total_mb {
            0..=4500 => 200 * 1024 * 1024,          // 4 GB devices: 200 MB
            4501..=6500 => 400 * 1024 * 1024,        // 6 GB devices: 400 MB
            6501..=12000 => 600 * 1024 * 1024,       // 8 GB devices: 600 MB
            _ => 2 * 1024 * 1024 * 1024,             // 16+ GB: 2 GB
        }
    }

    #[cfg(not(target_os = "ios"))]
    {
        match total_mb {
            0..=4500 => 200 * 1024 * 1024,           // 4 GB: 200 MB
            4501..=6500 => 400 * 1024 * 1024,        // 6 GB: 400 MB
            6501..=12000 => 2 * 1024 * 1024 * 1024,  // 8 GB: 2 GB
            12001..=20000 => 8_u64 * 1024 * 1024 * 1024,  // 16 GB: 8 GB
            20001..=40000 => 16_u64 * 1024 * 1024 * 1024, // 32 GB: 16 GB
            40001..=80000 => 32_u64 * 1024 * 1024 * 1024, // 64 GB: 32 GB
            _ => 64_u64 * 1024 * 1024 * 1024,        // 128+ GB: 64 GB
        }
    }
}

/// Recommend quantization and max model size based on the calibrated
/// memory estimator rather than a hardcoded lookup table.
///
/// Strategy: for each (quant, model_size) pair, compute estimated peak memory
/// using the same formula as load_model's pre-flight check. Pick the best
/// quant that allows the largest practical model to fit in the usable budget.
fn recommend_config(total_memory: u64, _has_metal: bool) -> (String, u32) {
    // Usable memory: iOS gets ~55% (jetsam headroom), macOS gets ~60% (swap OK)
    let usable_fraction: f64 = if cfg!(target_os = "ios") { 0.55 } else { 0.60 };
    let budget = (total_memory as f64 * usable_fraction) as u64;

    // Default context length for recommendations (conservative)
    let ctx = 2048u32;

    // Quant levels from best quality to most compressed.
    // We prefer higher quality at the same model size.
    let quant_levels: &[(&str, &str)] = &[
        ("Q8_0", "Q8_0"),
        ("Q6_K", "Q6_K"),
        ("Q4_K_M", "Q4_K_M"),
    ];

    // Common model sizes (billions) to probe, largest first
    let model_sizes: &[u32] = &[70, 30, 14, 13, 8, 7, 3, 2, 1];

    // Find the largest model size that fits, then the best quant for it
    for &size_b in model_sizes {
        for &(quant, quant_label) in quant_levels {
            let estimate = MemoryPressureManager::estimate_model_memory(
                size_b as f32, quant, ctx,
            );
            if estimate < budget {
                return (quant_label.to_string(), size_b);
            }
        }
    }

    // Fallback: very constrained device
    ("Q4_K_M".to_string(), 1)
}

/// Get total physical memory
/// On iOS this uses os_proc_available_memory or similar syscalls
fn get_total_memory() -> u64 {
    #[cfg(target_os = "macos")]
    {
        use std::mem;
        let mut size: u64 = 0;
        let mut len = mem::size_of::<u64>();
        let mib = [libc::CTL_HW, libc::HW_MEMSIZE];
        unsafe {
            libc::sysctl(
                mib.as_ptr() as *mut _,
                2,
                &mut size as *mut _ as *mut _,
                &mut len,
                std::ptr::null_mut(),
                0,
            );
        }
        size
    }

    #[cfg(target_os = "ios")]
    {
        // On iOS, use NSProcessInfo.processInfo.physicalMemory equivalent
        // via sysctl which works the same way
        use std::mem;
        let mut size: u64 = 0;
        let mut len = mem::size_of::<u64>();
        let mib = [libc::CTL_HW, libc::HW_MEMSIZE];
        unsafe {
            libc::sysctl(
                mib.as_ptr() as *mut _,
                2,
                &mut size as *mut _ as *mut _,
                &mut len,
                std::ptr::null_mut(),
                0,
            );
        }
        size
    }

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        // Fallback for development on Linux
        // Read from /proc/meminfo
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if line.starts_with("MemTotal:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if let Some(kb_str) = parts.get(1) {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            return kb * 1024;
                        }
                    }
                }
            }
        }
        4 * 1024 * 1024 * 1024 // Default 4GB fallback
    }
}

/// Get currently available memory
fn get_available_memory() -> u64 {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        // os_proc_available_memory() is the correct iOS API
        // It accounts for memory pressure and jetsam limits
        extern "C" {
            fn os_proc_available_memory() -> u64;
        }
        unsafe { os_proc_available_memory() }
    }

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        // Linux fallback
        if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
            for line in content.lines() {
                if line.starts_with("MemAvailable:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if let Some(kb_str) = parts.get(1) {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            return kb * 1024;
                        }
                    }
                }
            }
        }
        2 * 1024 * 1024 * 1024 // Default 2GB fallback
    }
}

/// Check if Metal GPU is available
fn check_metal_support() -> bool {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        // On Apple platforms, Metal is available on:
        // - All Apple Silicon (M1+)
        // - A8+ (iPhone 6+)
        // For our purposes (LLM inference), we want A14+ / M1+
        true // Safe assumption for our target devices
    }

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        false
    }
}

/// Check if Neural Engine is available
fn check_neural_engine() -> bool {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        // Neural Engine available on A11+ / M1+
        // Currently not directly accessible for LLM inference
        // but we track it for future CoreML integration
        true
    }

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        false
    }
}

/// Get device model string
fn get_device_model() -> String {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        use std::ffi::CStr;
        let mut size: usize = 0;
        unsafe {
            libc::sysctlbyname(
                b"hw.machine\0".as_ptr() as *const _,
                std::ptr::null_mut(),
                &mut size,
                std::ptr::null_mut(),
                0,
            );
            let mut buf = vec![0u8; size];
            libc::sysctlbyname(
                b"hw.machine\0".as_ptr() as *const _,
                buf.as_mut_ptr() as *mut _,
                &mut size,
                std::ptr::null_mut(),
                0,
            );
            CStr::from_bytes_until_nul(&buf)
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|_| "Unknown".to_string())
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        "Linux-Dev".to_string()
    }
}

/// Apple Silicon chip information
struct ChipInfo {
    chip_name: String,
    chip_variant: String,
}

/// Detect Apple Silicon chip name and variant via sysctl.
///
/// Uses `machdep.cpu.brand_string` which returns e.g. "Apple M4 Max".
/// Parses the string to extract generation (M1-M4) and variant (Pro/Max/Ultra).
fn get_chip_info() -> ChipInfo {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        let brand = sysctl_string(b"machdep.cpu.brand_string\0");
        if brand.starts_with("Apple M") {
            let variant = if brand.contains("Ultra") {
                "Ultra"
            } else if brand.contains("Max") {
                "Max"
            } else if brand.contains("Pro") {
                "Pro"
            } else {
                "Base"
            };
            return ChipInfo {
                chip_name: brand,
                chip_variant: variant.to_string(),
            };
        }
        ChipInfo {
            chip_name: if brand.is_empty() { "Unknown".to_string() } else { brand },
            chip_variant: String::new(),
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        ChipInfo {
            chip_name: "Unknown".to_string(),
            chip_variant: String::new(),
        }
    }
}

/// Read a sysctl string value by name.
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn sysctl_string(name: &[u8]) -> String {
    use std::ffi::CStr;
    let mut size: usize = 0;
    unsafe {
        libc::sysctlbyname(
            name.as_ptr() as *const _,
            std::ptr::null_mut(),
            &mut size,
            std::ptr::null_mut(),
            0,
        );
        if size == 0 {
            return String::new();
        }
        let mut buf = vec![0u8; size];
        libc::sysctlbyname(
            name.as_ptr() as *const _,
            buf.as_mut_ptr() as *mut _,
            &mut size,
            std::ptr::null_mut(),
            0,
        );
        CStr::from_bytes_until_nul(&buf)
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default()
    }
}

/// Recommend context window size based on available RAM.
///
///   8 GB → 4096
///  16 GB → 8192
///  32 GB → 16384
///  64+ GB → 32768
fn recommended_context_length(total_memory: u64) -> u32 {
    let total_gb = total_memory / (1024 * 1024 * 1024);
    match total_gb {
        0..=12 => 4096,
        13..=24 => 8192,
        25..=48 => 16384,
        _ => 32768,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_device() {
        let info = detect_device();
        assert!(info.total_memory_bytes > 0);
        assert!(!info.device_model.is_empty());
        assert!(!info.recommended_quant.is_empty());
    }

    #[test]
    fn test_recommend_config() {
        // 6GB device: budget ~3.3 GB (iOS) or ~3.6 GB (macOS)
        // Should recommend a small model (≤3B)
        let (_, max_b) = recommend_config(6 * 1024 * 1024 * 1024, true);
        assert!(max_b <= 3, "6GB device should recommend ≤3B, got {}B", max_b);
        assert!(max_b >= 1, "6GB device should recommend ≥1B, got {}B", max_b);

        // 8GB device: budget ~4.4 GB (iOS) or ~4.8 GB (macOS)
        let (_, max_b) = recommend_config(8 * 1024 * 1024 * 1024, true);
        assert!(max_b >= 3, "8GB device should recommend ≥3B, got {}B", max_b);
        assert!(max_b <= 8, "8GB device should recommend ≤8B, got {}B", max_b);
    }

    #[test]
    fn test_recommend_config_large_memory() {
        // 64GB device: budget ~38 GB — should comfortably fit 30B+ models
        let (_, max_b) = recommend_config(64 * 1024 * 1024 * 1024, true);
        assert!(max_b >= 30, "64GB device should recommend ≥30B, got {}B", max_b);
    }

    #[test]
    fn test_max_safe_model_file_size() {
        let max_size = max_safe_model_file_size();
        // On any real machine this should return a positive value
        assert!(max_size > 0);
        // It should be one of the known tier values (iOS or macOS)
        let valid_sizes: Vec<u64> = vec![
            200 * 1024 * 1024,
            400 * 1024 * 1024,
            600 * 1024 * 1024,
            2_u64 * 1024 * 1024 * 1024,
            8_u64 * 1024 * 1024 * 1024,
            16_u64 * 1024 * 1024 * 1024,
            32_u64 * 1024 * 1024 * 1024,
            64_u64 * 1024 * 1024 * 1024,
        ];
        assert!(
            valid_sizes.contains(&max_size),
            "max_safe_model_file_size returned {max_size}, expected one of {valid_sizes:?}"
        );
    }

    #[test]
    fn test_detect_device_includes_max_file_size() {
        let info = detect_device();
        assert!(info.max_model_file_size_bytes > 0);
    }

    #[test]
    fn test_detect_device_chip_info() {
        let info = detect_device();
        assert!(!info.chip_name.is_empty());
        assert!(info.recommended_context_length >= 4096);
    }

    #[test]
    fn test_recommended_context_length() {
        assert_eq!(recommended_context_length(8 * 1024 * 1024 * 1024), 4096);
        assert_eq!(recommended_context_length(16 * 1024 * 1024 * 1024), 8192);
        assert_eq!(recommended_context_length(32 * 1024 * 1024 * 1024), 16384);
        assert_eq!(recommended_context_length(64 * 1024 * 1024 * 1024), 32768);
        assert_eq!(recommended_context_length(128 * 1024 * 1024 * 1024), 32768);
    }
}
