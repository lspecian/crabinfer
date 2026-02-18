/// Device capability detection for iOS/macOS
/// Determines optimal model configuration based on hardware

use crate::DeviceInfo;

/// Detect the current device's capabilities
pub fn detect_device() -> DeviceInfo {
    let total_memory = get_total_memory();
    let available_memory = get_available_memory();
    let has_metal = check_metal_support();

    let (recommended_quant, max_model_size_b) = recommend_config(total_memory, has_metal);

    DeviceInfo {
        device_model: get_device_model(),
        total_memory_bytes: total_memory,
        available_memory_bytes: available_memory,
        has_metal_gpu: has_metal,
        has_neural_engine: check_neural_engine(),
        recommended_quant: recommended_quant,
        max_model_size_b: max_model_size_b,
    }
}

/// Recommend quantization and max model size based on available memory
fn recommend_config(total_memory: u64, has_metal: bool) -> (String, u32) {
    let gb = total_memory / (1024 * 1024 * 1024);

    // Reserve ~2GB for iOS system + app overhead
    // Model needs to fit in remaining memory
    // Rule of thumb: Q4_K_M ≈ 0.55 * params in GB (for the weights)
    // Plus KV cache which grows with context length

    match (gb, has_metal) {
        // 4GB devices (iPhone 12, 13, 14 base)
        // ~2GB usable → max 1B-2B models at Q4
        (0..=4, _) => ("Q4_K_M".to_string(), 2),

        // 6GB devices (iPhone 12-14 Pro, iPhone 15 base)
        // ~3.5GB usable → 3B models comfortable, 7B at Q2 risky
        (5..=6, true) => ("Q4_K_M".to_string(), 3),
        (5..=6, false) => ("Q4_0".to_string(), 2),

        // 8GB devices (iPhone 15 Pro, 16 Pro)
        // ~5GB usable → 7B at Q4 feasible
        (7..=8, true) => ("Q4_K_M".to_string(), 7),
        (7..=8, false) => ("Q4_K_M".to_string(), 3),

        // 16GB+ (iPads, Macs)
        (9..=16, true) => ("Q6_K".to_string(), 13),
        (9..=16, false) => ("Q4_K_M".to_string(), 7),

        // 32GB+ (Mac Studio, MacBook Pro)
        (17..=32, _) => ("Q6_K".to_string(), 30),

        // 64GB+ (high-end Macs)
        (33..=64, _) => ("Q8_0".to_string(), 70),

        // 128GB+ (Mac Pro, Mac Studio Max)
        (65.., _) => ("Q8_0".to_string(), 70),

        _ => ("Q4_0".to_string(), 1),
    }
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
        // 6GB device with Metal → should recommend 3B Q4_K_M
        let (quant, max_b) = recommend_config(6 * 1024 * 1024 * 1024, true);
        assert_eq!(quant, "Q4_K_M");
        assert_eq!(max_b, 3);

        // 8GB device with Metal → should recommend 7B Q4_K_M
        let (quant, max_b) = recommend_config(8 * 1024 * 1024 * 1024, true);
        assert_eq!(quant, "Q4_K_M");
        assert_eq!(max_b, 7);
    }
}
