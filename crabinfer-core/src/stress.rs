//! Stress test module for load/unload cycle memory leak detection.
//!
//! This is a child module of `engine` (declared via `#[path]`), so it has
//! access to all private fields and methods of `CrabInferEngine`.

use super::*;

impl CrabInferEngine {
    /// Run a load/unload stress test to detect memory leaks.
    ///
    /// Loads the model, generates `tokens_per_cycle` tokens, unloads,
    /// and repeats for `cycles` iterations.  Returns a log of memory
    /// measurements after each cycle so the caller can check for leaks.
    ///
    /// **Key design**: between cycles the model weights are dropped but
    /// the Metal buffer pool is deliberately NOT flushed.  Our Candle
    /// fork's `new_buffer_with_data` reuses pool buffers of the same size
    /// instead of allocating fresh Metal pages.  This keeps peak memory
    /// at ~1× model size across all cycles, preventing jetsam kills.
    ///
    /// The Metal device is reused across cycles (creating a new one each
    /// cycle leaks Metal framework overhead).
    pub fn stress_test(
        &self,
        model_path: String,
        cycles: u32,
        tokens_per_cycle: u32,
    ) -> Result<Vec<String>, CrabInferError> {
        let mut log: Vec<String> = Vec::new();

        // Flush any stale Metal buffers left over from a previous stress test
        // or model load. Without this, a failed prior stress test (which skips
        // final cleanup) can leave hundreds of MB of unreleased buffers that
        // push the next cold load over the jetsam limit.
        #[cfg(feature = "metal")]
        if let Some(ref device) = *self.metal_device.lock_recover() {
            if let Ok(metal) = device.as_metal_device() {
                let stale = metal.current_allocated_size() / (1024 * 1024);
                if stale > 0 {
                    log_debug!("[CrabInfer-StressTest] Flushing {} MB of stale Metal buffers", stale);
                    let _ = metal.wait_until_completed();
                    let _ = metal.release_unused_buffers();
                }
            }
        }
        Self::force_memory_reclaim();

        let file_size_mb = std::fs::metadata(&model_path)
            .map(|m| m.len() / (1024 * 1024))
            .unwrap_or(0);

        // Safety margins:
        //   Cycle 1 (cold): 1.5x file size — new Metal allocations + overhead
        //   Cycle 2+ (warm): 200 MB — buffers already exist in pool, only need
        //     headroom for tokenizer, GGUF parsing, and temporary copies
        let cold_safety_mb = (file_size_mb as f64 * 1.5) as u64;
        let warm_safety_mb: u64 = 200;

        let baseline_rss = Self::resident_memory_mb();
        let baseline_avail = Self::available_memory_mb();
        let baseline_metal = Self::metal_allocated_mb(&self.metal_device);
        log.push(format!(
            "Baseline: RSS {} MB, free {} MB, Metal {} MB  (model {} MB)",
            baseline_rss, baseline_avail, baseline_metal, file_size_mb
        ));
        log_debug!(
            "[CrabInfer-StressTest] Baseline RSS={} avail={} metal={}",
            baseline_rss, baseline_avail, baseline_metal
        );

        let mut completed_cycles = 0u32;

        for i in 0..cycles {
            log_debug!("[CrabInfer-StressTest] === Cycle {}/{} ===", i + 1, cycles);

            // Safety check: bail if available memory is too low.
            // After cycle 1, buffers exist in pool and will be reused via
            // memcpy — no new Metal allocations needed, so the margin is small.
            let safety_mb = if i == 0 { cold_safety_mb } else { warm_safety_mb };
            let available_mb = Self::available_memory_mb();
            if available_mb > 0 && available_mb < safety_mb {
                let msg = format!(
                    "Stopped before cycle {}: {} MB free < {} MB needed",
                    i + 1, available_mb, safety_mb
                );
                log_debug!("[CrabInfer-StressTest] {}", msg);
                log.push(msg);
                break;
            }

            // Load
            match self.load_model(model_path.clone()) {
                Ok(()) => {}
                Err(CrabInferError::OutOfMemory { reason }) => {
                    log.push(format!("Stopped at cycle {}: {}", i + 1, reason));
                    break;
                }
                Err(e) => return Err(e),
            }
            let after_load_rss = Self::resident_memory_mb();
            let after_load_metal = Self::metal_allocated_mb(&self.metal_device);

            // Generate tokens
            self.reset();
            let prompt = "Hello world".to_string();
            for t in 0..tokens_per_cycle {
                match self.next_token(prompt.clone())? {
                    Some(tok) if tok.is_end_of_sequence => {
                        log_debug!("[CrabInfer-StressTest] EOS at token {}", t + 1);
                        break;
                    }
                    None => break,
                    _ => {}
                }
            }

            // --- Unload weights but KEEP Metal buffers in pool for reuse ---
            //
            // We intentionally do NOT call self.unload_model() here because
            // that flushes the buffer pool.  Instead we drop only the model
            // state.  On the next load_model() call, Candle's
            // new_buffer_with_data() will find same-size unused buffers in
            // the pool and reuse them (memcpy into existing Metal pages)
            // instead of creating new Metal allocations.
            {
                let old_model = self.model.lock_recover().take();
                *self.model_info.lock_recover() = None;
                *self.streaming.lock_recover() = None;
                *self.last_stats.lock_recover() = None;
                if let Some(loaded) = old_model {
                    drop(loaded);
                    // Wait for any in-flight GPU work to finish before
                    // measuring, but do NOT release buffers.
                    #[cfg(feature = "metal")]
                    if let Some(ref device) = *self.metal_device.lock_recover() {
                        if let Ok(metal) = device.as_metal_device() {
                            let _ = metal.wait_until_completed();
                        }
                    }
                }
            }

            Self::force_memory_reclaim();
            std::thread::sleep(std::time::Duration::from_millis(100));

            let after_unload_rss = Self::resident_memory_mb();
            let after_unload_metal = Self::metal_allocated_mb(&self.metal_device);
            let available_after = Self::available_memory_mb();
            completed_cycles = i + 1;

            let entry = format!(
                "Cycle {}: load {}|{} MB → free {}|{} MB (avail {})",
                i + 1,
                after_load_rss, after_load_metal,
                after_unload_rss, after_unload_metal,
                available_after
            );
            log_debug!("[CrabInfer-StressTest] {}", entry);
            log.push(entry);
        }

        // Final cleanup: now actually release Metal buffers and unload
        self.unload_model();
        Self::force_memory_reclaim();
        std::thread::sleep(std::time::Duration::from_millis(200));

        let final_rss = Self::resident_memory_mb();
        let final_avail = Self::available_memory_mb();
        let final_metal = Self::metal_allocated_mb(&self.metal_device);
        let overhead = final_rss.saturating_sub(baseline_rss);
        let per_cycle = if completed_cycles > 0 {
            overhead / completed_cycles as u64
        } else {
            0
        };
        log.push(format!(
            "Done: {} cycles | RSS {} MB (+{}, ~{}/cyc) Metal {} MB | {} MB free",
            completed_cycles, final_rss, overhead, per_cycle, final_metal, final_avail
        ));

        Ok(log)
    }

    /// Get available memory in MB (jetsam-aware on iOS).
    fn available_memory_mb() -> u64 {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            extern "C" { fn os_proc_available_memory() -> u64; }
            return (unsafe { os_proc_available_memory() }) / (1024 * 1024);
        }
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        { 0 }
    }

    /// Get current process resident memory in MB (for stress test logging).
    fn resident_memory_mb() -> u64 {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
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
                    return info.resident_size / (1024 * 1024);
                }
            }
        }

        0
    }
}
