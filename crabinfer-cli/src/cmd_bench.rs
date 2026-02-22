use crabinfer_core::engine::{self, CrabInferEngine};
use crabinfer_core::memory::MemoryPressureManager;
use crabinfer_core::{EngineConfig, GenerationStats};

pub fn run(model: &str, prompt: &str, max_tokens: u32, temperature: f32, context_length: u32) {
    eprintln!("=== CrabInfer Benchmark ===");
    eprintln!("Model: {model}");
    eprintln!("Prompt: \"{prompt}\"");
    eprintln!("Max tokens: {max_tokens}");
    eprintln!("Temperature: {temperature}");
    eprintln!();

    // Print memory estimate using peek_model_metadata (reads GGUF header only)
    if let Ok((info, _emb_overhead)) = engine::peek_model_metadata(model, context_length) {
        let total_b = info.parameter_count as f32 / 1e9;
        let active_b = info.active_parameter_count as f32 / 1e9;
        let estimated = MemoryPressureManager::estimate_model_memory_moe(
            total_b,
            active_b,
            &info.quantization,
            context_length,
        );
        if info.is_moe {
            eprintln!(
                "Memory estimate: {:.2} GB (MoE: {:.1}B total / {:.1}B active, {}, ctx {context_length})",
                estimated as f64 / (1024.0 * 1024.0 * 1024.0),
                total_b,
                active_b,
                info.quantization,
            );
        } else {
            eprintln!(
                "Memory estimate: {:.2} GB (for {:.1}B params, {}, ctx {context_length})",
                estimated as f64 / (1024.0 * 1024.0 * 1024.0),
                total_b,
                info.quantization,
            );
        }
        eprintln!();
    }

    // Run Metal
    eprintln!("--- Metal GPU ---");
    let metal_stats = run_single(model, prompt, max_tokens, temperature, context_length, true);

    // Run CPU
    eprintln!("--- CPU ---");
    let cpu_stats = run_single(model, prompt, max_tokens, temperature, context_length, false);

    // Print comparison
    eprintln!();
    eprintln!("=== Comparison ===");
    eprintln!(
        "{:<25} {:>12} {:>12} {:>10}",
        "", "Metal", "CPU", "Speedup"
    );
    eprintln!("{:-<25} {:-^12} {:-^12} {:-^10}", "", "", "", "");

    match (&metal_stats, &cpu_stats) {
        (Some(m), Some(c)) => {
            let speedup_tps = m.tokens_per_second / c.tokens_per_second;
            let speedup_ttft = c.time_to_first_token_ms / m.time_to_first_token_ms;

            eprintln!(
                "{:<25} {:>10.1}/s {:>10.1}/s {:>9.2}x",
                "Tokens/sec", m.tokens_per_second, c.tokens_per_second, speedup_tps
            );
            eprintln!(
                "{:<25} {:>10.0}ms {:>10.0}ms {:>9.2}x",
                "Time to first token",
                m.time_to_first_token_ms,
                c.time_to_first_token_ms,
                speedup_ttft
            );
            eprintln!(
                "{:<25} {:>10.0}ms {:>10.0}ms {:>9.2}x",
                "Total time",
                m.total_time_ms,
                c.total_time_ms,
                c.total_time_ms / m.total_time_ms
            );
            eprintln!(
                "{:<25} {:>8} MB {:>8} MB",
                "Peak memory",
                m.peak_memory_bytes / (1024 * 1024),
                c.peak_memory_bytes / (1024 * 1024),
            );
            eprintln!(
                "{:<25} {:>12} {:>12}",
                "Tokens generated", m.tokens_generated, c.tokens_generated
            );
        }
        (Some(m), None) => {
            eprintln!("CPU run failed. Metal only:");
            eprintln!("  Tokens/sec:     {:.1}", m.tokens_per_second);
            eprintln!("  TTFT:           {:.0}ms", m.time_to_first_token_ms);
            eprintln!(
                "  Peak memory:    {} MB",
                m.peak_memory_bytes / (1024 * 1024)
            );
        }
        (None, Some(c)) => {
            eprintln!("Metal run failed. CPU only:");
            eprintln!("  Tokens/sec:     {:.1}", c.tokens_per_second);
            eprintln!("  TTFT:           {:.0}ms", c.time_to_first_token_ms);
            eprintln!(
                "  Peak memory:    {} MB",
                c.peak_memory_bytes / (1024 * 1024)
            );
        }
        (None, None) => {
            eprintln!("Both runs failed!");
        }
    }

    eprintln!("=== Done ===");
}

fn run_single(
    model: &str,
    prompt: &str,
    max_tokens: u32,
    temperature: f32,
    context_length: u32,
    use_metal: bool,
) -> Option<GenerationStats> {
    let backend = if use_metal { "Metal" } else { "CPU" };

    let config = EngineConfig {
        model_path: String::new(),
        max_tokens,
        temperature,
        top_p: 0.9,
        context_length,
        use_metal,
        memory_limit_bytes: 0,
        metallib_path: String::new(),
    };

    let engine = match CrabInferEngine::new(config) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("  Failed to create engine ({backend}): {e}");
            return None;
        }
    };

    eprintln!("  Loading model...");
    if let Err(e) = engine.load_model(model.to_string()) {
        eprintln!("  Failed to load model ({backend}): {e}");
        return None;
    }

    eprintln!("  Generating...");
    match engine.complete(prompt.to_string(), max_tokens, temperature) {
        Ok(response) => {
            let preview: String = response.chars().take(80).collect();
            eprintln!("  Output: {preview}...");
        }
        Err(e) => {
            eprintln!("  Generation failed ({backend}): {e}");
            return None;
        }
    }

    let stats = engine.last_stats();
    if let Some(ref s) = stats {
        eprintln!(
            "  [{} tokens, {:.1} tok/s, TTFT {:.0}ms, total {:.0}ms, peak {} MB]",
            s.tokens_generated,
            s.tokens_per_second,
            s.time_to_first_token_ms,
            s.total_time_ms,
            s.peak_memory_bytes / (1024 * 1024),
        );
    }
    eprintln!();

    engine.unload_model();
    stats
}

