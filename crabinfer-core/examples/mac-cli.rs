/// CrabInfer macOS CLI — fastest path to testing inference without Xcode
///
/// Usage:
///   cargo run --release --example mac-cli -- --model path/to/model.gguf --prompt "hello"
///   cargo run --release --example mac-cli -- --model path/to/model.gguf --prompt "hello" --max-tokens 100 --temperature 0.5
///   cargo run --release --example mac-cli -- --model path/to/model.gguf --prompt "hello" --cpu
///   cargo run --release --example mac-cli -- --model path/to/model.gguf --prompt "hello" --benchmark
///   cargo run --release --example mac-cli -- --model path/to/model.gguf --prompt "hello" --stream

use crabinfer_core::engine::CrabInferEngine;
use crabinfer_core::memory::MemoryPressureManager;
use crabinfer_core::EngineConfig;

fn main() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    let args = parse_args();

    if args.model.is_empty() {
        eprintln!("Usage: mac-cli --model <path/to/model.gguf> --prompt <text>");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --model <path>         Path to GGUF model file (required)");
        eprintln!("  --prompt <text>        Prompt text (required)");
        eprintln!("  --max-tokens <n>       Maximum tokens to generate (default: 128)");
        eprintln!("  --temperature <f>      Sampling temperature (default: 0.7)");
        eprintln!("  --top-p <f>            Top-p sampling (default: 0.9)");
        eprintln!("  --context-length <n>   Context window size (default: 4096)");
        eprintln!("  --cpu                  Force CPU-only (skip Metal)");
        eprintln!("  --benchmark            Run CPU vs Metal comparison benchmark");
        eprintln!("  --stream               Use token-by-token streaming (next_token API)");
        std::process::exit(1);
    }

    if args.benchmark {
        run_benchmark(&args);
    } else if args.stream {
        run_streaming(&args);
    } else {
        run_normal(&args);
    }
}

// ---------------------------------------------------------------------------
// Normal generation
// ---------------------------------------------------------------------------

fn run_normal(args: &CliArgs) {
    let config = EngineConfig {
        model_path: String::new(),
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        context_length: args.context_length,
        use_metal: !args.cpu,
        memory_limit_bytes: 0,
        metallib_path: String::new(),
    };

    let engine = match CrabInferEngine::new(config) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to create engine: {e}");
            std::process::exit(1);
        }
    };

    eprintln!("Loading model: {}", args.model);
    if let Err(e) = engine.load_model(args.model.clone()) {
        eprintln!("Failed to load model: {e}");
        std::process::exit(1);
    }

    if let Ok(info) = engine.model_info() {
        eprintln!(
            "Model: {} ({}, {}, {} params, ctx {})",
            info.model_name, info.architecture, info.quantization,
            format_params(info.parameter_count), info.context_length
        );
    }

    eprintln!("Generating...\n");
    match engine.complete(args.prompt.clone(), args.max_tokens, args.temperature) {
        Ok(response) => {
            println!("{response}");
        }
        Err(e) => {
            eprintln!("\nGeneration failed: {e}");
            std::process::exit(1);
        }
    }

    if let Some(stats) = engine.last_stats() {
        eprintln!();
        eprintln!(
            "[{} tokens, {:.1} tok/s, TTFT {:.0}ms, total {:.0}ms, peak mem {} MB, {} backend]",
            stats.tokens_generated,
            stats.tokens_per_second,
            stats.time_to_first_token_ms,
            stats.total_time_ms,
            stats.peak_memory_bytes / (1024 * 1024),
            stats.compute_backend,
        );
    }
}

// ---------------------------------------------------------------------------
// Streaming generation (next_token API)
// ---------------------------------------------------------------------------

fn run_streaming(args: &CliArgs) {
    let config = EngineConfig {
        model_path: String::new(),
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        context_length: args.context_length,
        use_metal: !args.cpu,
        memory_limit_bytes: 0,
        metallib_path: String::new(),
    };

    let engine = match CrabInferEngine::new(config) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to create engine: {e}");
            std::process::exit(1);
        }
    };

    eprintln!("Loading model: {}", args.model);
    if let Err(e) = engine.load_model(args.model.clone()) {
        eprintln!("Failed to load model: {e}");
        std::process::exit(1);
    }

    if let Ok(info) = engine.model_info() {
        eprintln!(
            "Model: {} ({}, {}, {} params, ctx {})",
            info.model_name, info.architecture, info.quantization,
            format_params(info.parameter_count), info.context_length
        );
    }

    eprintln!("Streaming...\n");
    use std::io::Write;
    let stdout = std::io::stdout();

    let mut token_count = 0u32;
    for _ in 0..args.max_tokens {
        match engine.next_token(args.prompt.clone()) {
            Ok(Some(tok)) => {
                if tok.is_end_of_sequence {
                    break;
                }
                print!("{}", tok.text);
                stdout.lock().flush().ok();
                token_count += 1;
            }
            Ok(None) => break,
            Err(e) => {
                eprintln!("\nStreaming failed: {e}");
                std::process::exit(1);
            }
        }
    }
    println!();

    if let Some(stats) = engine.last_stats() {
        eprintln!(
            "[{} tokens, {:.1} tok/s, TTFT {:.0}ms, total {:.0}ms, {} backend]",
            stats.tokens_generated,
            stats.tokens_per_second,
            stats.time_to_first_token_ms,
            stats.total_time_ms,
            stats.compute_backend,
        );
    } else {
        eprintln!("[{} tokens streamed]", token_count);
    }

    engine.reset();
}

// ---------------------------------------------------------------------------
// Benchmark: CPU vs Metal comparison
// ---------------------------------------------------------------------------

fn run_benchmark(args: &CliArgs) {
    eprintln!("=== CrabInfer Benchmark ===");
    eprintln!("Model: {}", args.model);
    eprintln!("Prompt: \"{}\"", args.prompt);
    eprintln!("Max tokens: {}", args.max_tokens);
    eprintln!("Temperature: {}", args.temperature);
    eprintln!();

    // Print memory estimate from the memory pressure manager
    if let Ok(info) = peek_model_info(&args.model) {
        let param_b = info.parameter_count as f32 / 1e9;
        let estimated = MemoryPressureManager::estimate_model_memory(
            param_b,
            &info.quantization,
            args.context_length,
        );
        eprintln!(
            "Memory estimate: {:.2} GB (for {:.1}B params, {}, ctx {})",
            estimated as f64 / (1024.0 * 1024.0 * 1024.0),
            param_b,
            info.quantization,
            args.context_length,
        );
        eprintln!();
    }

    // Run on Metal
    eprintln!("--- Metal GPU ---");
    let metal_stats = run_single_bench(args, true);

    // Run on CPU
    eprintln!("--- CPU ---");
    let cpu_stats = run_single_bench(args, false);

    // Print comparison table
    eprintln!();
    eprintln!("=== Comparison ===");
    eprintln!("{:<25} {:>12} {:>12} {:>10}", "", "Metal", "CPU", "Speedup");
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
                "Time to first token", m.time_to_first_token_ms, c.time_to_first_token_ms, speedup_ttft
            );
            eprintln!(
                "{:<25} {:>10.0}ms {:>10.0}ms {:>9.2}x",
                "Total time", m.total_time_ms, c.total_time_ms, c.total_time_ms / m.total_time_ms
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
            eprintln!("  Peak memory:    {} MB", m.peak_memory_bytes / (1024 * 1024));
        }
        (None, Some(c)) => {
            eprintln!("Metal run failed. CPU only:");
            eprintln!("  Tokens/sec:     {:.1}", c.tokens_per_second);
            eprintln!("  TTFT:           {:.0}ms", c.time_to_first_token_ms);
            eprintln!("  Peak memory:    {} MB", c.peak_memory_bytes / (1024 * 1024));
        }
        (None, None) => {
            eprintln!("Both runs failed!");
        }
    }

    eprintln!("=== Done ===");
}

fn run_single_bench(
    args: &CliArgs,
    use_metal: bool,
) -> Option<crabinfer_core::GenerationStats> {
    let backend = if use_metal { "Metal" } else { "CPU" };

    let config = EngineConfig {
        model_path: String::new(),
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        context_length: args.context_length,
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
    if let Err(e) = engine.load_model(args.model.clone()) {
        eprintln!("  Failed to load model ({backend}): {e}");
        return None;
    }

    eprintln!("  Generating...");
    match engine.complete(args.prompt.clone(), args.max_tokens, args.temperature) {
        Ok(response) => {
            // Print first 80 chars of response
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
            s.tokens_generated, s.tokens_per_second,
            s.time_to_first_token_ms, s.total_time_ms,
            s.peak_memory_bytes / (1024 * 1024),
        );
    }
    eprintln!();

    // Unload to free memory before next run
    engine.unload_model();

    stats
}

/// Quick peek at model info without full weight loading (just parse GGUF header)
fn peek_model_info(model_path: &str) -> Result<crabinfer_core::ModelInfo, String> {
    use candle_core::quantized::gguf_file;

    let mut file = std::fs::File::open(model_path).map_err(|e| e.to_string())?;
    let file_size = file.metadata().map(|m| m.len()).unwrap_or(0);
    let content = gguf_file::Content::read(&mut file).map_err(|e| format!("{e}"))?;

    let md = &content.metadata;
    let architecture = md.get("general.architecture")
        .and_then(|v| v.to_string().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let model_name = md.get("general.name")
        .and_then(|v| v.to_string().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let context_length = md.get(&format!("{architecture}.context_length"))
        .and_then(|v| match v {
            gguf_file::Value::U32(n) => Some(*n),
            gguf_file::Value::U64(n) => Some(*n as u32),
            _ => None,
        })
        .unwrap_or(4096);

    let vocab_size = content.tensor_infos
        .get("token_embd.weight")
        .map(|t| t.shape.dims()[0] as u32)
        .unwrap_or(0);

    let parameter_count: u64 = content.tensor_infos.values()
        .map(|t| t.shape.elem_count() as u64)
        .sum();

    // Detect quantization
    let mut counts = std::collections::HashMap::new();
    for info in content.tensor_infos.values() {
        *counts.entry(format!("{:?}", info.ggml_dtype)).or_insert(0u32) += 1;
    }
    let quantization = counts.into_iter()
        .filter(|(k, _)| k != "F32" && k != "F16")
        .max_by_key(|(_, v)| *v)
        .map(|(k, _)| k)
        .unwrap_or_else(|| "unknown".to_string());

    Ok(crabinfer_core::ModelInfo {
        model_name,
        architecture,
        parameter_count,
        quantization,
        file_size_bytes: file_size,
        context_length,
        vocab_size,
    })
}

// ---------------------------------------------------------------------------
// Arg parsing
// ---------------------------------------------------------------------------

struct CliArgs {
    model: String,
    prompt: String,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    context_length: u32,
    cpu: bool,
    benchmark: bool,
    stream: bool,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = std::env::args().collect();
    let mut cli = CliArgs {
        model: String::new(),
        prompt: String::new(),
        max_tokens: 128,
        temperature: 0.7,
        top_p: 0.9,
        context_length: 4096,
        cpu: false,
        benchmark: false,
        stream: false,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                if i < args.len() { cli.model = args[i].clone(); }
            }
            "--prompt" => {
                i += 1;
                if i < args.len() { cli.prompt = args[i].clone(); }
            }
            "--max-tokens" => {
                i += 1;
                if i < args.len() { cli.max_tokens = args[i].parse().unwrap_or(128); }
            }
            "--temperature" => {
                i += 1;
                if i < args.len() { cli.temperature = args[i].parse().unwrap_or(0.7); }
            }
            "--top-p" => {
                i += 1;
                if i < args.len() { cli.top_p = args[i].parse().unwrap_or(0.9); }
            }
            "--context-length" => {
                i += 1;
                if i < args.len() { cli.context_length = args[i].parse().unwrap_or(4096); }
            }
            "--cpu" => {
                cli.cpu = true;
            }
            "--benchmark" => {
                cli.benchmark = true;
            }
            "--stream" => {
                cli.stream = true;
            }
            _ => {}
        }
        i += 1;
    }

    cli
}

fn format_params(count: u64) -> String {
    if count >= 1_000_000_000 {
        format!("{:.1}B", count as f64 / 1e9)
    } else if count >= 1_000_000 {
        format!("{:.0}M", count as f64 / 1e6)
    } else {
        format!("{count}")
    }
}
