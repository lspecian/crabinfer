use crabinfer_core::engine::CrabInferEngine;
use crabinfer_core::EngineConfig;

pub fn run(
    model: &str,
    prompt: &str,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    context_length: u32,
    cpu: bool,
    stream: bool,
) {
    let config = EngineConfig {
        model_path: String::new(),
        max_tokens,
        temperature,
        top_p,
        context_length,
        use_metal: !cpu,
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

    eprintln!("Loading model: {model}");
    if let Err(e) = engine.load_model(model.to_string()) {
        eprintln!("Failed to load model: {e}");
        std::process::exit(1);
    }

    if let Ok(info) = engine.model_info() {
        eprintln!(
            "Model: {} ({}, {}, {} params, ctx {})",
            info.model_name,
            info.architecture,
            info.quantization,
            format_params(info.parameter_count),
            info.context_length,
        );
    }

    if stream {
        run_streaming(&engine, prompt, max_tokens);
    } else {
        run_normal(&engine, prompt, max_tokens, temperature);
    }
}

fn run_normal(engine: &CrabInferEngine, prompt: &str, max_tokens: u32, temperature: f32) {
    eprintln!("Generating...\n");
    match engine.complete(prompt.to_string(), max_tokens, temperature) {
        Ok(response) => println!("{response}"),
        Err(e) => {
            eprintln!("\nGeneration failed: {e}");
            std::process::exit(1);
        }
    }
    print_stats(engine);
}

fn run_streaming(engine: &CrabInferEngine, prompt: &str, max_tokens: u32) {
    use std::io::Write;

    eprintln!("Streaming...\n");
    let stdout = std::io::stdout();

    for _ in 0..max_tokens {
        match engine.next_token(prompt.to_string()) {
            Ok(Some(tok)) => {
                if tok.is_end_of_sequence {
                    break;
                }
                print!("{}", tok.text);
                stdout.lock().flush().ok();
            }
            Ok(None) => break,
            Err(e) => {
                eprintln!("\nStreaming failed: {e}");
                std::process::exit(1);
            }
        }
    }
    println!();
    print_stats(engine);
    engine.reset();
}

fn print_stats(engine: &CrabInferEngine) {
    if let Some(stats) = engine.last_stats() {
        eprintln!(
            "\n[{} tokens, {:.1} tok/s, TTFT {:.0}ms, total {:.0}ms, peak {} MB, {} backend]",
            stats.tokens_generated,
            stats.tokens_per_second,
            stats.time_to_first_token_ms,
            stats.total_time_ms,
            stats.peak_memory_bytes / (1024 * 1024),
            stats.compute_backend,
        );
    }
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
