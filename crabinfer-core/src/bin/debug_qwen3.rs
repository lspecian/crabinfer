/// Debug Qwen3-8B NaN logits issue
/// Tests: single token vs multi-token, Metal vs CPU
use crabinfer_core::{CrabInferError, EngineConfig};
use crabinfer_core::engine::CrabInferEngine;

fn test_with_config(label: &str, use_metal: bool, model_path: &str, prompt: &str) -> Result<(), CrabInferError> {
    println!("\n=== {} ===", label);

    let config = EngineConfig {
        model_path: String::new(),
        max_tokens: 10,
        temperature: 0.0, // greedy — avoids sampling issues
        top_p: 1.0,
        context_length: 4096,
        use_metal,
        memory_limit_bytes: 0,
        metallib_path: String::new(),
    };
    let engine = CrabInferEngine::new(config)?;
    engine.load_model(model_path.to_string())?;
    println!("[OK] Model loaded (metal={})", use_metal);

    let info = engine.model_info()?;
    println!("  Arch: {} | Quant: {}", info.architecture, info.quantization);

    println!("\nPrompt: {:?}", prompt);
    match engine.complete(prompt.to_string(), 10, 0.0) {
        Ok(resp) => println!("  Response: {:?}", resp),
        Err(e) => println!("  ERROR: {:?}", e),
    }

    engine.unload_model();
    Ok(())
}

fn main() -> Result<(), CrabInferError> {
    tracing_subscriber::fmt::init();

    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        let dir = env!("CARGO_MANIFEST_DIR");
        format!("{dir}/../models/Qwen3-8B-Q4_K_M.gguf")
    });

    println!("Model: {}", model_path);

    // Test around the threshold and beyond
    let prompts = [
        ("7 tok", "Tell me how to learn the Rust"),
        ("8 tok", "Write a haiku about Rust programming:"),
        ("9 tok", "Write a haiku about the Rust programming language"),
        ("16 tok", "Please write a very detailed and comprehensive essay about the history of the Rust programming language"),
    ];

    for (label, prompt) in &prompts {
        test_with_config(&format!("Metal {}", label), true, &model_path, prompt)?;
    }

    // Also test Phi3 with 8 tokens to confirm it's Qwen3-specific
    let phi3_path = {
        let dir = env!("CARGO_MANIFEST_DIR");
        format!("{dir}/../models/Phi-3-mini-4k-instruct-q4.gguf")
    };
    if std::path::Path::new(&phi3_path).exists() {
        test_with_config("Phi3 Metal (8 tok)", true, &phi3_path, "Write a haiku about Rust programming:")?;
    }

    Ok(())
}
