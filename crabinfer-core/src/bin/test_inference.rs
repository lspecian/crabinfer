/// Quick integration test for GGUF loading and inference
/// Usage: cargo run --bin test_inference -- /path/to/model.gguf
use crabinfer_core::{CrabInferError, EngineConfig};
use crabinfer_core::engine::CrabInferEngine;

fn main() -> Result<(), CrabInferError> {
    tracing_subscriber::fmt::init();

    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        // Default path for local dev
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        format!("{manifest_dir}/../models/Phi-3-mini-4k-instruct-q4.gguf")
    });

    println!("=== CrabInfer Integration Test ===");
    println!("Model: {model_path}");
    println!();

    // Step 1: Create engine
    let config = EngineConfig {
        model_path: String::new(), // load manually below
        max_tokens: 50,
        temperature: 0.7,
        top_p: 0.9,
        context_length: 4096,
        use_metal: true, // Will auto-fallback to CPU if Metal ops unsupported
        memory_limit_bytes: 0, // auto-detect
        metallib_path: String::new(),
    };
    let engine = CrabInferEngine::new(config)?;
    println!("[OK] Engine created");

    // Step 2: Load model
    println!("Loading model (this may take a moment)...");
    engine.load_model(model_path)?;
    println!("[OK] Model loaded");

    // Step 3: Print model info
    let info = engine.model_info()?;
    println!();
    println!("--- Model Info ---");
    println!("  Name:         {}", info.model_name);
    println!("  Architecture: {}", info.architecture);
    println!("  Parameters:   {}", info.parameter_count);
    println!("  Quantization: {}", info.quantization);
    println!("  File size:    {} MB", info.file_size_bytes / (1024 * 1024));
    println!("  Context len:  {}", info.context_length);
    println!("  Vocab size:   {}", info.vocab_size);
    println!();

    // Step 4: Run inference
    let prompt = "Write a haiku about Rust programming:";
    println!("--- Inference ---");
    println!("Prompt: {prompt}");
    println!();

    let response = engine.complete(prompt.to_string(), 50, 0.7)?;
    println!("Response: {response}");
    println!();

    // Step 5: Print stats
    if let Some(stats) = engine.last_stats() {
        println!("--- Stats ---");
        println!("  Tokens generated:     {}", stats.tokens_generated);
        println!("  Tokens/sec:           {:.2}", stats.tokens_per_second);
        println!("  Time to first token:  {:.2} ms", stats.time_to_first_token_ms);
        println!("  Total time:           {:.2} ms", stats.total_time_ms);
        println!("  Peak memory:          {} MB", stats.peak_memory_bytes / (1024 * 1024));
        println!("  Backend:              {}", stats.compute_backend);
    }

    // Step 6: Cleanup
    engine.unload_model();
    println!();
    println!("[OK] Model unloaded");
    println!("=== Test Complete ===");

    Ok(())
}
