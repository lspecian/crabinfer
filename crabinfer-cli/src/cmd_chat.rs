use crabinfer_core::chat_template::apply_chat_template;
use crabinfer_core::engine::CrabInferEngine;
use crabinfer_core::provider::ChatMessage;
use crabinfer_core::EngineConfig;
use std::io::Write;

pub fn run(model: &str, max_tokens: u32, temperature: f32, context_length: u32, cpu: bool) {
    let config = EngineConfig {
        model_path: String::new(),
        max_tokens,
        temperature,
        top_p: 0.9,
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

    let architecture = engine
        .model_info()
        .map(|i| {
            eprintln!(
                "Model: {} ({}, {}, ctx {})",
                i.model_name, i.architecture, i.quantization, i.context_length,
            );
            i.architecture
        })
        .unwrap_or_else(|_| "unknown".to_string());

    eprintln!();
    eprintln!("Type a message to chat. Commands: /help, /reset, /quit");
    eprintln!();

    let mut rl = match rustyline::DefaultEditor::new() {
        Ok(rl) => rl,
        Err(e) => {
            eprintln!("Failed to initialize line editor: {e}");
            std::process::exit(1);
        }
    };

    let mut history: Vec<ChatMessage> = Vec::new();

    loop {
        let input = match rl.readline(">>> ") {
            Ok(line) => line,
            Err(rustyline::error::ReadlineError::Interrupted | rustyline::error::ReadlineError::Eof) => break,
            Err(e) => {
                eprintln!("Input error: {e}");
                break;
            }
        };

        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }

        match trimmed {
            "/quit" | "/exit" => break,
            "/reset" => {
                engine.reset();
                history.clear();
                eprintln!("[Conversation reset]");
                continue;
            }
            "/help" => {
                eprintln!("Commands:");
                eprintln!("  /reset  — Clear conversation history");
                eprintln!("  /quit   — Exit chat");
                eprintln!("  /exit   — Exit chat");
                eprintln!("  /help   — Show this help");
                continue;
            }
            _ => {}
        }

        let _ = rl.add_history_entry(trimmed);

        history.push(ChatMessage {
            role: "user".to_string(),
            content: trimmed.to_string(),
        });

        // Format the full conversation with the chat template
        let prompt = apply_chat_template(&architecture, &history);

        // Reset engine state for fresh generation from the full prompt
        engine.reset();

        // Stream the response
        let stdout = std::io::stdout();
        let mut response = String::new();

        for _ in 0..max_tokens {
            match engine.next_token(prompt.clone()) {
                Ok(Some(tok)) => {
                    if tok.is_end_of_sequence {
                        break;
                    }
                    print!("{}", tok.text);
                    stdout.lock().flush().ok();
                    response.push_str(&tok.text);
                }
                Ok(None) => break,
                Err(e) => {
                    eprintln!("\nGeneration error: {e}");
                    break;
                }
            }
        }
        println!();

        // Print stats
        if let Some(stats) = engine.last_stats() {
            eprintln!(
                "[{} tokens, {:.1} tok/s, TTFT {:.0}ms]",
                stats.tokens_generated, stats.tokens_per_second, stats.time_to_first_token_ms,
            );
        }
        eprintln!();

        // Add assistant response to history
        if !response.is_empty() {
            history.push(ChatMessage {
                role: "assistant".to_string(),
                content: response,
            });
        }
    }
}
